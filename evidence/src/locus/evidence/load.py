"""One incremental load: every chunk under a prefix streamed from the store into the db, with its own accounting.

load_chunks is the whole store→db chain — every consumer comes through it; `locus load` adds only narration and the derivation pass on top. LoadStats is the load's own accounting and heartbeat.
"""

import threading
import time

from .hydrate import hydrate_stream, record_chunk_errors
from .retention import expired_key
from .speak import beating
from .store import GET_CONCURRENCY


class LoadStats:
    """A load's own accounting: where the time went — network round-trips vs decode vs the db — is the fact that decides how a slow load gets attacked. Counters accrue from the GET worker threads (get_s/decode_s are thread-seconds — compare against wall × window, not wall) and from hydration's flushes (insert_s); final() is the one-line breakdown for the end of the load. beating() is the load's pulse, fed by those counters."""

    def __init__(self, every_s: float | None = None):
        self._lock = threading.Lock()
        self._every = every_s
        self._started = time.perf_counter()
        self.counts = {
            "chunks": 0,
            "skipped": 0,
            "expired": 0,
            "corrupt": 0,
            "lossy": 0,
            "events": 0,
            "bytes": 0,
            "get_s": 0.0,
            "decode_s": 0.0,
            "insert_s": 0.0,
        }

    def add(self, **fields) -> None:
        with self._lock:
            for name, value in fields.items():
                self.counts[name] += value

    def beating(self):
        return beating("load", self.progress, self._every)

    def unclean(self) -> str:
        """The chunks this load could not take whole — corrupt (nothing decoded) or lossy (some events malformed) — which the manifest never records, so every later load fetches them again. Empty when there were none."""
        c = self.counts
        parts = []
        if c["corrupt"]:
            parts.append(f"{c['corrupt']} corrupt")
        if c["lossy"]:
            parts.append(f"{c['lossy']} lossy")
        return f"{' and '.join(parts)} skipped" if parts else ""

    def _tail(self) -> str:
        c = self.counts
        expired = f", {c['expired']} past the horizon" if c["expired"] else ""
        unclean = self.unclean()
        return f"{expired}, {unclean}" if unclean else expired

    def progress(self) -> str:
        c = self.counts
        return (
            f"{c['chunks']} chunks in ({c['bytes'] / 1e6:.1f} MB), "
            f"{c['skipped']} already held{self._tail()}, {c['events']} events"
        )

    def final(self) -> str:
        c = self.counts
        wall = time.perf_counter() - self._started
        return (
            f"load {wall:.0f}s wall: {c['chunks']} chunks "
            f"({c['bytes'] / 1e6:.1f} MB), {c['skipped']} already held{self._tail()}; "
            f"GET {c['get_s']:.0f} thread-s across a window of "
            f"{GET_CONCURRENCY}, decode {c['decode_s']:.1f}s, "
            f"db {c['insert_s']:.1f}s"
        )


def load_chunks(
    conn,
    store,
    prefix: str = "",
    stats: LoadStats | None = None,
    cutoff: str | None = None,
) -> dict:
    """One incremental load, whole: every chunk under the prefix streamed through events_under into the db, with the load's own state landing atomically alongside — the chunk manifest (`loaded_chunks`, keyed by ETag so an unchanged object never re-crosses the network) and the recorder's shipped error log (`chunk_errors`).

    `cutoff`, the deployment's retention horizon as a date (retention.horizon_cutoff), holds the intake to what the horizon keeps: an object carrying an older recording is passed over unfetched, so the sweep never has to undo a load. Without one the store is read whole — the horizon is the caller's to declare.

    The manifest and error rows ride hydration's flushes (on_flush runs inside each batch's transaction), so a load killed mid-stream keeps every flushed batch consistent with its manifest and loses only the in-flight batch. Returns the load's facts: events inserted, chunks fetched, chunks skipped as already held."""
    stats = stats if stats is not None else LoadStats()
    error_rows: list[tuple] = []
    fetched: list[tuple[str, str, int]] = []
    counts = {"fetched": 0}

    manifest = {
        row["key"]: row["etag"]
        for row in conn.execute(
            "SELECT key, etag FROM loaded_chunks WHERE key LIKE ?", (prefix + "%",)
        )
    }

    def on_errors(key, payload):
        for error in payload["errors"]:
            error_rows.append(
                (
                    key.split("/")[0],
                    payload["visitorId"],
                    payload.get("sliceId"),
                    key.split("/")[-1].removesuffix(".json.gz"),
                    str(error),
                )
            )

    def flush_load_state():
        # Runs inside each hydration flush, before its commit, so everything here lands atomically
        # with a batch that already holds the described events. Errors first is not ordering —
        # the transaction is one — just accounting: only cleanly-decoded chunks reach `fetched`,
        # so a manifest row never outruns its events, and a load that dies mid-stream keeps every
        # flushed batch with its manifest rows; the derivation pass finishes their derivations
        # on any later load, and the manifest spares that load the re-fetch.
        if error_rows:
            record_chunk_errors(conn, error_rows)
            error_rows.clear()
        if fetched:
            counts["fetched"] += len(fetched)
            conn.executemany(
                "INSERT INTO loaded_chunks (key, etag, uploaded_ms) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET etag = excluded.etag, "
                "uploaded_ms = excluded.uploaded_ms",
                fetched,
            )
            fetched.clear()

    events = store.events_under(
        prefix,
        on_errors=on_errors,
        loaded=lambda key, etag: manifest.get(key) == etag,
        on_fetched=lambda key, etag, uploaded_ms: fetched.append(
            (key, etag, uploaded_ms)
        ),
        expired=(None if cutoff is None else lambda key: expired_key(key, cutoff)),
        stats=stats,
    )
    with stats.beating():
        inserted = hydrate_stream(conn, events, on_flush=flush_load_state, stats=stats)
    return {
        "inserted": inserted,
        "fetched_chunks": counts["fetched"],
        "skipped_chunks": stats.counts["skipped"],
        "expired_chunks": stats.counts["expired"],
        "stats": stats,
    }
