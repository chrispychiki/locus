"""The retention horizon, applied to the local db.

The deployment declares one horizon (`retention_days` in store/deploy.config.toml) and it governs every copy of a recording. The bucket's copy is R2's own lifecycle rule, converged by the store's provision. This module is the same horizon over the other copy: `data/events.db` and the replay material derived beside it, which would otherwise keep forever what `locus load` ever pulled — a promise of "recordings are kept N days" that the local copy quietly broke.

A recording's age is its slice's, and a slice's moment is the one the store partitions by: the millisecond the recorder minted into the slice id, as a UTC date (store/src/keys.js derives the object key's date part from exactly that). So one predicate decides both ends of the load — an object past the horizon is never fetched, and a slice past it is dropped — and the two can never disagree about what "past" means.

Expiry drops a slice whole: the slices row, its events, the recorder's error log for it, and the manifest rows for the objects that carried it. Whole, because a slice is the unit that replays standalone — half a recording is an artifact that still claims to be one. Sessions are the one derivation that spans slices, so a visitor who lost anything has their sessions re-derived from what remains, leaving the db current rather than dirty. The replay material under `pages/` is a cache that regenerates on the next open, so an expiry clears it rather than reasoning about which payload embeds which slice.

Analyses (`data/analyses/`) are not recordings — they are the operator's findings, and they outlive what they cite. A citation into an expired recording then fails to open, which is the honest outcome: the finding stands, the evidence behind it is gone at the horizon like every other copy.

Deletion is done with SQLite's secure_delete on, so the freed pages are zeroed rather than left readable in the file. The file itself does not shrink — the space is reused by later loads.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import tomllib

from .chunk import parse_chunk_key, slice_date
from .speak import beating

CONFIG_FILE = "store/deploy.config.toml"

# Rows per DELETE ... IN (...): SQLite's parameter ceiling is what caps it.
_BATCH = 500


def horizon_days(root: Path) -> int:
    """The declared horizon, from the one declaration the store's provision and doctor read (CONFIG_FILE). Validated here because the number decides deletions: anything but a positive whole number of days is refused by name rather than silently floored into one."""
    path = root / CONFIG_FILE
    if not path.exists():
        raise SystemExit(
            f"no {path}: the retention horizon is declared there and the "
            f"deploy reads the same file — restore it"
        )
    declared = tomllib.loads(path.read_text()).get("retention_days")
    if not isinstance(declared, int) or isinstance(declared, bool) or declared < 1:
        raise SystemExit(
            f"{path}: retention_days must be a whole number of days, at least "
            f"1, not {declared!r}"
        )
    return declared


def cutoff_date(days: int, now: datetime | None = None) -> str:
    """The horizon as a UTC date, `YYYY-MM-DD`: a recording whose open date falls before it is past the horizon, one whose date is it or later is kept."""
    moment = now or datetime.now(timezone.utc)
    return (moment.astimezone(timezone.utc).date() - timedelta(days=days)).isoformat()


def horizon_cutoff(root: Path, now: datetime | None = None) -> str:
    """The deployment's own cutoff date — the declared horizon read and turned into the date both the load's intake and the sweep compare against."""
    return cutoff_date(horizon_days(root), now)


def past_horizon(slice_id: str, cutoff: str) -> bool:
    """Whether a recording is past the horizon: its open date against the cutoff — the one comparison every caller makes, so the store's objects, the db's slices, and a listing all place the same recording on the same side."""
    return slice_date(slice_id) < cutoff


def expired_key(key: str, cutoff: str) -> bool:
    """Whether a store object holds a recording past the horizon, read from the slice id its key carries. A key that is not a chunk key dates to nothing and is never called expired — the bucket is open-write, and guessing an age for something the layout does not describe would delete on a guess."""
    parsed = parse_chunk_key(key)
    return parsed is not None and past_horizon(parsed["slice_id"], cutoff)


def _expired_slice(row, cutoff: str) -> bool:
    """Whether a slice's recording is past the horizon, dated exactly as the store dates the objects that carried it: from the id the recorder minted. Every chunk that hydrates carries a well-formed slice id (the decode gate in chunk.py refuses one that does not), so a slice without one is a db no sweep can date — said rather than deleted on a guess."""
    if not row["recorder_slice"]:
        raise SystemExit(
            f"slice {row['id']} (visitor {row['visitor_id']}) carries no "
            f"recorder slice id, so nothing can date it against the retention "
            f"horizon — every loaded chunk stamps one, so this db did not come "
            f"from a load"
        )
    return past_horizon(row["recorder_slice"], cutoff)


def _batched(values: list):
    for start in range(0, len(values), _BATCH):
        yield values[start : start + _BATCH]


def expire(
    conn: sqlite3.Connection,
    cutoff: str,
    definition: dict,
    pages: Path | None = None,
) -> dict:
    """Drop every recording in the db that opened before `cutoff`, and leave what remains current. Returns what went: the slices and events dropped, the visitors touched, the chunk manifest rows released, and whether the replay cache was cleared. Idempotent — a second run over the same db finds nothing left to drop.

    The sweep commits in batches of recordings, each batch its own transaction and its visitors' sessions re-derived inside it, because a corpus-scale sweep is millions of rows: one transaction would hold the whole deletion in the WAL and leave nothing durable if it died. So every commit point is a db with whole recordings gone and everything derived from them current, and a killed sweep resumes by simply running again."""
    from .sessions import materialize_sessions

    expired = [
        row
        for row in conn.execute(
            "SELECT id, visitor_id, recorder_slice, absorbed FROM slices"
        )
        if _expired_slice(row, cutoff)
    ]
    stale_chunks = [
        row["key"]
        for row in conn.execute("SELECT key FROM loaded_chunks")
        if expired_key(row["key"], cutoff)
    ]
    if not expired and not stale_chunks:
        return {
            "cutoff": cutoff,
            "slices": 0,
            "events": 0,
            "visitors": 0,
            "chunks": 0,
            "pages_cleared": False,
        }

    visitors = {row["visitor_id"] for row in expired}
    events = 0
    done = 0
    conn.execute("PRAGMA secure_delete=ON")
    try:
        with beating(
            "expiring recordings past the horizon",
            lambda: f"{done}/{len(expired)} slices, {events} events",
        ):
            for batch in _batched(expired):
                ids = [row["id"] for row in batch]
                marks = ",".join("?" * len(ids))
                events += conn.execute(
                    f"DELETE FROM events WHERE slice_id IN ({marks})", ids
                ).rowcount
                conn.execute(f"DELETE FROM slices WHERE id IN ({marks})", ids)
                # A rescued slice kept its own row after its events were appended onto the slice
                # that absorbed it, and its own recorder id is what its error rows are keyed by —
                # so the recording takes both with it rather than leaving a row standing over
                # events that are gone.
                absorbed = [
                    (row["visitor_id"], name)
                    for row in batch
                    for name in json.loads(row["absorbed"] or "[]")
                ]
                conn.executemany(
                    "DELETE FROM slices WHERE visitor_id = ? AND recorder_slice = ?",
                    absorbed,
                )
                conn.executemany(
                    "DELETE FROM chunk_errors WHERE visitor_id = ? AND recorder_slice = ?",
                    [
                        *absorbed,
                        *((r["visitor_id"], r["recorder_slice"]) for r in batch),
                    ],
                )
                conn.commit()
                # Sessions span a visitor's slices, so a visitor who lost one is re-derived from
                # what is left, in the same breath as the deletion: the db never stands with
                # sessions grounded in events that are gone, not even between batches.
                for visitor_id in sorted({row["visitor_id"] for row in batch}):
                    materialize_sessions(conn, visitor_id, definition)
                done += len(batch)
            for batch in _batched(stale_chunks):
                conn.execute(
                    f"DELETE FROM loaded_chunks WHERE key IN ({','.join('?' * len(batch))})",
                    batch,
                )
                conn.commit()
    finally:
        conn.execute("PRAGMA secure_delete=OFF")

    cleared = False
    if pages is not None and pages.exists():
        for path in pages.iterdir():
            if path.is_file():
                path.unlink()
                cleared = True
    return {
        "cutoff": cutoff,
        "slices": len(expired),
        "events": events,
        "visitors": len(visitors),
        "chunks": len(stale_chunks),
        "pages_cleared": cleared,
    }
