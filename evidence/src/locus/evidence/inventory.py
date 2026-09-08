"""The store's exact plane, listed: every slice in the bucket, one row each — what `locus ls` reads."""

from collections.abc import Iterator

from .chunk import parse_chunk_key
from .retention import past_horizon
from .slices import slice_open_ms


class Inventory:
    """One LIST under an optional key prefix → every slice in the store's exact plane, one row each, handed over as the listing finds them. The prefix is the whole query language, because it is the store's own: any leading substring of the key scopes the LIST to whatever depth it reaches (`snippetid/`, `snippetid/2026-06-25/`, …); empty lists the whole store, and cost is legible in the asking — a date-partition prefix is cheap by inspection, a bare snippet prefix is visibly a full scan. Any scope that is not a prefix is the caller's own filter over the rows, which carry every field it needs. Each chunk key (parse_chunk_key) folds into its slice, carrying its snippet, visitor, chunk count, bytes, the slice's open time, and the window in which its chunks actually landed; a local map {visitor: {recorder_slice}} marks each slice loaded or new, and the deployment's retention cutoff marks the ones past the horizon — still in the bucket until R2's own lifecycle pass reaches them, but never loadable, so they are neither loaded nor new. Objects that are not chunks are counted as foreign rather than dropped.

    Every slice is produced, at every scale. This is the exact plane, and it is the arrivals side of every completeness question — an arrival only answers a birth if the two can be joined, and they join on `(snippet, visitor_id, slice_id)`, which a rollup destroys.

    A row is a fold over its slice's chunk keys, closed the moment its shard reaches a different slice. The fold's precondition — one shard holds all of a slice's keys, contiguously — is the store's key layout (every chunk files under its slice's start date, store/src/keys.js) times the shard's own lexicographic order; a slice seen again after its row closed breaks it and fails the listing loud. Rows arrive in discovery order; the reader that wants an order applies it. `totals` is complete once iteration is."""

    def __init__(
        self,
        store,
        local: dict[str, set[str]] | None = None,
        prefix: str = "",
        cutoff: str | None = None,
    ):
        self.store = store
        self.local = local or {}
        self.prefix = prefix
        self.cutoff = cutoff
        self.totals = self._empty()

    @staticmethod
    def _empty() -> dict:
        return {
            "visitors": 0,
            "slices": 0,
            "new": 0,
            "expired": 0,
            "chunks": 0,
            "bytes": 0,
            "first_ms": None,
            "last_ms": None,
            "foreign_keys": 0,
            "foreign_bytes": 0,
        }

    def __iter__(self) -> Iterator[dict]:
        self.totals = totals = self._empty()
        visitors: set[str] = set()
        standing: dict[str, tuple] = {}
        emitted: set[tuple] = set()

        def close(row: dict) -> dict:
            ident = (row["visitor_id"], row["slice_id"])
            if ident in emitted:
                raise RuntimeError(
                    f"slice {row['slice_id']} of visitor {row['visitor_id']} "
                    f"surfaced again after its row was emitted — its chunk "
                    f"keys span more than one listing shard, so the store's "
                    f"key layout (store/src/keys.js) no longer files a slice "
                    f"under one date partition"
                )
            emitted.add(ident)
            visitors.add(row["visitor_id"])
            totals["visitors"] = len(visitors)
            totals["slices"] += 1
            totals["expired"] += 1 if row["expired"] else 0
            totals["new"] += 0 if row["loaded"] or row["expired"] else 1
            totals["chunks"] += row["chunks"]
            totals["bytes"] += row["bytes"]
            first, last = totals["first_ms"], totals["last_ms"]
            totals["first_ms"] = (
                row["open_ms"] if first is None else min(first, row["open_ms"])
            )
            totals["last_ms"] = (
                row["open_ms"] if last is None else max(last, row["open_ms"])
            )
            return row

        for shard, (key, size, uploaded_ms, _etag) in self.store.shard_rows(
            self.prefix
        ):
            parsed = parse_chunk_key(key)
            if parsed is None:
                totals["foreign_keys"] += 1
                totals["foreign_bytes"] += size
                continue
            visitor_id, slice_id = parsed["visitor_id"], parsed["slice_id"]
            held = standing.get(shard)
            if held is not None and held[0] != (visitor_id, slice_id):
                yield close(held[1])
                held = None
            if held is None:
                row = {
                    "snippet": parsed["snippet"],
                    "date": parsed["date"],
                    "visitor_id": visitor_id,
                    "slice_id": slice_id,
                    "open_ms": slice_open_ms(slice_id),
                    "chunks": 0,
                    "bytes": 0,
                    # When the slice's chunks actually landed. A slice is never closed —
                    # a device can deliver into it days later — so the last upload is the
                    # freshest evidence about it, not a completion mark.
                    "first_upload_ms": uploaded_ms,
                    "last_upload_ms": uploaded_ms,
                    "loaded": slice_id in self.local.get(visitor_id, set()),
                    # Still in the bucket, past the horizon: no load will take it, so
                    # counting it as new would send a reader after something that is
                    # never coming, every listing, until R2's own lifecycle pass.
                    "expired": self.cutoff is not None
                    and past_horizon(slice_id, self.cutoff),
                }
                standing[shard] = ((visitor_id, slice_id), row)
            else:
                row = held[1]
            row["chunks"] += 1
            row["bytes"] += size
            row["first_upload_ms"] = min(row["first_upload_ms"], uploaded_ms)
            row["last_upload_ms"] = max(row["last_upload_ms"], uploaded_ms)

        for _ident, row in standing.values():
            yield close(row)
