"""Hydration — raw-first.

Lands raw rrweb events into events.db as the source of truth. Duplicate content is ignored via the content hash. The canonical stream is read back in db.py's CANONICAL_ORDER, which owns the order and its rationale.

raw_json is stored zlib-compressed (pack_raw/raw_event, with raw.js the bun-side twin), which puts the raw plane structurally behind the readers that know how to parse rrweb, the same way the numeric→name boundary is structural: the queryable plane is the flat columns, and a bare SELECT of raw_json returns bytes. content_hash is always over the uncompressed canonical bytes; the compressed form is storage, never identity.

The chunk decode (chunk.py) attaches an "_envelope" key (the device facts — device/os/browser/language/time_zone/screen_width/screen_height — plus script_version/recorder_slice/snippet: transport metadata, not rrweb content); it is popped into its own columns before canonicalization, so content hashes and raw_json stay pure rrweb. script_version is the producing recorder's version (the chunk payload's recorderVersion). recorder_slice is the slice identity the recorder stamped, carried on the chunk's sliceId — every event has one, and slice materialization keys slices by it. snippet is the site the events were recorded from — the chunk payload never carries it, so it rides in from the store key's leading part.
"""

import hashlib
import json
import sqlite3
import time
import zlib
from collections.abc import Iterable

from .db import CANONICAL_ORDER
from .rrweb_constants import EventType, IncrementalSource


def _canonical_json(event: dict) -> str:
    """The canonical form is frozen identity: content_hash is the SHA-256 of exactly these bytes and the dedup key every load leans on, so a change here — key order, separators, escaping — makes the same wire event hash as new and a later load duplicate it silently into a db that reads healthy. Pinned byte-for-byte in tests; a change is a rebuild of the db, never a heal.

    The default ensure_ascii=True is load-bearing: recorded text can carry lone surrogates (real, unpaired code points after json.loads of the wire's \\uXXXX escapes), and escaping them back keeps the canonical form pure ASCII — faithful to the recording, and encodable, which the hash and the stored bytes both require. Distillation sanitizes its flat columns separately (distill_worker.js); the canonical form never does."""
    return json.dumps(event, separators=(",", ":"), sort_keys=True)


def pack_raw(canonical: str) -> bytes:
    """The storage form of a raw_json value: the canonical bytes, zlib level 6 (~5× on real recordings; zlib because it is stdlib here and native in bun — raw.js is the read twin). Compression is storage only — content_hash is taken over the canonical bytes before this."""
    return zlib.compress(canonical.encode(), 6)


def raw_event(stored: bytes) -> dict:
    """A stored raw_json value back as the event dict — the one Python-side reader of the compressed raw plane; every code path that needs the raw event goes through here."""
    return json.loads(zlib.decompress(stored))


_MOVE_SOURCES = {
    IncrementalSource.MouseMove,
    IncrementalSource.TouchMove,
    IncrementalSource.Drag,
}


def canonical_ts(event: dict) -> int:
    """The event's place on the canonical clock: a move batch lands at its last position's moment (timestamp + timeOffset), every other event at its own timestamp. A derived value — the timestamp column re-derives from raw_json through this whenever the deriving code changes (derive.py owns that currency)."""
    data = event.get("data") or {}
    if (
        event.get("type") == EventType.IncrementalSnapshot
        and data.get("source") in _MOVE_SOURCES
    ):
        positions = data.get("positions") or []
        if positions:
            return event["timestamp"] + positions[-1]["timeOffset"]
    return event["timestamp"]


def event_skip_reason(event) -> str | None:
    """Why an event cannot land as a canonical row — None when it can. Checks exactly the shape row construction and canonical ordering touch: object shape, integer type, numeric timestamp, string counter, object data, and a numeric timeOffset on the last move position (it decides the canonical timestamp). The recorder stamps a counter on every event it emits, so an event without one is damaged: without the counter, same-millisecond order would be whichever chunk GET happened to return first. The store read path gates on this at decode, where the loss is still nameable by its chunk; hydration itself trusts its callers and checks nothing."""
    if not isinstance(event, dict):
        return "not an object"
    if type(event.get("type")) is not int:
        return "type is not an integer"
    if type(event.get("timestamp")) not in (int, float):
        return "timestamp is not a number"
    counter = event.get("counter")
    if counter is None:
        return "no counter"
    if not isinstance(counter, str):
        return "counter is not a string"
    data = event.get("data")
    if not isinstance(data, dict):
        return "data is not an object"
    if (
        event["type"] == EventType.IncrementalSnapshot
        and data.get("source") in _MOVE_SOURCES
    ):
        positions = data.get("positions")
        if positions is not None and not isinstance(positions, list):
            return "positions is not a list"
        if positions and not (
            isinstance(positions[-1], dict)
            and type(positions[-1].get("timeOffset")) in (int, float)
        ):
            return "last move position has no numeric timeOffset"
    return None


INSERT_SQL = (
    "INSERT OR IGNORE INTO events "
    "(visitor_id, timestamp, type, counter, raw_json, content_hash, "
    " device, os, browser, language, time_zone, screen_width, screen_height, "
    " script_version, recorder_slice, snippet) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _row(visitor_id: str, event: dict) -> tuple:
    envelope = event.get("_envelope") or {}
    if "_envelope" in event:
        event = {k: v for k, v in event.items() if k != "_envelope"}
    canonical = _canonical_json(event)
    content_hash = hashlib.sha256(canonical.encode()).hexdigest()
    return (
        visitor_id,
        canonical_ts(event),
        event["type"],
        event.get("counter"),
        pack_raw(canonical),
        content_hash,
        envelope.get("device"),
        envelope.get("os"),
        envelope.get("browser"),
        envelope.get("language"),
        envelope.get("time_zone"),
        envelope.get("screen_width"),
        envelope.get("screen_height"),
        envelope.get("script_version"),
        envelope.get("recorder_slice"),
        envelope.get("snippet"),
    )


def hydrate(conn: sqlite3.Connection, visitor_id: str, events: Iterable[dict]) -> int:
    return hydrate_stream(conn, ((visitor_id, event) for event in events))


def hydrate_stream(
    conn: sqlite3.Connection,
    pairs: Iterable[tuple[str, dict]],
    batch_size: int = 5000,
    on_flush=None,
    stats=None,
) -> int:
    """Land the stream in committed batches: every flush is durable, so a load killed mid-stream keeps everything flushed so far and loses only the in-flight batch — the WAL stays bounded at a batch instead of growing the whole load, and resuming is nothing special, just loading again, which content-hash dedup and the caller's chunk manifest make cheap. on_flush, when given, runs inside each flush after the batch's insert and before its commit — the caller's chance to land rows that must be durable only alongside the events they describe (the load's chunk manifest rides here: a chunk's manifest row commits atomically with a batch that already holds all its events). stats, a LoadStats when given, takes each flush's insert+commit time as insert_s."""
    inserted = 0
    batch = []

    def flush() -> None:
        nonlocal inserted
        t0 = time.perf_counter()
        if batch:
            inserted += conn.executemany(INSERT_SQL, batch).rowcount
            batch.clear()
        if on_flush is not None:
            on_flush()
        conn.commit()
        if stats is not None:
            stats.add(insert_s=time.perf_counter() - t0)

    for visitor_id, event in pairs:
        batch.append(_row(visitor_id, event))
        if len(batch) >= batch_size:
            flush()
    flush()
    return inserted


def record_chunk_errors(conn: sqlite3.Connection, rows: Iterable[tuple]) -> int:
    """Land (snippet, visitor_id, recorder_slice, chunk_key, error) rows into chunk_errors — the recorder's shipped error log, deduped like events so re-loads are free. Rides the caller's open transaction: the load lands these inside each hydration flush, so they commit atomically with the batch that carried them."""
    return conn.executemany(
        "INSERT OR IGNORE INTO chunk_errors "
        "(snippet, visitor_id, recorder_slice, chunk_key, error) "
        "VALUES (?, ?, ?, ?, ?)",
        list(rows),
    ).rowcount


def canonical_events(conn: sqlite3.Connection, visitor_id: str) -> list[dict]:
    rows = conn.execute(
        f"SELECT raw_json FROM events WHERE visitor_id = ? {CANONICAL_ORDER}",
        (visitor_id,),
    ).fetchall()
    return [raw_event(row["raw_json"]) for row in rows]
