import hashlib
import zlib

import pytest
from locus.evidence.db import connect
from locus.evidence.hydrate import (
    _canonical_json,
    canonical_events,
    event_skip_reason,
    hydrate,
    hydrate_stream,
    record_chunk_errors,
)
from locus.evidence.rrweb_constants import (
    EventType,
    IncrementalSource,
    MouseInteractions,
    NodeType,
)

ENVELOPE = {
    "device": "mobile",
    "os": "iOS",
    "browser": "Mobile Safari",
    "script_version": "locus-recorder/0.1.0",
    "recorder_slice": "00001749600000000-ab12",
    "snippet": "snip01",
}


def scroll(ts, y=10):
    return {
        "type": EventType.IncrementalSnapshot,
        "timestamp": ts,
        "data": {"source": IncrementalSource.Scroll, "id": 1, "x": 0, "y": y},
    }


def sample_events():
    """A tiny event stream exercising what hydrate must see through: an event
    arriving out of timestamp order and an exact duplicate."""
    return [
        {
            "type": EventType.Meta,
            "timestamp": 1000,
            "data": {"href": "https://example.com/", "width": 1280, "height": 800},
        },
        {
            "type": EventType.FullSnapshot,
            "timestamp": 1000,
            "data": {
                "node": {"type": NodeType.Document, "id": 1, "childNodes": []},
                "initialOffset": {"left": 0, "top": 0},
            },
        },
        {
            "type": EventType.IncrementalSnapshot,
            "timestamp": 1200,
            "data": {
                "source": IncrementalSource.MouseInteraction,
                "type": MouseInteractions.MouseDown,
                "id": 5,
                "x": 10,
                "y": 20,
            },
        },
        scroll(1100, y=300),
        scroll(1100, y=300),
    ]


def test_the_live_schema_says_what_its_columns_hold(tmp_path):
    # An agent querying this db reads `.schema` and nothing else, so a column whose name
    # does not say what it holds has to say it there. SQLite keeps the comments verbatim
    # in sqlite_master, which is the only reason that works — a reformat that drops them
    # takes the whole self-description with it, silently.
    conn = connect(tmp_path / "events.db")
    schema = "\n".join(
        row["sql"]
        for row in conn.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL")
    )
    for opaque in (
        "extra",
        "hidden",
        "page_url",
        "md",
        "diff",
        "counter",
        "absorbed",
        "uploaded_ms",
    ):
        line = next(l for l in schema.splitlines() if l.strip().startswith(opaque))
        assert "--" in line, f"{opaque} is named for nothing and explains nothing"


def test_hydrate_dedups_and_canonicalizes(tmp_path):
    conn = connect(tmp_path / "events.db")
    inserted = hydrate(conn, "v1", sample_events())
    assert inserted == 4

    canon = canonical_events(conn, "v1")
    assert [(e["type"], e["timestamp"]) for e in canon] == [
        (EventType.Meta, 1000),
        (EventType.FullSnapshot, 1000),
        (EventType.IncrementalSnapshot, 1100),
        (EventType.IncrementalSnapshot, 1200),
    ]


def test_hydrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "events.db")
    hydrate(conn, "v1", sample_events())
    reinserted = hydrate(conn, "v1", sample_events())
    assert reinserted == 0
    assert len(canonical_events(conn, "v1")) == 4


# The envelope: transport metadata the rrweb event cannot carry. It rides in on the store read
# path's "_envelope" key and must land in columns — never in raw_json, which is the source of
# truth and stays pure rrweb.


def test_every_envelope_fact_lands_in_its_own_column(tmp_path):
    conn = connect(tmp_path / "events.db")
    hydrate(conn, "v1", [{**scroll(1000), "_envelope": ENVELOPE}])
    row = conn.execute(
        "SELECT device, os, browser, script_version, recorder_slice, snippet FROM events"
    ).fetchone()
    assert dict(row) == ENVELOPE


def test_raw_json_is_pure_rrweb_and_so_is_the_hash(tmp_path):
    # The envelope is transport, not recording: it must not enter raw_json, or the source of
    # truth stops being the thing the browser emitted.
    conn = connect(tmp_path / "events.db")
    hydrate(conn, "v1", [{**scroll(1000), "_envelope": ENVELOPE}])
    events = canonical_events(conn, "v1")
    assert "_envelope" not in events[0]
    assert events[0] == scroll(1000)


def test_dedup_is_content_only_so_a_reload_never_rewrites_an_envelope(tmp_path):
    # The content hash covers the rrweb event alone, so the same event arriving under a
    # different envelope is the same row — the first envelope stands. This is what makes
    # re-loading free, and it is also why a deployment that changes what its envelope carries
    # must rebuild the db rather than re-load into it: the columns will not heal on their own.
    conn = connect(tmp_path / "events.db")
    hydrate(conn, "v1", [{**scroll(1000), "_envelope": {**ENVELOPE, "snippet": None}}])
    reinserted = hydrate(conn, "v1", [{**scroll(1000), "_envelope": ENVELOPE}])

    assert reinserted == 0
    rows = conn.execute("SELECT snippet FROM events").fetchall()
    assert [row["snippet"] for row in rows] == [None]


def test_the_canonical_form_is_frozen_dedup_identity():
    # content_hash is the SHA-256 over exactly these bytes, and the dedup identity every load
    # leans on: change the canonical form — key order, separators, escaping — and the same wire
    # event hashes as new, so a re-load duplicates every event silently into a db that reads
    # healthy. This pin failing means dedup identity broke against every standing db; the fix is
    # a rebuild of the db, never an accommodation in the canonicalization.
    event = {
        "type": EventType.IncrementalSnapshot,
        "timestamp": 1_700_000_000_000,
        "counter": "000007",
        "data": {
            "source": IncrementalSource.MouseMove,
            "positions": [{"x": 1.5, "y": 2, "id": 7, "timeOffset": -42}],
            "text": "café \ud83d",
        },
    }
    canonical = _canonical_json(event)
    assert canonical == (
        '{"counter":"000007","data":{"positions":[{"id":7,"timeOffset":-42,'
        '"x":1.5,"y":2}],'
        f'"source":{int(IncrementalSource.MouseMove)},'
        '"text":"caf\\u00e9 \\ud83d"},"timestamp":1700000000000,'
        f'"type":{int(EventType.IncrementalSnapshot)}}}'
    )
    assert hashlib.sha256(canonical.encode()).hexdigest() == (
        "a1246b8727cc351c742ee880d85f9d68988f1b1848b1e4a1806cde3bf96286bf"
    )


def test_raw_json_is_stored_zlib_compressed_and_identity_stays_uncompressed(tmp_path):
    # Storage is compressed; identity is not. The stored blob must decompress to exactly the
    # canonical bytes the content_hash was taken over — zlib format (RFC 1950), which the
    # bun-side reader (distill/raw.js, node:zlib) expects byte-for-byte.
    conn = connect(tmp_path / "events.db")
    hydrate(conn, "v1", [scroll(1000)])
    row = conn.execute("SELECT raw_json, content_hash FROM events").fetchone()
    assert isinstance(row["raw_json"], bytes)
    canonical = zlib.decompress(row["raw_json"])
    assert canonical == _canonical_json(scroll(1000)).encode()
    assert hashlib.sha256(canonical).hexdigest() == row["content_hash"]


def test_dedup_is_scoped_to_the_visitor(tmp_path):
    # Two visitors doing the identical thing at the identical millisecond testify separately:
    # identical bytes under different visitors are two events, never one.
    conn = connect(tmp_path / "events.db")
    inserted = hydrate_stream(conn, [("v1", scroll(1000)), ("v2", scroll(1000))])
    assert inserted == 2
    assert len(canonical_events(conn, "v1")) == 1
    assert len(canonical_events(conn, "v2")) == 1


def test_a_move_batch_is_timed_at_its_last_position_not_its_flush(tmp_path):
    # rrweb buffers motion and flushes it later with negative offsets, so the flush timestamp
    # is the batch's end, not its span. Ordering on the flush alone would file motion that
    # happened seconds ago as if it were the newest thing in the stream.
    move = {
        "type": EventType.IncrementalSnapshot,
        "timestamp": 5_000,
        "data": {
            "source": IncrementalSource.MouseMove,
            "positions": [
                {"x": 1, "y": 1, "id": 1, "timeOffset": -400},
                {"x": 2, "y": 2, "id": 1, "timeOffset": -100},
            ],
        },
    }
    conn = connect(tmp_path / "events.db")
    hydrate(conn, "v1", [move])
    row = conn.execute("SELECT timestamp FROM events").fetchone()
    assert row["timestamp"] == 4_900
    assert canonical_events(conn, "v1")[0]["timestamp"] == 5_000, (
        "raw_json keeps the flush; only the ordering column moves"
    )


def test_the_counter_orders_same_millisecond_events_across_chunk_arrival(tmp_path):
    # Two events in one millisecond, loaded in reverse: only the counter can recover emission
    # order, because load order is the arrival of chunks, which is not ordered at all.
    first = {**scroll(1000, y=1), "counter": "1000000001"}
    second = {**scroll(1000, y=2), "counter": "1000000002"}
    conn = connect(tmp_path / "events.db")
    hydrate(conn, "v1", [second, first])
    assert [e["data"]["y"] for e in canonical_events(conn, "v1")] == [1, 2]


def test_a_counter_tie_across_page_contexts_keeps_a_total_order(tmp_path):
    # The counter is minted per page context, so a visitor's two concurrent tabs can stamp
    # identical (timestamp, counter) on different events; id must totalize the order rather
    # than let the tie surface in an unstable read order.
    tab_a = {**scroll(1000, y=1), "counter": "1000000000"}
    tab_b = {**scroll(1000, y=2), "counter": "1000000000"}
    conn = connect(tmp_path / "events.db")
    hydrate(conn, "v1", [tab_a, tab_b])
    assert [e["data"]["y"] for e in canonical_events(conn, "v1")] == [1, 2]


# The shape gate — what the store read path checks before handing an event over.


@pytest.mark.parametrize(
    "event, reason",
    [
        (scroll(1000), "no counter"),
        ({**scroll(1000), "counter": "1000000001"}, None),
        (42, "not an object"),
        ({"timestamp": 1000, "data": {}}, "type is not an integer"),
        (
            {"type": str(EventType.IncrementalSnapshot), "timestamp": 1000, "data": {}},
            "type is not an integer",
        ),
        (
            {"type": EventType.IncrementalSnapshot, "data": {}},
            "timestamp is not a number",
        ),
        (
            {"type": EventType.IncrementalSnapshot, "timestamp": "soon", "data": {}},
            "timestamp is not a number",
        ),
        (
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 1000.5,
                "data": {},
                "counter": "1000000001",
            },
            None,
        ),
        (
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 1000,
                "data": {},
                "counter": 7,
            },
            "counter is not a string",
        ),
        (
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 1000,
                "counter": "1000000001",
            },
            "data is not an object",
        ),
        (
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 1000,
                "counter": "1000000001",
                "data": "nope",
            },
            "data is not an object",
        ),
        (
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 1000,
                "counter": "1000000001",
                "data": {"source": IncrementalSource.MouseMove, "positions": "nope"},
            },
            "positions is not a list",
        ),
        (
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 1000,
                "counter": "1000000001",
                "data": {
                    "source": IncrementalSource.MouseMove,
                    "positions": [{"x": 1}],
                },
            },
            "last move position has no numeric timeOffset",
        ),
        (
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 1000,
                "counter": "1000000001",
                "data": {"source": IncrementalSource.MouseMove, "positions": []},
            },
            None,
        ),
    ],
)
def test_the_shape_gate_names_exactly_what_row_construction_touches(event, reason):
    assert event_skip_reason(event) == reason


def test_a_bare_type_is_not_a_boolean(tmp_path):
    # True is an int in Python, and it would land as type 1 — a real rrweb event type.
    assert (
        event_skip_reason({"type": True, "timestamp": 1000, "data": {}})
        == "type is not an integer"
    )


# The recorder's shipped error log, landed beside the events it rode in with.


def test_chunk_errors_dedup_like_events_so_a_reload_is_free(tmp_path):
    conn = connect(tmp_path / "events.db")
    rows = [
        (
            "snip01",
            "v1",
            "00001749600000000-ab12",
            "k1",
            "dropped 2 malformed buffered records",
        )
    ]
    assert record_chunk_errors(conn, rows) == 1
    assert record_chunk_errors(conn, rows) == 0
    assert conn.execute("SELECT COUNT(*) c FROM chunk_errors").fetchone()["c"] == 1


def test_two_different_errors_on_one_chunk_are_two_records(tmp_path):
    conn = connect(tmp_path / "events.db")
    assert (
        record_chunk_errors(
            conn,
            [
                ("snip01", "v1", "00001749600000000-ab12", "k1", "evicted 4 records"),
                (
                    "snip01",
                    "v1",
                    "00001749600000000-ab12",
                    "k1",
                    "dropped oversize chunk",
                ),
            ],
        )
        == 2
    )


def test_each_flush_is_durable_and_on_flush_rides_its_transaction(tmp_path):
    # A load killed mid-stream keeps every flushed batch. The stream dies after
    # the fourth event; with batches of two, two flushes committed before the
    # death — a second connection (what a resumed load is) sees exactly those
    # rows, plus the on_flush writes that rode the same transactions (the chunk
    # manifest's atomicity in the real load).
    db = tmp_path / "events.db"
    conn = connect(db)
    flushes = []

    def on_flush():
        flushes.append(True)
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, 'x') ON CONFLICT(key) DO UPDATE SET value = value",
            (f"flush-{len(flushes)}",),
        )

    def dying_stream():
        for i in range(5):
            if i == 4:
                raise RuntimeError("connection dropped")
            yield (
                "v1",
                {"type": EventType.Meta, "timestamp": 1_000 + i, "data": {"i": i}},
            )

    with pytest.raises(RuntimeError, match="connection dropped"):
        hydrate_stream(conn, dying_stream(), batch_size=2, on_flush=on_flush)

    survivor = connect(db)
    assert survivor.execute("SELECT COUNT(*) n FROM events").fetchone()["n"] == 4, (
        "both full batches committed before the stream died"
    )
    assert [
        row["key"]
        for row in survivor.execute(
            "SELECT key FROM meta WHERE key LIKE 'flush-%' ORDER BY key"
        )
    ] == [
        "flush-1",
        "flush-2",
    ], "on_flush landed with each batch, atomically"
