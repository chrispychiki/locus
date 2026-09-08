import json
from pathlib import Path

import pytest
from _support import (
    click,
    env,
    full_snapshot,
    hidden,
    meta,
    read_recording,
    stamped,
    text_mutation,
)
from locus.evidence.db import CANONICAL_ORDER, connect
from locus.evidence.hydrate import canonical_events, hydrate, raw_event
from locus.evidence.rrweb_constants import EventType, IncrementalSource, NodeType
from locus.evidence.slices import (
    DAMAGED_SNAPSHOT_REASON,
    NO_SNAPSHOT_REASON,
    ORPHAN_REASON,
    materialize_slices,
    rescue_orphan_slices,
)

FIXTURE = Path(__file__).parent / "fixtures" / "demo_recording.json"


def test_every_event_is_accounted_for_across_the_demo_recording(tmp_path):
    """Slice materialization is loss accounting: every event lands in exactly one slice. Nothing may vanish between the canonical stream and the slices table."""
    visitor_id, events = read_recording(FIXTURE)
    conn = connect(tmp_path / "events.db")
    hydrate(conn, visitor_id, events)
    canon = canonical_events(conn, visitor_id)
    assert len(events) == 64
    assert len(canon) == 64

    summary = materialize_slices(conn, visitor_id)

    n_full = sum(1 for e in canon if e["type"] == EventType.FullSnapshot)
    n_meta = sum(1 for e in canon if e["type"] == EventType.Meta)
    assert summary == {"replayable": n_full, "discarded": n_meta - n_full}
    assert n_full == 4 and n_meta == 4

    in_a_slice = conn.execute(
        "SELECT COUNT(*) n FROM events WHERE slice_id IS NOT NULL"
    ).fetchone()["n"]
    assert in_a_slice == len(canon)
    assert (
        conn.execute("SELECT SUM(n_events) n FROM slices").fetchone()["n"] == in_a_slice
    )

    for row in conn.execute("SELECT id FROM slices"):
        kinds = [
            raw_event(r["raw_json"])["type"]
            for r in conn.execute(
                f"SELECT raw_json FROM events WHERE slice_id = ? {CANONICAL_ORDER} LIMIT 2",
                (row["id"],),
            )
        ]
        assert kinds == [EventType.Meta, EventType.FullSnapshot]


def test_an_event_with_no_recorder_slice_fails_loud(tmp_path):
    """The recorder stamps every event it emits. A missing stamp is a damaged stream, not a source variant, and the reader cannot recover the slicing — so it stops rather than invent one and file the damage as a supported case."""
    conn = connect(tmp_path / "events.db")
    hydrate(conn, "v1", [meta(1000), full_snapshot(1001)])

    with pytest.raises(ValueError, match="carries no recorder_slice"):
        materialize_slices(conn, "v1")


def rescue_run(tmp_path, orphan_events):
    db_path = tmp_path / "events.db"
    conn = connect(db_path)
    events = [meta(1000), full_snapshot(1001), click(2000, 3)] + orphan_events
    hydrate(conn, "v1", stamped(events))
    materialize_slices(conn, "v1")
    summary = rescue_orphan_slices(conn, str(db_path))
    return conn, summary


def test_orphan_slice_with_continuous_ids_is_rescued(tmp_path):
    conn, summary = rescue_run(
        tmp_path,
        [
            meta(3000),
            text_mutation(3100, 4, "hello again"),
            click(3200, 3),
        ],
    )
    assert summary == {"rescued": 1, "gated_out": 0, "ineligible": 0}

    prior = conn.execute("SELECT * FROM slices WHERE status = 'replayable'").fetchone()
    assert prior["n_events"] == 6
    assert prior["end_ts"] == 3200
    in_the_slice = conn.execute(
        "SELECT COUNT(*) n FROM events WHERE slice_id = ?", (prior["id"],)
    ).fetchone()["n"]
    assert in_the_slice == 6

    rescued = conn.execute("SELECT * FROM slices WHERE status = 'rescued'").fetchone()
    assert f"appended to slice {prior['recorder_slice']}" in rescued["reason"]
    assert json.loads(prior["absorbed"]) == [rescued["recorder_slice"]]


def test_orphan_slice_referencing_unknown_nodes_is_gated(tmp_path):
    conn, summary = rescue_run(
        tmp_path,
        [
            meta(3000),
            text_mutation(3100, 99, "ghost"),
        ],
    )
    assert summary == {"rescued": 0, "gated_out": 1, "ineligible": 0}

    orphan = conn.execute("SELECT * FROM slices WHERE status = 'discarded'").fetchone()
    assert "rescue gated" in orphan["reason"]
    assert "unknown node 99" in orphan["reason"]
    prior = conn.execute("SELECT * FROM slices WHERE status = 'replayable'").fetchone()
    assert prior["n_events"] == 3


def test_incremental_loads_never_renumber_slices(tmp_path):
    conn = connect(tmp_path / "events.db")
    rid_a = "00000000001000-aaaa"
    hydrate(
        conn,
        "v1",
        [
            env(meta(1000), rid_a),
            env(full_snapshot(1001), rid_a),
            env(click(2000, 3), rid_a),
        ],
    )
    materialize_slices(conn, "v1")
    first = conn.execute("SELECT * FROM slices").fetchone()
    assert first["recorder_slice"] == rid_a

    rid_b = "00000000005000-bbbb"
    hydrate(
        conn,
        "v1",
        [
            env(click(2500, 3), rid_a),
            env(meta(5000), rid_b),
            env(full_snapshot(5001), rid_b),
        ],
    )
    materialize_slices(conn, "v1")

    a = conn.execute(
        "SELECT * FROM slices WHERE recorder_slice = ?", (rid_a,)
    ).fetchone()
    assert a["id"] == first["id"]
    assert a["n_events"] == 4
    assert a["end_ts"] == 2500
    b = conn.execute(
        "SELECT * FROM slices WHERE recorder_slice = ?", (rid_b,)
    ).fetchone()
    assert b["id"] != first["id"]
    assert b["status"] == "replayable"


def test_other_page_or_late_orphans_are_ineligible(tmp_path):
    conn, summary = rescue_run(
        tmp_path,
        [
            meta(3000, href="https://x.test/other"),
            text_mutation(3100, 4, "x"),
            meta(60_000),
            text_mutation(60_100, 4, "y"),
        ],
    )
    assert summary == {"rescued": 0, "gated_out": 0, "ineligible": 2}
    assert (
        conn.execute(
            "SELECT COUNT(*) n FROM slices WHERE status = 'discarded'"
        ).fetchone()["n"]
        == 2
    )


def test_concurrent_page_contexts_keep_their_own_events(tmp_path):
    """One visitor with two tabs open interleaves two page contexts into one time-ordered stream. The recorder stamped each event with its own context's slice; a reader that cut the merged stream by position instead would file each context's events under the other's slice — and close tab A's span at tab B's Meta, discarding A as snapshotless while its FullSnapshot sat one row away in B."""
    a, b = "00000000001000-aaaa", "00000000001001-bbbb"
    conn = connect(tmp_path / "events.db")
    hydrate(
        conn,
        "v1",
        [
            env(meta(1000), a),
            env(meta(1001), b),  # the other tab opens 1ms later
            env(full_snapshot(1002), a),  # A's covering snapshot, behind B's Meta
            env(full_snapshot(1003), b),
            env(click(2000, 3), a),
            env(text_mutation(2001, 4, "b types"), b),
            env(click(2002, 3), a),
        ],
    )

    summary = materialize_slices(conn, "v1")

    assert summary == {"replayable": 2, "discarded": 0}
    for rid, n in ((a, 4), (b, 3)):
        row = conn.execute(
            "SELECT * FROM slices WHERE recorder_slice = ?", (rid,)
        ).fetchone()
        assert row["status"] == "replayable"
        assert row["n_events"] == n
        kinds = [
            raw_event(r["raw_json"])["type"]
            for r in conn.execute(
                f"SELECT raw_json FROM events WHERE slice_id = ? {CANONICAL_ORDER}",
                (row["id"],),
            )
        ]
        assert kinds.count(EventType.FullSnapshot) == 1


def test_a_slice_whose_first_event_beat_the_snapshot_is_still_replayable(tmp_path):
    """A page context opens its slice the moment it starts, so a marker can precede the Meta inside it. The covering rule is that the slice *contains* the Meta+FullSnapshot pair, not that it opens on it."""
    rid = "00000000000900-aaaa"
    conn = connect(tmp_path / "events.db")
    hydrate(
        conn,
        "v1",
        [
            env(hidden(950), rid),  # the page was hidden before rrweb snapshotted
            env(meta(1000), rid),
            env(full_snapshot(1001), rid),
            env(click(2000, 3), rid),
        ],
    )

    summary = materialize_slices(conn, "v1")

    assert summary == {"replayable": 1, "discarded": 0}
    row = conn.execute("SELECT * FROM slices").fetchone()
    assert row["start_ts"] == 950 and row["n_events"] == 4


def test_a_snapshot_with_no_root_is_discarded_not_projected_as_an_empty_page(tmp_path):
    """A FullSnapshot whose payload carries no root node cannot be rebuilt into a DOM. Projected, it
    yields nothing — and nothing is indistinguishable from an empty page, which is precisely what the
    model would be told the visitor sat looking at, with every later mutation diffed against it. There
    is no honest reading of the slice, so it is discarded rather than testified to."""
    conn = connect(tmp_path / "events.db")
    rootless = {"type": EventType.FullSnapshot, "timestamp": 1001, "data": {"node": {}}}
    hydrate(conn, "v1", stamped([meta(1000), rootless, click(2000, 3)]))

    summary = materialize_slices(conn, "v1")

    assert summary == {"replayable": 0, "discarded": 1}
    row = conn.execute("SELECT * FROM slices").fetchone()
    assert row["reason"] == DAMAGED_SNAPSHOT_REASON


def test_a_zero_dimension_viewport_is_discarded_as_a_page_no_human_saw(tmp_path):
    """A Meta recording a 0x0 viewport is a hidden prerender context — a page no human ever saw. Its covering pair is intact, so this gate is the only thing standing between it and analysis."""
    conn = connect(tmp_path / "events.db")
    prerender = meta(1000)
    prerender["data"].update(width=0, height=0)
    hydrate(conn, "v1", stamped([prerender, full_snapshot(1001), click(2000, 3)]))

    summary = materialize_slices(conn, "v1")

    assert summary == {"replayable": 0, "discarded": 1}
    row = conn.execute("SELECT * FROM slices").fetchone()
    assert row["reason"] == "zero-dimension viewport in Meta (0x0)"


def test_chained_orphans_rescue_in_sequence_gated_on_the_prior_rescues_state(tmp_path):
    """A rescue rewrites the prior slice's events and end time, and the next orphan's verdict is taken against that new state. The second orphan here references a node that exists only because the first orphan's events added it — it can pass the gate only if the first rescue's writes are already in the replayed state."""
    db_path = tmp_path / "events.db"
    conn = connect(db_path)
    adds_a_span = {
        "type": EventType.IncrementalSnapshot,
        "timestamp": 3100,
        "data": {
            "source": IncrementalSource.Mutation,
            "texts": [],
            "attributes": [],
            "removes": [],
            "adds": [
                {
                    "parentId": 3,
                    "nextId": None,
                    "node": {
                        "type": NodeType.Element,
                        "tagName": "span",
                        "id": 99,
                        "attributes": {},
                        "childNodes": [],
                    },
                }
            ],
        },
    }
    touches_the_added_node = {
        "type": EventType.IncrementalSnapshot,
        "timestamp": 4100,
        "data": {
            "source": IncrementalSource.Mutation,
            "texts": [],
            "attributes": [{"id": 99, "attributes": {"class": "lit"}}],
            "removes": [],
            "adds": [],
        },
    }
    hydrate(
        conn,
        "v1",
        stamped(
            [
                meta(1000),
                full_snapshot(1001),
                click(2000, 3),
                meta(3000),
                adds_a_span,
                meta(4000),
                touches_the_added_node,
            ]
        ),
    )
    materialize_slices(conn, "v1")

    summary = rescue_orphan_slices(conn, str(db_path))

    assert summary == {"rescued": 2, "gated_out": 0, "ineligible": 0}
    prior = conn.execute("SELECT * FROM slices WHERE status = 'replayable'").fetchone()
    assert prior["n_events"] == 7
    assert prior["end_ts"] == 4100
    rescued = conn.execute(
        "SELECT recorder_slice FROM slices WHERE status = 'rescued' ORDER BY start_ts"
    ).fetchall()
    assert json.loads(prior["absorbed"]) == [r["recorder_slice"] for r in rescued]


def test_a_reload_is_never_welded_onto_the_page_it_replaced(tmp_path):
    """The rescue gate's blind spot, and why the node-id check can never see it.

    A visitor reloads. The new document is a new rrweb mirror, so its node ids restart at 1 — its
    body is id 3 and its text node is id 4, exactly as the previous page's were, and they are not
    the same nodes. The reload's FullSnapshot chunk is lost, so it arrives as an orphan whose
    mutations reference ids 3 and 4. Every one of them resolves against the prior page's DOM, the
    gate reports continuity verified, and two unrelated documents are welded into one slice that
    replays the second page's mutations against the first page's DOM — and asserts it is replayable.

    Continuity and coincidence are the same answer to the question the gate asks. The slice id is
    what tells them apart: a page context mints its head slice before rrweb runs, so the id opens
    strictly earlier than its own Meta, while a checkout's id is minted from that Meta."""
    db_path = tmp_path / "events.db"
    conn = connect(db_path)
    page_a = "00000000001000-aaaa"
    reload_b = (
        "00000000002999-bbbb"  # minted at start(), a millisecond before its own Meta
    )
    hydrate(
        conn,
        "v1",
        [
            env(meta(1000), page_a),
            env(full_snapshot(1001), page_a),
            env(click(2000, 3), page_a),
            env(meta(3000), reload_b),  # same URL, one second later
            env(text_mutation(3100, 4, "a different page"), reload_b),
        ],
    )
    materialize_slices(conn, "v1")

    summary = rescue_orphan_slices(conn, str(db_path))

    assert summary == {"rescued": 0, "gated_out": 0, "ineligible": 1}
    prior = conn.execute(
        "SELECT * FROM slices WHERE recorder_slice = ?", (page_a,)
    ).fetchone()
    assert prior["n_events"] == 3  # page A absorbed nothing
    assert prior["absorbed"] is None
    assert (
        conn.execute(
            "SELECT status FROM slices WHERE recorder_slice = ?", (reload_b,)
        ).fetchone()["status"]
        == "discarded"
    )


def test_a_checkout_of_the_same_document_is_still_rescued(tmp_path):
    """The refusal must cut the reload out and leave the real case standing: a parked tab's periodic
    checkout is the same document, and its id is minted from its own Meta — so it opens at that Meta,
    not before it, and goes on to the node-id gate exactly as it did."""
    _conn, summary = rescue_run(
        tmp_path,
        [
            meta(3000),
            text_mutation(3100, 4, "still the same page"),
        ],
    )
    assert summary == {"rescued": 1, "gated_out": 0, "ineligible": 0}


def test_a_snapshot_that_does_not_immediately_follow_its_meta_is_no_covering_pair(
    tmp_path,
):
    """rrweb emits Meta and FullSnapshot back to back — the pair is adjacency in canonical order, not mere containment of both types. A slice whose Meta is followed by other events before any snapshot has lost the snapshot the Meta heralded; a later snapshot in the same slice captured a different DOM moment, and seeking the slice's head would paint it at a time its pixels never held."""
    db_path = tmp_path / "events.db"
    conn = connect(db_path)
    torn = "00000000001000-aaaa"
    hydrate(
        conn,
        "v1",
        [
            env(meta(1000), torn),
            env(click(1500, 3), torn),
            env(full_snapshot(2000), torn),
        ],
    )
    materialize_slices(conn, "v1")

    row = conn.execute(
        "SELECT status, reason FROM slices WHERE recorder_slice = ?", (torn,)
    ).fetchone()
    assert row["status"] == "discarded"
    assert row["reason"] == ORPHAN_REASON


def test_a_context_that_died_before_its_snapshot_is_discarded_and_never_rescued(
    tmp_path,
):
    """A page that ends before rrweb captures its DOM has a slice but no Meta. Nothing was lost in transit — there was never a snapshot — and the events belong to a fresh document, so no prior slice's DOM can cover them. It must never enter the rescue path, whose node-id continuity check would happily match a reload's restarted ids against the previous page's."""
    db_path = tmp_path / "events.db"
    conn = connect(db_path)
    prior, dead = "00000000001000-aaaa", "00000000003000-bbbb"
    hydrate(
        conn,
        "v1",
        [
            env(meta(1000), prior),
            env(full_snapshot(1001), prior),
            env(click(2000, 3), prior),
            env(hidden(3100), dead),  # a reload; its rrweb never emitted a Meta
        ],
    )
    materialize_slices(conn, "v1")

    row = conn.execute(
        "SELECT * FROM slices WHERE recorder_slice = ?", (dead,)
    ).fetchone()
    assert row["status"] == "discarded"
    assert row["reason"] == NO_SNAPSHOT_REASON

    summary = rescue_orphan_slices(conn, str(db_path))

    assert summary == {"rescued": 0, "gated_out": 0, "ineligible": 0}
    assert (
        conn.execute(
            "SELECT status FROM slices WHERE recorder_slice = ?", (dead,)
        ).fetchone()["status"]
        == "discarded"
    )


# The slice-id grammar has a write-side twin: the store's path gate. One case list through both
# regexes keeps read and write sides from drifting apart silently — the same pin test_text.py
# holds over the URL rule's twins.
SLICE_ID_CASES = [
    "20260801120000-ab3z",
    "00000000000000-0000",
    "20260801120000-AB3Z",  # uppercase disambiguator: minted lowercase only
    "2026080112000-ab3z",  # 13 digits
    "202608011200000-ab3z",  # 15 digits
    "20260801120000-ab3",  # 3-char disambiguator
    "20260801120000-ab3zz",  # 5-char disambiguator
    "20260801120000ab3z",  # no dash
    "٠١٢٣٤٥٦٧٨٩٠١٢٣-abcd",  # Unicode digits: JS \d is ASCII; the twin must be too
    " 20260801120000-ab3z",
    "20260801120000-ab3z ",
    "20260801120000-ab3z\n",  # trailing newline: JS $ is \Z; Python's $ would take it
]


def test_the_slice_id_shape_matches_the_store_gate_case_for_case():
    from locus.evidence.slices import SLICE_ID_SHAPE

    keys_js = Path(__file__).parents[2] / "store" / "src" / "keys.js"
    script = (
        f"import {{ SLICE_ID_SHAPE }} from {json.dumps(str(keys_js))};"
        f"const cases = {json.dumps(SLICE_ID_CASES)};"
        f"console.log(JSON.stringify(cases.map((c) => SLICE_ID_SHAPE.test(c))));"
    )
    import subprocess

    out = subprocess.run(
        ["bun", "-e", script], check=True, capture_output=True, text=True
    ).stdout
    verdicts = json.loads(out.strip())
    assert [bool(SLICE_ID_SHAPE.match(c)) for c in SLICE_ID_CASES] == verdicts


def test_materialize_slices_demo_recording(tmp_path):
    visitor_id, events = read_recording(FIXTURE)
    conn = connect(tmp_path / "events.db")
    hydrate(conn, visitor_id, events)

    summary = materialize_slices(conn, visitor_id)
    assert summary == {"replayable": 4, "discarded": 0}

    rows = conn.execute(
        "SELECT status, COUNT(*) n, SUM(n_events) ev FROM slices GROUP BY status"
    ).fetchall()
    assert [(r["status"], r["n"]) for r in rows] == [("replayable", 4)]

    unmaterialized = conn.execute(
        "SELECT COUNT(*) n FROM events WHERE slice_id IS NULL"
    ).fetchone()["n"]
    assert unmaterialized == 0

    stamped = conn.execute(
        "SELECT COUNT(*) n FROM events WHERE slice_id IS NOT NULL"
    ).fetchone()["n"]
    assert stamped == rows[0]["ev"]

    head_types = conn.execute(
        "SELECT e.type, MIN(e.id) FROM events e JOIN slices s ON e.slice_id = s.id "
        "WHERE s.status = 'replayable' GROUP BY s.id"
    ).fetchall()
    assert len(head_types) == 4
    assert all(r["type"] == EventType.Meta for r in head_types)


def test_rematerialize_is_stable(tmp_path):
    visitor_id, events = read_recording(FIXTURE)
    conn = connect(tmp_path / "events.db")
    hydrate(conn, visitor_id, events)

    first = materialize_slices(conn, visitor_id)
    second = materialize_slices(conn, visitor_id)
    assert first == second
    n_slices = conn.execute("SELECT COUNT(*) n FROM slices").fetchone()["n"]
    assert n_slices == 4


def test_a_slice_carries_its_own_facts_so_no_site_or_device_read_walks_the_events(
    tmp_path,
):
    """One page context is one site on one device in one browser, so those are the slice's facts, not per-event ones: materialization lifts the envelope testimony and the Meta's address onto the slices row, and a slice that never had a Meta names no page."""
    conn = connect(tmp_path / "events.db")
    events = stamped(
        [
            meta(1000, href="https://www.example.com/products/a?ref=x"),
            full_snapshot(1001),
            click(2000, 3),
            meta(5000, href="https://www.example.com/cart"),
        ],
        snippet="abc123",
        device="mobile",
        os="iOS",
        browser="Mobile Safari",
        language="en-US",
        time_zone="America/Denver",
        screen_width=390,
        screen_height=844,
        script_version="locus-recorder/0.5.3",
    )
    hydrate(conn, "v1", events)
    materialize_slices(conn, "v1")

    rows = conn.execute(
        "SELECT status, snippet, url, device, os, browser, language, time_zone, "
        "screen_width, screen_height, script_version FROM slices ORDER BY start_ts"
    ).fetchall()
    assert [dict(r) for r in rows] == [
        {
            "status": "replayable",
            "snippet": "abc123",
            "url": "https://www.example.com/products/a?ref=x",
            "device": "mobile",
            "os": "iOS",
            "browser": "Mobile Safari",
            "language": "en-US",
            "time_zone": "America/Denver",
            "screen_width": 390,
            "screen_height": 844,
            "script_version": "locus-recorder/0.5.3",
        },
        {
            "status": "discarded",
            "snippet": "abc123",
            "url": "https://www.example.com/cart",
            "device": "mobile",
            "os": "iOS",
            "browser": "Mobile Safari",
            "language": "en-US",
            "time_zone": "America/Denver",
            "screen_width": 390,
            "screen_height": 844,
            "script_version": "locus-recorder/0.5.3",
        },
    ]


def test_a_db_from_before_a_column_existed_is_widened_on_connect(tmp_path):
    """CREATE TABLE IF NOT EXISTS leaves an old table as it was, so a db created before a column existed gets it on connect, with the values landing on its next derivation."""
    import sqlite3

    db = tmp_path / "events.db"
    old = sqlite3.connect(db)
    old.executescript(
        "CREATE TABLE slices (id INTEGER PRIMARY KEY, visitor_id TEXT NOT NULL, "
        "recorder_slice TEXT, start_ts INTEGER NOT NULL, end_ts INTEGER NOT NULL, "
        "n_events INTEGER NOT NULL, status TEXT NOT NULL, reason TEXT, absorbed TEXT);"
        "CREATE TABLE events (id INTEGER PRIMARY KEY, visitor_id TEXT NOT NULL, "
        "timestamp INTEGER NOT NULL, type INTEGER NOT NULL, counter TEXT, "
        "raw_json BLOB NOT NULL, content_hash TEXT NOT NULL, slice_id INTEGER, "
        "type_str TEXT, snippet TEXT)"
    )
    old.close()

    conn = connect(db)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(slices)")}
    assert {"snippet", "url", "device", "screen_width", "script_version"} <= columns
    assert "session_id" in {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    indexes = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    assert {"slices_site", "events_session"} <= indexes
    stored = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert "snippet TEXT /* the site's snippet id" in stored["slices"]
    assert "session_id INTEGER /* sessions.id:" in stored["events"]


def test_every_declared_column_reads_as_the_schema_states_it():
    """The widening adds a column by the words the CREATE gives it, so the declared set is exactly the live set of a fresh db, and every slice fact is a declared slices column."""
    import sqlite3

    from locus.evidence.db import SCHEMA, SLICE_FACTS, declared_columns

    declared = declared_columns()
    fresh = sqlite3.connect(":memory:")
    fresh.executescript(SCHEMA)
    for table, columns in declared.items():
        live = [row[1] for row in fresh.execute(f"PRAGMA table_info({table})")]
        assert list(columns) == live, table
    assert set(SLICE_FACTS) <= set(declared["slices"])


def test_a_column_the_widening_cannot_add_is_refused_by_name(tmp_path, monkeypatch):
    """SQLite's ALTER refuses a NOT NULL column on a table holding rows, and a PRIMARY KEY or UNIQUE one always, with a message naming nothing; the widening says which table and column the schema declares beyond what a standing db can take, and that the db is rebuilt by dropping it and loading again."""
    import sqlite3

    from locus.evidence import db as db_module

    path = tmp_path / "events.db"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE slices (id INTEGER PRIMARY KEY, visitor_id TEXT NOT NULL, "
        "recorder_slice TEXT, start_ts INTEGER NOT NULL, end_ts INTEGER NOT NULL, "
        "n_events INTEGER NOT NULL, status TEXT NOT NULL, reason TEXT, absorbed TEXT, "
        "snippet TEXT);"
        "INSERT INTO slices VALUES (1, 'v', 's', 0, 1, 1, 'replayable', NULL, NULL, 'x');"
        "CREATE TABLE events (id INTEGER PRIMARY KEY, visitor_id TEXT NOT NULL, "
        "timestamp INTEGER NOT NULL, type INTEGER NOT NULL, counter TEXT, "
        "raw_json BLOB NOT NULL, content_hash TEXT NOT NULL, slice_id INTEGER, "
        "type_str TEXT, snippet TEXT, session_id INTEGER)"
    )
    old.close()
    monkeypatch.setattr(
        db_module,
        "declared_columns",
        lambda: {"slices": {"opened_by": "opened_by TEXT NOT NULL /* who */"}},
    )
    with pytest.raises(RuntimeError, match=r"slices\.opened_by.*NOT NULL.*locus load"):
        connect(path)

    monkeypatch.setattr(
        db_module,
        "declared_columns",
        lambda: {"slices": {"opened_by": "opened_by TEXT /* who */"}},
    )
    conn = connect(path)
    assert "opened_by" in {
        row[1] for row in conn.execute("PRAGMA table_info(slices)")
    }, "a plain column is added to a populated table as before"
