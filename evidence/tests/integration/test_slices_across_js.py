import json
from pathlib import Path

import pytest
from _support import (
    click,
    full_snapshot,
    meta,
    rescue_run,
    stamped,
    text_mutation,
)
from locus.evidence.db import connect
from locus.evidence.hydrate import hydrate
from locus.evidence.rrweb_constants import EventType, IncrementalSource, NodeType
from locus.evidence.slices import (
    materialize_slices,
    rescue_orphan_slices,
    slice_events,
    viewport_at,
    viewport_events,
)


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


def _viewport_resize(ts, width, height):
    return {
        "type": EventType.IncrementalSnapshot,
        "timestamp": ts,
        "data": {
            "source": IncrementalSource.ViewportResize,
            "width": width,
            "height": height,
        },
    }


def test_the_viewport_at_a_moment_is_the_last_one_stated_at_or_before_it(tmp_path):
    """The replayer sizes its frame at every Meta and ViewportResize it applies, so a visitor who resizes mid-slice changes the size of everything shown after; a resize reported at zero width or height leaves the frame showing nothing, so the size last shown stands."""
    from locus.evidence.derive import run_distill

    db = tmp_path / "events.db"
    conn = connect(db)
    hydrate(
        conn,
        "v1",
        stamped(
            [
                meta(1000),
                full_snapshot(1001),
                click(2000, 3),
                _viewport_resize(3000, 390, 663),
                _viewport_resize(4000, 0, 663),
                click(5000, 3),
            ]
        ),
    )
    materialize_slices(conn, "v1")
    conn.close()
    run_distill(str(db))
    conn = connect(db)
    (slice_id,) = [row["id"] for row in conn.execute("SELECT id FROM slices")]

    events = viewport_events(conn, slice_id)
    assert [event["timestamp"] for event in events] == [1000, 3000, 4000], (
        "only the events that state a viewport"
    )
    assert viewport_at(events, 2999) == (1280, 800)
    assert viewport_at(events, 3000) == (390, 663), "a resize at the moment applies"
    assert viewport_at(events, 5000) == (390, 663), (
        "a zero-size resize shows nothing new"
    )
    with pytest.raises(ValueError, match="no viewport"):
        viewport_at(events, 999)


def test_a_move_batch_reaches_the_replayer_at_its_last_positions_moment(tmp_path):
    """A move batch sent after a resize but moving before it reaches the replayer at its canonical moment, so timestamps rise in the order the replayer receives them, and the viewport read stops at the moment with the resize applied."""
    from locus.evidence.derive import run_distill

    db = tmp_path / "events.db"
    conn = connect(db)
    touch_moves = {
        "type": EventType.IncrementalSnapshot,
        "timestamp": 3200,
        "data": {
            "source": IncrementalSource.TouchMove,
            "positions": [{"x": 10, "y": 10, "id": 3, "timeOffset": -400}],
        },
    }
    hydrate(
        conn,
        "v1",
        stamped(
            [
                meta(1000),
                full_snapshot(1001),
                touch_moves,
                _viewport_resize(3000, 390, 663),
            ]
        ),
    )
    materialize_slices(conn, "v1")
    conn.close()
    run_distill(str(db))
    conn = connect(db)
    (slice_id,) = [row["id"] for row in conn.execute("SELECT id FROM slices")]

    events = slice_events(conn, slice_id)
    assert [event["timestamp"] for event in events][-2:] == [2800, 3000]
    assert events[-2]["data"]["positions"] == [
        {"x": 10, "y": 10, "id": 3, "timeOffset": 0}
    ], "the position keeps its moment, 2800"
    assert viewport_at(events, 3100) == (390, 663)


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

    keys_js = Path(__file__).parents[3] / "store" / "src" / "keys.js"
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
