import json

import pytest
from _support import (
    URL,
    click,
    env,
    full_snapshot,
    meta,
    replayable_slices,
    stamped,
)
from locus.evidence.db import connect
from locus.evidence.hydrate import hydrate, pack_raw
from locus.evidence.replay import (
    analysis_slices,
    disjoint_lanes,
    materialize,
    payload_js,
    set_name,
)
from locus.evidence.rrweb_constants import EventType
from locus.evidence.slices import materialize_slices


def test_the_payload_is_an_inert_registration_script(tmp_path):
    conn = replayable_slices(tmp_path)
    sid = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' LIMIT 1"
    ).fetchone()["id"]

    script = payload_js(conn, [sid])
    assert script.startswith("(window.LOCUS_REPLAYS ??= []).push(")
    assert "</" not in script, (
        "escaped so an inlined payload cannot close a <script> block"
    )


# The payload's refusals. A citation resolves to a slice set, and a payload the player
# cannot honestly play is worse than one that refuses: an interleaved two-visitor replay
# looks like a session and is not, concurrent tabs merged onto one clock replay as
# neither tab, and a player sized off nothing shows a plausible wrong page.


def test_a_set_naming_a_slice_the_db_does_not_hold_is_refused(tmp_path):
    conn = replayable_slices(tmp_path)
    real = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' LIMIT 1"
    ).fetchone()["id"]

    with pytest.raises(ValueError, match="unknown slice"):
        payload_js(conn, [real, 9999])


def test_a_set_spanning_two_visitors_is_refused(tmp_path):
    conn = connect(tmp_path / "events.db")
    for visitor in ("v1", "v2"):
        hydrate(
            conn, visitor, stamped([meta(1000), full_snapshot(1001), click(1500, 3)])
        )
        materialize_slices(conn, visitor)
    slice_ids = [r["id"] for r in conn.execute("SELECT id FROM slices")]
    assert len(slice_ids) == 2

    with pytest.raises(ValueError, match="visitor"):
        payload_js(conn, slice_ids)


def test_a_time_overlapping_set_is_refused(tmp_path):
    # Two page contexts (tabs) recording at once: each slice is honest alone, but their
    # merged stream puts tab B's snapshot mid-way through tab A's events, and one player
    # rebuilds on it — from there A's timeline is gone. Refused, not rendered.
    conn = connect(tmp_path / "events.db")
    tab_a, tab_b = "00000000001000-taba", "00000000002000-tabb"
    hydrate(
        conn,
        "v1",
        [
            env(meta(1000), tab_a),
            env(full_snapshot(1001), tab_a),
            env(meta(2000), tab_b),
            env(full_snapshot(2001), tab_b),
            env(click(5000, 3), tab_a),
            env(click(5500, 3), tab_b),
        ],
    )
    materialize_slices(conn, "v1")
    rows = conn.execute("SELECT id, status FROM slices ORDER BY start_ts").fetchall()
    assert [r["status"] for r in rows] == ["replayable", "replayable"]

    with pytest.raises(ValueError, match="overlap"):
        payload_js(conn, [r["id"] for r in rows])


def test_a_set_with_no_meta_viewport_is_refused(tmp_path):
    conn = connect(tmp_path / "events.db")
    hydrate(conn, "v1", stamped([meta(1000), full_snapshot(1001)]))
    materialize_slices(conn, "v1")
    dimensionless = {"type": EventType.Meta, "timestamp": 1000, "data": {"href": URL}}
    conn.execute(
        "UPDATE events SET raw_json = ? WHERE type = ?",
        (pack_raw(json.dumps(dimensionless)), EventType.Meta),
    )
    conn.commit()
    [sid] = [r["id"] for r in conn.execute("SELECT id FROM slices")]

    with pytest.raises(ValueError, match="viewport"):
        payload_js(conn, [sid])


# The analysis-directory seam: window.json's slice table is the citation→replay
# resolution contract — rows of {label, visitor, slice, slice_id, start_ts,
# end_ts} in window order, exactly as the engine writes it. Opening consumes
# label, visitor, and slice, and fails loud when the table is not there.


def test_analysis_slices_reads_the_slice_table_in_window_order(tmp_path):
    (tmp_path / "window.json").write_text(
        json.dumps(
            {
                "slices": [
                    {
                        "label": "S1",
                        "visitor": "a1b2c3d4-full",
                        "slice": "s1",
                        "slice_id": 1,
                        "start_ts": 1000,
                        "end_ts": 2000,
                    },
                    {
                        "label": "S2",
                        "visitor": "e5f6a7b8-full",
                        "slice": "s3",
                        "slice_id": 3,
                        "start_ts": 2000,
                        "end_ts": 3000,
                    },
                ]
            }
        )
    )
    table = analysis_slices(tmp_path)
    assert [s["label"] for s in table] == ["S1", "S2"]
    assert table[0]["visitor"] == "a1b2c3d4-full"
    assert table[0]["slice"] == "s1"


def test_disjoint_lanes_split_only_where_time_overlaps():
    a, b, c, d = (
        {"start_ts": 0, "end_ts": 100},
        {"start_ts": 10, "end_ts": 20},
        {"start_ts": 30, "end_ts": 40},
        {"start_ts": 150, "end_ts": 160},
    )
    assert disjoint_lanes([a, d]) == [[a, d]], (
        "a visitor's sequential slices are one lane — one player plays them through"
    )
    assert disjoint_lanes([d, b, c, a]) == [[a, d], [b, c]], (
        "concurrent lanes split; each lane stays time-disjoint and start-ordered"
    )


def test_a_directory_without_a_manifest_is_not_an_analysis(tmp_path):
    with pytest.raises(ValueError, match="window.json"):
        analysis_slices(tmp_path)


def test_a_manifest_without_a_slice_table_fails_loud(tmp_path):
    (tmp_path / "window.json").write_text(json.dumps({"slice_ids": [1]}))
    with pytest.raises(ValueError, match="table"):
        analysis_slices(tmp_path)


def test_materialized_names_derive_from_the_set(tmp_path):
    conn = replayable_slices(tmp_path)
    rows = conn.execute(
        "SELECT * FROM slices WHERE status='replayable' ORDER BY start_ts"
    ).fetchall()

    single = set_name([rows[0]])
    assert single == f"{rows[0]['visitor_id']}_{rows[0]['recorder_slice']}"

    # The digest keys the whole set: two sets sharing a first slice and a count are
    # different payloads, and _plusN alone would let one silently replace the other
    # under a page that still embeds it.
    first_pair = set_name([rows[0], rows[1]])
    second_pair = set_name([rows[0], rows[2]])
    assert first_pair.startswith(f"{single}_plus1_")
    assert first_pair != second_pair

    # Deterministic derivation, not evidence: re-materializing refreshes in place.
    written = materialize(conn, tmp_path / "pages", [rows[0]])
    before = written["payload"].read_text()
    again = materialize(conn, tmp_path / "pages", [rows[0]])
    assert again["payload"] == written["payload"]
    assert again["payload"].read_text() == before
    assert written["component"].name == "locus-replay.js"
    assert written["component"].stat().st_size > 100_000, (
        "the component carries the whole player — nothing left to a CDN"
    )
