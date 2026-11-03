"""The renderer's refusals are the replayability reject rule at the render boundary: what it cannot render faithfully it must refuse, never capture a wrong screenshot from. Each is a construction-time refusal — before a browser is ever launched."""

import pytest
from _support import dom_content_loaded, sized_meta, synthetic
from locus.evidence.db import connect
from locus.evidence.render import RenderSession
from locus.evidence.rrweb_constants import EventType
from locus.evidence.slices import (
    ORPHAN_REASON,
    materialize_slices,
)


def _snapshot(ts):
    return {
        "type": EventType.FullSnapshot,
        "timestamp": ts,
        "data": {"node": {}, "initialOffset": {"left": 0, "top": 0}},
    }


def test_an_empty_window_is_refused(corpus, tmp_path):
    with pytest.raises(ValueError, match="slice"):
        RenderSession(corpus, {}, tmp_path / "screenshots")


def test_an_unknown_slice_is_refused(corpus, tmp_path):
    with pytest.raises(ValueError, match="events"):
        RenderSession(corpus, {"S1": 9999}, tmp_path / "screenshots")


def test_a_slice_without_its_own_snapshot_is_refused(tmp_path):
    conn = connect(tmp_path / "events.db")
    synthetic(conn, [sized_meta(1000)])
    materialize_slices(conn, "v1")
    slice_id = conn.execute("SELECT id FROM slices").fetchone()["id"]
    # A lone Meta is exactly the slice the guard exists for, and it is refused by the reason
    # materialize_slices discarded it with — one criterion, quoted, never a second one re-derived here.
    with pytest.raises(ValueError, match=ORPHAN_REASON):
        RenderSession(conn, {"S1": slice_id}, tmp_path / "screenshots")


def test_a_zero_viewport_slice_is_refused(tmp_path):
    """The viewport is read off the covering Meta, so a slice that opened before its snapshot is
    still judged on the Meta that sizes the render — never on whatever event happens to be first."""
    conn = connect(tmp_path / "events.db")
    synthetic(
        conn,
        [dom_content_loaded(999), sized_meta(1000, width=0, height=0), _snapshot(1001)],
    )
    materialize_slices(conn, "v1")
    slice_id = conn.execute("SELECT id FROM slices").fetchone()["id"]
    with pytest.raises(ValueError, match="viewport"):
        RenderSession(conn, {"S1": slice_id}, tmp_path / "screenshots")
