"""The screenshot-moment selector: which moments a window renders, and why each one is there.

Every screenshot is a browser replay, so what this picks is what analysis costs and what
it can see. What counts as user activity is locus.evidence.user_activity's, pinned in its own suite;
this one pins where the marks land around it.
"""

import pytest
from _support import stamped
from locus.analysis.select_screenshots import (
    screenshot_moments,
    select_screenshots,
)
from locus.analysis.window import resolve_window
from locus.evidence.db import connect
from locus.evidence.hydrate import hydrate
from locus.evidence.rrweb_constants import (
    EventType,
    IncrementalSource,
    MouseInteractions,
)
from locus.evidence.slices import materialize_slices


def _moments(conn, slice_ids, ws=None, we=None, **kwargs):
    return screenshot_moments(conn, resolve_window(conn, slice_ids, ws, we), **kwargs)


HEAD = 100_000


def meta(ts):
    return {
        "type": EventType.Meta,
        "timestamp": ts,
        "data": {"href": "https://x.test/", "width": 1280, "height": 800},
    }


def snapshot(ts):
    return {
        "type": EventType.FullSnapshot,
        "timestamp": ts,
        "data": {"node": {"id": 1}, "initialOffset": {"left": 0, "top": 0}},
    }


def incremental(ts, source, **data):
    return {
        "type": EventType.IncrementalSnapshot,
        "timestamp": ts,
        "data": {"source": source, **data},
    }


def click(ts):
    return incremental(
        ts,
        IncrementalSource.MouseInteraction,
        type=MouseInteractions.Click,
        id=1,
        x=0,
        y=0,
    )


def page_load(ts):
    return {
        "type": EventType.PageLoad,
        "timestamp": ts,
        "data": {"url": "https://x.test/", "title": "x", "referrer": ""},
    }


def marker(ts, event_type):
    return {"type": event_type, "timestamp": ts, "data": {}}


def build(tmp_path, *event_lists):
    """One db, one visitor, one slice per event list (each carries a Meta+FullSnapshot pair)."""
    conn = connect(tmp_path / "events.db")
    events = [event for group in event_lists for event in group]
    hydrate(conn, "v1", stamped(events))
    materialize_slices(conn, "v1")
    ids = [row["id"] for row in conn.execute("SELECT id FROM slices ORDER BY start_ts")]
    return conn, ids


def one_slice(tmp_path, *events):
    """A slice whose covering snapshot lands exactly on HEAD — the Meta ahead of it is the
    slice's first event, as rrweb really emits the pair."""
    conn, ids = build(tmp_path, [meta(HEAD - 1), snapshot(HEAD), *events])
    return conn, ids[0]


# A head slice's real shape: the page context opens, and emits, before rrweb has a DOM to
# capture, so markers and live user activity precede the covering pair inside the slice they
# belong to — the ~18-second lead below is one recorded in the wild, with rrweb's own
# serialization gap between the Meta and the snapshot it heralds.
LEAD_MS = 17_866
SNAPSHOT_LAG_MS = 64
SNAPSHOT = HEAD + SNAPSHOT_LAG_MS


def lead_slice(tmp_path, *events):
    conn, ids = build(
        tmp_path,
        [
            marker(HEAD - LEAD_MS, EventType.DomContentLoaded),
            marker(HEAD - LEAD_MS + 100, EventType.Load),
            meta(HEAD),
            snapshot(SNAPSHOT),
            *events,
        ],
    )
    return conn, ids[0]


# Where the marks land.


def test_a_script_set_input_earns_no_screenshot(tmp_path):
    conn, slice_id = one_slice(
        tmp_path,
        incremental(
            HEAD + 5_000, IncrementalSource.Input, id=1, text="tok", userTriggered=False
        ),
        incremental(HEAD + 20_000, IncrementalSource.Mutation, adds=[]),
    )
    selection = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    )
    assert selection.timestamps == [HEAD, HEAD + 20_000]
    assert selection.user_activity_events == 0


def test_a_quiet_slice_yields_only_its_endpoints(tmp_path):
    conn, slice_id = one_slice(
        tmp_path, incremental(HEAD + 5_000, IncrementalSource.Mutation, adds=[])
    )
    selection = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    )
    assert selection.timestamps == [HEAD, HEAD + 5_000]
    assert selection.user_activity_events == 0


def test_screenshot_moments_anchor_at_the_slice_head_not_at_epoch(tmp_path):
    conn, slice_id = one_slice(
        tmp_path,
        click(HEAD + 3_500),
        incremental(HEAD + 9_000, IncrementalSource.Mutation, adds=[]),
    )
    selection = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    )
    # HEAD is 100_000, so epoch-anchored marks would be indistinguishable; the
    # click at +3500 must keep the +3000 and +4000 marks off the head, not +3500.
    assert HEAD + 3_000 in selection.timestamps
    assert HEAD + 4_000 in selection.timestamps
    assert HEAD + 3_500 not in selection.timestamps


def test_only_moments_within_the_activity_radius_survive(tmp_path):
    conn, slice_id = one_slice(
        tmp_path,
        click(HEAD + 5_000),
        incremental(HEAD + 20_000, IncrementalSource.Mutation, adds=[]),
    )
    selection = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    )
    assert selection.timestamps == [
        HEAD,
        HEAD + 4_000,
        HEAD + 5_000,
        HEAD + 6_000,
        HEAD + 20_000,
    ]
    assert selection.user_activity_events == 1


def test_a_page_load_buys_a_wider_radius_than_a_click(tmp_path):
    conn, slice_id = one_slice(
        tmp_path,
        page_load(HEAD + 10_000),
        incremental(HEAD + 30_000, IncrementalSource.Mutation, adds=[]),
    )
    selection = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    )
    around = [ts for ts in selection.timestamps if HEAD + 7_000 <= ts <= HEAD + 13_000]
    assert around == [
        HEAD + 8_000,
        HEAD + 9_000,
        HEAD + 10_000,
        HEAD + 11_000,
        HEAD + 12_000,
    ]


def test_screenshot_moments_never_run_past_the_slice(tmp_path):
    # User activity in the last second: the radius reaches beyond the final event, and
    # nothing may be selected there — there is no slice left to replay.
    conn, slice_id = one_slice(tmp_path, click(HEAD + 2_500))
    selection = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    )
    assert max(selection.timestamps) == HEAD + 2_500
    assert selection.timestamps == [HEAD, HEAD + 2_000, HEAD + 2_500]


def test_screenshot_moments_never_run_before_the_slice_head(tmp_path):
    conn, slice_id = one_slice(
        tmp_path,
        click(HEAD + 200),
        incremental(HEAD + 8_000, IncrementalSource.Mutation, adds=[]),
    )
    selection = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    )
    assert min(selection.timestamps) == HEAD
    assert selection.timestamps == [HEAD, HEAD + 1_000, HEAD + 8_000]


# The screenshot plane's floor. A rendered screenshot is testimony of what the page showed, and
# before the covering snapshot the slice holds no recorded DOM — the replayer paints the
# later snapshot's state there, so a screenshot addressed at those moments would be stamped
# with a time the pixels never held. The events themselves are untouched: they are real
# testimony of *when*, and they ride the event stream.


def test_no_screenshot_is_offered_before_the_covering_snapshot(tmp_path):
    conn, slice_id = lead_slice(
        tmp_path,
        click(HEAD + 2_064),
        incremental(HEAD + 9_000, IncrementalSource.Mutation, adds=[]),
    )
    selection = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    )
    assert min(selection.timestamps) == SNAPSHOT, (
        "the slice opens 17.9s earlier, and none of that lead can be shown"
    )


def test_screenshot_moments_anchor_at_the_covering_snapshot(tmp_path):
    conn, slice_id = lead_slice(
        tmp_path,
        click(SNAPSHOT + 2_000),
        incremental(HEAD + 9_000, IncrementalSource.Mutation, adds=[]),
    )
    selection = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    )
    assert selection.timestamps == [
        SNAPSHOT,
        SNAPSHOT + 1_000,
        SNAPSHOT + 2_000,
        SNAPSHOT + 3_000,
        HEAD + 9_000,
    ], "the marks sit on the snapshot's lattice, not the first event's"


def test_user_activity_before_the_covering_snapshot_earns_no_screenshot_before_it(
    tmp_path,
):
    # A real head slice carries live incrementals ahead of the pair. Their radius reaches
    # forward into the addressable bounds, and nothing reaches back out of them.
    conn, ids = build(
        tmp_path,
        [
            marker(HEAD - 5_000, EventType.DomContentLoaded),
            incremental(HEAD - 500, IncrementalSource.Scroll, id=1, x=0, y=40),
            meta(HEAD),
            snapshot(SNAPSHOT),
            incremental(HEAD + 9_000, IncrementalSource.Mutation, adds=[]),
        ],
    )
    selection = select_screenshots(
        conn,
        ids[0],
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    )
    assert selection.user_activity_events == 1
    assert selection.timestamps == [SNAPSHOT, HEAD + 9_000]


def test_a_slice_with_no_covering_snapshot_has_no_screenshot_plane(tmp_path):
    conn, ids = build(
        tmp_path,
        [meta(HEAD - 1), snapshot(HEAD)],
        [meta(HEAD + 5_000), click(HEAD + 6_000)],
    )
    assert len(ids) == 2
    with pytest.raises(ValueError, match="covering"):
        select_screenshots(conn, ids[1])


def test_user_activity_events_counts_events_not_screenshots(tmp_path):
    conn, slice_id = one_slice(
        tmp_path,
        click(HEAD + 1_000),
        click(HEAD + 1_100),
        click(HEAD + 1_200),
        incremental(HEAD + 9_000, IncrementalSource.Mutation, adds=[]),
    )
    selection = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    )
    assert selection.user_activity_events == 3
    assert len(selection.timestamps) < 3 + 2  # the three clicks share their marks


def test_widening_the_interval_is_the_only_way_to_pay_less(tmp_path):
    # Selection is blind to pixels by design, so the one cost lever is the interval.
    events = [click(HEAD + t) for t in range(1_000, 20_000, 1_000)]
    conn, slice_id = one_slice(tmp_path, *events)
    fine = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    )
    coarse = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=5000,
        activity_radius_ms=5000,
        pageload_radius_ms=5000,
    )
    assert len(coarse.timestamps) < len(fine.timestamps)
    assert coarse.user_activity_events == fine.user_activity_events


# Refusals — a radius under the interval cannot guarantee coverage, so it is never
# silently accepted.


def test_an_activity_radius_under_the_screenshot_interval_is_refused(tmp_path):
    conn, slice_id = one_slice(tmp_path, click(HEAD + 1_000))
    with pytest.raises(ValueError, match="activity_radius_ms"):
        select_screenshots(
            conn,
            slice_id,
            screenshot_interval_ms=1000,
            activity_radius_ms=999,
            pageload_radius_ms=2000,
        )


def test_a_pageload_radius_under_the_screenshot_interval_is_refused(tmp_path):
    conn, slice_id = one_slice(tmp_path, click(HEAD + 1_000))
    with pytest.raises(ValueError, match="pageload_radius_ms"):
        select_screenshots(
            conn,
            slice_id,
            screenshot_interval_ms=1000,
            activity_radius_ms=1000,
            pageload_radius_ms=999,
        )


def test_a_slice_with_no_events_is_refused(tmp_path):
    conn, _ = one_slice(tmp_path, click(HEAD + 1_000))
    with pytest.raises(ValueError, match="events"):
        select_screenshots(conn, 9999)


# The engine's moments: the selector as the engine actually calls it — per slice,
# because a screenshot's identity is (label, moment) and slices may overlap in
# time, so a bare timestamp names no screenshot.


def test_each_slice_anchors_its_own_moments(tmp_path):
    # The second slice opens at a non-multiple offset from the first, so a
    # window-anchored marks and per-slice-anchored ones cannot agree.
    conn, ids = build(
        tmp_path,
        [
            meta(HEAD - 1),
            snapshot(HEAD),
            click(HEAD + 2_000),
            incremental(HEAD + 4_000, IncrementalSource.Mutation, adds=[]),
        ],
        [
            meta(HEAD + 10_499),
            snapshot(HEAD + 10_500),
            click(HEAD + 12_500),
            incremental(HEAD + 14_500, IncrementalSource.Mutation, adds=[]),
        ],
    )
    assert len(ids) == 2
    moments = _moments(conn, ids, screenshot_interval_ms=1000)
    assert sorted(moments) == sorted(ids), "one moments per slice"
    assert HEAD + 2_000 in moments[ids[0]]  # slice 1's moments: head + k*1000
    assert HEAD + 12_500 in moments[ids[1]]  # slice 2's: its own head + k*1000
    assert HEAD + 12_000 not in moments[ids[1]]


def test_the_window_clamp_is_start_inclusive_and_end_exclusive(tmp_path):
    conn, slice_id = one_slice(
        tmp_path, *[click(HEAD + t) for t in range(1_000, 10_000, 1_000)]
    )
    full = _moments(conn, [slice_id], screenshot_interval_ms=1000)[slice_id]
    cut = _moments(
        conn, [slice_id], HEAD + 3_000, HEAD + 6_000, screenshot_interval_ms=1000
    )[slice_id]
    assert HEAD + 3_000 in full and HEAD + 6_000 in full
    assert cut == [HEAD + 3_000, HEAD + 4_000, HEAD + 5_000]


def test_a_coarse_screenshot_moments_never_orphans_user_activity(tmp_path):
    # The engine widens both radii to at least the interval, so widening the interval
    # degrades coverage uniformly instead of dropping user activity that falls between marks.
    conn, slice_id = one_slice(
        tmp_path,
        click(HEAD + 12_000),
        incremental(HEAD + 40_000, IncrementalSource.Mutation, adds=[]),
    )
    moments = _moments(conn, [slice_id], screenshot_interval_ms=5000)
    assert HEAD + 10_000 in moments[slice_id]
    with pytest.raises(ValueError):
        select_screenshots(
            conn,
            slice_id,
            screenshot_interval_ms=5000,
            activity_radius_ms=1000,
            pageload_radius_ms=2000,
        )


def test_a_bounded_window_anchors_on_its_own_first_and_last_events(tmp_path):
    # A split window's slice endpoints are clamped away, so without these anchors the
    # window's own opening and closing moments would have no screenshot.
    conn, slice_id = one_slice(
        tmp_path,
        *[click(HEAD + t) for t in range(1_000, 30_000, 1_000)],
        incremental(HEAD + 30_500, IncrementalSource.Mutation, adds=[]),
    )
    moments = _moments(
        conn, [slice_id], HEAD + 10_500, HEAD + 20_500, screenshot_interval_ms=1000
    )[slice_id]
    assert min(moments) == HEAD + 11_000  # the window's first in-bounds event
    assert max(moments) == HEAD + 20_000  # its last
    assert all(HEAD + 11_000 <= ts <= HEAD + 20_000 for ts in moments), (
        "the moments stays inside the slice's own event bounds"
    )


def test_an_unbounded_screenshot_moments_adds_no_anchors_of_its_own(tmp_path):
    conn, slice_id = one_slice(
        tmp_path,
        click(HEAD + 2_000),
        incremental(HEAD + 9_000, IncrementalSource.Mutation, adds=[]),
    )
    moments = _moments(conn, [slice_id], screenshot_interval_ms=1000)[slice_id]
    expected = select_screenshots(
        conn,
        slice_id,
        screenshot_interval_ms=1000,
        activity_radius_ms=1000,
        pageload_radius_ms=2000,
    ).timestamps
    assert moments == expected


def test_the_screenshot_momentss_bounds_open_at_the_covering_snapshot(tmp_path):
    conn, slice_id = lead_slice(
        tmp_path,
        click(SNAPSHOT + 2_000),
        incremental(HEAD + 9_000, IncrementalSource.Mutation, adds=[]),
    )
    moments = _moments(conn, [slice_id], screenshot_interval_ms=1000)[slice_id]
    assert min(moments) == SNAPSHOT, (
        "the bounds' opening anchor is the snapshot, not the slice's first event"
    )
    assert max(moments) == HEAD + 9_000


def test_a_window_reaching_into_the_lead_still_opens_at_the_snapshot(tmp_path):
    conn, slice_id = lead_slice(
        tmp_path,
        click(SNAPSHOT + 2_000),
        incremental(HEAD + 9_000, IncrementalSource.Mutation, adds=[]),
    )
    moments = _moments(
        conn, [slice_id], HEAD - LEAD_MS, HEAD + 9_001, screenshot_interval_ms=1000
    )[slice_id]
    assert min(moments) == SNAPSHOT


def test_a_bounded_window_with_no_in_bounds_events_is_refused(tmp_path):
    conn, slice_id = one_slice(tmp_path, click(HEAD + 2_000))
    with pytest.raises(ValueError, match="events"):
        _moments(
            conn, [slice_id], HEAD + 50_000, HEAD + 60_000, screenshot_interval_ms=1000
        )
