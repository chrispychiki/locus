"""Activity-screenshot moment selection — regular interval steps, gated by user activity.

Moments at regular steps of the screenshot interval across screenshot-addressable bounds — the slice's own by default, from its covering snapshot, the first moment it has a recorded DOM at all (slices.covering_snapshot_ts), through its last event; a window's per-slice bounds when the caller passes them — kept only within an activity radius of user activity (a wider radius around PageLoads), those bounds' endpoints always. What counts as user activity is locus.evidence.user_activity's one definition. Selection is blind to what a screenshot contains: semantic coverage is bought by spending screenshots near user activity, and a cull that judges pixels deletes whole classes of moment — a character typed into a field, an error line appearing, a price ticking over all sit below any usable pixel threshold. Pare cost by widening the interval, which degrades coverage uniformly in time.
"""

import math
import sqlite3
from dataclasses import dataclass

from locus.evidence.rrweb_constants import EventType
from locus.evidence.slices import covering_snapshot_ts, slice_events
from locus.evidence.user_activity import is_user_activity

SCREENSHOT_INTERVAL_MS = 1000
ACTIVITY_RADIUS_MS = 1000
PAGELOAD_RADIUS_MS = 2000


@dataclass
class Selection:
    timestamps: list[int]
    user_activity_events: int


def select_screenshots(
    conn: sqlite3.Connection,
    slice_id: int,
    *,
    bounds: tuple[int, int] | None = None,
    screenshot_interval_ms: int = SCREENSHOT_INTERVAL_MS,
    activity_radius_ms: int = ACTIVITY_RADIUS_MS,
    pageload_radius_ms: int = PAGELOAD_RADIUS_MS,
) -> Selection:
    """`bounds` is the screenshot-addressable extent to mark, absolute ms inclusive — a window's per-slice [screenshot_start_ts, end_ts] when given, the slice's own otherwise. `screenshot_interval_ms` is the spacing between marks, never an extent."""
    if activity_radius_ms < screenshot_interval_ms:
        raise ValueError(
            f"activity_radius_ms ({activity_radius_ms}) must be >= "
            f"screenshot_interval_ms ({screenshot_interval_ms}) to guarantee "
            "coverage around user activity"
        )
    if pageload_radius_ms < screenshot_interval_ms:
        raise ValueError(
            f"pageload_radius_ms ({pageload_radius_ms}) must be >= "
            f"screenshot_interval_ms ({screenshot_interval_ms}) to guarantee "
            "coverage around PageLoads"
        )

    events = slice_events(conn, slice_id)
    if not events:
        raise ValueError(f"slice {slice_id} has no events")

    # The marks cover what the bounds can show, not when the slice ran: they
    # open no earlier than the covering snapshot, and the events ahead of
    # that — a page context's first instants, before rrweb had a DOM to
    # capture — get no marks. User activity outside the bounds still counts and
    # still reaches inward, which is why the loop runs over every event and
    # only the marks are bounded.
    first_ts, last_ts = (
        bounds
        if bounds is not None
        else (covering_snapshot_ts(conn, slice_id), events[-1]["timestamp"])
    )
    kept = {first_ts, last_ts}
    acts = 0
    for event in events:
        if not is_user_activity(event):
            continue
        acts += 1
        radius = (
            pageload_radius_ms
            if event["type"] == EventType.PageLoad
            else activity_radius_ms
        )
        act_ts = event["timestamp"]
        k_min = math.ceil((act_ts - radius - first_ts) / screenshot_interval_ms)
        k_max = math.floor((act_ts + radius - first_ts) / screenshot_interval_ms)
        for k in range(max(0, k_min), k_max + 1):
            mark_ts = first_ts + k * screenshot_interval_ms
            if mark_ts <= last_ts:
                kept.add(mark_ts)

    return Selection(timestamps=sorted(kept), user_activity_events=acts)


def screenshot_moments(
    conn: sqlite3.Connection,
    window,
    *,
    screenshot_interval_ms: int = SCREENSHOT_INTERVAL_MS,
) -> dict[int, list[int]]:
    """The engine's activity-screenshot moments for a resolved window (window.resolve_window), in one home so composition-time counting and render-time capture can never disagree: {slice_id: sorted absolute timestamps}, per slice because a screenshot's identity is (slice label, moment) — slices may overlap in time, so a bare timestamp names no screenshot. Each slice's moments are its own user-activity-gated selection over the screenshot-addressable bounds its window row states — [screenshot_start_ts, end_ts], the same ones a pulled moment clamps into — with both radii widened to at least the screenshot interval (widening that interval must not orphan user activity)."""
    moments: dict[int, list[int]] = {}
    for s in window.slices:
        sel = select_screenshots(
            conn,
            s.slice_id,
            bounds=(s.screenshot_start_ts, s.end_ts),
            screenshot_interval_ms=screenshot_interval_ms,
            activity_radius_ms=max(screenshot_interval_ms, ACTIVITY_RADIUS_MS),
            pageload_radius_ms=max(screenshot_interval_ms, PAGELOAD_RADIUS_MS),
        )
        moments[s.slice_id] = sel.timestamps
    return moments
