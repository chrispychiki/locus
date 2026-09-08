"""Sessions — the operator's definition of a session applied to the recording, derived into the db like every other derived surface.

The definition is the operator's, three numbers in config/definitions.toml under the deployment root (deployment.definitions): the inactivity gap that ends a session, and the duration and pageview floors that make one engaged.
Everything else is invariant and lives here once: which events are user activity (user_activity.py — the one definition, PageLoad among them), that a session is a run of those split wherever two sit further apart than the gap, that a session begins at its first user activity — a page load, or for a page context that outlived the gap, the act that resumed it — and ends at its last, that its pageviews are its page loads, and that a session not engaged is a bounce.
Nothing downstream re-derives any of it; a count is a query over `sessions`, and a cut is a join.

A page load is a PageLoad event — one per full load and one per route change — and also a page context that opened and died before rrweb captured its DOM: its head slice holds no PageLoad, but the context did open, and its open is the one act a bounce that fast leaves behind (slices.opened_a_page). A prerendered context nobody saw is not a page load.

Active time is a downstream quantity, computed over the boundary the definition draws: the stretches of user activity inside a session — acts no further apart than user_activity.py's grain, the gap the player skips as inactivity — summed. A session of one act has none.

materialize_sessions stamps the derivation into the db the way materialize_slices does: one summary row per session, and `session_id` on every event of the visitor — an event inside a session's span, or within the gap after its last act (the inactivity timer still running), or within the gap before its first (a page context's Meta and markers precede the page load that starts the session) takes that session, the preceding one winning where both reach; an event no session reaches stays NULL. Sessions re-derive per visitor whenever the visitor's events change — a chunk landing days late can extend a session, merge two, or add a pageview — and for every visitor when the definition or this code changes, which the derivation vintage detects (derive.py). Session ids are re-minted on every derivation; a session is addressed by its visitor and its start.
"""

import sqlite3

from .db import CANONICAL_ORDER
from .kinds import Kind
from .slices import opened_a_page
from .user_activity import INACTIVE_PERIOD_MS, user_activity_clause


def split_sessions(
    moments: list[tuple[int, bool]], gap_ms: int
) -> list[list[tuple[int, bool]]]:
    """One visitor's moments on one site — (ts_ms, is_page_load), any order — as runs split wherever consecutive moments sit more than gap_ms apart."""
    runs: list[list[tuple[int, bool]]] = []
    for moment in sorted(moments):
        if runs and moment[0] - runs[-1][-1][0] <= gap_ms:
            runs[-1].append(moment)
        else:
            runs.append([moment])
    return runs


def user_activity_ms(timestamps: list[int]) -> int:
    """Time inside stretches of user activity: consecutive moments no further apart than the grain form a stretch, and the stretches' spans sum. A lone moment spans nothing."""
    total = 0
    stretch_start = previous = None
    for ts in sorted(timestamps):
        if previous is None or ts - previous > INACTIVE_PERIOD_MS:
            if previous is not None:
                total += previous - stretch_start
            stretch_start = ts
        previous = ts
    if previous is not None:
        total += previous - stretch_start
    return total


def judge(run: list[tuple[int, bool]], definition: dict) -> dict:
    """One session's row from its run of moments, under the definition's numbers."""
    timestamps = [ts for ts, _ in run]
    start, end = timestamps[0], timestamps[-1]
    pageviews = sum(1 for _, is_page_load in run if is_page_load)
    duration = end - start
    engaged = (
        duration > definition["engaged"]["min_seconds"] * 1000
        or pageviews >= definition["engaged"]["min_pageviews"]
    )
    return {
        "start_ts": start,
        "end_ts": end,
        "duration_ms": duration,
        "pageviews": pageviews,
        "user_activity_ms": user_activity_ms(timestamps),
        "engaged": int(engaged),
    }


def materialize_sessions(
    conn: sqlite3.Connection, visitor_id: str, definition: dict
) -> dict:
    """One visitor's sessions, derived from their distilled events and written to `sessions` and `events.session_id`. Reads the user activity off the flat columns (user_activity_clause) and the visitor's slices for page contexts that opened without a page load; the rest of the visitor's rows are stamped by their place against the sessions, never decoded."""
    gap_ms = definition["session"]["inactivity_minutes"] * 60_000
    clause, params = user_activity_clause()
    moments: dict[str | None, list[tuple[int, bool]]] = {}
    for row in conn.execute(
        f"SELECT timestamp, type_str, snippet FROM events "
        f"WHERE visitor_id = ? AND ({clause}) {CANONICAL_ORDER}",
        (visitor_id, *params),
    ):
        moments.setdefault(row["snippet"], []).append(
            (row["timestamp"], row["type_str"] == Kind.PAGE_LOAD)
        )
    for slice_row in conn.execute(
        "SELECT * FROM slices WHERE visitor_id = ? AND status != 'rescued'",
        (visitor_id,),
    ).fetchall():
        if (
            opened_a_page(conn, slice_row)
            and not conn.execute(
                "SELECT 1 FROM events WHERE slice_id = ? AND type_str = ? LIMIT 1",
                (slice_row["id"], Kind.PAGE_LOAD),
            ).fetchone()
        ):
            moments.setdefault(slice_row["snippet"], []).append(
                (slice_row["start_ts"], True)
            )

    conn.execute("DELETE FROM sessions WHERE visitor_id = ?", (visitor_id,))
    conn.execute(
        "UPDATE events SET session_id = NULL WHERE visitor_id = ?", (visitor_id,)
    )
    count = 0
    for snippet, on_site in moments.items():
        reach_floor = None
        for run in split_sessions(on_site, gap_ms):
            session = judge(run, definition)
            session_id = conn.execute(
                "INSERT INTO sessions (visitor_id, snippet, start_ts, end_ts, "
                "duration_ms, pageviews, user_activity_ms, engaged) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    visitor_id,
                    snippet,
                    session["start_ts"],
                    session["end_ts"],
                    session["duration_ms"],
                    session["pageviews"],
                    session["user_activity_ms"],
                    session["engaged"],
                ),
            ).lastrowid
            # A session reaches one gap either side of its span; where the gap after one
            # session meets the gap before the next, the earlier session's timer wins. The
            # reaches are therefore disjoint ranges of the visitor's clock, each stamped by
            # one range update over the visitor's own index (`+snippet` keeps the planner
            # off the site index, whose range would be every visitor's rows in the window).
            low = session["start_ts"] - gap_ms
            if reach_floor is not None:
                low = max(low, reach_floor)
            high = session["end_ts"] + gap_ms
            conn.execute(
                "UPDATE events SET session_id = ? WHERE visitor_id = ? AND +snippet IS ? "
                "AND timestamp BETWEEN ? AND ?",
                (session_id, visitor_id, snippet, low, high),
            )
            reach_floor = high + 1
            count += 1
    conn.commit()
    return {"sessions": count}
