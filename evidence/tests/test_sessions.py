"""Sessions — the operator's definition applied to the recording and derived into the db."""

from _support import click, full_snapshot, hidden, meta, stamped, text_mutation
from locus.evidence.db import connect
from locus.evidence.derive import run_distill
from locus.evidence.hydrate import hydrate
from locus.evidence.kinds import Kind
from locus.evidence.rrweb_constants import EventType, IncrementalSource
from locus.evidence.sessions import (
    judge,
    materialize_sessions,
    split_sessions,
    user_activity_ms,
)
from locus.evidence.slices import materialize_slices
from locus.evidence.user_activity import INACTIVE_PERIOD_MS

DEFINITION = {
    "session": {"inactivity_minutes": 30},
    "engaged": {"min_seconds": 10, "min_pageviews": 2},
}
MIN = 60_000
GAP = 30 * MIN


def test_a_gap_over_the_definition_splits_and_a_gap_at_it_does_not():
    runs = split_sessions([(0, True), (GAP, False), (2 * GAP + 1, True)], GAP)
    assert runs == [[(0, True), (GAP, False)], [(2 * GAP + 1, True)]]
    assert split_sessions([], GAP) == []


def test_active_time_sums_stretches_and_a_lone_act_spans_nothing():
    assert user_activity_ms([5]) == 0
    assert user_activity_ms([0, INACTIVE_PERIOD_MS]) == INACTIVE_PERIOD_MS
    assert user_activity_ms([0, INACTIVE_PERIOD_MS + 1]) == 0, (
        "past the grain the two are separate stretches, each of one act"
    )
    assert user_activity_ms([0, 4_000, 9_000, 60_000, 62_000]) == 9_000 + 2_000


def test_each_engaged_clause_stands_on_its_own_and_the_rest_is_a_bounce():
    pages = judge([(0, True), (5_000, True)], DEFINITION)
    assert pages["pageviews"] == 2 and pages["engaged"] == 1
    late = judge([(0, True), (10_001, False)], DEFINITION)
    assert (late["duration_ms"], late["engaged"]) == (10_001, 1)
    at_the_floor = judge([(0, True), (10_000, False)], DEFINITION)
    assert at_the_floor["engaged"] == 0, "exceeds, not reaches"
    bounce = judge([(0, True)], DEFINITION)
    assert bounce == {
        "start_ts": 0,
        "end_ts": 0,
        "duration_ms": 0,
        "pageviews": 1,
        "user_activity_ms": 0,
        "engaged": 0,
    }


def page_load(ts, url="https://x.com/"):
    return {
        "type": EventType.PageLoad,
        "timestamp": ts,
        "data": {"url": url, "title": "", "referrer": ""},
    }


def script_input(ts):
    return {
        "type": EventType.IncrementalSnapshot,
        "timestamp": ts,
        "data": {
            "source": IncrementalSource.Input,
            "id": 4,
            "text": "token",
            "isChecked": False,
            "userTriggered": False,
        },
    }


def _derived(tmp_path, visitors: dict[str, list[dict]]):
    """A db with each visitor's events hydrated, sliced, distilled, and sessioned under the test definition."""
    db = tmp_path / "events.db"
    conn = connect(db)
    for visitor_id, events in visitors.items():
        hydrate(conn, visitor_id, events)
        materialize_slices(conn, visitor_id)
    conn.close()
    run_distill(str(db))
    conn = connect(db)
    for visitor_id in visitors:
        materialize_sessions(conn, visitor_id, DEFINITION)
    return conn


def _sessions(conn, visitor_id):
    return [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM sessions WHERE visitor_id = ? ORDER BY start_ts",
            (visitor_id,),
        )
    ]


def test_sessions_derive_from_user_activity_and_every_event_takes_the_session_that_held_it(
    tmp_path,
):
    """User activity splits at the gap into sessions; a script's write is no act and opens nothing. The page context's Meta and snapshot precede its page load and count to the session it starts; a mutation after the last act counts to that session while its timer runs, and to none once the gap has passed."""
    t0 = 1_700_000_000_000
    events = stamped(
        [
            meta(t0, href="https://x.com/"),
            full_snapshot(t0 + 1),
            page_load(t0 + 50),
            click(t0 + 15_000, 3),
            hidden(t0 + 20 * MIN),
            script_input(t0 + 3 * 60 * MIN),
            text_mutation(t0 + 4 * 60 * MIN, 4, "later"),
            click(t0 + 5 * 60 * MIN, 3),
        ],
        snippet="snip",
    )
    conn = _derived(tmp_path, {"v1": events})
    first, second = _sessions(conn, "v1")
    assert (first["start_ts"], first["end_ts"], first["pageviews"]) == (
        t0 + 50,
        t0 + 15_000,
        1,
    )
    assert first["engaged"] == 1, "duration 14.95 s exceeds the 10 s floor"
    assert first["user_activity_ms"] == 0, "two acts further apart than the grain"
    assert (second["start_ts"], second["pageviews"], second["engaged"]) == (
        t0 + 5 * 60 * MIN,
        0,
        0,
    ), "a page context outliving the gap resumes on its next act, with no page load"
    assert second["snippet"] == "snip"

    stamps = {
        row["timestamp"]: row["session_id"]
        for row in conn.execute("SELECT timestamp, session_id FROM events")
    }
    assert stamps[t0] == stamps[t0 + 1] == stamps[t0 + 50] == first["id"], (
        "the context's opening events count to the session its page load starts"
    )
    assert stamps[t0 + 20 * MIN] == first["id"], "inside the gap after the last act"
    assert stamps[t0 + 3 * 60 * MIN] is None and stamps[t0 + 4 * 60 * MIN] is None, (
        "past the gap, no session reaches these"
    )
    assert stamps[t0 + 5 * 60 * MIN] == second["id"]


def test_a_page_context_that_died_before_its_snapshot_is_a_page_load_and_a_prerender_is_not(
    tmp_path,
):
    """A stub — a page opened and hidden before rrweb captured its DOM — holds no PageLoad, but the context opened: one page, then gone, a bounce. A zero-dimension viewport is a context nobody saw."""
    t0 = 1_700_000_000_000
    stub = stamped([hidden(t0 + 800)], snippet="snip")
    flat = meta(t0, href="https://x.com/")
    flat["data"].update(width=0, height=0)
    prerender = stamped([flat, full_snapshot(t0 + 1)], snippet="snip")
    conn = _derived(tmp_path, {"bounced": stub, "ghost": prerender})
    (session,) = _sessions(conn, "bounced")
    slice_start = conn.execute(
        "SELECT start_ts, status FROM slices WHERE visitor_id = 'bounced'"
    ).fetchone()
    assert slice_start["status"] == "discarded"
    assert (session["start_ts"], session["pageviews"], session["engaged"]) == (
        slice_start["start_ts"],
        1,
        0,
    )
    assert (
        conn.execute(
            "SELECT session_id FROM events WHERE visitor_id = 'bounced'"
        ).fetchone()["session_id"]
        == session["id"]
    ), "the stub's marker is the session's"
    assert _sessions(conn, "ghost") == []
    assert all(
        row["session_id"] is None
        for row in conn.execute(
            "SELECT session_id FROM events WHERE visitor_id = 'ghost'"
        )
    )


def test_a_head_with_a_page_load_counts_it_once(tmp_path):
    t0 = 1_700_000_000_000
    events = stamped(
        [meta(t0, href="https://x.com/"), full_snapshot(t0 + 1), page_load(t0 + 50)],
        snippet="snip",
    )
    conn = _derived(tmp_path, {"v": events})
    (session,) = _sessions(conn, "v")
    assert session["pageviews"] == 1


def test_sessions_are_per_site_and_re_derive_whole_when_a_late_chunk_joins_two(
    tmp_path,
):
    """One visitor id on two snippets is two runs. A chunk landing later with an act inside the gap between two sessions merges them: the derivation replaces the visitor's rows rather than appending."""
    t0 = 1_700_000_000_000
    on_a = stamped(
        [
            meta(t0, href="https://a.com/"),
            full_snapshot(t0 + 1),
            page_load(t0 + 50, "https://a.com/"),
        ],
        snippet="a",
    )
    on_b = stamped(
        [
            meta(t0 + 10, href="https://b.com/"),
            full_snapshot(t0 + 11),
            page_load(t0 + 60, "https://b.com/"),
        ],
        snippet="b",
    )
    later = stamped(
        [
            meta(t0 + 50 * MIN, href="https://a.com/x"),
            full_snapshot(t0 + 50 * MIN + 1),
            page_load(t0 + 50 * MIN + 50, "https://a.com/x"),
        ],
        snippet="a",
    )
    conn = _derived(tmp_path, {"v": on_a + on_b + later})
    sessions = _sessions(conn, "v")
    assert [(s["snippet"], s["pageviews"]) for s in sessions] == [
        ("a", 1),
        ("b", 1),
        ("a", 1),
    ]

    bridge = [
        {
            **click(t0 + 25 * MIN, 3),
            "_envelope": {**on_a[0]["_envelope"]},
        }
    ]
    hydrate(conn, "v", bridge)
    materialize_slices(conn, "v")
    conn.close()
    run_distill(str(tmp_path / "events.db"))
    conn = connect(tmp_path / "events.db")
    materialize_sessions(conn, "v", DEFINITION)
    merged = [s for s in _sessions(conn, "v") if s["snippet"] == "a"]
    assert (
        len(merged) == 1 and merged[0]["pageviews"] == 2 and merged[0]["engaged"] == 1
    )
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM events WHERE type_str = ? AND session_id IS NULL",
            (Kind.PAGE_LOAD,),
        ).fetchone()[0]
        == 0
    ), "every act of user activity is in a session"
