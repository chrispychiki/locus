"""Sessions — the operator's definition applied to a visitor's events."""

from _support import DEFINITION, MIN
from locus.evidence.sessions import (
    judge,
    split_sessions,
    user_activity_ms,
)
from locus.evidence.user_activity import INACTIVE_PERIOD_MS

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
