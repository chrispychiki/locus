"""The one definition of user activity: rrweb's rule as the base, pinned to rrweb's own source, with Locus's additions and exclusions; the raw predicate and the SQL clause agree."""

import re
from pathlib import Path

import pytest
from locus.evidence import user_activity
from locus.evidence.db import connect
from locus.evidence.kinds import Kind
from locus.evidence.rrweb_constants import (
    EventType,
    IncrementalSource,
    MouseInteractions,
)
from locus.evidence.user_activity import (
    INACTIVE_PERIOD_MS,
    RRWEB_USER_INTERACTION_SOURCES,
    USER_ACTIVITY_KINDS,
    is_user_activity,
    user_activity_clause,
)

RRWEB_DIST = (
    Path(__file__).resolve().parents[1]
    / "distill"
    / "node_modules"
    / "rrweb"
    / "dist"
    / "rrweb.js"
)


def incremental(source, **data):
    return {
        "type": EventType.IncrementalSnapshot,
        "timestamp": 0,
        "data": {"source": source, **data},
    }


def interaction(kind):
    return incremental(IncrementalSource.MouseInteraction, type=kind, id=1, x=0, y=0)


def test_the_base_is_rrwebs_own_rule_read_out_of_its_source():
    """rrweb's replayer decides user interaction by one enum range; the base here is that range through the generated enums, and this reads the rule out of the vendored dist so an upgrade that changes it fails here instead of drifting."""
    assert RRWEB_DIST.exists(), (
        f"{RRWEB_DIST} is missing — install the distill toolchain (bun install in "
        "evidence/distill) so rrweb's rule can be pinned"
    )
    source = RRWEB_DIST.read_text()
    body = re.search(r"isUserInteraction\(event\) \{(.*?)\n  \}", source, re.DOTALL)
    assert body, "rrweb no longer defines isUserInteraction(event) — re-derive the base"
    assert (
        "event.data.source > IncrementalSource.Mutation && "
        "event.data.source <= IncrementalSource.Input"
    ) in body.group(1), f"rrweb's rule changed:\n{body.group(1)}"
    assert RRWEB_USER_INTERACTION_SOURCES == {
        IncrementalSource.MouseMove,
        IncrementalSource.MouseInteraction,
        IncrementalSource.Scroll,
        IncrementalSource.ViewportResize,
        IncrementalSource.Input,
    }


def test_the_grain_is_rrwebs_own_inactivity_threshold():
    """The gap that ends a stretch of user activity is the gap the vendored player skips as inactivity, read out of its default config so the two cannot drift apart."""
    threshold = re.search(
        r"inactivePeriodThreshold: ([0-9e* .]+),", RRWEB_DIST.read_text()
    )
    assert threshold, (
        "rrweb no longer declares inactivePeriodThreshold — re-derive the grain"
    )
    assert eval(threshold.group(1)) == INACTIVE_PERIOD_MS


@pytest.mark.parametrize(
    "event, active",
    [
        ({"type": EventType.PageLoad, "timestamp": 0, "data": {}}, True),
        ({"type": EventType.PageVisible, "timestamp": 0, "data": {}}, True),
        ({"type": EventType.PageHidden, "timestamp": 0, "data": {}}, False),
        ({"type": EventType.Meta, "timestamp": 0, "data": {}}, False),
        ({"type": EventType.FullSnapshot, "timestamp": 0, "data": {}}, False),
        (interaction(MouseInteractions.Click), True),
        (interaction(MouseInteractions.TouchStart), True),
        (interaction(MouseInteractions.TouchCancel), True),
        (interaction(MouseInteractions.ContextMenu), True),
        (interaction(MouseInteractions.Focus), False),
        (interaction(MouseInteractions.Blur), False),
        (incremental(IncrementalSource.MouseMove, positions=[]), True),
        (incremental(IncrementalSource.TouchMove, positions=[]), True),
        (incremental(IncrementalSource.Scroll, id=1, x=0, y=0), True),
        (incremental(IncrementalSource.ViewportResize, width=1, height=1), True),
        (incremental(IncrementalSource.Drag, positions=[]), True),
        (incremental(IncrementalSource.MediaInteraction, id=1, type=0), True),
        (incremental(IncrementalSource.Selection, ranges=[]), True),
        (incremental(IncrementalSource.Input, id=1, text="a"), True),
        (
            incremental(IncrementalSource.Input, id=1, text="a", userTriggered=True),
            True,
        ),
        (
            incremental(IncrementalSource.Input, id=1, text="a", userTriggered=False),
            False,
        ),
        (incremental(IncrementalSource.Mutation, adds=[]), False),
        (incremental(IncrementalSource.StyleSheetRule, id=1), False),
        (incremental(IncrementalSource.Log), False),
    ],
)
def test_what_is_user_activity(event, active):
    assert is_user_activity(event) is active


def test_the_sql_clause_selects_exactly_what_the_predicate_does():
    """The distilled kinds the clause names are the predicate's sources and interactions by name, and the flag rule reads the same off `extra`."""
    conn = connect(":memory:")
    rows = [
        (Kind.CLICK, None, True),
        (Kind.FOCUS, None, False),
        (Kind.MOUSE_MOVE, None, True),
        (Kind.TOUCH_CANCEL, None, True),
        (Kind.SELECTION, None, True),
        (Kind.MEDIA_INTERACTION, None, True),
        (Kind.PAGE_VISIBLE, None, True),
        (Kind.PAGE_HIDDEN, None, False),
        (Kind.MUTATION, None, False),
        (Kind.INPUT, None, True),
        (Kind.INPUT, '{"userTriggered": true}', True),
        (Kind.INPUT, '{"userTriggered": false}', False),
    ]
    for i, (kind, extra, _) in enumerate(rows):
        conn.execute(
            "INSERT INTO events (id, visitor_id, timestamp, type, raw_json, "
            "content_hash, type_str, extra) VALUES (?, 'v', ?, 0, x'00', ?, ?, ?)",
            (i, i, str(i), kind, extra),
        )
    clause, params = user_activity_clause()
    selected = {
        row[0] for row in conn.execute(f"SELECT id FROM events WHERE {clause}", params)
    }
    assert selected == {i for i, (_, _, active) in enumerate(rows) if active}
    assert (
        Kind.FOCUS not in USER_ACTIVITY_KINDS and Kind.BLUR not in USER_ACTIVITY_KINDS
    )
    assert (
        Kind.MOUSE_MOVE in USER_ACTIVITY_KINDS and Kind.PAGE_LOAD in USER_ACTIVITY_KINDS
    )


def test_no_consumer_keeps_a_list_of_its_own():
    """Every module that judges user activity imports this one; a second list is the drift this definition exists to end."""
    src = Path(__file__).resolve().parents[1] / "src" / "locus" / "evidence"
    analysis = Path(__file__).resolve().parents[2] / "analysis" / "src"
    offenders = []
    for path in [*src.rglob("*.py"), *analysis.rglob("*.py")]:
        if path.name == "user_activity.py":
            continue
        text = path.read_text()
        if re.search(r"USER_ACTIVITY_(SOURCES|INTERACTIONS|KINDS)\s*=", text):
            offenders.append(str(path))
    assert not offenders, offenders
    assert user_activity.__doc__
