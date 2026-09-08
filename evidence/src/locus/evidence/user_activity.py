"""User activity — the one definition of a person doing something at the device, as the recording testifies it.

rrweb's replayer carries its own rule (`isUserInteraction` in the vendored rrweb dist, used to skip inactivity): an IncrementalSnapshot whose source lies in the enum range Mutation < source <= Input — MouseMove, MouseInteraction, Scroll, ViewportResize, Input. That range is the base here, written through the generated enums so an enum change flows through, and pinned by a test that reads the rule out of rrweb's source, so an upgrade that changes it fails there instead of drifting.

What rrweb's range leaves out that still means a person: TouchMove, Drag, MediaInteraction, and Selection sit above Input in the enum and are in; a PageLoad is the visitor arriving and a PageVisible is the tab brought to the front — someone came back. What the range lets in that means nothing on its own: Focus and Blur fire from script (autofocus, `el.focus()` on load) and on tab return, and when a person caused one the click or tap that did is already counted, so both are out; an Input the recorder flagged `userTriggered: false` is a script writing a field — a tracking token, a timer — and is out. A ViewportResize stays in: rotation, a keyboard opening, a window dragged are a person's, and a resize can precede a long visibly-broken render with no interaction near it, which a gate that ignored resizes would leave unshown.

Two readings of the same sets: `is_user_activity` judges a raw event; `user_activity_clause` is the same test as a SQL fragment over the distilled columns (`type_str`, `extra`). Every consumer — session derivation, activity-screenshot selection — reads one of the two, never its own list.

User activity also has a grain: two acts further apart than INACTIVE_PERIOD_MS are separate stretches of doing, with nothing attested between them. The number is rrweb's own `inactivePeriodThreshold` — the gap its player skips as inactivity — pinned to the vendored dist by the same test, so the stretches a session's active time sums are exactly the stretches the replay plays through without skipping.
"""

from .kinds import Kind
from .rrweb_constants import (
    EVENTTYPE_NAMES,
    INCREMENTALSOURCE_NAMES,
    MOUSEINTERACTIONS_NAMES,
    EventType,
    IncrementalSource,
    MouseInteractions,
)

INACTIVE_PERIOD_MS = 10_000

RRWEB_USER_INTERACTION_SOURCES = frozenset(
    source
    for source in INCREMENTALSOURCE_NAMES
    if IncrementalSource.Mutation < source <= IncrementalSource.Input
)

ADDED_SOURCES = frozenset(
    {
        IncrementalSource.TouchMove,
        IncrementalSource.Drag,
        IncrementalSource.MediaInteraction,
        IncrementalSource.Selection,
    }
)
ADDED_EVENT_TYPES = frozenset({EventType.PageLoad, EventType.PageVisible})
EXCLUDED_INTERACTIONS = frozenset({MouseInteractions.Focus, MouseInteractions.Blur})

USER_ACTIVITY_SOURCES = RRWEB_USER_INTERACTION_SOURCES | ADDED_SOURCES
USER_ACTIVITY_INTERACTIONS = frozenset(MOUSEINTERACTIONS_NAMES) - EXCLUDED_INTERACTIONS


def is_user_activity(event: dict) -> bool:
    """Whether a raw rrweb event is user activity."""
    if event["type"] in ADDED_EVENT_TYPES:
        return True
    if event["type"] != EventType.IncrementalSnapshot:
        return False
    data = event.get("data") or {}
    source = data.get("source")
    if source == IncrementalSource.MouseInteraction:
        return data.get("type") in USER_ACTIVITY_INTERACTIONS
    if source == IncrementalSource.Input:
        return data.get("userTriggered") is not False
    return source in USER_ACTIVITY_SOURCES


# The distilled kind names the same test selects: a MouseInteraction distills to its subtype's
# name, every other source and event type to its own.
USER_ACTIVITY_KINDS = frozenset(
    {EVENTTYPE_NAMES[t] for t in ADDED_EVENT_TYPES}
    | {
        INCREMENTALSOURCE_NAMES[s]
        for s in USER_ACTIVITY_SOURCES
        if s != IncrementalSource.MouseInteraction
    }
    | {MOUSEINTERACTIONS_NAMES[i] for i in USER_ACTIVITY_INTERACTIONS}
)


def user_activity_clause() -> tuple[str, list]:
    """The same test over the flat columns, as a WHERE fragment and its parameters: the kind is one the definition names, and an Input carries no `userTriggered: false` in `extra` (`IS 0`, so an Input with no flag at all — a NULL `extra` — stays in, as the predicate keeps it)."""
    kinds = ", ".join("?" * len(USER_ACTIVITY_KINDS))
    clause = (
        f"type_str IN ({kinds}) AND NOT (type_str = ? "
        "AND json_extract(extra, '$.userTriggered') IS 0)"
    )
    return clause, [*sorted(USER_ACTIVITY_KINDS), Kind.INPUT]
