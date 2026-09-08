"""Kind names must stay identical to the strings distillation actually writes into type_str — the coupling that lets stream/selection logic match on kinds without drifting from the db. A typo or a stale rrweb rename here is a silent miss; this pins each Kind to the canonical generated mapping."""

from locus.evidence.kinds import Kind
from locus.evidence.rrweb_constants import (
    EVENTTYPE_NAMES,
    INCREMENTALSOURCE_NAMES,
    MOUSEINTERACTIONS_NAMES,
)


def test_every_kind_is_a_real_distillation_string():
    canonical = (
        set(EVENTTYPE_NAMES.values())
        | set(INCREMENTALSOURCE_NAMES.values())
        | set(MOUSEINTERACTIONS_NAMES.values())
    )
    declared = {
        value
        for name, value in vars(Kind).items()
        if not name.startswith("_") and isinstance(value, str)
    }
    assert declared, "Kind declares no names"
    assert declared <= canonical, (
        f"Kind names absent from the generated mapping: {declared - canonical}"
    )
    assert Kind.STREAM <= canonical, (
        f"STREAM names absent from the generated mapping: {Kind.STREAM - canonical}"
    )


def test_kinds_used_by_live_stream_logic_resolve():
    for attr in (
        "PAGE_LOAD",
        "MUTATION",
        "CLICK",
        "FOCUS",
        "BLUR",
        "MOUSE_DOWN",
        "MOUSE_UP",
    ):
        assert isinstance(getattr(Kind, attr), str)
