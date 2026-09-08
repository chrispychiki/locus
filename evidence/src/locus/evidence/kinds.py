"""Event-kind names exactly as distillation emits them into type_str — derived from the canonical generated mapping, never typed, so event-stream logic that matches on kinds cannot drift from the strings actually in the db (an attribute typo here fails at import, an rrweb rename flows through on regeneration)."""

from .rrweb_constants import (
    EVENTTYPE_NAMES,
    INCREMENTALSOURCE_NAMES,
    MOUSEINTERACTIONS_NAMES,
    EventType,
    IncrementalSource,
    MouseInteractions,
)


class Kind:
    META = EVENTTYPE_NAMES[EventType.Meta]
    FULL_SNAPSHOT = EVENTTYPE_NAMES[EventType.FullSnapshot]
    PAGE_LOAD = EVENTTYPE_NAMES[EventType.PageLoad]
    PAGE_VISIBLE = EVENTTYPE_NAMES[EventType.PageVisible]
    PAGE_HIDDEN = EVENTTYPE_NAMES[EventType.PageHidden]
    MUTATION = INCREMENTALSOURCE_NAMES[IncrementalSource.Mutation]
    MOUSE_MOVE = INCREMENTALSOURCE_NAMES[IncrementalSource.MouseMove]
    VIEWPORT_RESIZE = INCREMENTALSOURCE_NAMES[IncrementalSource.ViewportResize]
    SCROLL = INCREMENTALSOURCE_NAMES[IncrementalSource.Scroll]
    INPUT = INCREMENTALSOURCE_NAMES[IncrementalSource.Input]
    TOUCH_MOVE = INCREMENTALSOURCE_NAMES[IncrementalSource.TouchMove]
    MEDIA_INTERACTION = INCREMENTALSOURCE_NAMES[IncrementalSource.MediaInteraction]
    DRAG = INCREMENTALSOURCE_NAMES[IncrementalSource.Drag]
    SELECTION = INCREMENTALSOURCE_NAMES[IncrementalSource.Selection]
    CLICK = MOUSEINTERACTIONS_NAMES[MouseInteractions.Click]
    DBL_CLICK = MOUSEINTERACTIONS_NAMES[MouseInteractions.DblClick]
    MOUSE_DOWN = MOUSEINTERACTIONS_NAMES[MouseInteractions.MouseDown]
    MOUSE_UP = MOUSEINTERACTIONS_NAMES[MouseInteractions.MouseUp]
    TOUCH_START = MOUSEINTERACTIONS_NAMES[MouseInteractions.TouchStart]
    TOUCH_END = MOUSEINTERACTIONS_NAMES[MouseInteractions.TouchEnd]
    TOUCH_CANCEL = MOUSEINTERACTIONS_NAMES[MouseInteractions.TouchCancel]
    CONTEXT_MENU = MOUSEINTERACTIONS_NAMES[MouseInteractions.ContextMenu]
    FOCUS = MOUSEINTERACTIONS_NAMES[MouseInteractions.Focus]
    BLUR = MOUSEINTERACTIONS_NAMES[MouseInteractions.Blur]
    # What an event stream prints as a line of its own: the visitor acting, and the page's
    # presentation changing. Every other kind rrweb records — a stylesheet rule or adoption, a
    # font load, a console line, a custom-element definition, a canvas draw, and any kind a
    # later rrweb adds — is the page's own machinery, counted and never printed.
    STREAM = frozenset(
        {
            META,
            FULL_SNAPSHOT,
            PAGE_LOAD,
            PAGE_VISIBLE,
            PAGE_HIDDEN,
            MUTATION,
            MOUSE_MOVE,
            VIEWPORT_RESIZE,
            SCROLL,
            INPUT,
            TOUCH_MOVE,
            MEDIA_INTERACTION,
            DRAG,
            SELECTION,
            CLICK,
            DBL_CLICK,
            MOUSE_DOWN,
            MOUSE_UP,
            TOUCH_START,
            TOUCH_END,
            TOUCH_CANCEL,
            CONTEXT_MENU,
            FOCUS,
            BLUR,
        }
    )
