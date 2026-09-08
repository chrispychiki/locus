"""Compose one analysis's evidence window into a conversation.

The window — the slice set the caller named, resolved, labeled, and placed on one shared clock — is window.py's; this module consumes the resolved Window. It turns the window's distilled rows and rendered screenshots into one conversation: site context first (one WEBSITE block per site the window spans, attributed when there are several), then per visitor (their slices in time order) the recording facts, the session context, and the event stream interleaved with screenshots at their timestamps. The question is not composed here at all — it rides last, entire, on whichever turn writes the answer (prompts.question_task), sent by the caller and never kept in the history.

The event stream is the distilled representation, formatted as one line per event with timestamps relative to the window start; markdown projections ride with their FullSnapshot, literal content diffs with their mutation run. In a multi-slice window every stream line and screenshot anchor is stamped with its slice's label — the form the answer's citations take. The mapping back to absolute time (and so to a replay page's citation seek) is the returned manifest's window_start_ts: absolute = window_start_ts + offset.

The event-stream composer (event_stream) applies mechanical noise passes — content-blind, never semantic:
  - events sort by their canonical timestamps (batched moves were corrected to true time at hydration)
  - MouseDown/MouseUp that resolve into a Click on the same target collapse into the Click; unresolved presses (drags) stay
  - a Focus a recorded pointer interaction itself caused (same target, within the dispatch interval) folds into it, and a Blur within that interval of any pointer interaction folds — leaving the old element is the interaction's mechanical shadow, while a different-target Focus rides because it alone names what lit up; keyboard-driven focus, script focus-steals, and a blur to nowhere are behavior and keep their lines
  - churn that leaves the page text unchanged (empty-diff mutations, hidden-target writes, page machinery) never prints content, and each unbroken run leaves one counted trace line — the page may still have changed on screen, and the count is what lets a screenshot's change be placed
  - consecutive Selection events collapse to the final selection state
  - Input events superseded on the same element within the sub-perceptual settle interval collapse to the settled value (input state is last-wins — the replayer's own semantics); human keystrokes keep their cadence
  - a typing run names its field once, states its value whole at both ends, and between them prints only the span that differs from the value before it; a value identical to that one says so where the stored prefix hides the edit and drops where nothing changed
  - a Scroll target's text prints once per consecutive run, not per tick
  - a TouchStart target's text prints once per same-target run (intervening scrolls/moves don't reset it; a content change does)
  - class strings are capped
  - within a diff, a removed block above a threshold summarizes to its boundaries and count, and any other removed line prints as its size and boundaries wherever that is shorter than the line (the content was on screen before, by definition); a re-added block byte-identical to an earlier block in the same stream becomes a reference to that timestamp, and a block whose lines are almost all already in the stream emits only its novel lines — content appears in full at least once, repeats never pay full freight (the repeat memory spans a visitor's whole group of slices, threaded by the composer)
  - chains of single-line text flips at the same spot (timers, counters) collapse to first → last with a count
  - a projected link target's query string collapses to `?…` — where a link points is origin, path, and fragment; the arrival url keeps its full query on its own Meta/PageLoad line
"""

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from locus.evidence.db import CANONICAL_ORDER
from locus.evidence.hydrate import raw_event
from locus.evidence.kinds import Kind
from locus.evidence.text import describe_url, escape_value

from .model.protocol import Conversation
from .session_context import session_context_block, window_pages
from .window import Window, WindowManifest, WindowSlice, format_offset

PRESS_TYPES = {Kind.MOUSE_DOWN, Kind.MOUSE_UP}
POINTER_TYPES = {
    Kind.CLICK,
    Kind.DBL_CLICK,
    Kind.MOUSE_DOWN,
    Kind.MOUSE_UP,
    Kind.TOUCH_START,
    Kind.TOUCH_END,
    Kind.CONTEXT_MENU,
}
# A Focus/Blur a recorded pointer interaction itself caused is the interaction's own effect —
# duplicate testimony. The two fold asymmetrically. Focus folds only into a same-target
# interaction: a click on a label or on the icon inside a button focuses a *different* element,
# and that Focus is the only line naming what actually lit up. Blur folds into any pointer
# interaction in the fold interval: leaving the old element is the mechanical shadow of engaging
# elsewhere, whatever was engaged — Blur's own signal (an abandoned field, a script stealing
# focus) exists only when no pointer caused it. Browser dispatch stamps the caused event a few ms
# from its pointer event, in no guaranteed order, so the interval bounds that latency on both sides
# with slack while staying far under deliberate-action scale; it errs tight by design — a miss
# costs a duplicate line, never behavior.
FOCUS_FOLD_MS = 50

CLASS_CAP = 60
BOUNDARY_CAP = 60
REMOVED_SUMMARY_MIN_LINES = 10
REPEAT_REF_MIN_LINES = 5
NOVEL_FRACTION_MAX = 0.1
TICK_COLLAPSE_MIN = 3
TRIAD_RESOLVE_MS = 1_000
# The pointer drifts between press and release, and the browser stamps the click at the
# release — so a press/click pair on one element routinely differs by a few pixels.
# Two bounds fix the tolerance. Below: it must cover the platform's own click slop — the
# drift a browser accepts before a press stops being a click — which runs from Android's
# 8dp ViewConfiguration slop into the mid-teens on desktop engines, so anything a click
# event can carry fits under 15. Above: the check exists only to split two
# identically-rendered elements, and interactive targets are spaced no tighter than the
# 24px WCAG minimum, so the tolerance must stay under 24. 15 is the top of the slop
# range and safely under the spacing floor.
PRESS_DRIFT_PX = 15
SELECTION_SETTLE_MS = 1_000
# Below any human inter-keystroke interval, so only sub-perceptual churn — a script re-setting
# the same field faster than anyone can perceive — collapses, and every real keystroke keeps its line.
INPUT_SETTLE_MS = 50


def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# A projected link's target, when it carries a query string. The query on a
# link is addressing machinery — campaign and click ids, variant params the
# page template stamps into every self-referencing anchor — restated per
# anchor, and an ad-arrival page repeats its own several-hundred-byte query
# in each one. Where the link points is origin, path, and fragment; the query
# collapses to a `?…` residue so the fact it existed stays. The page the
# visitor actually arrived at keeps its full url on its own Meta/PageLoad
# line — attribution reads there, never off an anchor.
_LINK_QUERY = re.compile(r"\]\((https?://[^)?\s]*)\?[^)#\s]*(#[^)\s]*)?\)")


def stream_body(text: str) -> str:
    """A multi-line body (a snapshot's projection, a mutation's diff) as it rides under its event line: blank lines dropped — markdown's paragraph separators carry no testimony, and a diff line whose whole content is a blank ("+" / "-" alone) asserts only that a separator moved — and every link target's query string collapsed to `?…` (_LINK_QUERY says why). The body rides flush left; the event line's own header ("— page content:", "— content change:") is the delimiter."""
    return "\n".join(
        _LINK_QUERY.sub(r"](\1?…\2)", line)
        for line in text.split("\n")
        if line.strip() not in ("", "+", "-")
    )


def _named(row: sqlite3.Row) -> str | None:
    """An element showing no text of its own — an icon button, an image, an empty field — carries the page's name for it, derived once at distillation by the projection's rule (extra.label) and printed in the projection's own form, so the event line and the page call the control the same thing. A name is not what the visitor saw, so the line says the element is named X and never that the word X was on screen."""
    attrs = json.loads(row["extra"]) if row["extra"] else {}
    name = attrs.get("label")
    return f'label="{name}"' if isinstance(name, str) and name.strip() else None


def _delta(before: str, after: str) -> str:
    """What one value change actually changed. The longest common prefix and then the longest common suffix are what the reader is already holding, so only the differing span is stated, at the offset it starts. Exact string arithmetic — no unit to choose, so a paste shares nothing and states itself whole, while a one-character edit states one character. On repetitive text the two runs can overlap; the prefix is taken first, which yields a minimal edit reaching the new value rather than necessarily the keystroke that was made."""
    p = 0
    while p < len(before) and p < len(after) and before[p] == after[p]:
        p += 1
    s = 0
    while (
        s < len(before) - p and s < len(after) - p and before[-1 - s] == after[-1 - s]
    ):
        s += 1
    return f"@{p} {before[p : len(before) - s]!r}→{after[p : len(after) - s]!r}"


def _size_note(row: sqlite3.Row, key: str) -> str:
    """A value cut at the ceiling testifies its true size beside the visible cut — distillation stamps the uncut length (input_chars/text_chars) when and only when it cut, so the note rides exactly the values that lost something and the fact of a huge value is itself evidence."""
    if not row["extra"]:
        return ""
    total = json.loads(row["extra"]).get(key)
    if not total:
        return ""
    shown = len(row["input" if key == "input_chars" else "text"]) - 1
    return f" (first {shown} of {total} chars)"


def _element(row: sqlite3.Row) -> str | None:
    if not row["tag"]:
        return None
    element = f"<{row['tag']}"
    if row["class"]:
        element += f' class="{_cap(row["class"], CLASS_CAP)}"'
    named = _named(row) if not row["text"] else None
    if named:
        element += f" {named}"
    return element + ">"


def stream_visible(row: sqlite3.Row) -> bool:
    """Whether a distilled row prints as its own stream line. A False row is not silently gone — event_stream rolls each unbroken run of them into a counted trace line, so the fact of invisible churn stays testimony while its content never rides. Fails loud on an undistilled row: a silent skip there would thin the evidence without saying so."""
    kind = row["type_str"]
    if kind is None:
        raise ValueError(
            f"event {row['id']} has no distillation — run distill/distill.js first"
        )
    # Churn on an element the page never displays (a hidden field, a tracking
    # pixel's sandbox frame) is machinery writing state, not the visitor acting
    # — the same boundary that keeps it out of the projection keeps its content
    # out of the stream. Only an affirmative judgment (hidden == 1) drops; an
    # unresolved target (NULL) stays.
    if kind in (Kind.INPUT, Kind.FOCUS, Kind.BLUR) and row["hidden"] == 1:
        return False
    if kind not in Kind.STREAM:
        return False
    return not (kind == Kind.MUTATION and not row["diff"])


def format_event_line(
    row: sqlite3.Row,
    start_ms: int,
    timestamp: int | None = None,
    element: str | None | bool = True,
    label: str | None = None,
    value: str | None | bool = True,
    seen_values: set[str] | None = None,
) -> str:
    """One stream line for a row the stream carries (stream_visible is the gate, and event_stream applies it). `seen_values` is the held-value memory (event_stream threads it): a snapshot's field values the reader already holds print as held content."""
    kind = row["type_str"]
    ts = format_offset(
        timestamp if timestamp is not None else row["timestamp"], start_ms, label
    )
    if kind == Kind.FULL_SNAPSHOT:
        md = row["md"]
        if md is None:
            # "" is a projection of an empty page — a fact. None is the absence of a projection,
            # and rendering it as an empty page would tell the model the visitor sat looking at
            # nothing, which is a claim no evidence was ever produced for.
            raise ValueError(
                f"the snapshot at {ts} was never projected: its md column is NULL, so there is "
                f"no page content to testify to. Run `locus doctor`."
            )
        body = stream_body(md or "(empty page)")
        if seen_values is not None:
            body = _reference_values(body, seen_values)
        return f"{ts} {kind} — page content:\n{body}"
    if kind == Kind.MUTATION:
        return f"{ts} {kind} — content change:\n{stream_body(row['diff'])}"

    parts = [ts, kind]
    if row["url"]:
        parts.append(describe_url(row["url"]))
    if element is True:
        element = _element(row)
    if element:
        parts.append(element)
        if row["text"]:
            parts.append(f'"{row["text"]}"{_size_note(row, "text_chars")}')
    elif row["text"] and not row["tag"]:
        parts.append(f'"{row["text"]}"{_size_note(row, "text_chars")}')
    if value is True:
        value = (
            None
            if row["input"] is None
            else f"input={row['input']!r}{_size_note(row, 'input_chars')}"
        )
    if value:
        parts.append(value)
    if kind in (Kind.META, Kind.VIEWPORT_RESIZE) and row["extra"]:
        attrs = json.loads(row["extra"])
        if attrs.get("width") is not None and attrs.get("height") is not None:
            parts.append(f"{attrs['width']}×{attrs['height']}")
    if row["x"] is not None:
        coords = f"({row['x']},{row['y']})"
        parts.append(f"to offset {coords}" if kind == Kind.SCROLL else f"@{coords}")
    return " ".join(parts)


def _content(line: str) -> str:
    return line[2:] if line[1:2] == " " else line[1:]


def _boundaries(run: list[str]) -> tuple[str, str]:
    solid = [_content(l).strip() for l in run if _content(l).strip()]
    if not solid:
        return "", ""
    return _cap(solid[0], BOUNDARY_CAP), _cap(solid[-1], BOUNDARY_CAP)


def _reference(text: str) -> str:
    """The one form a reference to content the reader already holds takes: its size and its first and last characters, enough to recognize it."""
    return f"[{len(text)} chars: '{text[:BOUNDARY_CAP]}' … '{text[-BOUNDARY_CAP:]}']"


def _held(content: str) -> str:
    """Content the reader already holds — a removed line, the value a flip started from — as a reference when that is shorter than quoting it whole, and quoted whole otherwise."""
    ref = _reference(content)
    return ref if len(ref) < len(repr(content)) else repr(content)


# A field's value as the projection writes it: value="…", closed by the cut note, the checked
# state, the name, and the bracket that can follow it on the line (project.js). The value ends
# at the first quote that closes it that way; a quote inside the value followed by exactly that
# closing costs a missed reference, never a wrong one — a reference is only ever made of a
# value the memory holds whole.
_FIELD_VALUE = re.compile(
    r'value="(.*?)"((?: \(first \d+ of \d+ chars\))?(?: checked=(?:true|false))?(?: label="[^"]*")?)\]'
)


def _value_form(key: str, seen_values: set[str], whole: str) -> str:
    """A field's value where the stream states it whole: its size and boundaries when the reader already holds it — the page or an earlier Input stated it, or stated a value it is part of, a field trimmed at its cap being the same text ending earlier — and that is shorter than `whole` — the value as this line would state it — and `whole` otherwise, entering the memory either way. `key` is the projection's written form of the value (escape_value), the one form the page and the Input line share, and the form the reference quotes."""
    if any(key in held for held in seen_values):
        ref = _reference(key)
        if len(ref) < len(whole):
            return ref
    seen_values.add(key)
    return whole


def _reference_values(body: str, seen_values: set[str]) -> str:
    """A projection body — a snapshot's page, a diff's added lines — with every field value the reader already holds printed as held content, and every value it does not yet hold entering the memory."""
    if 'value="' not in body:
        return body

    def sub(m: re.Match) -> str:
        key, tail = m.group(1), m.group(2)
        return f"value={_value_form(key, seen_values, f'"{key}"')}{tail}]"

    return "\n".join(
        _FIELD_VALUE.sub(sub, line) if 'value="' in line else line
        for line in body.split("\n")
    )


def _line_ref(line: str) -> str:
    """A removed diff line printed as held content (_held): every removed line already stands in the stream, in the snapshot or the diff that added it."""
    content = _content(line).strip()
    held = _held(content)
    return f"- {held}" if held.startswith("[") else line


def _rewrite_diff(
    diff: str,
    ts_label: str,
    seen_blocks: dict[tuple[str, ...], str],
    seen_lines: set[str],
    seen_values: set[str],
) -> str:
    lines = diff.split("\n")
    out: list[str] = []
    run_prefix = None
    run: list[str] = []

    def flush() -> None:
        nonlocal run, run_prefix
        if not run:
            return
        if run_prefix == "-" and len(run) >= REMOVED_SUMMARY_MIN_LINES:
            first, last = _boundaries(run)
            out.append(f"- [{len(run)} lines: {first!r} … {last!r}]")
        elif run_prefix == "+" and len(run) >= REPEAT_REF_MIN_LINES:
            # The block itself is the key. A digest would let a collision assert
            # "identical to the content at 03:12" about a block that isn't —
            # a noise pass may never claim two different things are the same.
            key = tuple(run)
            novel = [l for l in run if _content(l) not in seen_lines]
            first, last = _boundaries(run)
            if key in seen_blocks:
                out.append(f"+ [{len(run)} lines, as at {seen_blocks[key]}]")
            elif len(novel) <= len(run) * NOVEL_FRACTION_MAX:
                out.append(
                    f"+ [{len(run)} lines, {len(novel) or 'none'} new: {first!r} … {last!r}]"
                )
                out.extend(_reference_values(l, seen_values) for l in novel)
            else:
                out.extend(_reference_values(l, seen_values) for l in run)
            seen_blocks.setdefault(key, ts_label)
            seen_lines.update(_content(l) for l in run)
        elif run_prefix == "-":
            out.extend(_line_ref(l) for l in run)
        else:
            seen_lines.update(_content(l) for l in run)
            out.extend(_reference_values(l, seen_values) for l in run)
        run, run_prefix = [], None

    for line in lines:
        prefix = line[0] if line and line[0] in "+-" else None
        if prefix != run_prefix:
            flush()
            run_prefix = prefix
        if prefix is None:
            out.append(line)
            run_prefix = None
        else:
            run.append(line)
    flush()
    return "\n".join(out)


def _single_flip(diff: str) -> tuple[str, str] | None:
    """If the diff is exactly one removed line and one added line, return (old, new) — the shape of a ticking timer or counter."""
    lines = diff.split("\n")
    minus = [l for l in lines if l.startswith("-")]
    plus = [l for l in lines if l.startswith("+")]
    hunks = [l for l in lines if l.startswith("@@")]
    if len(minus) == 1 and len(plus) == 1 and len(hunks) <= 1:
        return minus[0][1:].strip(), plus[0][1:].strip()
    return None


def event_stream(
    rows: list[sqlite3.Row],
    start_ms: int,
    label: str | None = None,
    seen_blocks: dict[tuple[str, ...], str] | None = None,
    seen_lines: set[str] | None = None,
    seen_values: set[str] | None = None,
) -> list[tuple[int, str]]:
    """The event-stream composer for one slice's rows → [(ts, line)], noise passes applied. `label` stamps every line with the slice's label (multi-slice windows); `seen_blocks`/`seen_lines`/`seen_values` are the repeat-collapse memory, threaded in by a caller composing several slices so a block or a field value shown in one slice is referenced, not re-printed, in the next."""
    seen_blocks = {} if seen_blocks is None else seen_blocks
    seen_lines = set() if seen_lines is None else seen_lines
    seen_values = set() if seen_values is None else seen_values
    # Rows that print partition from rows that don't; the latter are counted,
    # never silent — an unbroken run of churn that left the page text unchanged
    # (empty-diff mutations, hidden-target writes) becomes one trace line, so
    # a screenshot that changed with no diff behind it has a moment to land on.
    items = []
    churn_runs: list[dict] = []
    run = None
    for row in sorted(rows, key=lambda r: r["timestamp"]):
        if stream_visible(row):
            run = None
            items.append({"ts": row["timestamp"], "row": row, "kind": row["type_str"]})
        elif run is None:
            run = {"first": row["timestamp"], "last": row["timestamp"], "count": 1}
            churn_runs.append(run)
        else:
            run["last"] = row["timestamp"]
            run["count"] += 1

    next_selection_ts: list[int | None] = [None] * len(items)
    upcoming = None
    for i in range(len(items) - 1, -1, -1):
        next_selection_ts[i] = upcoming
        if items[i]["kind"] == Kind.SELECTION:
            upcoming = items[i]["ts"]

    # Input state is last-wins on the wire — the replayer overwrites the same
    # element's value on each Input event — and a value superseded on its own
    # element within the sub-perceptual settle interval was never a state anyone
    # read: a script re-setting a field every few milliseconds collapses to
    # what it settled on, while every human keystroke keeps its line. Identity
    # is the event's own target node id.
    next_same_input_ts: list[int | None] = [None] * len(items)
    latest_input_ts: dict = {}
    for i in range(len(items) - 1, -1, -1):
        if items[i]["kind"] == Kind.INPUT:
            node = raw_event(items[i]["row"]["raw_json"]).get("data", {}).get("id")
            if node is not None:
                next_same_input_ts[i] = latest_input_ts.get(node)
                latest_input_ts[node] = items[i]["ts"]

    def same_target(press: sqlite3.Row, click: sqlite3.Row) -> bool:
        if press["tag"] != click["tag"] or press["text"] != click["text"]:
            return False
        if press["x"] is not None and click["x"] is not None:
            return (
                abs(int(press["x"]) - int(click["x"])) <= PRESS_DRIFT_PX
                and abs(int(press["y"]) - int(click["y"])) <= PRESS_DRIFT_PX
            )
        return True

    def caused_by(other: dict, focus: sqlite3.Row, kind: str) -> bool:
        if other["kind"] not in POINTER_TYPES:
            return False
        if kind == Kind.BLUR:
            return True
        a = other["row"]
        return (
            a["tag"] == focus["tag"]
            and a["class"] == focus["class"]
            and a["text"] == focus["text"]
        )

    kept: list[dict] = []
    for i, item in enumerate(items):
        kind, row = item["kind"], item["row"]

        if kind in PRESS_TYPES:
            resolved = any(
                later["kind"] == Kind.CLICK and same_target(row, later["row"])
                for later in items[i + 1 : i + 6]
                if later["ts"] - item["ts"] <= TRIAD_RESOLVE_MS
            )
            if resolved:
                continue

        if kind in (Kind.FOCUS, Kind.BLUR):
            caused = False
            for j in range(i - 1, -1, -1):
                if item["ts"] - items[j]["ts"] > FOCUS_FOLD_MS:
                    break
                if caused_by(items[j], row, kind):
                    caused = True
                    break
            if not caused:
                for j in range(i + 1, len(items)):
                    if items[j]["ts"] - item["ts"] > FOCUS_FOLD_MS:
                        break
                    if caused_by(items[j], row, kind):
                        caused = True
                        break
            if caused:
                continue

        if kind == Kind.SELECTION:
            upcoming_ts = next_selection_ts[i]
            if (
                upcoming_ts is not None
                and upcoming_ts - item["ts"] <= SELECTION_SETTLE_MS
            ):
                continue

        if kind == Kind.INPUT:
            upcoming_ts = next_same_input_ts[i]
            if upcoming_ts is not None and upcoming_ts - item["ts"] <= INPUT_SETTLE_MS:
                continue

        kept.append(item)

    # rrweb resends a field's whole value on every change, so a run of keystrokes is one
    # document restated once per keystroke. A run states its value whole at both ends —
    # the reader needs the finished text, not a reconstruction of it — and between them
    # only what changed. A value identical to the one before it changed nothing the
    # recording can show: past the stored prefix it says so, and otherwise it says nothing
    # and the line goes, the same rule mutation churn with no content change already meets.
    inputs = [i for i, it in enumerate(kept) if it["kind"] == Kind.INPUT]
    prior = None
    silent = set()
    for i in inputs:
        value = kept[i]["row"]["input"]
        if value is not None and value == prior and not value.endswith("…"):
            silent.add(i)
        prior = value
    kept = [it for i, it in enumerate(kept) if i not in silent]

    # A run of events on one element cannot outlive the document it was in: past a snapshot
    # or a Meta the ids are re-minted and the elements are new ones, so an identical-looking
    # field is a different field. Without this an empty field on the page after a navigation
    # would state the whole previous page's text as deleted.
    epoch = 0
    for item in kept:
        if item["kind"] in (Kind.FULL_SNAPSHOT, Kind.META):
            epoch += 1
        item["epoch"] = epoch

    inputs = [i for i, it in enumerate(kept) if it["kind"] == Kind.INPUT]
    prior, prior_field = None, None

    # A field's identity is the event's own target node id, scoped to its document
    # epoch — the identity the settle pass already keys on. The rendered element
    # string is not identity: distinct fields can render to the same string (two
    # bare `<input class="form-control">`s), and chaining them into one run states
    # one field's value as an edit of another's.
    def field_of(item) -> tuple | None:
        node = raw_event(item["row"]["raw_json"]).get("data", {}).get("id")
        return None if node is None else (item["epoch"], node)

    for n, i in enumerate(inputs):
        value = kept[i]["row"]["input"]
        field = field_of(kept[i])
        after = inputs[n + 1] if n + 1 < len(inputs) else None
        whole = (
            field is None
            or value is None
            or prior is None
            or field != prior_field
            or after is None
            or field_of(kept[after]) != field
        )
        kept[i]["whole"] = value if whole else None
        kept[i]["value"] = (
            None
            if value is None or whole
            else "…"
            if value == prior
            else _delta(prior, value)
        )
        prior, prior_field = value, field

    stream: list[tuple[int, str]] = []
    prev_scroll_key = None
    prev_touch_key = None
    prev_input_key = None
    pending_flip: dict | None = None

    def flush_flip() -> None:
        nonlocal pending_flip
        if pending_flip is None:
            return
        if pending_flip["count"] >= TICK_COLLAPSE_MIN:
            first_ts = format_offset(pending_flip["first_ts"], start_ms, label)
            last_ts = format_offset(pending_flip["last_ts"], start_ms, label)
            stream.append(
                (
                    pending_flip["first_ts"],
                    (
                        f"{first_ts} Mutation — text flipped {pending_flip['count']}× "
                        f"through {last_ts}: "
                        f"{_held(pending_flip['first_old'])} → {pending_flip['last_new']!r}"
                    ),
                )
            )
        else:
            stream.extend(pending_flip["lines"])
        pending_flip = None

    for item in kept:
        kind, row, ts = item["kind"], item["row"], item["ts"]

        if kind == Kind.MUTATION:
            body = stream_body(row["diff"])
            ts_label = format_offset(ts, start_ms, label)
            flip = _single_flip(body)
            if flip is not None:
                old, new = flip
                rewritten = _rewrite_diff(
                    body, ts_label, seen_blocks, seen_lines, seen_values
                )
                line = (ts, f"{ts_label} Mutation — content change:\n{rewritten}")
                if pending_flip and pending_flip["last_new"] == old:
                    pending_flip["count"] += 1
                    pending_flip["last_new"] = new
                    pending_flip["last_ts"] = ts
                    pending_flip["lines"].append(line)
                else:
                    flush_flip()
                    pending_flip = {
                        "count": 1,
                        "first_old": old,
                        "last_new": new,
                        "first_ts": ts,
                        "last_ts": ts,
                        "lines": [line],
                    }
                continue
            flush_flip()
            rewritten = _rewrite_diff(
                body, ts_label, seen_blocks, seen_lines, seen_values
            )
            stream.append(
                (
                    ts,
                    f"{ts_label} Mutation — content change:\n{rewritten}",
                )
            )
            prev_scroll_key = None
            prev_touch_key = None
            continue

        if kind == Kind.FULL_SNAPSHOT and row["md"]:
            seen_lines.update(stream_body(row["md"]).split("\n"))
        if kind == Kind.SCROLL:
            key = (item["epoch"], row["tag"], row["text"])
            element = _element(row) if key != prev_scroll_key else None
            prev_scroll_key = key
            line = format_event_line(
                row, start_ms, timestamp=ts, element=element, label=label
            )
        elif kind == Kind.TOUCH_START:
            key = (item["epoch"], row["tag"], row["text"])
            element = _element(row) if key != prev_touch_key else None
            prev_touch_key = key
            prev_scroll_key = None
            line = format_event_line(
                row, start_ms, timestamp=ts, element=element, label=label
            )
        elif kind == Kind.INPUT:
            # Typing is a run of events on one field: naming it on every keystroke is the
            # field restated a hundred times, not a hundred facts. The run ends when the
            # field does and not before — an app that re-renders on every keystroke puts a
            # mutation between each pair, and none of them changed which field is being typed in.
            key = field_of(item)
            element = _element(row) if key is None or key != prev_input_key else None
            prev_input_key = key
            value = item["value"]
            if item["whole"] is not None:
                whole = item["whole"]
                stated = _value_form(escape_value(whole), seen_values, repr(whole))
                value = f"input={stated}{_size_note(row, 'input_chars')}"
            line = format_event_line(
                row,
                start_ms,
                timestamp=ts,
                element=element,
                label=label,
                value=value,
            )
        else:
            prev_scroll_key = None
            line = format_event_line(
                row, start_ms, timestamp=ts, label=label, seen_values=seen_values
            )
        stream.append((ts, line))
    flush_flip()

    for run in churn_runs:
        stamp = format_offset(run["first"], start_ms, label)
        if run["count"] == 1:
            stream.append(
                (run["first"], f"{stamp} 1 event, no change to the page text")
            )
        else:
            last = format_offset(run["last"], start_ms, label)
            stream.append(
                (
                    run["first"],
                    (
                        f"{stamp} {run['count']} events, no change to the page text, through {last}"
                    ),
                )
            )

    stream.sort(key=lambda entry: entry[0])
    return stream


def recording_facts(
    rows: list[sqlite3.Row],
    slice_pages: list[tuple[str | None, str, int | None]] | None = None,
    visitor: str | None = None,
    slices: list[WindowSlice] | None = None,
    window_start_ts: int | None = None,
    site: str | None = None,
) -> str:
    """Mechanical facts about one visitor's recording in the window, for orientation: wall-clock date range, totals, event-type distribution, device. `visitor` names the group when the window spans more than one, so each visitor's group is attributed in the evidence itself; `site` names the group's site when the window spans more than one, matching its WEBSITE block's attribution; `slices` lists the group's labeled slices with their window-clock bounds when the window is multi-slice, so the labels the citations use are declared in the evidence. Pages visited are the ordered page arrivals (window_pages): each slice's head plus the in-slice route PageLoads an SPA navigation fires in place. An arrival is counted when the url differs from the previous page's or it came by page-load — a route navigation or a real load is a new arrival, a checkout continuation of the same url is not. The list appears only when the group holds more than one page."""
    from locus.evidence.clock import utc_stamp

    duration_s = (rows[-1]["timestamp"] - rows[0]["timestamp"]) / 1000

    kinds = Counter(row["type_str"] for row in rows)
    distribution = ", ".join(f"{kind} ({count})" for kind, count in kinds.most_common())

    lines = (
        ([f"Site: {site}"] if site else [])
        + ([f"Visitor: {visitor}"] if visitor else [])
        + [
            (
                f"Date Range: {utc_stamp(rows[0]['timestamp'])} – {utc_stamp(rows[-1]['timestamp'])}"
            ),
            f"Recording: {len(rows)} events over {duration_s:.1f} seconds",
            f"Event Types: {distribution}",
        ]
    )

    if slices:
        bare = lambda ms: format_offset(ms, window_start_ts)[1:-1]
        lines.append(
            "Slices: "
            + "; ".join(
                f"{s.label} [{bare(s.start_ts)}, {bare(s.end_ts)}]" for s in slices
            )
        )

    device = next(
        (
            (row["device"], row["os"], row["browser"])
            for row in rows
            if row["device"] or row["os"] or row["browser"]
        ),
        None,
    )
    if device:
        lines.append("Device: " + " / ".join(p for p in device if p))

    pages = [(url, kind) for url, kind, _ in (slice_pages or []) if url]
    if len(pages) > 1:
        arrival_counts = Counter(
            url
            for i, (url, kind) in enumerate(pages)
            if i == 0 or url != pages[i - 1][0] or kind == "page-load"
        )
        total = sum(arrival_counts.values())
        unique = len(arrival_counts)
        if unique == 1:
            lines.append(
                "Pages Visited: 1" + (f" ({total} arrivals)" if total > 1 else "")
            )
        else:
            lines.append(f"Pages Visited: {unique} unique ({total} total arrivals)")
        for url, count in arrival_counts.most_common(5):
            lines.append(
                f"  - {describe_url(url)}"
                + (f" ({count} arrivals)" if count > 1 else "")
            )
        if unique > 5:
            lines.append(f"  ... and {unique - 5} more")

    return "<SUMMARY>\n" + "\n".join(lines) + "\n</SUMMARY>"


def screenshot_index(screenshot_dirs: list[str | Path]) -> dict[tuple[str, int], Path]:
    """Rendered screenshots on disk → {(slice label, absolute ts): path}. A screenshot's file name is its identity — `{label}_screenshot_{ts}.png` — minted by the render service from the same slice table that stamps the stream, so the screenshots beside an answer read in the answer's own citation vocabulary."""
    screenshots: dict[tuple[str, int], Path] = {}
    for directory in screenshot_dirs:
        for png in Path(directory).glob("*_screenshot_*.png"):
            label, _, ts = png.stem.rpartition("_screenshot_")
            screenshots[(label, int(ts))] = png
    return screenshots


def compose_window(
    conn: sqlite3.Connection,
    conversation: Conversation,
    window: Window,
    screenshots: dict[tuple[str, int], Path],
    site_contexts: dict[str, str],
) -> WindowManifest:
    """`window` is the resolved Window (resolve_window). Its bounds, when set, cut the window inside a slice — the route-boundary-piece case, when one slice alone out-prices the model's context. The stream cuts at any timestamp; screenshots for a later piece still render by replaying the slice forward from its head snapshot, so a piece that doesn't start at the snapshot carries no page-content projection — screenshots and the session-context block carry that context.

    The set may span visitors and may overlap in time: slice_table labels every slice on the one window clock, slices are grouped by visitor (each group opening with its own attributed SUMMARY and SESSION_CONTEXT blocks, then its RECORDING) and emitted slice by slice in time order — delimited in the evidence itself, never event-interleaved across slices. In a multi-slice window every stream line and screenshot anchor is stamped with its slice's label, which is the form the answer's citations take.

    `site_contexts` is the operator-stated business frame per site, keyed by snippet. A single entry is the window's one frame, emitted bare. A window spanning sites gets one attributed WEBSITE block per site and each visitor group's SUMMARY block names its site, so no site's slices ever read under another site's frame — a group whose snippet has no entry fails loud rather than composing misattributed evidence.

    `screenshots` is the {(slice label, absolute ts): path} map of the window's screenshots — the engine renders the activity-screenshots straight into it; screenshots already on disk index in via screenshot_index. Every screenshot composed here is pushed — it samples the session beside the stream — and is added as such; how that rides the wire (resized to a local card's per-image token ceiling, or at a Gemini card's activity_screenshots tier) is the conversation's own card's business."""
    table = list(window.slices)
    window_start, window_end = window.window_start, window.window_end
    multi = len(table) > 1
    by_label = {s.label: s for s in table}
    start_ms = window.start_ms

    def in_bounds(ts: int) -> bool:
        return (window_start is None or ts >= window_start) and (
            window_end is None or ts < window_end
        )

    rows_by_label: dict[str, list[sqlite3.Row]] = {}
    for s in table:
        rows = conn.execute(
            f"SELECT * FROM events WHERE slice_id = ? {CANONICAL_ORDER}", (s.slice_id,)
        ).fetchall()
        rows_by_label[s.label] = [row for row in rows if in_bounds(row["timestamp"])]

    screenshot_map: dict[tuple[str, int], Path] = {}
    for (label, ts), path in screenshots.items():
        if not in_bounds(ts):
            continue
        s = by_label.get(label)
        if s is None or not s.screenshot_start_ts <= ts <= s.end_ts:
            raise ValueError(
                f"screenshot ({label!r}, {ts}) lies outside every slice's "
                f"screenshot-addressable bounds in this window and would silently "
                f"vanish from the evidence — screenshots are rendered from the "
                f"window's slices, each from its covering snapshot through its last "
                f"event, so a screenshot no slice's bounds hold is a caller bug"
            )
        screenshot_map[(label, ts)] = path
    conversation.prefetch_images([str(path) for path in screenshot_map.values()])

    groups: dict[str, list[WindowSlice]] = {}
    for s in table:
        groups.setdefault(s.visitor, []).append(s)
    ordered_groups = sorted(
        groups.values(), key=lambda group: min(s.start_ts for s in group)
    )
    multi_visitor = len(ordered_groups) > 1

    site_of_group: dict[str, str | None] = {}
    for members in ordered_groups:
        snippets = {
            row["snippet"] for s in members for row in rows_by_label[s.label]
        } - {None}
        if len(snippets) > 1:
            raise ValueError(
                f"visitor {members[0].visitor}'s slices span snippets "
                f"{sorted(snippets)} — two sites' recordings sharing one "
                f"visitor id; a visitor group reads under exactly one site's "
                f"frame, so this group cannot be attributed and composing it "
                f"would file one site's slices under the other's"
            )
        site_of_group[members[0].visitor] = snippets.pop() if snippets else None
    multi_site = len(site_contexts) > 1
    if multi_site:
        unmapped = [v for v, s in site_of_group.items() if s not in site_contexts]
        if unmapped:
            raise ValueError(
                f"visitor group(s) {unmapped} resolve to no entry in "
                f"site_contexts {sorted(site_contexts)} — a multi-site window "
                f"composes each site's own frame beside its slices, and a "
                f"group without one would silently read under another site's"
            )

    # The payload is composed least-often-varying first: site context
    # first, then each visitor's mechanical facts and event stream with its
    # screenshots. The question is not composed here; it is sent on the answer
    # turn (prompts.question_task). Two runs over the same window therefore
    # send byte-identical evidence prefixes, and Gemini's implicit cache
    # sometimes discounts the repeat. Nothing arranges or relies on that.
    if not site_contexts:
        raise ValueError(
            "a window composes at least one site frame — site_contexts is empty"
        )
    if multi_site:
        seen_sites: list[str] = []
        for members in ordered_groups:
            site = site_of_group[members[0].visitor]
            if site not in seen_sites:
                seen_sites.append(site)
        for site in seen_sites:
            conversation.add_user_text(
                f'<WEBSITE site="{site}">\n{site_contexts[site]}\n</WEBSITE>'
            )
    else:
        conversation.add_user_text(
            f"<WEBSITE>\n{next(iter(site_contexts.values()))}\n</WEBSITE>"
        )

    n_screenshots = 0
    n_events = 0
    for members in ordered_groups:
        member_ids = [s.slice_id for s in members]
        group_rows = sorted(
            (row for s in members for row in rows_by_label[s.label]),
            key=lambda r: r["timestamp"],
        )
        n_events += len(group_rows)
        slice_pages = window_pages(conn, member_ids, window_start, window_end)
        conversation.add_user_text(
            recording_facts(
                group_rows,
                slice_pages,
                visitor=members[0].visitor if multi_visitor else None,
                slices=members if multi else None,
                window_start_ts=start_ms,
                site=site_of_group[members[0].visitor] if multi_site else None,
            )
        )
        conversation.add_user_text(
            session_context_block(
                conn,
                member_ids,
                window_start,
                window_end,
                labels={s.slice_id: s.label for s in members} if multi else None,
            )
        )

        seen_blocks: dict[tuple[str, ...], str] = {}
        seen_lines: set[str] = set()
        seen_values: set[str] = set()
        stream: list[tuple[int, int, int, str | Path]] = []
        for rank, s in enumerate(members):
            label = s.label if multi else None
            lines = event_stream(
                rows_by_label[s.label],
                start_ms,
                label=label,
                seen_blocks=seen_blocks,
                seen_lines=seen_lines,
                seen_values=seen_values,
            )
            for ts, line in lines:
                stream.append((rank, ts, 0, line))
            # Every screenshot rides under its own stamped anchor line: its moment
            # is stated in the stream's citation clock.
            for (screenshot_label, ts), path in screenshot_map.items():
                if screenshot_label != s.label:
                    continue
                stream.append(
                    (rank, ts, 1, f"{format_offset(ts, start_ms, label)} Screenshot")
                )
                stream.append((rank, ts, 2, path))
        stream.sort(key=lambda item: (item[0], item[1], item[2]))

        batch: list[str] = ["<RECORDING>"]
        for _, _, _, item in stream:
            if isinstance(item, Path):
                if batch:
                    conversation.add_user_text("\n".join(batch))
                    batch.clear()
                conversation.add_user_image(str(item), pushed=True)
                n_screenshots += 1
            else:
                batch.append(item)
        batch.append("</RECORDING>")
        conversation.add_user_text("\n".join(batch))

    return WindowManifest(
        window_start_ts=start_ms,
        n_events=n_events,
        n_screenshots=n_screenshots,
        slices=table,
    )
