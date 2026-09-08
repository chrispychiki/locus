"""The evidence window: the slice set one analysis reads, resolved, labeled, and addressable.

A window is the slice set the caller named — whole slices, or one route-boundary piece of one slice via [window_start, window_end) bounds — placed on one shared clock: every offset is milliseconds from the window's first event, wherever that event lives, so concurrent recordings — a second tab, overlapping visitors — read as concurrent, their offsets genuinely overlapping instead of being serialized apart. Each named slice is labeled S1..Sn in time order by slice_table — the label is a window-local abbreviation of the slice id, nothing more — and the slice table (label → visitor, slice, bounds) rides the manifest into window.json, the expansion back to full ids.

A single-slice window stamps and cites bare offsets ([MM:SS.mmm]); a multi-slice window stamps every stream line and screenshot anchor with its slice's label ([S2 04:31.220]), and citations follow the stamps (citation_forms names the window's concrete forms for the validator's messages and the oracle). Citation → replay resolution is resolve_citation, a pure table lookup implemented once for every consumer: extraction, the judge, the engine's screenshot pulls, and anything opening a replay page. Wall-clock never becomes an address — resolution returns the recorder slice id and the absolute instant, and callers keep addressing by slice ids.
"""

import re
import sqlite3
from dataclasses import asdict, dataclass

from locus.evidence.slices import covering_snapshot_ts

from .session_context import window_pages


@dataclass(frozen=True)
class WindowSlice:
    """One row of the window's slice table: one named slice on the window clock. `label` is the window-local abbreviation (S1..Sn) citations use; `snippet` is the site the slice was recorded on, so the persisted table answers which analyses touched a site without a visitor id in hand; `slice` is the recorder slice id — the durable address callers replay by; `slice_id` is the db key analysis queries by within the composing process — persisted into window.json it is a record, not an address, because the db is a deletable cache whose rowids re-mint on rebuild, so later db reads re-resolve (visitor, slice) instead (ground.judge.current_slice_ids); `start_ts`/`end_ts` are the first and last of the slice's in-bounds events, absolute epoch ms. `screenshot_start_ts` opens the bounds the screenshot plane offers — [screenshot_start_ts, end_ts], from the covering snapshot, the first moment with a recorded DOM (slices.covering_snapshot_ts); the events ahead of it still ride the stream from start_ts."""

    label: str
    snippet: str
    visitor: str
    slice: str
    slice_id: int
    start_ts: int
    end_ts: int
    screenshot_start_ts: int


@dataclass
class WindowManifest:
    window_start_ts: int
    n_events: int
    n_screenshots: int
    slices: list[WindowSlice]


def slice_table(
    conn: sqlite3.Connection,
    slice_ids: list[int],
    window_start: int | None = None,
    window_end: int | None = None,
) -> list[WindowSlice]:
    """The window's slice table: each named slice labeled S1..Sn in time order (start, then end, then recorder id — fully data-determined, so the caller's argument order never changes a label). Every consumer of the window builds the table through here — composition, pricing, the engine, the render service — so one slice set always means one table. Every wall on what a window may name fires here: an unknown or non-replayable slice, a named slice contributing no in-bounds events, and one whose in-bounds events all fall in its lead, ahead of the covering snapshot — such a window could hold no screenshot of that slice at all."""
    if not slice_ids:
        raise ValueError("a window names at least one slice")
    if len(set(slice_ids)) != len(slice_ids):
        raise ValueError(
            f"a window lists each slice once; got duplicates in {slice_ids} "
            f"(a repeated slice would silently double its events in the stream)"
        )
    entries = []
    for slice_id in slice_ids:
        row = conn.execute(
            "SELECT recorder_slice, visitor_id, snippet, status FROM slices WHERE id = ?",
            (slice_id,),
        ).fetchone()
        if row is None:
            if conn.execute(
                "SELECT 1 FROM events WHERE slice_id = ? LIMIT 1", (slice_id,)
            ).fetchone():
                raise ValueError(
                    f"slice {slice_id} has events but no slices row — slices were never materialized; run `locus doctor`"
                )
            raise ValueError(f"unknown slice {slice_id}")
        if row["status"] != "replayable":
            raise ValueError(
                f"slice {slice_id} is not replayable ({row['status']}) — only replayable slices can be analyzed"
            )
        clauses, params = ["slice_id = ?"], [slice_id]
        if window_start is not None:
            clauses.append("timestamp >= ?")
            params.append(window_start)
        if window_end is not None:
            clauses.append("timestamp < ?")
            params.append(window_end)
        bounds = conn.execute(
            f"SELECT MIN(timestamp) lo, MAX(timestamp) hi FROM events WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()
        if bounds["lo"] is None:
            raise ValueError(
                f"window [{window_start}, {window_end}) holds no events from slice {slice_id}"
            )
        snapshot = covering_snapshot_ts(conn, slice_id)
        if snapshot > bounds["hi"]:
            raise ValueError(
                f"window [{window_start}, {window_end}) ends before slice "
                f"{slice_id}'s covering snapshot at {snapshot} — it holds only "
                f"the slice's lead, which has no recorded DOM, so no screenshot in "
                f"it could show anything"
            )
        entries.append(
            {
                "snippet": row["snippet"],
                "visitor": row["visitor_id"],
                "slice": row["recorder_slice"],
                "slice_id": slice_id,
                "start_ts": bounds["lo"],
                "end_ts": bounds["hi"],
                "screenshot_start_ts": max(bounds["lo"], snapshot),
            }
        )
    entries.sort(key=lambda e: (e["start_ts"], e["end_ts"], e["slice"]))
    return [WindowSlice(label=f"S{k}", **entry) for k, entry in enumerate(entries, 1)]


@dataclass(frozen=True)
class Window:
    """One resolved evidence window — the value every consumer takes (composition, the screenshot moments, pricing, the engine), minted by resolve_window so every wall has already fired by the time a Window exists. `slices` is the labeled slice table; `window_start`/`window_end` (absolute ms, half-open) are set exactly when the window is one route-boundary piece of a single slice."""

    slices: tuple[WindowSlice, ...]
    window_start: int | None = None
    window_end: int | None = None

    @property
    def slice_ids(self) -> list[int]:
        return [s.slice_id for s in self.slices]

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.slices]

    @property
    def start_ms(self) -> int:
        """The window clock's origin: the window's first in-bounds event."""
        return min(s.start_ts for s in self.slices)


def resolve_window(
    conn: sqlite3.Connection,
    slice_ids: list[int],
    window_start: int | None = None,
    window_end: int | None = None,
) -> Window:
    """A caller's slice set — whole slices, or one slice cut to a piece's [window_start, window_end) — resolved into the Window everything downstream shares. The walls are slice_table's, and resolution is where they fire: at the address, not deep inside composition or pricing."""
    return Window(
        tuple(slice_table(conn, slice_ids, window_start, window_end)),
        window_start,
        window_end,
    )


def slice_pieces(
    conn: sqlite3.Connection, slice_id: int
) -> list[tuple[str | None, int | None, int | None]]:
    """One slice's route-boundary pieces as (page url, window_start, window_end) — the structural units an over-window slice divides into, cut where a route arrival turned the page over wholesale. A cut is legal only strictly after the slice's covering snapshot: a window must hold at least one moment a screenshot can show, and everything ahead of the snapshot is lead — real stream testimony with no recorded DOM — so a route arrival in the lead folds into the first piece instead of opening one of its own. Every piece returned here resolves as a window by construction. A single piece means the slice is indivisible."""
    snapshot = covering_snapshot_ts(conn, slice_id)
    pieces: list[list] = []
    for k, (url, _, ts) in enumerate(window_pages(conn, [slice_id])):
        if k and ts <= snapshot:
            continue
        if pieces:
            pieces[-1][2] = ts
        pieces.append([url, ts if k else None, None])
    return [tuple(piece) for piece in pieces]


def citation_forms(labels: list[str]) -> dict[str, str]:
    """The window's concrete citation bracket forms, from its real slice labels. A single-slice window cites bare offsets; a multi-slice window's citation opens with the label of the slice it cites, exactly as the evidence is stamped (format_offset). The validator's defect messages and the oracle's extraction prompt name a window's forms through here."""
    if len(labels) <= 1:
        return {"cite": "[MM:SS.mmm]", "cite_range": "[MM:SS.mmm, MM:SS.mmm]"}
    example = labels[1]
    return {
        "cite": f"[{example} MM:SS.mmm]",
        "cite_range": f"[{example} MM:SS.mmm, MM:SS.mmm]",
    }


def resolve_citation(
    slices,
    window_start_ts: int,
    *,
    label: str | None = None,
    start_offset_ms: int | None = None,
    end_offset_ms: int | None = None,
) -> dict:
    """A citation → its replay address: a pure lookup in the window's slice table, the one implementation shared by extraction, the judge, the engine's screenshot pulls, and any caller opening a replay page at a cited moment.

    `slices` is the table exactly as window.json carries it (dicts) or as composed (WindowSlice rows). A bare citation resolves only in a single-slice window; a multi-slice window's citation must name its slice's label, and an unknown label fails loud. Returns {"slice": the table row, "start_ts": ..., "end_ts": ...} with the cited instants absolute (window_start_ts + offset) — the row's `slice` is the recorder slice id a replay page takes, so wall-clock never becomes an address."""
    rows = [asdict(s) if isinstance(s, WindowSlice) else dict(s) for s in slices]
    if not rows:
        raise ValueError("an empty slice table resolves nothing")
    roster = ", ".join(r["label"] for r in rows)
    if label is None:
        if len(rows) > 1:
            raise ValueError(
                f"a citation into a multi-slice window names its slice's label; this window's labels are {roster}"
            )
        row = rows[0]
    else:
        matches = [r for r in rows if r["label"] == label]
        if not matches:
            raise ValueError(
                f"unknown label {label!r} — this window's labels are {roster}"
            )
        row = matches[0]
    absolute = lambda off: window_start_ts + off if off is not None else None
    return {
        "slice": row,
        "start_ts": absolute(start_offset_ms),
        "end_ts": absolute(end_offset_ms),
    }


def format_offset(ms: int, start_ms: int, label: str | None = None) -> str:
    elapsed = max(0, ms - start_ms)
    seconds, millis = divmod(elapsed, 1000)
    minutes, seconds = divmod(seconds, 60)
    stamp = f"{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"[{label} {stamp}]" if label else f"[{stamp}]"


# A citation attempt is any bracketed span whose content carries a digits:digits
# core — the loose net that catches malformed forms (00:99.123, a missing
# millis field, a stray label) instead of letting them pass as prose. The exact
# form then holds each attempt to the citation format the model was given: an optional
# slice label, minutes:seconds.millis with seconds under 60 and exactly three
# millis digits, optionally a comma-joined second moment for a range.
_CITE_ATTEMPT = re.compile(r"\[([^\[\]]*\d+:\d{2}[^\[\]]*)\]")
_CITE_EXACT = re.compile(
    r"^(?:(?P<label>[A-Za-z]\w*) )?(?P<m1>\d{2,}):(?P<s1>[0-5]\d)\.(?P<ms1>\d{3})"
    r"(?:,\s*(?:(?P<label2>[A-Za-z]\w*) )?(?P<m2>\d{2,}):(?P<s2>[0-5]\d)\.(?P<ms2>\d{3}))?$"
)


def _offset_ms(minutes: str, seconds: str, millis: str) -> int:
    return int(minutes) * 60000 + int(seconds) * 1000 + int(millis)


def citation_violations(text, slices, window_start_ts: int) -> list[str]:
    """Every citation-shaped span in `text` that fails the citation format, names a slice label wrongly for this window, or addresses a moment outside its slice — each violation the offending span plus its defect, in window-offset vocabulary, ready to hand back to the writer. Mechanical by design: format, label, and bounds are all it judges — whether a valid citation supports its sentence is judgment, and no code here attempts it.

    `slices` is the window's slice table (WindowSlice rows or window.json dicts); bounds are each slice's own recorded span [start_ts, end_ts] — a moment in a slice's lead, ahead of its covering snapshot, is a real recorded moment and cites fine."""
    rows = [asdict(s) if isinstance(s, WindowSlice) else dict(s) for s in slices]
    by_label = {r["label"]: r for r in rows}
    roster = ", ".join(r["label"] for r in rows)
    multi = len(rows) > 1
    form = citation_forms([r["label"] for r in rows])

    def span_of(row) -> str:
        lo = format_offset(row["start_ts"], window_start_ts)
        hi = format_offset(row["end_ts"], window_start_ts)
        return f"{lo} to {hi}"

    violations: list[str] = []

    def flag(raw: str, defect: str) -> None:
        entry = f"[{raw}] — {defect}"
        if entry not in violations:
            violations.append(entry)

    for raw in _CITE_ATTEMPT.findall(text):
        exact = _CITE_EXACT.match(raw)
        if not exact:
            flag(raw, f"not the citation form {form['cite']}")
            continue
        label = exact.group("label")
        if exact.group("label2") not in (None, label):
            flag(raw, "a range's two moments must lie in one slice; these name two")
            continue
        if multi and label is None:
            flag(
                raw,
                f"has no slice label; this recording's citations start with one of {roster}",
            )
            continue
        if not multi and label is not None:
            flag(
                raw,
                "carries a slice label, but this recording has one slice and cites bare times",
            )
            continue
        if multi and label not in by_label:
            flag(raw, f"{label} is not one of the recording's slices ({roster})")
            continue
        row = by_label[label] if multi else rows[0]
        start = _offset_ms(exact.group("m1"), exact.group("s1"), exact.group("ms1"))
        end = (
            _offset_ms(exact.group("m2"), exact.group("s2"), exact.group("ms2"))
            if exact.group("m2")
            else None
        )
        if end is not None and end < start:
            flag(raw, "the range ends before it starts")
            continue
        where = label if multi else "the recording"
        for offset in (start, end):
            if offset is None:
                continue
            absolute = window_start_ts + offset
            if not row["start_ts"] <= absolute <= row["end_ts"]:
                flag(raw, f"outside {where}, which runs {span_of(row)}")
                break
    return violations
