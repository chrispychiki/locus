"""The groundedness judge — claims scored against the evidence in their spans.

Evidence comes straight from the db and the rendered screenshot files, scoped by slice: a claim's citation resolves through the window's slice table (window.resolve_citation — the same lookup replay-opening callers use), so its span draws events and screenshots from the cited slice alone. The table may be older than the db — an analysis's window.json outlives the deletable, re-loadable cache it was composed against, and a rebuilt db re-mints rowids — so db reads never trust the table's persisted slice_id: each slice's durable (visitor, recorder slice) identity resolves to the current rowid at assembly (current_slice_ids), and a slice the db no longer holds replayable fails loud rather than letting a stale rowid read another slice's events as evidence. Two slices may cover the same instant — a second tab, a concurrent visitor — and only label scoping keeps a claim about one lane from being judged against the other's evidence. Events inside the union of a slice's claim spans (±EVIDENCE_SPAN_MS, overlaps merged) are formatted by the same line formatter the analyzed event stream uses, stamped with the slice's label exactly as that stream was, plus the screenshots whose (label, timestamp) fall in those spans. One batched multimodal call judges every structurally valid claim; structurally invalid claims (inference without supports, a citation naming no resolvable label) score 0 without spending a token.

The judge holds evidence parity: each screenshot is re-shown at the tier it rode in the analysis — activity-screenshots as activity-screenshots, under the analysis's own recorded tier table — never sharper. A judge seeing better pixels than the model stops measuring groundedness: it can confirm a lucky guess the model could not have grounded, and refute a fair reading of what the model was actually shown.

Verdict reconciliation: the batch schema pins the verdict COUNT to the claim count (an unconstrained list lets the grammar close early — see claims.py), but no grammar can pin id distinctness, and a model decoding deterministically reproduces the same id glitch on every identical retry. So an id-set mismatch gets exactly one repair turn in the still-live conversation — the complaint names the malformed ids, the judge re-emits, both calls persist in the payloads — and a second mismatch raises. The repair is format-only by instruction: verdicts and reasoning are kept, ids fixed.
"""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from locus.evidence.db import CANONICAL_ORDER

from ..analyze import event_stream, screenshot_index
from ..model.protocol import Conversation
from ..session_context import session_context_block
from ..window import format_offset, resolve_citation
from .claims import (
    Claim,
    ClaimVerdict,
    batch_verdicts_schema,
    merge_spans,
    structural_failures,
)
from .prompts import SET_A, PromptSet

EVIDENCE_SPAN_MS = 2000


def current_slice_ids(conn: sqlite3.Connection, slices: list[dict]) -> dict[str, int]:
    """Each window slice's current db rowid, by label. The persisted slice_id is testimony about the db the window was composed against; the identity that survives a rebuild is (visitor, recorder slice) — the db's own slices_identity key — so that is what evidence reads address by."""
    ids: dict[str, int] = {}
    for s in slices:
        row = conn.execute(
            "SELECT id, status, reason FROM slices WHERE visitor_id = ? AND recorder_slice = ?",
            (s["visitor"], s["slice"]),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"slice {s['visitor']}/{s['slice']} ({s['label']}) is not in this db — "
                f"the analysis predates the db's current contents; `locus load` the visitor "
                f"and re-evaluate"
            )
        if row["status"] != "replayable":
            raise ValueError(
                f"slice {s['visitor']}/{s['slice']} ({s['label']}) is no longer "
                f"replayable in this db ({row['status']}: {row['reason']}) — its events "
                f"are not addressable as the slice the analysis read"
            )
        ids[s["label"]] = row["id"]
    return ids


@dataclass
class Evidence:
    """screenshots entries are (label, ts, path, was_pushed) — was_pushed says the screenshot rode the analysis as an activity-screenshot, so the judge can re-show it as one (parity: the instrument sees what the model saw)."""

    event_stream: str
    screenshots: list[tuple[str | None, int, Path, bool]]
    session_context: str
    screenshot_distance_ms: dict[int, int | None] = field(default_factory=dict)


def claim_spans(
    claims: list[Claim], slices: list[dict], window_start_ts: int
) -> dict[int, tuple[str, int, int]]:
    """Each resolvable claim's cited span as (slice label, absolute lo, absolute hi), resolved through the slice table. A claim whose citation cannot resolve — no timestamps, or no nameable label in a multi-slice window — gets no span; structural_failures is what scores the mechanically unverifiable ones."""
    labels = [s["label"] for s in slices]
    spans: dict[int, tuple[str, int, int]] = {}
    for claim in claims:
        start = claim.start_timestamp_components
        end = claim.end_timestamp_components
        if start is None and end is None:
            continue
        if claim.label is None and len(labels) > 1:
            continue
        if claim.label is not None and claim.label not in labels:
            continue
        resolved = resolve_citation(
            slices,
            window_start_ts,
            label=claim.label,
            start_offset_ms=start.to_ms() if start is not None else None,
            end_offset_ms=end.to_ms() if end is not None else None,
        )
        lo = (
            resolved["start_ts"]
            if resolved["start_ts"] is not None
            else resolved["end_ts"]
        )
        hi = resolved["end_ts"] if resolved["end_ts"] is not None else lo
        spans[claim.claim_id] = (resolved["slice"]["label"], min(lo, hi), max(lo, hi))
    return spans


def assemble_evidence(
    conn: sqlite3.Connection,
    slices: list[dict],
    window_start_ts: int,
    screenshot_dirs: list[str | Path],
    claims: list[Claim],
    window_bounds: tuple[int | None, int | None] = (None, None),
    pushed_moments: set[tuple[str | None, int]] = frozenset(),
) -> Evidence:
    multi = len(slices) > 1
    live_ids = current_slice_ids(conn, slices)
    spans = claim_spans(claims, slices, window_start_ts)

    per_label: dict[str, list[tuple[int, int]]] = {}
    for label, lo, hi in spans.values():
        per_label.setdefault(label, []).append((lo, hi))
    merged = {
        label: merge_spans(pairs, EVIDENCE_SPAN_MS)
        for label, pairs in per_label.items()
    }

    lines: list[str] = []
    n_spans = 0
    for s in slices:
        label = s["label"]
        merged_spans = merged.get(label, [])
        n_spans += len(merged_spans)
        for start, end in merged_spans:
            rows = conn.execute(
                f"SELECT * FROM events WHERE slice_id = ? AND timestamp BETWEEN ? AND ? {CANONICAL_ORDER}",
                (live_ids[label], start, end),
            ).fetchall()
            lines.extend(
                line
                for _, line in event_stream(
                    rows, window_start_ts, label=label if multi else None
                )
            )
    if not lines and n_spans:
        raise ValueError(
            "no events found in any claim span — the spans fall outside the "
            "analyzed slices; check the slice table / window-clock mapping"
        )

    all_screenshots = screenshot_index(screenshot_dirs)
    screenshots = sorted(
        (label, ts, path, (label, ts) in pushed_moments)
        for (label, ts), path in all_screenshots.items()
        if any(start <= ts <= end for start, end in merged.get(label, []))
    )

    distances: dict[int, int | None] = {}
    for claim_id, (label, start, end) in spans.items():
        best = None
        for screenshot_label, ts in all_screenshots:
            if screenshot_label != label:
                continue
            distance = 0 if start <= ts <= end else min(abs(ts - start), abs(ts - end))
            if best is None or distance < best:
                best = distance
        distances[claim_id] = (
            best if best is not None and best <= EVIDENCE_SPAN_MS else None
        )

    visitors: dict[str, list[int]] = {}
    for s in slices:
        visitors.setdefault(s["visitor"], []).append(live_ids[s["label"]])
    context = "\n".join(
        session_context_block(conn, slice_ids, *window_bounds)
        for slice_ids in visitors.values()
    )

    return Evidence(
        event_stream="\n".join(lines)
        if lines
        else "No claim cited a resolvable timestamp; no events in span.",
        screenshots=screenshots,
        session_context=context,
        screenshot_distance_ms=distances,
    )


def _claims_section(claims: list[Claim]) -> str:
    parts = []
    for claim in claims:
        span = ""
        if claim.start_timestamp_components or claim.end_timestamp_components:
            fmt = lambda c: (
                f"{c.minutes:02d}:{c.seconds:02d}.{c.milliseconds:03d}" if c else "?"
            )
            span = f' timestamp="[{fmt(claim.start_timestamp_components)}, {fmt(claim.end_timestamp_components)}]"'
        label = f' label="{claim.label}"' if claim.label else ""
        supports = (
            f' supported_by="{",".join(map(str, claim.supporting_claim_ids))}"'
            if claim.supporting_claim_ids
            else ""
        )
        parts.append(
            f'<Claim id="{claim.claim_id}" type="{claim.claim_type.value}"'
            f"{label}{span}{supports}>{claim.claim_text}</Claim>"
        )
    return "\n".join(parts)


def judge_claims(
    claims: list[Claim],
    evidence: Evidence,
    slices: list[dict],
    window_start_ts: int,
    conversation: Conversation,
    prompt_set: PromptSet = SET_A,
) -> tuple[list[ClaimVerdict], dict]:
    """Judge the claims against evidence already assembled for them — one assembly per evaluation, since scanning the db and indexing the screenshots twice for one measurement would be two chances to disagree with itself."""
    multi = len(slices) > 1
    failures = structural_failures(claims, [s["label"] for s in slices])
    verdicts = [
        ClaimVerdict(claim_id=claim_id, reasoning=reason, groundedness_pct=0)
        for claim_id, reason in failures.items()
    ]
    valid = [claim for claim in claims if claim.claim_id not in failures]
    if not valid:
        return verdicts, {}

    conversation.add_user_text(
        prompt_set.validation_opening.format(
            session_context=evidence.session_context,
            event_stream=evidence.event_stream,
        )
    )
    for label, ts, path, was_pushed in evidence.screenshots:
        stamp = format_offset(ts, window_start_ts, label if multi else None)
        conversation.add_user_text(f"Screenshot at {stamp}")
        conversation.add_user_image(str(path), pushed=was_pushed)
    conversation.add_user_text(
        prompt_set.validation_closing.format(claims_section=_claims_section(valid))
    )

    schema = batch_verdicts_schema(len(valid), prompt_set.binary_judge)
    valid_ids = {claim.claim_id for claim in valid}
    usage: dict = {}

    for attempt in ("verdicts", "verdicts_repair"):
        response = conversation.get_response(response_schema=schema, label=attempt)
        usage = dict(response.usage_metadata)
        judged: list[ClaimVerdict] = [
            verdict.to_pct() if prompt_set.binary_judge else verdict
            for verdict in response.parsed.evaluations
        ]
        judged_ids = {verdict.claim_id for verdict in judged}
        if judged_ids == valid_ids:
            break
        missing = sorted(valid_ids - judged_ids)
        unknown = sorted(judged_ids - valid_ids)
        complaint = (
            f"Your verdict list is malformed: it must contain exactly "
            f"one verdict per claim id {sorted(valid_ids)}, but "
            f"{f'ids {missing} are missing' if missing else ''}"
            f"{' and ' if missing and unknown else ''}"
            f"{f'ids {unknown} do not exist' if unknown else ''}"
            f" (a claim id may be duplicated). Re-emit the complete "
            f"corrected JSON; keep your verdicts and reasoning, fix "
            f"only the ids."
        )
        if attempt == "verdicts_repair":
            raise ValueError(
                f"judge verdict ids wrong after one repair turn: missing {missing}, unknown {unknown}"
            )
        conversation.add_user_text(complaint)

    verdicts.extend(sorted(judged, key=lambda verdict: verdict.claim_id))
    return verdicts, usage
