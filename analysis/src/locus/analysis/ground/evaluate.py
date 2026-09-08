"""The evaluation: a finished answer decomposed into claims, each judged against the evidence in its cited span, and the whole scored."""

import math
import sqlite3
from pathlib import Path

from ..model.factory import make_conversation
from .claims import Claim, ClaimType, ClaimVerdict
from .extract import extract_claims
from .judge import assemble_evidence, judge_claims
from .prompts import SET_A, PromptSet

DISTANT_SCREENSHOT_THRESHOLD_MS = 1000


def groundedness(
    claims: list[Claim],
    verdicts: list[ClaimVerdict],
    screenshot_distance_ms: dict[int, int | None],
) -> dict:
    """The measurement, reported in its parts.

    answer_score is the one composite: mean groundedness × sqrt(claim count) ÷ 100. It exists because groundedness alone rewards a timid answer — a single trivially-grounded claim scores 100 — so the count enters the score, under a square root so that saying more is worth something but never worth saying it loosely. It is a comparator between answers over the same window, not a quantity with units.
    """
    if not claims:
        return {
            "total_claims": 0,
            "claims_missing_citation": 0,
            "avg_groundedness_pct": None,
            "min_groundedness_pct": None,
            "answer_score": 0.0,
            "by_type": {},
            "claims_with_span": 0,
            "claims_no_nearby_screenshot": 0,
            "claims_with_distant_screenshots": 0,
            "avg_screenshot_distance_ms": None,
        }
    scores = [verdict.groundedness_pct for verdict in verdicts]
    by_type: dict[str, list[int]] = {}
    verdict_by_id = {verdict.claim_id: verdict for verdict in verdicts}
    for claim in claims:
        by_type.setdefault(claim.claim_type.value, []).append(
            verdict_by_id[claim.claim_id].groundedness_pct
        )

    spanned = list(screenshot_distance_ms.values())
    known = [distance for distance in spanned if distance is not None]
    return {
        "total_claims": len(claims),
        "claims_missing_citation": sum(
            1
            for claim in claims
            if claim.claim_type is ClaimType.OBSERVATION
            and claim.start_timestamp_components is None
            and claim.end_timestamp_components is None
        ),
        "avg_groundedness_pct": round(sum(scores) / len(scores), 1),
        "min_groundedness_pct": min(scores),
        "answer_score": round(
            (sum(scores) / len(scores)) * math.sqrt(len(scores)) / 100, 2
        ),
        "by_type": {
            kind: round(sum(values) / len(values), 1)
            for kind, values in by_type.items()
        },
        "claims_with_span": len(spanned),
        "claims_no_nearby_screenshot": sum(1 for d in spanned if d is None),
        "claims_with_distant_screenshots": sum(
            1 for d in known if d > DISTANT_SCREENSHOT_THRESHOLD_MS
        ),
        "avg_screenshot_distance_ms": (
            round(sum(known) / len(known)) if known else None
        ),
    }


def judged_claims(evaluation: dict) -> list[dict]:
    """Each claim carrying the verdict it drew — the measurement's per-claim half, persisted beside the aggregates so a score can be doubted and checked claim by claim."""
    verdict_by_id = {v.claim_id: v for v in evaluation["verdicts"]}
    return [
        {
            **claim.model_dump(mode="json"),
            "groundedness_pct": verdict_by_id[claim.claim_id].groundedness_pct,
            "reasoning": verdict_by_id[claim.claim_id].reasoning,
        }
        for claim in evaluation["claims"]
    ]


def evaluate_answer(
    conn: sqlite3.Connection,
    slices: list[dict],
    window_start_ts: int,
    screenshot_dirs: list,
    answer_text: str,
    model: str,
    *,
    record_root: Path,
    conversation_factory=make_conversation,
    prompt_set: PromptSet = SET_A,
    window_bounds: tuple[int | None, int | None] = (None, None),
    pushed_moments: set[tuple[str | None, int]] = frozenset(),
    analysis_media_resolution: dict | None = None,
) -> dict:
    """`slices` is the analyzed window's slice table exactly as window.json carries it: it interpolates the extraction prompt's citation format and scopes every claim span to its cited slice through the shared resolution lookup. record_root is where the measurement's own conversations leave their transcripts — extraction/ and judge/ beneath it, assigned here by the same motion that runs them. The measurement is as auditable as what it measures; there is no unrecorded arm.

    Parity: `pushed_moments` is the set of (label, ts) the analysis pushed as activity-screenshots, and `analysis_media_resolution` the tier table its conversation ran with — both from the analysis's record. The judge's conversation is built on that table and re-shows each screenshot as the kind it rode, so the instrument reads the evidence at exactly the resolution the model did. An analysis whose record states no tier table (a local backend) passes neither, and the judge's screenshots ride the one economics that backend has."""
    labels = [s["label"] for s in slices]
    extraction = conversation_factory(
        model,
        prompt_set.extraction_system,
        record_dir=Path(record_root) / "extraction",
    )
    claims, extract_usage = extract_claims(
        answer_text, extraction, prompt_set, labels=labels
    )

    if not claims:
        return {
            "claims": [],
            "verdicts": [],
            "groundedness": groundedness([], [], {}),
            "usage": {"extraction": extract_usage, "judge": None},
        }

    judge_kwargs = (
        {"media_resolution": analysis_media_resolution}
        if analysis_media_resolution
        else {}
    )
    judgement = conversation_factory(
        model,
        prompt_set.validation_system,
        record_dir=Path(record_root) / "judge",
        **judge_kwargs,
    )
    evidence = assemble_evidence(
        conn,
        slices,
        window_start_ts,
        screenshot_dirs,
        claims,
        window_bounds,
        pushed_moments=pushed_moments,
    )
    verdicts, judge_usage = judge_claims(
        claims, evidence, slices, window_start_ts, judgement, prompt_set
    )

    return {
        "claims": claims,
        "verdicts": verdicts,
        "groundedness": groundedness(claims, verdicts, evidence.screenshot_distance_ms),
        "usage": {"extraction": extract_usage, "judge": judge_usage},
    }
