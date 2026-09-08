"""The measurement surface — the objective the engine is tuned against, an importable instrument.

It reads one analysis's directory — the transcript whose final response IS the answer, the screenshots,
window.json — and scores its groundedness: is each claim supported by the evidence in the window
it cites? Extraction and the judge (ground/) both run on a model, the instrument of record, and
their own transcripts land beside their results.

It imports the grounding internals and nothing from engine.py, and the engine imports nothing
from here: the thing being optimized must not be able to touch the thing that scores it.
"""

import json
import sqlite3
from pathlib import Path

from .ground.evaluate import evaluate_answer, judged_claims
from .ground.prompts import SET_A
from .model.factory import make_conversation
from .model.protocol import latest_answer_response, response_text
from .prompts import split_meta


def _analysis_answer(analysis_dir: Path) -> str:
    """An analysis's answer is its transcript's final answer-labeled response — the model's own last word, thoughts stripped, the contract's meta channel excluded (it is the model's words to the tuner, not claims about the session)."""
    return split_meta(response_text(latest_answer_response(analysis_dir).read_text()))[
        0
    ]


def evaluate_analysis(
    conn: sqlite3.Connection,
    analysis_dir: str | Path,
    *,
    model: str,
    prompt_set=SET_A,
    conversation_factory=make_conversation,
) -> dict:
    """Score one analysis. The measurement lands in the analysis's grounding/ — keyed by the
    instrument that made it (prompt set + model), written once, never into the
    record it reads. Returns the groundedness aggregates, every claim with the
    verdict it drew, and what it cost."""
    analysis_dir = Path(analysis_dir)
    manifest = json.loads((analysis_dir / "window.json").read_text())
    if "slices" not in manifest or "window_start_ts" not in manifest:
        raise ValueError(
            f"{analysis_dir / 'window.json'} carries no slice table — not an "
            f"engine analysis"
        )

    answer = _analysis_answer(analysis_dir)
    measurement = analysis_dir / "grounding" / f"{prompt_set.name}_{model}"
    if measurement.exists():
        raise FileExistsError(
            f"output dir already exists: {measurement} — outputs are never overwritten"
        )
    measurement.mkdir(parents=True)

    # Parity provenance, from the analysis's own record: which moments rode as
    # activity-screenshots (a moment the model re-pulled was last seen at the
    # pull tier, so it does not count), and the tier table the analysis's
    # conversation ran with (the first answer call's persisted config). The
    # judge re-shows the evidence exactly as the model saw it.
    pushed_moments = {
        (label, ts)
        for label, stamps in (manifest.get("activity_screenshots") or {}).items()
        for ts in stamps
    }
    pushed_moments -= {(label, ts) for label, ts in manifest.get("pulled") or []}
    metas = sorted(
        analysis_dir.glob("*_answer_meta.json"),
        key=lambda p: int(p.name.split("_", 1)[0]),
    )
    analysis_media_resolution = {}
    if metas:
        config = json.loads(metas[0].read_text()).get("config") or {}
        analysis_media_resolution = config.get("media_resolution") or {}

    evaluation = evaluate_answer(
        conn,
        manifest["slices"],
        manifest["window_start_ts"],
        [analysis_dir / "screenshots"],
        answer,
        model,
        prompt_set=prompt_set,
        record_root=measurement,
        conversation_factory=conversation_factory,
        window_bounds=(manifest.get("window_start"), manifest.get("window_end")),
        pushed_moments=pushed_moments,
        analysis_media_resolution=analysis_media_resolution,
    )

    from locus.evidence.clock import utc_stamp

    result = {
        "analysis": str(analysis_dir),
        "dir": str(measurement),
        "measured": utc_stamp(),
        "model": model,
        "prompt_set": prompt_set.name,
        "groundedness": evaluation["groundedness"],
        "claims": judged_claims(evaluation),
        "usage": evaluation["usage"],
    }
    (measurement / "oracle.json").write_text(json.dumps(result, indent=2, default=str))
    return result
