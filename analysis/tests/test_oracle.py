import json

import pytest
from _analysis_support import (
    ScriptedConversation,
    Verdicts,
    claim,
    slice_rows,
)
from _support import (
    best_slice,
    slice_bounds,
)
from locus.analysis.ground.claims import ClaimsFound, ClaimVerdict
from locus.analysis.model.protocol import Response
from locus.analysis.oracle import evaluate_analysis
from locus.evidence.db import connect


def _make_analysis(
    conn, tmp_path, *, answer="an answer", thoughts=None, transcript=True
):
    """An analysis directory on disk — what the oracle is pointed at: window.json,
    screenshots/, and the conversation transcript whose final answer response IS the
    answer."""
    slice_id = best_slice(conn)
    lo, _ = slice_bounds(conn, slice_id)
    analysis_dir = tmp_path / "2026-07-16T22-14-05Z_0f6e2a7c-sy9t"
    (analysis_dir / "screenshots").mkdir(parents=True)
    for i in (1, 2):
        (
            analysis_dir / "screenshots" / f"S1_screenshot_{lo + i * 1000}.png"
        ).write_bytes(b"\x89PNG")
    (analysis_dir / "window.json").write_text(
        json.dumps(
            {
                "question": "what happened?",
                "slice_ids": [slice_id],
                "slices": slice_rows(conn, [slice_id]),
                "window_start_ts": lo,
            }
        )
    )
    if transcript:
        body = (
            answer
            if thoughts is None
            else f"=== thoughts ===\n{thoughts}\n\n=== response ===\n{answer}"
        )
        (analysis_dir / "1_answer_input.txt").write_text("=== user ===\n\nevidence\n")
        (analysis_dir / "1_answer_response.txt").write_text(body + "\n")
    return analysis_dir


def _factory(*, claims, verdicts):
    def make(model, system, *, record_dir=None, **kwargs):
        if "claim extractor" in system:
            return ScriptedConversation(
                [
                    Response(
                        parsed=ClaimsFound(claims=claims),
                        text="x",
                        usage_metadata={},
                    )
                ],
                record_dir=record_dir,
            )
        if "evaluator" in system:
            return ScriptedConversation(
                [Response(parsed=Verdicts(verdicts), text="x")], record_dir=record_dir
            )
        raise AssertionError(f"unexpected system prompt: {system[:40]!r}")

    return make


def _scripted():
    return _factory(
        claims=[claim(1, start=1), claim(2, start=2)],
        verdicts=[
            ClaimVerdict(claim_id=1, reasoning="ok", groundedness_pct=100),
            ClaimVerdict(claim_id=2, reasoning="ok", groundedness_pct=80),
        ],
    )


def test_oracle_scores_the_analysis_and_lands_in_grounding(distilled_db, tmp_path):
    conn = connect(distilled_db)
    analysis_dir = _make_analysis(conn, tmp_path)

    result = evaluate_analysis(
        conn, analysis_dir, model="scripted", conversation_factory=_scripted()
    )

    assert result["groundedness"]["total_claims"] == 2
    assert result["groundedness"]["avg_groundedness_pct"] == 90
    measurement = analysis_dir / "grounding" / "a_scripted"
    persisted = json.loads((measurement / "oracle.json").read_text())
    assert persisted["groundedness"]
    assert (measurement / "extraction" / "1_claims_input.txt").exists(), (
        "the extraction conversation's own transcript lands in grounding/"
    )
    assert (measurement / "judge" / "1_verdicts_input.txt").exists()


def test_the_oracle_reads_the_answer_from_the_transcript(distilled_db, tmp_path):
    """The answer is the transcript's final response — the oracle judges what
    the model actually wrote, thoughts stripped."""
    conn = connect(distilled_db)
    analysis_dir = _make_analysis(
        conn, tmp_path, answer="the cited answer", thoughts="let me think"
    )

    captured = {}

    def factory(model, system, *, record_dir=None, **kwargs):
        if "claim extractor" in system:
            conv = ScriptedConversation(
                [
                    Response(
                        parsed=ClaimsFound(claims=[claim(1, start=1)]),
                        text="x",
                        usage_metadata={},
                    )
                ],
                record_dir=record_dir,
            )
            captured["extraction"] = conv
            return conv
        return ScriptedConversation(
            [
                Response(
                    parsed=Verdicts(
                        [ClaimVerdict(claim_id=1, reasoning="ok", groundedness_pct=100)]
                    ),
                    text="x",
                )
            ],
            record_dir=record_dir,
        )

    evaluate_analysis(
        conn, analysis_dir, model="scripted", conversation_factory=factory
    )
    fed = "\n".join(captured["extraction"].texts)
    assert "the cited answer" in fed
    assert "let me think" not in fed, "thoughts are not the answer"
    assert "# what happened?" not in fed, (
        "the measurement reads the record, not the assembled product"
    )


def test_the_oracle_excludes_the_meta_channel_from_extraction(distilled_db, tmp_path):
    """The contract's channel is the model's words to the tuner, not claims
    about the session — extraction never sees it."""
    from locus.analysis.prompts import META_HEADER

    conn = connect(distilled_db)
    analysis_dir = _make_analysis(
        conn,
        tmp_path,
        answer=(
            f"the cited answer\n\n{META_HEADER}\n- the event stream cut off mid-table\n"
        ),
    )

    captured = {}

    def factory(model, system, *, record_dir=None, **kwargs):
        if "claim extractor" in system:
            conv = ScriptedConversation(
                [
                    Response(
                        parsed=ClaimsFound(claims=[claim(1, start=1)]),
                        text="x",
                        usage_metadata={},
                    )
                ],
                record_dir=record_dir,
            )
            captured["extraction"] = conv
            return conv
        return ScriptedConversation(
            [
                Response(
                    parsed=Verdicts(
                        [ClaimVerdict(claim_id=1, reasoning="ok", groundedness_pct=100)]
                    ),
                    text="x",
                )
            ],
            record_dir=record_dir,
        )

    evaluate_analysis(
        conn, analysis_dir, model="scripted", conversation_factory=factory
    )
    fed = "\n".join(captured["extraction"].texts)
    assert "the cited answer" in fed
    assert "cut off mid-table" not in fed
    assert META_HEADER not in fed


def test_the_measurement_never_writes_into_the_analysis(distilled_db, tmp_path):
    conn = connect(distilled_db)
    analysis_dir = _make_analysis(conn, tmp_path)
    before = sorted(p.name for p in analysis_dir.iterdir())

    evaluate_analysis(
        conn, analysis_dir, model="scripted", conversation_factory=_scripted()
    )

    after = sorted(p.name for p in analysis_dir.iterdir())
    assert after == sorted(before + ["grounding"]), (
        "the measurement adds grounding/ and touches nothing else"
    )


def test_two_instruments_scoring_one_analysis_coexist(distilled_db, tmp_path):
    conn = connect(distilled_db)
    analysis_dir = _make_analysis(conn, tmp_path)

    evaluate_analysis(
        conn, analysis_dir, model="scripted", conversation_factory=_scripted()
    )
    evaluate_analysis(
        conn, analysis_dir, model="other", conversation_factory=_scripted()
    )

    assert {p.name for p in (analysis_dir / "grounding").iterdir()} == {
        "a_scripted",
        "a_other",
    }

    with pytest.raises(FileExistsError):
        evaluate_analysis(
            conn, analysis_dir, model="scripted", conversation_factory=_scripted()
        )


def test_judge_holds_evidence_parity_with_the_analysis(distilled_db, tmp_path):
    """The judge sees the evidence exactly as the model did: its conversation
    is built on the analysis's recorded tier table, a screenshot it pushed
    re-rides pushed, and a pushed moment the model re-pulled was last seen at
    the pull tier, so it re-rides as a pull."""
    from locus.analysis.ground.claims import ClaimVerdict

    conn = connect(distilled_db)
    analysis_dir = _make_analysis(conn, tmp_path)
    manifest = json.loads((analysis_dir / "window.json").read_text())
    lo = manifest["window_start_ts"]
    manifest["activity_screenshots"] = {"S1": [lo + 1000, lo + 2000]}
    manifest["pulled"] = [["S1", lo + 2000]]
    (analysis_dir / "window.json").write_text(json.dumps(manifest))
    table = {"activity_screenshots": "medium", "pulled_screenshots": "high"}
    (analysis_dir / "1_answer_meta.json").write_text(
        json.dumps({"config": {"media_resolution": table}})
    )

    captured = {}

    def factory(model, system, *, record_dir=None, **kwargs):
        if "claim extractor" in system:
            return ScriptedConversation(
                [
                    Response(
                        parsed=ClaimsFound(
                            claims=[claim(1, start=1), claim(2, start=2)]
                        ),
                        text="x",
                        usage_metadata={},
                    )
                ],
                record_dir=record_dir,
            )
        captured["judge_kwargs"] = kwargs
        conv = ScriptedConversation(
            [
                Response(
                    parsed=Verdicts(
                        [
                            ClaimVerdict(
                                claim_id=1, reasoning="ok", groundedness_pct=100
                            ),
                            ClaimVerdict(
                                claim_id=2, reasoning="ok", groundedness_pct=100
                            ),
                        ]
                    ),
                    text="x",
                )
            ],
            record_dir=record_dir,
        )
        captured["judge"] = conv
        return conv

    evaluate_analysis(
        conn, analysis_dir, model="scripted", conversation_factory=factory
    )

    assert captured["judge_kwargs"]["media_resolution"] == table, (
        "the judge runs on the tier table the analysis's record states"
    )
    assert captured["judge"].image_pushed == [True, False], (
        "the pushed moment re-rides pushed; the re-pulled one as a pull"
    )


def test_oracle_fails_loud_on_a_missing_transcript(distilled_db, tmp_path):
    conn = connect(distilled_db)
    analysis_dir = _make_analysis(conn, tmp_path, transcript=False)
    with pytest.raises(ValueError, match="answer"):
        evaluate_analysis(
            conn,
            analysis_dir,
            model="scripted",
            conversation_factory=lambda *a, **k: None,
        )


def test_oracle_fails_loud_without_a_slice_table(distilled_db, tmp_path):
    conn = connect(distilled_db)
    analysis_dir = _make_analysis(conn, tmp_path)
    manifest = json.loads((analysis_dir / "window.json").read_text())
    del manifest["slices"]
    (analysis_dir / "window.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="slice table"):
        evaluate_analysis(
            conn,
            analysis_dir,
            model="scripted",
            conversation_factory=lambda *a, **k: None,
        )
