import pytest
from _analysis_support import (
    ScriptedConversation,
    Verdicts,
    slice_rows,
)
from _analysis_support import claim as _claim
from locus.analysis.ground.claims import (
    BinaryVerdict,
    ClaimsFound,
    ClaimVerdict,
    NoClaims,
)
from locus.analysis.ground.evaluate import evaluate_answer
from locus.analysis.ground.judge import assemble_evidence, judge_claims
from locus.analysis.ground.prompts import SET_B
from locus.analysis.model.protocol import Response
from locus.evidence.db import connect


def test_assemble_evidence_windows_real_slice(distilled_db, tmp_path):
    conn = connect(distilled_db)
    row = conn.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()
    table = slice_rows(conn, [row["id"]])
    epoch = table[0]["start_ts"]

    mid_offset_s = int((row["end_ts"] - epoch) / 2000)
    claims = [_claim(1, start=mid_offset_s, end=mid_offset_s + 2)]

    screenshot_dir = tmp_path / "screenshots"
    screenshot_dir.mkdir()
    in_window_ts = epoch + mid_offset_s * 1000 + 500
    far_ts = epoch
    (screenshot_dir / f"S1_activity_screenshot_{in_window_ts}.png").write_bytes(b"png")
    (screenshot_dir / f"S1_activity_screenshot_{far_ts}.png").write_bytes(b"png")

    evidence = assemble_evidence(
        conn,
        table,
        epoch,
        [screenshot_dir],
        claims,
        pushed_moments={("S1", in_window_ts)},
    )
    assert [
        (label, ts, was_pushed) for label, ts, _, was_pushed in evidence.screenshots
    ] == [("S1", in_window_ts, True)]
    assert evidence.screenshot_distance_ms[1] == 0
    assert evidence.session_context.startswith("<SESSION_CONTEXT>")
    for line in evidence.event_stream.splitlines()[:3]:
        assert line.startswith("[")


def test_evaluate_answer_extracts_then_judges_and_persists_both_halves(
    distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_id = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]
    table = slice_rows(conn, [slice_id])
    epoch = table[0]["start_ts"]

    claims = [_claim(1, start=1), _claim(2, start=2)]
    verdicts = Verdicts(
        [
            ClaimVerdict(claim_id=1, reasoning="ok", groundedness_pct=95),
            ClaimVerdict(claim_id=2, reasoning="weak", groundedness_pct=10),
        ]
    )

    def factory(model_name, system_prompt, *, record_dir=None, **kwargs):
        if "claim extractor" in system_prompt:
            return ScriptedConversation(
                [Response(parsed=ClaimsFound(claims=claims), text="x")],
                record_dir=record_dir,
            )
        return ScriptedConversation(
            [Response(parsed=verdicts, text="x")], record_dir=record_dir
        )

    evaluation = evaluate_answer(
        conn,
        table,
        epoch,
        [],
        "an answer",
        "scripted",
        conversation_factory=factory,
        record_root=tmp_path / "payloads",
    )

    assert evaluation["groundedness"]["min_groundedness_pct"] == 10
    assert evaluation["groundedness"]["avg_groundedness_pct"] == 52.5
    assert [v.claim_id for v in evaluation["verdicts"]] == [1, 2]
    assert (tmp_path / "payloads" / "extraction" / "1_claims_input.txt").exists()
    assert (tmp_path / "payloads" / "judge" / "1_verdicts_input.txt").exists()


def _binary_batch(ids):
    return Verdicts(
        [BinaryVerdict(claim_id=i, reasoning="r", grounded=True) for i in ids]
    )


def test_judge_repairs_malformed_verdict_ids_once(distilled_db):
    conn = connect(distilled_db)
    row = conn.execute(
        "SELECT id, start_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()
    table = slice_rows(conn, [row["id"]])
    epoch = row["start_ts"]
    claims = [_claim(1, start=1), _claim(2, start=2), _claim(3, start=3)]

    evidence = assemble_evidence(conn, table, epoch, [], claims)

    conversation = ScriptedConversation(
        [
            Response(text="x", parsed=_binary_batch([1, 2, 2])),
            Response(text="x", parsed=_binary_batch([1, 2, 3])),
        ]
    )
    verdicts, _ = judge_claims(claims, evidence, table, epoch, conversation, SET_B)
    assert [v.claim_id for v in verdicts] == [1, 2, 3]
    assert any("malformed" in t for t in conversation.texts)

    stubborn = ScriptedConversation(
        [
            Response(text="x", parsed=_binary_batch([1, 2, 2])),
            Response(text="x", parsed=_binary_batch([1, 2, 9])),
        ]
    )
    with pytest.raises(ValueError, match="after one repair turn"):
        judge_claims(claims, evidence, table, epoch, stubborn, SET_B)


def test_zero_claims_skips_the_judge(distilled_db, tmp_path):
    conn = connect(distilled_db)
    slice_id = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]
    table = slice_rows(conn, [slice_id])

    judge_calls = []

    def factory(model_name, system_prompt, **kwargs):
        if "claim extractor" in system_prompt:
            return ScriptedConversation(
                [
                    Response(
                        parsed=NoClaims(no_claims="true", reason="no activity"),
                        text='{"no_claims": "true", "reason": "no activity"}',
                    )
                ]
            )
        judge_calls.append(model_name)
        return ScriptedConversation([])

    evaluation = evaluate_answer(
        conn,
        table,
        table[0]["start_ts"],
        [],
        "an answer with no claims",
        "scripted",
        record_root=tmp_path / "record",
        conversation_factory=factory,
    )

    assert judge_calls == [], "judge was called despite zero claims"
    measured = evaluation["groundedness"]
    assert measured["total_claims"] == 0
    assert measured["avg_groundedness_pct"] is None
    assert measured["min_groundedness_pct"] is None
