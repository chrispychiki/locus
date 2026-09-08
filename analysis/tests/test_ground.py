import json as _json

import pytest
from _analysis_support import (
    ScriptedConversation,
    Verdicts,
    slice_rows,
)
from _analysis_support import claim as _claim
from locus.analysis.ground.claims import (
    BinaryVerdict,
    ClaimExtraction,
    ClaimsFound,
    ClaimType,
    ClaimVerdict,
    NoClaims,
    batch_verdicts_schema,
    merge_spans,
    structural_failures,
)
from locus.analysis.ground.evaluate import evaluate_answer, groundedness
from locus.analysis.ground.extract import extract_claims
from locus.analysis.ground.judge import assemble_evidence, claim_spans, judge_claims
from locus.analysis.ground.prompts import SET_B
from locus.analysis.model.protocol import Response
from locus.evidence.db import connect
from locus.evidence.hydrate import pack_raw
from pydantic import TypeAdapter, ValidationError


def test_structural_failures():
    claims = [
        _claim(1, start=1),
        _claim(2, ClaimType.INFERENCE, supports=[1]),
        _claim(3, ClaimType.INFERENCE),
        _claim(4, ClaimType.INFERENCE, supports=[99]),
    ]
    failures = structural_failures(claims, ["S1"])
    assert set(failures) == {3, 4}
    assert "unknown claim ids [99]" in failures[4]


def test_structural_failures_catch_unresolvable_citations():
    claims = [
        _claim(1, start=1, label="S2"),  # resolvable
        _claim(2, start=2, label="S9"),  # no such label
        _claim(3, start=3),  # bare citation, multi window
        _claim(4),  # uncited — not a structural matter
    ]
    failures = structural_failures(claims, ["S1", "S2"])
    assert set(failures) == {2, 3}
    assert "unknown label 'S9'" in failures[2]
    assert "without naming its slice's label" in failures[3]

    assert structural_failures([_claim(1, start=1)], ["S1"]) == {}, (
        "a single-slice window's bare citations are fully resolvable"
    )


def test_merge_spans():
    assert merge_spans([(10_000, 11_000), (12_000, 13_000)], 2000) == [(8_000, 15_000)]
    assert merge_spans([(0, 1_000), (20_000, 21_000)], 1000) == [
        (-1_000, 2_000),
        (19_000, 22_000),
    ]


_TABLE = [
    {
        "label": "S1",
        "visitor": "va",
        "slice": "s-a",
        "slice_id": 1,
        "start_ts": 1_000,
        "end_ts": 60_000,
    },
    {
        "label": "S2",
        "visitor": "vb",
        "slice": "s-b",
        "slice_id": 2,
        "start_ts": 2_000,
        "end_ts": 61_000,
    },
]


def test_claim_spans_resolve_through_the_slice_table():
    spans = claim_spans(
        [
            _claim(1, start=5, label="S1"),
            _claim(2, end=7, label="S2"),
            _claim(3, label="S1"),
            _claim(4, start=1),
            _claim(5, start=1, label="S9"),
        ],
        _TABLE,
        1_000,
    )
    assert spans == {1: ("S1", 6_000, 6_000), 2: ("S2", 8_000, 8_000)}, (
        "uncited, bare-in-multi, and unknown-label claims draw no span"
    )

    single = claim_spans([_claim(1, start=5), _claim(2, end=7)], _TABLE[:1], 1_000)
    assert single == {1: ("S1", 6_000, 6_000), 2: ("S1", 8_000, 8_000)}, (
        "a single-slice window resolves bare citations to its one slice"
    )


def test_groundedness_counts_uncovered_screenshots():
    claims = [_claim(1, start=1), _claim(2, start=2)]
    verdicts = [
        ClaimVerdict(claim_id=1, reasoning="ok", groundedness_pct=90),
        ClaimVerdict(claim_id=2, reasoning="weak", groundedness_pct=30),
    ]
    measured = groundedness(claims, verdicts, {1: 500, 2: None})
    assert measured["min_groundedness_pct"] == 30
    assert measured["claims_no_nearby_screenshot"] == 1
    assert measured["avg_screenshot_distance_ms"] == 500


def test_groundedness_pins_the_composite_citation_hygiene_and_distance_threshold():
    """The headline numbers the oracle reports: answer_score (mean × sqrt(count) ÷ 100), the uncited-OBSERVATION count (inferences owe no citation), by_type means, and the distant-screenshot count strictly beyond DISTANT_SCREENSHOT_THRESHOLD_MS."""
    claims = [
        _claim(1, start=1),
        _claim(2),
        _claim(3, ClaimType.INFERENCE),
    ]
    verdicts = [
        ClaimVerdict(claim_id=1, reasoning="ok", groundedness_pct=90),
        ClaimVerdict(claim_id=2, reasoning="weak", groundedness_pct=30),
        ClaimVerdict(claim_id=3, reasoning="fair", groundedness_pct=60),
    ]
    measured = groundedness(claims, verdicts, {1: 1000, 2: 1001, 3: None})
    assert measured["answer_score"] == 1.04, "60.0 × sqrt(3) ÷ 100, rounded to 2"
    assert measured["claims_missing_citation"] == 1
    assert measured["by_type"] == {"OBSERVATION": 60.0, "INFERENCE": 60.0}
    assert measured["claims_with_distant_screenshots"] == 1, (
        "exactly at the threshold is near; one past it is distant"
    )
    assert measured["claims_no_nearby_screenshot"] == 1
    assert measured["avg_screenshot_distance_ms"] == 1000


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
    (screenshot_dir / f"S1_screenshot_{in_window_ts}.png").write_bytes(b"png")
    (screenshot_dir / f"S1_screenshot_{far_ts}.png").write_bytes(b"png")

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


def _synthetic_overlap_db(tmp_path):
    """Two slices covering the same instants under two visitors, with
    distinguishable events — the shape that proves the judge scopes a claim's
    evidence to its cited slice."""
    from locus.evidence.rrweb_constants import EventType

    conn = connect(tmp_path / "events.db")
    rec = "r/1.0"
    for sid, visitor, marker in ((1, "va", "alpha"), (2, "vb", "beta")):
        conn.execute(
            "INSERT INTO slices (id, visitor_id, recorder_slice, start_ts, "
            "end_ts, n_events, status) VALUES (?, ?, ?, ?, ?, ?, 'replayable')",
            (sid, visitor, f"s-{marker}", 1_000, 9_000, 3),
        )
        kinds = {
            "Meta": EventType.Meta,
            "FullSnapshot": EventType.FullSnapshot,
            "Click": EventType.IncrementalSnapshot,
        }
        for ts, kind, text in (
            (1_000, "Meta", None),
            (1_010, "FullSnapshot", None),
            (5_000, "Click", marker),
        ):
            raw = {"type": kinds[kind], "timestamp": ts, "data": {}}
            conn.execute(
                "INSERT INTO events (visitor_id, timestamp, type, raw_json, "
                "content_hash, slice_id, type_str, text, tag, script_version, "
                "md) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    visitor,
                    ts,
                    raw["type"],
                    pack_raw(_json.dumps(raw)),
                    f"h{sid}_{ts}",
                    sid,
                    kind,
                    text,
                    "button" if text else None,
                    rec,
                    f"# {marker}" if kind == "FullSnapshot" else None,
                ),
            )
    conn.commit()
    return conn


def test_evidence_is_scoped_to_the_cited_label(tmp_path):
    """Two slices cover the same instants; a claim citing S2 must be judged
    against S2's slice alone — the concurrent lane's events at the very same
    moment are not its evidence."""
    conn = _synthetic_overlap_db(tmp_path)
    table = slice_rows(conn, [1, 2])
    assert [s["label"] for s in table] == ["S1", "S2"]
    epoch = 1_000

    screenshot_dir = tmp_path / "screenshots"
    screenshot_dir.mkdir()
    (screenshot_dir / "S1_screenshot_5000.png").write_bytes(b"png")
    (screenshot_dir / "S2_screenshot_5000.png").write_bytes(b"png")

    claims = [_claim(1, start=4, label="S2")]
    evidence = assemble_evidence(conn, table, epoch, [screenshot_dir], claims)

    assert "beta" in evidence.event_stream and "alpha" not in evidence.event_stream, (
        "only the cited slice testifies"
    )
    assert evidence.event_stream.splitlines()[0].startswith("[S2 "), (
        "the judge's event stream is stamped in the window's own format"
    )
    assert [
        (label, ts, was_pushed) for label, ts, _, was_pushed in evidence.screenshots
    ] == [("S2", 5_000, False)], (
        "the concurrent slice's screenshot at the same instant stays out, and a "
        "screenshot outside the recorded pushed set reads as a pull"
    )
    assert evidence.screenshot_distance_ms[1] == 0
    assert evidence.session_context.count("<SESSION_CONTEXT>") == 2, (
        "each visitor's context block rides the session context"
    )


def test_evidence_survives_a_rebuild_that_reminted_rowids(tmp_path):
    """The slice table in window.json outlives the db it was composed against. A re-loaded db re-mints rowids — here swapped between two slices, the worst case, where the stale slice_id silently addresses the *other* slice — and evidence must still come from the cited slice, resolved by (visitor, recorder slice)."""
    conn = _synthetic_overlap_db(tmp_path)
    table = slice_rows(conn, [1, 2])

    for old, new in ((1, 3), (2, 1), (3, 2)):
        conn.execute("UPDATE slices SET id = ? WHERE id = ?", (new, old))
        conn.execute("UPDATE events SET slice_id = ? WHERE slice_id = ?", (new, old))
    conn.commit()

    evidence = assemble_evidence(
        conn, table, 1_000, [], [_claim(1, start=4, label="S2")]
    )
    assert "beta" in evidence.event_stream and "alpha" not in evidence.event_stream, (
        "the stale rowid must not decide which slice testifies"
    )


def test_evidence_fails_loud_when_the_db_no_longer_holds_the_slice(tmp_path):
    conn = _synthetic_overlap_db(tmp_path)
    table = slice_rows(conn, [1, 2])
    claims = [_claim(1, start=4, label="S2")]

    conn.execute("UPDATE slices SET status = 'rescued', reason = 'x' WHERE id = 2")
    conn.commit()
    with pytest.raises(ValueError, match="no longer replayable"):
        assemble_evidence(conn, table, 1_000, [], claims)

    conn.execute("DELETE FROM slices WHERE id = 2")
    conn.commit()
    with pytest.raises(ValueError, match="not in this db"):
        assemble_evidence(conn, table, 1_000, [], claims)


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


def test_extraction_prompt_is_interpolated_per_window():
    single = ScriptedConversation(
        [
            Response(
                parsed=ClaimsFound(claims=[_claim(1, start=1)]),
                text="x",
                usage_metadata={},
            )
        ]
    )
    extract_claims("an answer", single, labels=["S1"])
    assert "`label`" not in "\n".join(single.texts), (
        "a single-slice extraction has no label field to populate"
    )

    multi = ScriptedConversation(
        [
            Response(
                parsed=ClaimsFound(claims=[_claim(1, start=1, label="S2")]),
                text="x",
                usage_metadata={},
            )
        ]
    )
    extract_claims("an answer", multi, labels=["S1", "S2", "S3"])
    fed = "\n".join(multi.texts)
    assert "[S2 MM:SS.mmm]" in fed, (
        "the multi-slice extraction states the labeled format concretely"
    )
    assert "S1" in fed and "S3" in fed, "with the window's real label roster"


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


def test_extraction_schema_forbids_silent_empty():
    adapter = TypeAdapter(ClaimExtraction)
    with pytest.raises(ValidationError):
        adapter.validate_python([])
    asserted = adapter.validate_python(
        {"no_claims": "true", "reason": "the answer describes no visitor activity"}
    )
    assert isinstance(asserted, NoClaims)
    with pytest.raises(ValidationError):
        adapter.validate_python({"claims": []})
    found = adapter.validate_python({"claims": [_claim(1).model_dump(mode="json")]})
    assert found.claims[0].claim_id == 1


def test_verdict_batch_schema_forbids_underdelivery():
    sized = batch_verdicts_schema(3, binary=True)
    short = {
        "evaluations": [
            {"claim_id": 1, "reasoning": "r", "grounded": True},
            {"claim_id": 2, "reasoning": "r", "grounded": False},
        ]
    }
    with pytest.raises(ValidationError):
        TypeAdapter(sized).validate_python(short)
    full = TypeAdapter(sized).validate_python(
        {
            "evaluations": [
                {"claim_id": i, "reasoning": "r", "grounded": True} for i in (1, 2, 3)
            ]
        }
    )
    assert all(isinstance(v, BinaryVerdict) for v in full.evaluations)
    scaled = batch_verdicts_schema(1, binary=False)
    schema = TypeAdapter(scaled).json_schema()
    items = schema["properties"]["evaluations"]
    assert items["minItems"] == items["maxItems"] == 1


def test_extract_claims_returns_empty_on_explicit_no_claims_assertion():
    conversation = ScriptedConversation(
        [
            Response(
                parsed=NoClaims(no_claims="true", reason="nothing substantive"),
                text='{"no_claims": "true", "reason": "nothing substantive"}',
                usage_metadata={"total_tokens": 7},
            )
        ]
    )
    claims, usage = extract_claims("answer text", conversation)
    assert claims == []
    assert usage == {"total_tokens": 7}


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
