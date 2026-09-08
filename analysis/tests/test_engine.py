import ast
import json
import re
from pathlib import Path

import pytest
from _analysis_support import ScriptedConversation
from _support import (
    FakeRenderSession,
    best_slice,
    slice_bounds,
)
from locus.analysis import engine
from locus.analysis.budget import CostModel
from locus.analysis.model.protocol import Response, ToolCall
from locus.analysis.prompts import (
    EMPTY_ANSWER_REPAIR,
    META_HEADER,
    SCREENSHOTS_ALREADY_SERVED,
    request_screenshots_tool,
    system_prompt,
)
from locus.analysis.window import format_offset
from locus.evidence.db import connect


def _pull(*offsets, label=None):
    """A request_screenshots argument object in the tool's one shape: bare offsets, or labeled moments when the window has several slices."""
    return {
        "screenshots": [
            {"timestamp": off, **({"label": label} if label else {})} for off in offsets
        ]
    }


SYSTEM = system_prompt()

CONTEXT = "demo site — a test fixture."
QUESTION = "What was this visitor trying to do, and where did they struggle?"
ANSWER = "The visitor did something [00:00.100], then paused [00:00.100, 00:00.200]."
ANSWER_LABELED = "The visitor did something [S1 00:00.000]."


def _homed(conv, kwargs):
    """The engine assigns every conversation its record home at creation; the
    fake factory threads it onto the pre-built double so transcripts land where
    the real path would put them."""
    record_dir = kwargs.get("record_dir")
    if record_dir is not None:
        conv._persist_dir = Path(record_dir)
    return conv


def _word_cost_model():
    return CostModel(
        count_text=lambda texts: sum(len(t.split()) for t in texts),
        verify_text=None,
        screenshot_tokens=lambda c, moments: sum(len(v) for v in moments.values()),
    )


def _wire(monkeypatch, conv, context_tokens=10_000_000):
    """Every analysis prices itself before running, so the cost model rides along
    with the conversation double: word-counted text, one token per screenshot."""
    from locus.analysis import budget as budget_mod

    monkeypatch.setattr(engine, "make_conversation", lambda *a, **k: _homed(conv, k))
    monkeypatch.setattr(engine, "RenderSession", FakeRenderSession)
    model = _word_cost_model()
    monkeypatch.setattr(
        budget_mod, "cost_model_for", lambda name: (model, context_tokens, 1.0)
    )
    return model


_WIRE_RECORD = re.compile(r"\d+_.+_(input\.txt|response\.txt|meta\.json)")


def _strays(out):
    """Whatever sits in an analysis dir beyond the record itself — the record being window.json, the numbered wire transcripts, and screenshots/."""
    return {
        p.name
        for p in out.iterdir()
        if p.name not in {"window.json", "screenshots"}
        and not _WIRE_RECORD.fullmatch(p.name)
    }


def _analysis(conn, slice_ids, out, window_start=None, window_end=None, **kwargs):
    from locus.analysis.window import resolve_window

    window = resolve_window(conn, slice_ids, window_start, window_end)
    out = engine.home_analysis(out)
    return engine.run_analysis(
        conn,
        window,
        question=QUESTION,
        model="scripted",
        site_contexts={"site": CONTEXT},
        out_dir=out,
        **kwargs,
    )


def test_engine_answers_directly_when_no_screenshots_requested(
    monkeypatch, distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    conv = ScriptedConversation([Response(text=ANSWER)], system_prompt=SYSTEM)
    _wire(monkeypatch, conv)
    out = tmp_path / "run"

    manifest = _analysis(conn, [slice_id], out)

    assert len(conv.calls) == 1, (
        "no tool call and a clean answer mean exactly one model turn"
    )
    assert conv.calls[0]["tools"] == ["request_screenshots"], "turn 1 offers the tool"
    task = conv.texts[-1]
    assert QUESTION in task, (
        "the entire question rides last on the turn that writes the answer"
    )
    assert "request_screenshots" in task, (
        "and on turn 1 it closes by telling the model to go and look first"
    )
    assert not any(QUESTION in t for t in conv.texts[:-1]), (
        "the analysis is never recorded into the window itself"
    )
    assert manifest["question"] == QUESTION
    assert manifest["n_pulled_screenshots"] == 0
    assert manifest["n_activity_screenshots"] >= 1
    assert (out / "1_answer_input.txt").exists(), (
        "the transcript is the analysis's record, at its root"
    )
    assert all(not isinstance(v, (bytes, bytearray)) for v in manifest.values()), (
        "the manifest is paths and counts — never image bytes"
    )


def test_an_analysis_narrates_where_its_run_is_speaking(
    monkeypatch, distilled_db, tmp_path, capsys
):
    """An analysis runs for minutes and narrates the whole way, onto the run's own stream. It opens no log and names no address; where the narration lands is the caller's."""
    import functools
    import time

    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    conv = ScriptedConversation([Response(text=ANSWER)], system_prompt=SYSTEM)
    _wire(monkeypatch, conv)
    # The pulse is brought within the test's patience and a call pushed past it,
    # which is the state it exists to make legible: in flight, not wedged.
    monkeypatch.setattr(
        engine, "beating", functools.partial(engine.beating, every_s=0.02)
    )
    posing = conv.get_response
    monkeypatch.setattr(
        conv, "get_response", lambda *a, **k: (time.sleep(0.1), posing(*a, **k))[1]
    )
    out = tmp_path / "run"

    _analysis(conn, [slice_id], out)

    said = capsys.readouterr()
    assert said.out == "", (
        "the analysis names no address of its own — the run that drove it did"
    )
    assert "asking scripted:" in said.err, (
        "the analysis says it is alive while a call is in flight"
    )
    assert "round-trip(s), 0 screenshot(s) served" in said.err, (
        "and how far it has got, so a wedge reads as standing detail"
    )
    assert not list(out.glob("*.log")), "and opens no log of its own"


def test_the_analysis_dir_holds_the_record_and_its_manifest(
    monkeypatch, distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    conv = ScriptedConversation([Response(text=ANSWER)], system_prompt=SYSTEM)
    _wire(monkeypatch, conv)
    out = tmp_path / "run"

    manifest = _analysis(conn, [slice_id], out)

    final = Path(manifest["response_path"])
    assert final.name.endswith("_answer_response.txt") and final.parent == out, (
        "the answer is the record itself — the final response transcript, never an assembled second copy"
    )
    assert ANSWER in final.read_text()
    assert not _strays(out), (
        "the directory is the record — the manifest, the wire transcripts, screenshots/ — with nothing assembled beside it"
    )
    recorded = json.loads((out / "window.json").read_text())
    assert recorded["question"] == QUESTION
    assert recorded["window_start_ts"] == 1781254694465, (
        "citations are offsets, so the manifest must state the clock anchor"
    )
    assert "price" not in recorded, (
        "the prediction is stale once real usage exists in the metas"
    )


def test_price_is_the_default_and_writes_nothing(monkeypatch, distilled_db, tmp_path):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    conv = ScriptedConversation([], system_prompt=SYSTEM)
    _wire(monkeypatch, conv)

    from locus.analysis.window import resolve_window

    price = engine.price_analysis(
        conn,
        resolve_window(conn, [slice_id]),
        question=QUESTION,
        model="scripted",
        site_contexts={"site": CONTEXT},
    )

    turn1 = price["turn1"]
    assert turn1["text_tokens"] > 0 and price["n_activity_screenshots"] > 0
    assert turn1["total_tokens"] == turn1["text_tokens"] + turn1["screenshot_tokens"]
    assert turn1["context_pct"] == pytest.approx(
        turn1["total_tokens"] / price["context_tokens"] * 100, abs=0.1
    )
    assert turn1["fits"] is True
    assert conv.calls == [], "pricing calls no model"
    assert list(tmp_path.iterdir()) == [], "pricing writes nothing"


def test_an_analysis_over_the_window_is_refused_with_its_price(
    monkeypatch, distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    conv = ScriptedConversation([], system_prompt=SYSTEM)
    _wire(monkeypatch, conv, context_tokens=10)
    out = tmp_path / "run"

    with pytest.raises(engine.PayloadOverflow) as exc:
        _analysis(conn, [slice_id], out)

    price = exc.value.price
    assert price["turn1"]["fits"] is False
    assert "pieces" in price, (
        "an over-window single slice offers its route-boundary pieces (or says it has none)"
    )
    assert list(out.iterdir()) == [], (
        "a refused analysis leaves its home as empty as it found it"
    )
    assert conv.calls == [], "a refused analysis calls no model"


def test_a_multi_slice_price_carries_each_slice_standalone(
    monkeypatch, distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM slices WHERE status='replayable' ORDER BY start_ts LIMIT 2"
        ).fetchall()
    ]
    assert len(slice_ids) == 2
    conv = ScriptedConversation([], system_prompt=SYSTEM)
    _wire(monkeypatch, conv)

    from locus.analysis.window import resolve_window

    price = engine.price_analysis(
        conn,
        resolve_window(conn, slice_ids),
        question=QUESTION,
        model="scripted",
        site_contexts={"site": CONTEXT},
    )

    assert len(price["per_slice"]) == 2, (
        "the whittling surface: every slice priced alone"
    )
    for entry in price["per_slice"]:
        assert entry["standalone_tokens"] > 0 and entry["slice"]


def test_engine_pulls_arbitrary_screenshots_then_answers(
    monkeypatch, distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, hi = slice_bounds(conn, slice_id)
    duration = hi - lo
    offsets = [duration // 3, 2 * duration // 3]  # arbitrary moments inside the slice
    conv = ScriptedConversation(
        [
            Response(
                text="",
                tool_calls=[ToolCall("c1", "request_screenshots", _pull(*offsets))],
            ),
            Response(text=ANSWER),
        ],
        system_prompt=SYSTEM,
    )
    _wire(monkeypatch, conv)
    out = tmp_path / "run"

    manifest = _analysis(conn, [slice_id], out)

    assert len(conv.calls) == 2, "the tool call forces a second turn for the answer"
    assert conv.calls[0]["tools"] == ["request_screenshots"]
    assert conv.calls[1]["tools"] == ["request_screenshots"], (
        "the declaration never leaves the request — withdrawing it would "
        "rewrite the head of a conversation that otherwise only extends"
    )
    resume = conv.texts[-1]
    assert QUESTION in resume, (
        "the question rides last again, now after the screenshots it asked for"
    )
    assert "request_screenshots" not in resume, (
        "without the go-and-look step: the task is what forces the answer"
    )
    assert [cid for cid, _ in conv.tool_results] == ["c1"], "the tool call is answered"
    assert manifest["n_pulled_screenshots"] == 2
    assert manifest["render_faults"] == {}, "clean screenshots leave no fault entries"
    assert manifest["pulled"] == [
        ["S1", max(lo, min(lo + off, hi))] for off in sorted(offsets)
    ], "the record states which moments rode as pulls"
    assert manifest["activity_screenshots"]["S1"], (
        "and which rode pushed — per-screenshot provenance, not just counts"
    )
    for off in offsets:
        ts = max(lo, min(lo + off, hi))
        assert (out / "screenshots" / f"S1_screenshot_{ts}.png").exists(), (
            "the requested arbitrary timestamp was rendered fresh"
        )


def test_the_renderer_is_released_after_every_round_of_captures(
    monkeypatch, distilled_db, tmp_path
):
    """An analysis waiting on the model holds no browser: the renderer is given up after the activity-screenshot burst and again after each served pull round, and the next round opens it afresh."""
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, hi = slice_bounds(conn, slice_id)
    conv = ScriptedConversation(
        [
            Response(
                text="",
                tool_calls=[
                    ToolCall("c1", "request_screenshots", _pull((hi - lo) // 2))
                ],
            ),
            Response(text=ANSWER),
        ],
        system_prompt=SYSTEM,
    )
    _wire(monkeypatch, conv)
    sessions = []

    def remembering(*args, **kwargs):
        session = FakeRenderSession(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(engine, "RenderSession", remembering)

    _analysis(conn, [slice_id], tmp_path / "run")

    (session,) = sessions
    pushed = len(session.captured) - 1
    assert session.releases[:2] == [pushed, pushed + 1], (
        "released once the burst was rendered, and again once the pull was served"
    )
    assert session.open is False


def test_engine_serves_every_pull_the_context_holds(
    monkeypatch, distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, hi = slice_bounds(conn, slice_id)
    duration = max(hi - lo, 6)
    offsets = [duration * k // 6 for k in range(1, 6)]  # five, and nothing caps them
    conv = ScriptedConversation(
        [
            Response(
                text="",
                tool_calls=[ToolCall("c1", "request_screenshots", _pull(*offsets))],
            ),
            Response(text=ANSWER),
        ],
        system_prompt=SYSTEM,
    )
    _wire(monkeypatch, conv)

    manifest = _analysis(conn, [slice_id], tmp_path / "run")

    assert manifest["n_pulled_screenshots"] == 5, "an ample context serves them all"
    assert manifest["refused_screenshots"] == 0


def test_engine_refuses_the_pulls_the_context_cannot_hold(
    monkeypatch, distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, hi = slice_bounds(conn, slice_id)
    duration = max(hi - lo, 6)
    offsets = [duration * k // 6 for k in range(1, 6)]
    from locus.analysis.window import resolve_window

    activity_screenshots = len(
        engine.screenshot_moments(conn, resolve_window(conn, [slice_id]))[slice_id]
    )
    # Every image in the double costs one token; leave room for exactly two
    # beyond the pushed set the window already spends.
    conv = ScriptedConversation(
        [
            Response(
                text="",
                tool_calls=[ToolCall("c1", "request_screenshots", _pull(*offsets))],
            ),
            Response(text=ANSWER),
        ],
        system_prompt=SYSTEM,
        room=activity_screenshots + 2,
    )
    _wire(monkeypatch, conv)

    manifest = _analysis(conn, [slice_id], tmp_path / "run")

    assert manifest["n_pulled_screenshots"] == 2, "only what the context held"
    assert manifest["refused_screenshots"] == 3
    for off in offsets[:2]:
        ts = max(lo, min(lo + off, hi))
        assert (
            tmp_path / "run" / "screenshots" / f"S1_screenshot_{ts}.png"
        ).exists(), "the earliest moments are the ones served"
    pushed = engine.screenshot_moments(conn, resolve_window(conn, [slice_id]))[slice_id]
    shown = {f"S1_screenshot_{ts}.png" for ts in pushed} | {
        f"S1_screenshot_{max(lo, min(lo + off, hi))}.png" for off in offsets[:2]
    }
    on_disk = {p.name for p in (tmp_path / "run" / "screenshots").iterdir()}
    assert on_disk == shown, (
        "the record holds exactly the screenshots the model was shown — a "
        "moment rendered for a refused pull must not survive as evidence"
    )
    note = conv.tool_results[0][1]
    assert re.search(r"\b3\b", note), (
        "the refusal is reported back to the model with its count, not silently swallowed"
    )


def test_engine_fails_a_second_round_of_requests(monkeypatch, distilled_db, tmp_path):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, hi = slice_bounds(conn, slice_id)
    duration = max(hi - lo, 4)
    conv = ScriptedConversation(
        [
            Response(
                text="",
                tool_calls=[
                    ToolCall("c1", "request_screenshots", _pull(duration // 3))
                ],
            ),
            Response(
                text="",
                tool_calls=[
                    ToolCall("c2", "request_screenshots", _pull(duration // 2))
                ],
            ),
            Response(text=ANSWER),
        ],
        system_prompt=SYSTEM,
    )
    _wire(monkeypatch, conv)

    manifest = _analysis(conn, [slice_id], tmp_path / "run")

    assert manifest["n_pulled_screenshots"] == 1, (
        "one round of screenshots per posing of the task, and no more"
    )
    assert conv.tool_results[1] == ("c2", SCREENSHOTS_ALREADY_SERVED), (
        "a second round fails in the tool's own channel, telling it to write"
    )


def test_engine_fails_loud_when_the_model_never_stops_requesting(
    monkeypatch, distilled_db, tmp_path
):
    """Tool calls all the way down: after the served round, each further round is refused in the tool's channel at most MAX_REFUSAL_ROUNDS times, then the analysis dies loud rather than looping forever on the caller's money."""
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, hi = slice_bounds(conn, slice_id)
    duration = max(hi - lo, 4)
    pull = lambda cid: Response(
        text="",
        tool_calls=[ToolCall(cid, "request_screenshots", _pull(duration // 3))],
    )
    conv = ScriptedConversation(
        [pull("c1"), pull("c2"), pull("c3"), pull("c4")], system_prompt=SYSTEM
    )
    _wire(monkeypatch, conv)

    with pytest.raises(RuntimeError, match="requested screenshots again"):
        _analysis(conn, [slice_id], tmp_path / "run")

    refused = [r for r in conv.tool_results if r[1] == SCREENSHOTS_ALREADY_SERVED]
    assert [cid for cid, _ in refused] == ["c2", "c3"], (
        "exactly MAX_REFUSAL_ROUNDS refusals are fed back before the loud stop"
    )


def test_a_pull_past_its_slices_end_is_failed_and_a_corrected_retry_is_served(
    monkeypatch, distilled_db, tmp_path
):
    """An offset past the slice's recording is an addressing error: the call fails in the tool's own channel naming the defect, nothing is served or silently substituted, and the corrected retry still gets the task's one served round — a failed attempt spends nothing."""
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, hi = slice_bounds(conn, slice_id)
    good = (hi - lo) // 2
    conv = ScriptedConversation(
        [
            Response(
                text="",
                tool_calls=[
                    ToolCall(
                        "c1",
                        "request_screenshots",
                        _pull((hi - lo) + 5000),
                    )
                ],
            ),
            Response(
                text="",
                tool_calls=[ToolCall("c2", "request_screenshots", _pull(good))],
            ),
            Response(text=ANSWER),
        ],
        system_prompt="x",
    )
    _wire(monkeypatch, conv)

    manifest = _analysis(conn, [slice_id], tmp_path / "run")

    failed = conv.tool_results[0][1]
    assert format_offset(lo + (hi - lo) + 5000, lo) in failed, (
        "the defect names the moment"
    )
    assert format_offset(hi, lo) in failed, (
        "the defect names the recording's own span, in window offsets"
    )
    assert manifest["pulled"] == [["S1", lo + good]], (
        "the corrected retry was served — the failed attempt spent the round on nothing"
    )


def test_unparseable_tool_arguments_fail_in_channel_and_the_retry_is_served(
    monkeypatch, distilled_db, tmp_path
):
    """A call whose arguments never parsed (the backend marks them None) is a bad request like any other: failed in the tool's own channel with the defect named, nothing served, and the corrected retry gets the round."""
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, hi = slice_bounds(conn, slice_id)
    good = (hi - lo) // 2
    conv = ScriptedConversation(
        [
            Response(
                text="",
                tool_calls=[ToolCall("c1", "request_screenshots", None)],
            ),
            Response(
                text="",
                tool_calls=[ToolCall("c2", "request_screenshots", _pull(good))],
            ),
            Response(text=ANSWER),
        ],
        system_prompt="x",
    )
    _wire(monkeypatch, conv)

    manifest = _analysis(conn, [slice_id], tmp_path / "run")

    failed = conv.tool_results[0][1]
    assert "JSON" in failed
    assert manifest["pulled"] == [["S1", lo + good]]


def test_a_model_that_keeps_pulling_invalid_moments_fails_loud(
    monkeypatch, distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, hi = slice_bounds(conn, slice_id)
    bad = lambda cid: Response(
        text="",
        tool_calls=[ToolCall(cid, "request_screenshots", _pull((hi - lo) + 5000))],
    )
    conv = ScriptedConversation([bad("c1"), bad("c2"), bad("c3")], system_prompt="x")
    _wire(monkeypatch, conv)

    with pytest.raises(RuntimeError, match="invalid moments"):
        _analysis(conn, [slice_id], tmp_path / "run")


def test_an_empty_answer_gets_the_same_repair_turn(monkeypatch, distilled_db, tmp_path):
    """An empty answer is a mechanical defect like any other: the one repair turn names it and asks for the complete answer, and what comes back ships."""
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    conv = ScriptedConversation(
        [Response(text="   "), Response(text=ANSWER)], system_prompt=SYSTEM
    )
    _wire(monkeypatch, conv)
    out = tmp_path / "run"

    manifest = _analysis(conn, [slice_id], out)

    assert len(conv.calls) == 2
    repair = conv.texts[-1]
    assert EMPTY_ANSWER_REPAIR.strip() in repair, (
        "the emptiness is named to the model as its own repair"
    )
    assert ANSWER in Path(manifest["response_path"]).read_text()


def test_a_repair_that_is_also_empty_fails_loud(monkeypatch, distilled_db, tmp_path):
    """Only when the repair itself returns no answer does the run error — nothing exists to ship."""
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    conv = ScriptedConversation(
        [Response(text="   "), Response(text=" ")], system_prompt=SYSTEM
    )
    _wire(monkeypatch, conv)

    with pytest.raises(RuntimeError, match="empty answer"):
        _analysis(conn, [slice_id], tmp_path / "run")

    assert len(conv.calls) == 2, "one repair turn, then the loud stop"
    recorded = json.loads((tmp_path / "run" / "window.json").read_text())
    assert [s["slice_id"] for s in recorded["slices"]] == [slice_id], (
        "the window's record lands before the model is asked, so the transcripts of a run that dies still resolve through it"
    )
    assert "pulled" not in recorded, "what the run found is written only when it ends"


def test_engine_refuses_to_overwrite_output(monkeypatch, distilled_db, tmp_path):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    conv = ScriptedConversation([Response(text=ANSWER)], system_prompt=SYSTEM)
    _wire(monkeypatch, conv)
    out = tmp_path / "run"
    out.mkdir()

    with pytest.raises(FileExistsError):
        _analysis(conn, [slice_id], out)


def test_engine_window_spans_multiple_slices(monkeypatch, distilled_db, tmp_path):
    conn = connect(distilled_db)
    slice_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM slices WHERE status='replayable' ORDER BY start_ts LIMIT 3"
        ).fetchall()
    ]
    assert len(slice_ids) >= 2, "the fixture carries a multi-slice visitor"
    conv = ScriptedConversation([Response(text=ANSWER_LABELED)], system_prompt=SYSTEM)
    _wire(monkeypatch, conv)
    out = tmp_path / "run"

    manifest = _analysis(conn, slice_ids, out)

    assert [s["slice_id"] for s in manifest["slices"]] == slice_ids
    assert manifest["n_activity_screenshots"] >= len(slice_ids), (
        "the activity-screenshots span every slice in the window"
    )


def test_engine_piece_window_anchors_screenshot_moments_at_its_bounds(
    monkeypatch, distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, hi = slice_bounds(conn, slice_id)
    ws, we = lo + (hi - lo) // 4, hi - (hi - lo) // 4  # a piece inside the slice
    bounds = conn.execute(
        "SELECT MIN(timestamp) lo, MAX(timestamp) hi FROM events WHERE slice_id=? AND timestamp>=? AND timestamp<?",
        (slice_id, ws, we),
    ).fetchone()
    window_lo, window_hi = bounds["lo"], bounds["hi"]
    conv = ScriptedConversation([Response(text=ANSWER)], system_prompt=SYSTEM)
    _wire(monkeypatch, conv)
    out = tmp_path / "run"

    _analysis(conn, [slice_id], out, window_start=ws, window_end=we)

    screenshot_ts = sorted(
        int(p.stem.rpartition("_screenshot_")[2])
        for p in (out / "screenshots").glob("*_screenshot_*.png")
    )
    assert window_lo in screenshot_ts and window_hi in screenshot_ts, (
        "the piece's own in-bounds endpoints anchor the pushed set"
    )
    assert screenshot_ts[0] >= ws and screenshot_ts[-1] < we, (
        "no pushed screenshot escapes the window bounds"
    )


def test_engine_rejects_duplicate_slices_in_a_window(
    monkeypatch, distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    conv = ScriptedConversation([Response(text=ANSWER)], system_prompt=SYSTEM)
    _wire(monkeypatch, conv)
    with pytest.raises(ValueError, match="twice|duplicate"):
        _analysis(conn, [slice_id, slice_id], tmp_path / "run")


def test_a_pushed_moment_re_requested_is_served_as_a_detail_purchase(
    monkeypatch, distilled_db, tmp_path
):
    """The pushed set rides the cheap tier, so a pull of a moment it
    already showed buys real detail and is served; only a duplicate within the
    round is dropped — the same moment twice buys nothing — and the model is
    told, not silently ignored."""
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    conv = ScriptedConversation(
        [
            Response(
                text="",
                tool_calls=[ToolCall("c1", "request_screenshots", _pull(0, 0))],
            ),
            Response(text=ANSWER),
        ],
        system_prompt=SYSTEM,
    )
    _wire(monkeypatch, conv)

    manifest = _analysis(conn, [slice_id], tmp_path / "run")

    assert manifest["n_pulled_screenshots"] == 1, (
        "offset 0 is the window's first pushed screenshot — re-served at full detail"
    )
    assert "once" in conv.tool_results[0][1], (
        "the model is told the duplicate was folded"
    )


def test_request_screenshots_tool_has_one_shape_for_every_window():
    tool = request_screenshots_tool()
    items = tool["parameters"]["properties"]["screenshots"]["items"]
    assert items["required"] == ["timestamp"], (
        "a moment is an offset; the label rides only where the window has several slices"
    )
    assert "label" in items["properties"] and "enum" not in items["properties"]["label"]


def _two_slice_window(conn):
    slice_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM slices WHERE status='replayable' ORDER BY start_ts LIMIT 2"
        ).fetchall()
    ]
    assert len(slice_ids) == 2, "the fixture carries a multi-slice visitor"
    return slice_ids


def test_a_multi_slice_analysis_reads_the_one_contract(
    monkeypatch, distilled_db, tmp_path
):
    """The system prompt and the closing of a two-slice analysis are the one
    contract every window reads — both citation cases stated — and the
    manifest carries the slice table the labeled citations resolve through."""
    conn = connect(distilled_db)
    slice_ids = _two_slice_window(conn)
    conv = ScriptedConversation([Response(text=ANSWER_LABELED)])
    from locus.analysis import budget as budget_mod

    monkeypatch.setattr(
        engine,
        "make_conversation",
        lambda model, system, **k: _homed(conv, k),
    )
    monkeypatch.setattr(engine, "RenderSession", FakeRenderSession)
    monkeypatch.setattr(
        budget_mod, "cost_model_for", lambda name: (_word_cost_model(), 10_000_000, 1.0)
    )
    out = tmp_path / "run"

    manifest = _analysis(conn, slice_ids, out)

    assert [s["label"] for s in manifest["slices"]] == ["S1", "S2"]
    assert [s["slice_id"] for s in manifest["slices"]] == slice_ids
    recorded = json.loads((out / "window.json").read_text())
    assert recorded["slices"] == manifest["slices"], (
        "window.json carries the slice table — the answer's citations resolve through it"
    )


def test_engine_serves_label_addressed_pulls(monkeypatch, distilled_db, tmp_path):
    """A multi-slice pull names its slice label; the engine resolves it through
    the table and renders from that slice. A request naming a label outside the
    window's roster fails the whole call in the tool's own channel — nothing
    served, the round unspent — and the corrected retry is served."""
    conn = connect(distilled_db)
    slice_ids = _two_slice_window(conn)
    lo1, _ = slice_bounds(conn, slice_ids[0])
    lo2, hi2 = slice_bounds(conn, slice_ids[1])
    off2 = (lo2 + hi2) // 2 - lo1
    conv = ScriptedConversation(
        [
            Response(
                text="",
                tool_calls=[
                    ToolCall(
                        "c1",
                        "request_screenshots",
                        {
                            "screenshots": [
                                {"label": "S2", "timestamp": off2},
                                {"label": "S9", "timestamp": 0},
                            ]
                        },
                    )
                ],
            ),
            Response(
                text="",
                tool_calls=[
                    ToolCall(
                        "c2",
                        "request_screenshots",
                        {"screenshots": [{"label": "S2", "timestamp": off2}]},
                    )
                ],
            ),
            Response(text=ANSWER_LABELED),
        ],
        system_prompt="x",
    )
    _wire(monkeypatch, conv)
    out = tmp_path / "run"

    manifest = _analysis(conn, slice_ids, out)

    failed = conv.tool_results[0][1]
    assert "S9" in failed and "S1" in failed and "S2" in failed, (
        "the defect names the bad label against the window's own roster"
    )
    assert manifest["n_pulled_screenshots"] == 1
    ts = max(lo2, min(lo1 + off2, hi2))
    assert (out / "screenshots" / f"S2_screenshot_{ts}.png").exists(), (
        "the corrected retry rendered from the named slice"
    )
    anchors = [t for t in conv.texts if t.startswith("screenshot at ")]
    assert anchors and anchors[0].startswith("screenshot at [S2 "), (
        "the served screenshot is anchored with its slice label"
    )


def test_a_pull_outside_its_slices_recording_is_failed_in_channel(
    monkeypatch, distilled_db, tmp_path
):
    """An offset pointing before S2's recording is an addressing error — the moment exists in the window but not in the slice the request named. It is never silently served from a neighboring slice or moved to a moment the model did not ask for: the call fails in the tool's own channel, naming the slice's real span."""
    conn = connect(distilled_db)
    slice_ids = _two_slice_window(conn)
    lo1, _ = slice_bounds(conn, slice_ids[0])
    lo2, _ = slice_bounds(conn, slice_ids[1])
    assert lo1 < lo2, "offset 0 is inside the window but ahead of S2's recording"
    conv = ScriptedConversation(
        [
            Response(
                text="",
                tool_calls=[
                    ToolCall(
                        "c1",
                        "request_screenshots",
                        {"screenshots": [{"label": "S2", "timestamp": 0}]},
                    )
                ],
            ),
            Response(text=ANSWER_LABELED),
        ],
        system_prompt="x",
    )
    _wire(monkeypatch, conv)
    out = tmp_path / "run"

    manifest = _analysis(conn, slice_ids, out)

    failed = conv.tool_results[0][1]
    assert "S2" in failed and format_offset(lo2, lo1) in failed, (
        "the defect names the slice and its real span"
    )
    assert manifest["n_pulled_screenshots"] == 0
    assert not list((out / "screenshots").glob("S2_screenshot_*.png")) or all(
        int(p.stem.rpartition("_screenshot_")[2]) >= lo2
        for p in (out / "screenshots").glob("S2_screenshot_*.png")
    ), "nothing was rendered for the failed request beyond the pushed set"


def test_a_pull_into_a_slices_lead_clamps_to_its_covering_snapshot(
    monkeypatch, distilled_db, tmp_path
):
    """The window opens at the slice's first event, but nothing was recorded of the page
    until the covering snapshot behind it — so offset 0 buys the earliest screenshot that can
    testify, never one stamped ahead of the pixels it shows."""
    from locus.evidence.rrweb_constants import EventType

    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, _ = slice_bounds(conn, slice_id)
    snapshot_ts = conn.execute(
        "SELECT MIN(timestamp) t FROM events WHERE slice_id = ? AND type = ?",
        (slice_id, EventType.FullSnapshot),
    ).fetchone()["t"]
    assert lo < snapshot_ts
    conv = ScriptedConversation(
        [
            Response(
                text="",
                tool_calls=[ToolCall("c1", "request_screenshots", _pull(0))],
            ),
            Response(text=ANSWER),
        ],
        system_prompt="x",
    )
    _wire(monkeypatch, conv)

    manifest = _analysis(conn, [slice_id], tmp_path / "run")

    assert manifest["pulled"] == [["S1", snapshot_ts]]
    assert min(manifest["activity_screenshots"]["S1"]) == snapshot_ts, (
        "the pushed set opens there too"
    )


def test_an_invalid_citation_gets_a_repair_turn_and_the_repair_ships(
    monkeypatch, distilled_db, tmp_path
):
    """Citation validity is mechanical, so the engine checks it — and hands back exactly the defects: the repaired answer is the deliverable, the first stays an earlier transcript, and the repair task names each bad citation without inviting any other change."""
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, hi = slice_bounds(conn, slice_id)
    bad = "The visitor did something [00:99.100], then left far later [59:59.999]."
    conv = ScriptedConversation(
        [Response(text=bad), Response(text=ANSWER)], system_prompt=SYSTEM
    )
    _wire(monkeypatch, conv)
    out = tmp_path / "run"

    manifest = _analysis(conn, [slice_id], out)

    answer = Path(manifest["response_path"]).read_text()
    assert ANSWER in answer and "[00:99.100]" not in answer, (
        "the deliverable is the repaired text, never the invalid one"
    )
    assert (out / "1_answer_response.txt").exists() and (
        out / "2_answer_response.txt"
    ).exists(), "both turns persist — the invalid answer is auditable in the record"
    repair = conv.texts[-1]
    assert "<REPAIR>" in repair
    assert "[00:99.100]" in repair and "[MM:SS.mmm]" in repair, (
        "a malformed time is named with the form it failed"
    )
    assert "[59:59.999]" in repair and format_offset(hi, lo) in repair, (
        "a well-formed citation past the recording is named with the real span"
    )
    assert repair.rindex("[MM:SS.mmm]") > repair.index("[59:59.999]"), (
        "the bracket form is demanded again after the defects — probed live, a "
        "rewrite abandons the format unless it is demanded at the moment of writing"
    )
    assert QUESTION not in repair, (
        "the question already sits in the retained history; the repair adds only the defects"
    )


def test_a_clean_answer_ships_with_no_repair_turn(monkeypatch, distilled_db, tmp_path):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    conv = ScriptedConversation([Response(text=ANSWER)], system_prompt=SYSTEM)
    _wire(monkeypatch, conv)

    _analysis(conn, [slice_id], tmp_path / "run")

    assert len(conv.calls) == 1, "valid citations cost zero extra calls"
    assert not any("<REPAIR>" in t for t in conv.texts)


def test_the_repairs_answer_ships_even_when_still_invalid(
    monkeypatch, distilled_db, tmp_path
):
    """One repair turn, and what it returns is the deliverable: a completed analysis is never destroyed over residual citation defects — they stand in the record as what they are."""
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    still_bad = "Something else happened [00:98.200]."
    conv = ScriptedConversation(
        [Response(text="Something happened [00:99.100]."), Response(text=still_bad)],
        system_prompt=SYSTEM,
    )
    _wire(monkeypatch, conv)
    out = tmp_path / "run"

    manifest = _analysis(conn, [slice_id], out)

    assert len(conv.calls) == 2, "one repair turn, never a second"
    assert still_bad in Path(manifest["response_path"]).read_text(), (
        "the repair's answer is the deliverable, residual defects and all"
    )


def test_a_bare_citation_in_a_labeled_window_gets_repaired(
    monkeypatch, distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_ids = _two_slice_window(conn)
    conv = ScriptedConversation(
        [Response(text=ANSWER), Response(text=ANSWER_LABELED)], system_prompt=SYSTEM
    )
    _wire(monkeypatch, conv)

    _analysis(conn, slice_ids, tmp_path / "run")

    repair = conv.texts[-1]
    assert "[00:00.100]" in repair and "S1" in repair and "S2" in repair, (
        "a bare citation in a multi-slice window is a defect naming the roster"
    )


def _imported_modules(path):
    tree = ast.parse(Path(path).read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
    return mods


def test_policy_measurement_boundary_holds():
    from locus.analysis import oracle

    assert not any("oracle" in m for m in _imported_modules(engine.__file__)), (
        "the engine (policy) must not import the oracle (measurement)"
    )
    assert not any(m.endswith("engine") for m in _imported_modules(oracle.__file__)), (
        "the oracle (measurement) must not import the engine (policy)"
    )


def _locus_import_closure(module: str) -> set[str]:
    """The locus modules a fresh interpreter loads for `import <module>` — the whole transitive closure, which a file-level import scan cannot see."""
    import subprocess
    import sys

    out = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys, {module}; "
                "print('\\n'.join(m for m in sys.modules if m.startswith('locus')))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(out.stdout.split())


def test_the_boundary_holds_across_the_whole_import_closure():
    """An indirect route — the engine importing a module that itself imports the oracle, or the grounding internals the oracle judges with — reads the same forbidden material through a middleman. The import closure is the fact the file-level scan approximates."""
    engine_side = _locus_import_closure("locus.analysis.engine")
    assert "locus.analysis.oracle" not in engine_side, (
        "loading the engine loaded the oracle — the policy can read its own objective"
    )
    assert not any(m.startswith("locus.analysis.ground") for m in engine_side), (
        "loading the engine loaded the grounding internals — the policy can read "
        "the rubric it is measured by"
    )
    oracle_side = _locus_import_closure("locus.analysis.oracle")
    assert "locus.analysis.engine" not in oracle_side, (
        "loading the oracle loaded the engine — the measurement depends on the policy"
    )


def test_the_answer_is_the_record_itself(monkeypatch, distilled_db, tmp_path):
    """The answer is the record itself: the final response transcript with the
    contract's meta section in place — nothing carved into a side file, nothing
    assembled beside the record, so the model's whole utterance sits on the
    read path."""
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    reply = f"{ANSWER}\n\n{META_HEADER}\n- The projection cut off mid-table on the second snapshot.\n"
    conv = ScriptedConversation(
        [Response(text=reply, thoughts_text="the cursor hesitates here")],
        system_prompt=SYSTEM,
    )
    _wire(monkeypatch, conv)
    out = tmp_path / "run"

    manifest = _analysis(conn, [slice_id], out)

    final = Path(manifest["response_path"])
    assert final == out / "1_answer_response.txt", (
        "the manifest names the final answer-labeled transcript"
    )
    answer = final.read_text()
    assert ANSWER in answer
    assert META_HEADER in answer and "cut off mid-table" in answer, (
        "the meta section stays in place in the one record"
    )
    assert "the cursor hesitates here" in answer, (
        "thoughts the backend exposes ride the record"
    )
    assert not _strays(out), (
        "the directory is the record — the manifest, the wire transcripts, screenshots/ — with nothing assembled beside it"
    )


def test_a_channel_only_reply_is_an_empty_answer(monkeypatch, distilled_db, tmp_path):
    """A reply that is only the contract's meta section answered nothing: it gets the empty-answer repair, and a repair that again carries no answer above the channel fails the run."""
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    channel_only = Response(text=f"{META_HEADER}\n- only complaints, no answer\n")
    conv = ScriptedConversation([channel_only, channel_only], system_prompt=SYSTEM)
    _wire(monkeypatch, conv)

    with pytest.raises(RuntimeError, match="empty answer"):
        _analysis(conn, [slice_id], tmp_path / "run")

    assert EMPTY_ANSWER_REPAIR.strip() in conv.texts[-1], (
        "the emptiness was named to the model before the run gave up"
    )


def _table(*rows):
    return [{"label": label, "start_ts": lo, "end_ts": hi} for label, lo, hi in rows]


def test_citation_violations_is_the_format_made_mechanical():
    """The validator holds every citation-shaped span to the citation format, the window's labels, and its bounds, and nothing else — a valid citation of a wrong moment is judgment, not its business."""
    from locus.analysis.window import citation_violations

    single = _table(("S1", 1_000_000, 1_600_000))  # a 10-minute recording

    assert citation_violations("plain prose, no brackets", single, 1_000_000) == []
    assert (
        citation_violations(
            "ok [00:10.500] and a range [00:10.500, 05:00.000]", single, 1_000_000
        )
        == []
    )
    assert citation_violations("aspect ratio [3:2] is prose", single, 1_000_000) == []

    (malformed,) = citation_violations("bad [00:99.123]", single, 1_000_000)
    assert malformed.startswith("[00:99.123]") and "[MM:SS.mmm]" in malformed
    (missing_millis,) = citation_violations("bad [00:10]", single, 1_000_000)
    assert missing_millis.startswith("[00:10]") and "[MM:SS.mmm]" in missing_millis

    (past_end,) = citation_violations("late [59:59.999]", single, 1_000_000)
    assert past_end.startswith("[59:59.999]") and "[10:00.000]" in past_end
    (bad_range,) = citation_violations("swap [00:20.000, 00:10.000]", single, 1_000_000)
    assert (
        bad_range.startswith("[00:20.000, 00:10.000]")
        and "[10:00.000]" not in bad_range
    )
    (labeled,) = citation_violations("label [S1 00:10.000]", single, 1_000_000)
    assert labeled.startswith("[S1 00:10.000]")

    (repeated,) = citation_violations(
        "twice [00:99.123] and again [00:99.123]", single, 1_000_000
    )
    assert repeated == malformed, "one defect however often the span repeats"

    multi = _table(("S1", 1_000_000, 1_300_000), ("S2", 1_200_000, 1_600_000))
    assert citation_violations("fine [S2 04:00.000]", multi, 1_000_000) == [], (
        "a moment inside the labeled slice passes"
    )
    (bare,) = citation_violations("bare [00:10.000]", multi, 1_000_000)
    assert bare.startswith("[00:10.000]") and "S1" in bare and "S2" in bare
    (unknown,) = citation_violations("ghost [S9 00:10.000]", multi, 1_000_000)
    assert unknown.startswith("[S9 00:10.000]") and "S1" in unknown and "S2" in unknown
    (crossed,) = citation_violations("crossed [S1 05:30.000]", multi, 1_000_000)
    assert crossed.startswith("[S1 05:30.000]") and "[05:00.000]" in crossed, (
        "a moment the window holds but the cited slice does not is a defect"
    )

    assert (
        citation_violations("both [S2 04:00.000, S2 04:30.000]", multi, 1_000_000) == []
    ), "a range naming its one slice twice is the form, redundantly"
    (split,) = citation_violations(
        "split [S1 01:00.000, S2 04:00.000]", multi, 1_000_000
    )
    assert split.startswith("[S1 01:00.000, S2 04:00.000]") and "two" in split


def test_a_citation_is_valid_as_far_out_as_the_payload_stamps_it():
    """The payload prints a moment's minutes unbounded — a comparative window's later slices sit days after its start — and the model cites the stamp it was shown, so the citation form admits what the payload emits."""
    from locus.analysis.window import citation_violations, format_offset

    late = 1_000_000 + 2079 * 60_000 + 37_620
    multi = _table(("S1", 1_000_000, 1_300_000), ("S3", late - 60_000, late + 60_000))
    stamp = format_offset(late, 1_000_000, "S3")
    assert stamp == "[S3 2079:37.620]"
    assert citation_violations(f"seen {stamp}", multi, 1_000_000) == []
    (early,) = citation_violations("off [S3 00:10.000]", multi, 1_000_000)
    assert early.startswith("[S3 00:10.000]") and "outside S3" in early
