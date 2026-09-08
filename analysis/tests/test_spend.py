"""Spend governance — the operator's dollar walls and the ledger they are enforced against.

The bar: the refusal fires at the boundary on every paid path, the ledger survives what a real deployment does (concurrent analyses, torn writes, clock boundaries), and the arithmetic is exactly usage × the declared card prices — never an estimate.
"""

import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from _analysis_support import GEMINI
from _support import (
    best_slice,
    slice_bounds,
)
from locus.analysis import spend
from locus.analysis.spend import (
    SpendWall,
    call_cost,
    check_walls,
    load_declaration,
    period_bounds,
    read_ledger,
    spend_report,
    verify_prices,
)
from locus.evidence.db import connect

PRICES = {
    "input_per_mtok": 0.25,
    "cached_input_per_mtok": 0.025,
    "output_per_mtok": 1.5,
}


@pytest.fixture(autouse=True)
def gemini_key(monkeypatch):
    """The Gemini fakes never reach the wire, but the SDK client refuses to construct keyless."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-not-used")


REGISTRY_MATCH = {
    f"gemini/{GEMINI}": {
        "input_cost_per_token": 2.5e-07,
        "cache_read_input_token_cost": 2.5e-08,
        "output_cost_per_token": 1.5e-06,
        "output_cost_per_reasoning_token": 1.5e-06,
    }
}


def _ms(*args):
    return int(datetime(*args, tzinfo=timezone.utc).timestamp() * 1000)


# ── period semantics: UTC calendar-aligned, stated precisely ───────────────


def test_periods_are_utc_calendar_aligned():
    noon_wednesday = _ms(2026, 7, 15, 12, 0, 0)
    assert period_bounds("day", noon_wednesday) == (_ms(2026, 7, 15), _ms(2026, 7, 16))
    assert period_bounds("week", noon_wednesday) == (
        _ms(2026, 7, 13),
        _ms(2026, 7, 20),
    ), "ISO week: Monday 00:00 UTC"
    assert period_bounds("month", noon_wednesday) == (_ms(2026, 7, 1), _ms(2026, 8, 1))


def test_periods_cross_year_boundaries():
    new_years_thursday = _ms(2026, 1, 1, 5, 0, 0)
    assert period_bounds("week", new_years_thursday) == (
        _ms(2025, 12, 29),
        _ms(2026, 1, 5),
    ), "the week containing Jan 1 opens the prior year's last Monday"
    assert period_bounds("month", _ms(2026, 12, 15)) == (
        _ms(2026, 12, 1),
        _ms(2027, 1, 1),
    )


def test_a_millisecond_before_the_period_is_outside_it():
    now = _ms(2026, 7, 15, 12)
    start, _ = period_bounds("day", now)
    assert period_bounds("day", start)[0] == start, "the boundary is inclusive"
    assert period_bounds("day", start - 1)[0] == _ms(2026, 7, 14), (
        "one ms earlier belongs to the prior day"
    )


def test_unknown_period_refuses():
    with pytest.raises(ValueError, match="unknown period"):
        period_bounds("fortnight", _ms(2026, 7, 15))


# ── the declaration: silent non-limiting is the defect ──────────────────────


def test_the_shipped_declaration_walls_a_fresh_clone():
    from pathlib import Path

    declared = load_declaration(Path(__file__).resolve().parents[2])
    assert declared, (
        "the repo ships spend.toml with a real cap — a fresh clone is walled"
    )


def test_absent_declaration_is_no_walls(spend_isolation):
    (spend_isolation / "config" / "spend.toml").unlink()
    assert load_declaration(spend_isolation) == {}
    _seed(spend_isolation, _ms(2026, 7, 15), 1_000_000.0)
    check_walls(_ms(2026, 7, 15, 12))
    # An empty declaration declares no caps — the same no-walls as no file.
    (spend_isolation / "config" / "spend.toml").write_text("")
    check_walls(_ms(2026, 7, 15, 12))


def test_a_typoed_cap_key_refuses_instead_of_capping_nothing(spend_isolation):
    (spend_isolation / "config" / "spend.toml").write_text("weekly_usd = 20.0\n")
    with pytest.raises(ValueError, match="weekly_usd"):
        check_walls()


def test_a_non_dollar_cap_refuses(spend_isolation):
    (spend_isolation / "config" / "spend.toml").write_text("week_usd = -5\n")
    with pytest.raises(ValueError, match="finite number of dollars"):
        check_walls()
    (spend_isolation / "config" / "spend.toml").write_text("week_usd = true\n")
    with pytest.raises(ValueError, match="finite number of dollars"):
        check_walls()


# ── the arithmetic: usage × declared prices, mechanical ─────────────────────


def test_call_cost_prices_cached_input_at_the_cached_rate():
    usd = call_cost(
        {
            "input_tokens": 1_000_000,
            "cached_tokens": 200_000,
            "output_tokens": 100_000,
            "thoughts_tokens": 50_000,
        },
        PRICES,
    )
    assert (
        usd
        == round((800_000 * 0.25 + 200_000 * 0.025 + 150_000 * 1.5) / 1e6, 8)
        == 0.43
    )


def test_call_cost_refuses_a_usage_record_missing_a_field():
    complete = {
        "input_tokens": 1000,
        "cached_tokens": 0,
        "output_tokens": 100,
        "thoughts_tokens": 0,
    }
    for field in spend.USAGE_FIELDS:
        partial = {k: v for k, v in complete.items() if k != field}
        with pytest.raises(ValueError, match=field):
            call_cost(partial, PRICES)


def test_call_cost_ignores_a_ledger_entrys_own_metadata():
    entry = {
        "ts": 1,
        "model": "gemini-x",
        "label": "answer",
        "input_tokens": 1_000_000,
        "cached_tokens": 0,
        "output_tokens": 0,
        "thoughts_tokens": 0,
        "record": None,
    }
    assert call_cost(entry, PRICES) == 0.25, (
        "a ledger entry re-prices as written, its metadata beside the counts"
    )


def test_a_usage_record_outside_the_vocabulary_refuses_naming_key_and_backend(
    spend_isolation,
):
    from locus.analysis.spend import record_paid_call, validate_usage

    smuggled = {
        "input_tokens": 1000,
        "cached_tokens": 0,
        "output_tokens": 100,
        "thoughts_tokens": 0,
        "prompt_tokens": 999,
    }
    with pytest.raises(ValueError) as refusal:
        validate_usage(smuggled, "OpenAICompatConversation")
    assert "prompt_tokens" in str(refusal.value)
    assert "OpenAICompatConversation" in str(refusal.value)

    billing = {"model": "gemini-x", "prices": PRICES}
    with pytest.raises(ValueError, match="GeminiConversation"):
        record_paid_call(
            billing,
            label=None,
            usage=smuggled,
            backend="GeminiConversation",
            record=None,
        )
    assert read_ledger(spend_isolation) == [], (
        "a refused usage record must never land on the ledger"
    )


# ── the ledger: append-only truth that survives a real deployment ───────────


def test_ledger_survives_concurrent_appends(spend_isolation):
    def worker(i):
        for j in range(50):
            spend._append_ledger(
                spend_isolation, {"ts": 1, "usd": 0.001, "who": i, "n": j}
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    entries = read_ledger(spend_isolation)
    assert len(entries) == 400, "every append landed as its own intact line"
    assert round(sum(e["usd"] for e in entries), 8) == 0.4


def test_a_torn_ledger_line_fails_every_read_loud(spend_isolation):
    spend._append_ledger(spend_isolation, {"ts": 1, "usd": 0.5})
    with open(spend_isolation / "data" / "spend.jsonl", "a") as ledger:
        ledger.write('{"ts": 17')
    with pytest.raises(ValueError, match="line 2"):
        read_ledger(spend_isolation)
    # A wall must never pass on spend truth it cannot read.
    (spend_isolation / "config" / "spend.toml").write_text("day_usd = 100\n")
    with pytest.raises(ValueError, match="not read as less spend"):
        check_walls()


# ── the walls: check-then-spend against the period's recorded spend ────────


def _seed(root, ts, usd):
    spend._append_ledger(
        root,
        {
            "ts": ts,
            "usd": usd,
            "model": "m",
            "label": None,
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "thoughts_tokens": 0,
            "record": None,
        },
    )


def test_wall_refuses_exactly_at_the_cap(spend_isolation):
    (spend_isolation / "config" / "spend.toml").write_text("day_usd = 1.0\n")
    now = _ms(2026, 7, 15, 12)
    _seed(spend_isolation, _ms(2026, 7, 15), 0.999)
    check_walls(now)
    _seed(spend_isolation, _ms(2026, 7, 15, 6), 0.001)
    with pytest.raises(SpendWall, match="day spend cap reached"):
        check_walls(now)


def test_the_wall_reaches_a_cap_set_to_the_reported_figure(spend_isolation):
    """The wall sums entries with the same rounding spend_report states, so a cap the operator copied from the report is reached by the wall's own arithmetic.
    Raw float summation sits one ulp under it: 0.1 + 0.7 is 0.7999999999999999, and an 0.8 cap would never trip."""
    (spend_isolation / "config" / "spend.toml").write_text("day_usd = 0.8\n")
    now = _ms(2026, 7, 15, 12)
    _seed(spend_isolation, _ms(2026, 7, 15), 0.1)
    _seed(spend_isolation, _ms(2026, 7, 15, 6), 0.7)
    with pytest.raises(SpendWall, match="day spend cap reached"):
        check_walls(now)


def test_spend_outside_the_period_does_not_count(spend_isolation):
    (spend_isolation / "config" / "spend.toml").write_text("day_usd = 1.0\n")
    now = _ms(2026, 7, 15, 12)
    _seed(spend_isolation, _ms(2026, 7, 15) - 1, 50.0)
    check_walls(now)  # yesterday's spend is yesterday's period
    _seed(spend_isolation, _ms(2026, 7, 16), 50.0)
    check_walls(now)  # the next period's spend is not this one's


def test_the_refusal_names_everything_the_operator_needs(spend_isolation):
    # The cap reads as declared.
    (spend_isolation / "config" / "spend.toml").write_text("week_usd = 0.003\n")
    now = _ms(2026, 7, 15, 12)
    _seed(spend_isolation, _ms(2026, 7, 13), 12.3456)
    with pytest.raises(SpendWall) as refusal:
        check_walls(now)
    message = str(refusal.value)
    assert "$12.3456" in message and "$0.003" in message
    assert "week_usd" in message and "spend.toml" in message
    assert "spend.jsonl" in message
    assert "2026-07-13T00:00:00Z → 2026-07-20T00:00:00Z" in message
    assert "edit" in message, "raising the cap is an edit to the declaration"


def test_a_zero_cap_is_the_kill_switch(spend_isolation):
    (spend_isolation / "config" / "spend.toml").write_text("month_usd = 0\n")
    with pytest.raises(SpendWall, match="month spend cap"):
        check_walls()


def test_an_edit_to_the_declaration_takes_effect_on_the_next_check(spend_isolation):
    check_walls()  # the shipped default stands
    (spend_isolation / "config" / "spend.toml").write_text("day_usd = 0\n")
    with pytest.raises(SpendWall):
        check_walls()
    (spend_isolation / "config" / "spend.toml").unlink()
    check_walls()


# ── the conversation layer: every billed call inside the wall, on the ledger ─


def _reply(text, usage, status=None):
    """A reply as the real SDK parses it, so the ledger reads the shapes the wire returns."""
    from google.genai import interactions as gi

    steps = (
        [{"type": "model_output", "content": [{"type": "text", "text": text}]}]
        if text
        else []
    )
    return gi.Interaction.model_validate(
        {
            "status": status or "completed",
            "id": "v1_spend",
            "usage": usage,
            "steps": steps,
        }
    )


def _usage(**fields):
    return {k: v for k, v in fields.items() if v is not None}


def _paid_conversation(tmp_path, replies, wire_log=None):
    from locus.analysis.model.gemini import GeminiConversation

    conversation = GeminiConversation(
        GEMINI,
        "sys",
        cache_path=tmp_path / "cache.db",
        record_dir=tmp_path / "payloads",
    )

    def create(**kwargs):
        if wire_log is not None:
            wire_log.append(kwargs)
        return replies.pop(0)

    conversation.client = SimpleNamespace(interactions=SimpleNamespace(create=create))
    return conversation


def test_the_wall_fires_at_the_layer_before_the_wire(tmp_path, spend_isolation):
    (spend_isolation / "config" / "spend.toml").write_text("day_usd = 0\n")
    wire = []
    conversation = _paid_conversation(tmp_path, [], wire_log=wire)
    conversation.add_user_text("q")
    with pytest.raises(SpendWall, match="day spend cap"):
        conversation.get_response(label="answer")
    assert not wire, "the refusal precedes the generation request"
    assert (tmp_path / "payloads" / "1_answer_input.txt").exists(), (
        "the refused call's input stands on the record"
    )
    meta = json.loads((tmp_path / "payloads" / "1_answer_meta.json").read_text())
    assert "SpendWall" in meta["error"], "and its meta carries the refusal"
    assert not (spend_isolation / "data" / "spend.jsonl").exists(), (
        "nothing was spent, nothing is ledgered"
    )


def test_every_billed_call_lands_on_the_ledger(tmp_path, spend_isolation):
    conversation = _paid_conversation(
        tmp_path,
        [
            _reply(
                "the answer",
                _usage(
                    total_input_tokens=1_000_000,
                    total_cached_tokens=200_000,
                    total_output_tokens=100_000,
                    total_thought_tokens=50_000,
                    total_tokens=1_150_000,
                ),
            )
        ],
    )
    conversation.add_user_text("q")
    conversation.get_response(label="answer")

    (entry,) = read_ledger(spend_isolation)
    assert entry["usd"] == 0.621, "usage × the card's declared prices"
    assert entry["model"] == GEMINI
    assert entry["label"] == "answer"
    assert entry["cached_tokens"] == 200_000
    assert entry["record"].endswith("1_answer_meta.json"), (
        "the entry names the persisted usage it was computed from"
    )
    assert (
        json.loads((tmp_path / "payloads" / "1_answer_meta.json").read_text())["usage"][
            "total_input_tokens"
        ]
        == 1_000_000
    )


def test_the_burned_retry_attempt_is_ledgered_with_its_reply(tmp_path, spend_isolation):
    conversation = _paid_conversation(
        tmp_path,
        [
            _reply(
                "",
                _usage(
                    total_input_tokens=100,
                    total_thought_tokens=32_768,
                    total_tokens=32_868,
                ),
                status="incomplete",
            ),
            _reply(
                "the answer",
                _usage(
                    total_input_tokens=100, total_output_tokens=40, total_tokens=140
                ),
            ),
        ],
    )
    conversation.thinking_level = "high"
    conversation.add_user_text("q")
    conversation.get_response(label="answer")

    (entry,) = read_ledger(spend_isolation)
    assert entry["usd"] == round(
        ((100 * 0.3 + 32_768 * 2.5) + (100 * 0.3 + 40 * 2.5)) / 1e6, 8
    ), "both paid attempts are one call's spend"
    assert entry["thoughts_tokens"] == 32_768


def test_a_breach_mid_conversation_refuses_the_next_paid_call(
    tmp_path, spend_isolation
):
    (spend_isolation / "config" / "spend.toml").write_text("week_usd = 0.5\n")
    expensive = _usage(
        total_input_tokens=100, total_output_tokens=400_000, total_tokens=400_100
    )
    conversation = _paid_conversation(
        tmp_path, [_reply("first", expensive), _reply("never", expensive)]
    )
    conversation.add_user_text("q")
    conversation.get_response(label="answer")
    (entry,) = read_ledger(spend_isolation)
    assert entry["usd"] > 0.5, "one call may carry spend past the cap"

    conversation.add_user_text("follow-up")
    with pytest.raises(SpendWall, match="week spend cap"):
        conversation.get_response(label="answer")
    assert len(read_ledger(spend_isolation)) == 1, (
        "the refused call spent nothing and is not ledgered"
    )


def test_a_priced_card_declares_its_calls_billed(tmp_path):
    from locus.analysis.model.cards import load_card
    from locus.analysis.model.openai_compat import OpenAICompatConversation

    card = {**load_card("qwen3.6-35b-a3b-6bit"), "pricing": PRICES}
    conversation = OpenAICompatConversation(
        "sys", card=card, record_dir=tmp_path / "record"
    )
    assert conversation._billing() == {
        "model": "qwen3.6-35b-a3b-6bit",
        "model_id": "mlx-community/Qwen3.6-35B-A3B-6bit",
        "prices": PRICES,
    }, "the pricing block alone is what makes a card's calls governed"


def test_free_backends_never_touch_the_spend_layer(tmp_path, spend_isolation):
    from _analysis_support import ScriptedConversation
    from locus.analysis.model.cards import load_card
    from locus.analysis.model.openai_compat import OpenAICompatConversation
    from locus.analysis.model.protocol import Response

    local = OpenAICompatConversation(
        "sys", card=load_card("qwen3.6-35b-a3b-6bit"), record_dir=tmp_path / "local"
    )
    assert local._billing() is None, "a card with no prices declares its calls free"

    (spend_isolation / "config" / "spend.toml").write_text("day_usd = 0\n")
    scripted = ScriptedConversation(
        [Response(text="free")], record_dir=tmp_path / "scripted"
    )
    scripted.add_user_text("q")
    scripted.get_response()
    assert not (spend_isolation / "data" / "spend.jsonl").exists(), (
        "a free call is neither walled nor ledgered"
    )


# ── the other paid paths: engine and oracle stand inside the same wall ──────


def test_the_engine_refuses_a_capped_paid_analysis_before_rendering(
    monkeypatch, distilled_db, tmp_path, spend_isolation
):
    from locus.analysis import budget, engine

    (spend_isolation / "config" / "spend.toml").write_text("week_usd = 0\n")

    def bomb(*args, **kwargs):
        raise AssertionError("the refusal must precede pricing and render")

    monkeypatch.setattr(engine, "RenderSession", bomb)
    monkeypatch.setattr(budget, "cost_model_for", bomb)

    conn = connect(distilled_db)
    from locus.analysis.window import resolve_window

    with pytest.raises(SpendWall, match="week spend cap"):
        engine.run_analysis(
            conn,
            resolve_window(conn, [best_slice(conn)]),
            question="what happened?",
            model=GEMINI,
            site_contexts={"site": "w"},
            out_dir=tmp_path / "analysis",
        )
    assert not (tmp_path / "analysis").exists()


def test_the_oracle_is_inside_the_wall(distilled_db, tmp_path, spend_isolation):
    from locus.analysis.ground.evaluate import evaluate_answer

    (spend_isolation / "config" / "spend.toml").write_text("day_usd = 0\n")
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    lo, _ = slice_bounds(conn, slice_id)

    def factory(model, system, *, record_dir, **kwargs):
        return _paid_conversation(tmp_path, [])

    from _analysis_support import slice_rows

    with pytest.raises(SpendWall, match="day spend cap"):
        evaluate_answer(
            conn,
            slice_rows(conn, [slice_id]),
            lo,
            [],
            "A claim. [00:00.100]",
            GEMINI,
            record_root=tmp_path / "grounding",
            conversation_factory=factory,
        )


# ── the observability surface ───────────────────────────────────────────────


def test_spend_report_teaches_the_wall_when_none_is_declared(spend_isolation):
    (spend_isolation / "config" / "spend.toml").unlink()
    report = spend_report(_ms(2026, 7, 15, 12))
    assert report["entries"] == 0 and report["lifetime_usd"] == 0
    assert report["declaration"] is None
    assert "spend.toml" in report["note"] and "week_usd" in report["note"]
    assert set(report["periods"]) == {"day", "week", "month"}
    assert report["periods"]["week"]["bounds"] == [
        "2026-07-13T00:00:00Z",
        "2026-07-20T00:00:00Z",
    ]


def test_spend_report_states_each_periods_spend_against_its_cap(spend_isolation):
    (spend_isolation / "config" / "spend.toml").write_text("week_usd = 20.0\n")
    now = _ms(2026, 7, 15, 12)
    _seed(spend_isolation, _ms(2026, 7, 13), 3.25)
    _seed(spend_isolation, _ms(2026, 7, 12), 100.0)
    report = spend_report(now)
    week = report["periods"]["week"]
    assert week["spent_usd"] == 3.25
    assert week["cap_usd"] == 20.0 and week["remaining_usd"] == 16.75
    assert report["periods"]["day"]["cap_usd"] is None
    assert "remaining_usd" not in report["periods"]["day"]
    assert report["lifetime_usd"] == 103.25
    assert "note" not in report


# ── price verification: the card is the cache, the registry the check ───────


def test_declared_prices_verify_against_the_registry(spend_isolation, monkeypatch):
    monkeypatch.setattr(spend, "_fetch_registry", lambda: REGISTRY_MATCH)
    verdict = verify_prices(GEMINI, PRICES)
    assert verdict["status"] == "verified"
    assert "litellm" in verdict["source"]


def test_price_drift_names_both_numbers(spend_isolation, monkeypatch):
    repriced = {
        f"gemini/{GEMINI}": {
            **REGISTRY_MATCH[f"gemini/{GEMINI}"],
            "input_cost_per_token": 3e-07,
        }
    }
    monkeypatch.setattr(spend, "_fetch_registry", lambda: repriced)
    verdict = verify_prices(GEMINI, PRICES)
    assert verdict["status"] == "drift"
    (mismatch,) = verdict["mismatches"]
    assert mismatch == {
        "field": "input_per_mtok",
        "declared_usd_per_mtok": 0.25,
        "registry_usd_per_mtok": pytest.approx(0.3),
    }


def test_a_reasoning_rate_apart_from_output_is_drift(spend_isolation, monkeypatch):
    split = {
        f"gemini/{GEMINI}": {
            **REGISTRY_MATCH[f"gemini/{GEMINI}"],
            "output_cost_per_reasoning_token": 3e-06,
        }
    }
    monkeypatch.setattr(spend, "_fetch_registry", lambda: split)
    verdict = verify_prices(GEMINI, PRICES)
    assert verdict["status"] == "drift"
    (mismatch,) = verdict["mismatches"]
    assert mismatch["field"] == "output_per_mtok (reasoning)", (
        "thinking bills at the card's one output rate; a registry that "
        "prices reasoning apart is drift against that rate"
    )


def test_an_absent_registry_key_is_unverified_never_wrong(spend_isolation, monkeypatch):
    monkeypatch.setattr(spend, "_fetch_registry", dict)
    verdict = verify_prices(GEMINI, PRICES)
    assert verdict["status"] == "unverified"
    assert f"gemini/{GEMINI}" in verdict["reason"]


def test_an_unreachable_registry_is_unverified(spend_isolation):
    verdict = verify_prices(GEMINI, PRICES)
    assert verdict["status"] == "unverified"
    assert "unreachable" in verdict["reason"]


def test_the_registry_fetch_is_memoized_for_the_day(spend_isolation, monkeypatch):
    fetches = []

    def fetch():
        fetches.append(1)
        return REGISTRY_MATCH

    monkeypatch.setattr(spend, "_fetch_registry", fetch)
    verify_prices(GEMINI, PRICES)
    verify_prices(GEMINI, PRICES)
    assert len(fetches) == 1, "one attempt per UTC day per machine"


def test_a_failed_fetch_is_also_memoized(spend_isolation, monkeypatch):
    attempts = []

    def fetch():
        attempts.append(1)
        raise OSError("offline")

    monkeypatch.setattr(spend, "_fetch_registry", fetch)
    verify_prices(GEMINI, PRICES)
    verify_prices(GEMINI, PRICES)
    assert len(attempts) == 1, (
        "an offline machine pays one attempt a day, not one per priced call"
    )


def test_drift_is_said_once_per_process_at_the_paid_call(
    tmp_path, spend_isolation, monkeypatch, capsys
):
    repriced = {
        f"gemini/{GEMINI}": {
            **REGISTRY_MATCH[f"gemini/{GEMINI}"],
            "input_cost_per_token": 3e-07,
        }
    }
    monkeypatch.setattr(spend, "_fetch_registry", lambda: repriced)
    usage = _usage(total_input_tokens=10, total_output_tokens=5, total_tokens=15)
    conversation = _paid_conversation(
        tmp_path, [_reply("a", usage), _reply("b", usage)]
    )
    conversation.add_user_text("q")
    conversation.get_response()
    conversation.add_user_text("again")
    conversation.get_response()
    stderr = capsys.readouterr().err
    assert stderr.count("price drift") == 1
    assert f"config/cards/{GEMINI}.toml" in stderr


# ── the price report's dollar side ──────────────────────────────────────────


def test_price_report_carries_dollars_for_a_priced_model(distilled_db, spend_isolation):
    from locus.analysis.budget import CostModel, price_payload

    conn = connect(distilled_db)
    cost_model = CostModel(
        count_text=lambda texts: 100,
        verify_text=None,
        screenshot_tokens=lambda conn, moments: 10,
    )
    common = {
        "prompts": lambda labels: ("s", "t"),
        "site_contexts": {"site": "w"},
        "cost_model": cost_model,
        "context_tokens": 1_000_000,
        "headroom": 0.5,
    }
    from locus.analysis.window import resolve_window

    price = price_payload(
        conn, resolve_window(conn, [best_slice(conn)]), model=GEMINI, **common
    )
    block = price["pricing"]
    total = price["turn1"]["total_tokens"]
    assert block["turn1_input_usd"] == round(total * 0.3 / 1e6, 6)
    assert block["declared_usd_per_mtok"]["output_per_mtok"] == 2.5
    assert block["verification"]["status"] == "unverified"

    free = price_payload(
        conn, resolve_window(conn, [best_slice(conn)]), model="injected", **common
    )
    assert "pricing" not in free, "a free model's report has no dollar side"


# ── live: the whole spend machinery against the real local server ───────────
# A priced copy of the local card makes real analyses billed — the pricing block
# is the declaration — at fictional rates: the entire governed path runs for
# zero actual dollars.

LIVE_PRICES = {
    "input_per_mtok": 1000.0,
    "cached_input_per_mtok": 100.0,
    "output_per_mtok": 5000.0,
}


def _price_the_local_card(tmp_path, monkeypatch):
    from _analysis_support import card_roster
    from locus.analysis.model import cards

    shipped = cards.cards_dir()
    roster = {path.stem: path.read_text() for path in sorted(shipped.glob("*.toml"))}
    roster["qwen3.6-35b-a3b-6bit"] += (
        "\n[pricing]\n"
        "input_per_mtok = 1000.0\n"
        "cached_input_per_mtok = 100.0\n"
        "output_per_mtok = 5000.0\n"
    )
    card_roster(monkeypatch, tmp_path, roster)


@pytest.mark.needs_mlx
def test_live_priced_analyses_are_ledgered_and_walled_at_the_boundary(
    distilled_db, tmp_path, spend_isolation, monkeypatch
):
    from pathlib import Path

    from locus.analysis import engine
    from locus.analysis.model.factory import make_conversation

    _price_the_local_card(tmp_path, monkeypatch)
    # The fictional rates make one real analysis cost tens of fake dollars, which the shipped default cap would refuse mid-test — so the test declares its own generous wall and then narrows it to the exact spend below.
    (spend_isolation / "config" / "spend.toml").write_text("week_usd = 100000.0\n")
    conn = connect(distilled_db)
    slice_id = best_slice(conn)

    # A real analysis through the real engine and server: every model call it made
    # is on the ledger, dollars computed from the usage its meta.json persisted.
    from locus.analysis.window import resolve_window

    result = engine.run_analysis(
        conn,
        resolve_window(conn, [slice_id]),
        model="qwen3.6-35b-a3b-6bit",
        question="In one sentence: what does this visitor do?",
        site_contexts={"site": "a demo fixture page"},
        out_dir=engine.home_analysis(tmp_path / "analysis"),
    )
    entries = read_ledger(spend_isolation)
    assert entries, "a billed analysis cannot leave the ledger empty"
    for entry in entries:
        assert entry["model"] == "qwen3.6-35b-a3b-6bit"
        assert entry["input_tokens"] > 0 and entry["output_tokens"] > 0
        meta = json.loads(Path(entry["record"]).read_text())
        assert entry["input_tokens"] == meta["usage"]["prompt_tokens"], (
            "the ledger is computed from the call's own persisted usage"
        )
        assert entry["usd"] == call_cost(entry, LIVE_PRICES)
    assert Path(result["response_path"]).exists()

    # Cap the week at exactly what is spent: the wall is reached, and a second
    # analysis refuses up front — before pricing, rendering, or any model call.
    spent = round(sum(entry["usd"] for entry in entries), 8)
    assert spent > 0
    (spend_isolation / "config" / "spend.toml").write_text(f"week_usd = {spent}\n")
    with pytest.raises(SpendWall, match="week spend cap"):
        engine.run_analysis(
            conn,
            resolve_window(conn, [slice_id]),
            model="qwen3.6-35b-a3b-6bit",
            question="again?",
            site_contexts={"site": "a demo fixture page"},
            out_dir=tmp_path / "analysis2",
        )
    assert not (tmp_path / "analysis2").exists()

    # The conversation layer refuses too — the wall at the wire, with the
    # refused call's input and meta on the record and nothing new ledgered.
    conversation = make_conversation(
        "qwen3.6-35b-a3b-6bit", "sys", record_dir=tmp_path / "probe"
    )
    conversation.add_user_text("Reply with the single word ok.")
    with pytest.raises(SpendWall, match="week spend cap"):
        conversation.get_response(label="probe")
    assert (tmp_path / "probe" / "1_probe_input.txt").exists()
    assert len(read_ledger(spend_isolation)) == len(entries)

    # Raise the cap and the same call passes, live, and lands on the ledger.
    (spend_isolation / "config" / "spend.toml").write_text(
        f"week_usd = {spent + 1000.0}\n"
    )
    conversation = make_conversation(
        "qwen3.6-35b-a3b-6bit", "sys", record_dir=tmp_path / "probe2"
    )
    conversation.add_user_text("Reply with the single word ok.")
    response = conversation.get_response(label="probe")
    assert response.text.strip()
    assert len(read_ledger(spend_isolation)) == len(entries) + 1
