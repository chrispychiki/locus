"""Spend governance over a real distilled db: the walls refuse on the engine's and the oracle's paid paths, and the price report carries dollars for a priced model."""

import pytest
from _analysis_support import (
    GEMINI,
    gemini_key,  # noqa: F401 — autouse fixture
    paid_conversation,
)
from _support import (
    best_slice,
    slice_bounds,
)
from locus.analysis.spend import (
    SpendWall,
)
from locus.evidence.db import connect

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
        return paid_conversation(tmp_path, [])

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


# ── the price report's dollar side ──────────────────────────────────────────


def test_price_report_carries_dollars_for_a_priced_model(distilled_db, spend_isolation):
    from locus.analysis.budget import CostModel, Count, price_payload

    conn = connect(distilled_db)
    cost_model = CostModel(lambda conn, sets, _: [Count(100, 10, 1) for _ in sets])
    common = {
        "prompts": lambda labels: ("s", "t", []),
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
