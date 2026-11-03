"""Spend governance against the running local server.

A pricing block on a copy of the local card makes its analyses billed, at fictional rates, so the spend walls and the ledger run against real calls that cost nothing.
"""

import json

import pytest
from _analysis_support import (
    gemini_key,  # noqa: F401 — autouse fixture
)
from _support import (
    best_slice,
)
from locus.analysis.spend import (
    SpendWall,
    call_cost,
    read_ledger,
)
from locus.evidence.db import connect

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
