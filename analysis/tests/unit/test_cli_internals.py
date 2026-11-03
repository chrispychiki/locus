import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _cli_support import (
    artifact,
    data_home,
    inline_echo,
    local_default_card,  # noqa: F401 — autouse fixture, registered by import
    offline_count,  # noqa: F401 — autouse fixture, registered by import
    output,
    shipped_definition,  # noqa: F401 — autouse fixture, registered by import
)
from locus.evidence.db import connect


def test_logs_minted_in_one_instant_never_collide(tmp_path, monkeypatch):
    """The clock names a log, so the instant several agents share is where a name must not repeat."""
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    instant = datetime(2026, 7, 25, 9, 8, 27, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "locus.evidence.speak.datetime",
        type("OneInstant", (), {"now": staticmethod(lambda tz: instant)}),
    )

    runs = [output("ae", {"run": i}) for i in range(25)]
    assert len(set(runs)) == 25
    assert [json.loads(p.read_text())["run"] for p in sorted(runs)] == list(
        range(25)
    ), "each run's own answer, in its own file"


def test_log_history_is_bounded_in_bytes(tmp_path, monkeypatch):
    """Logs are dropped oldest-first once a verb's history passes the disk budget."""
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    monkeypatch.setattr("locus.evidence.speak.OUTPUT_HISTORY_BYTES", 4096)

    heavy = [output("ls", {"rows": "x" * 3000}) for _ in range(4)]
    light = [output("ae", {"n": i}) for i in range(40)]
    assert [p for p in heavy if p.exists()] == heavy[-2:], (
        "a verb whose every run fills the budget keeps its newest, plus the "
        "history the budget holds — the newest is the answer, not history"
    )
    assert [p for p in light if p.exists()] == light, (
        "small runs all fit — and a heavy verb never evicts a light one"
    )

    alone = output("ls", {"rows": "z" * 9000})
    assert alone.exists(), (
        "a run whose log alone outgrows the budget still has it — the caller holds that address and has not read it yet"
    )


def test_trim_never_drops_a_log_a_run_is_still_speaking_into(tmp_path, monkeypatch):
    """A run holds its log locked while it speaks; a concurrent run's trim must leave that log alone however far past the budget it sits."""
    import fcntl

    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    monkeypatch.setattr("locus.evidence.speak.OUTPUT_HISTORY_BYTES", 4096)

    speaking = output("ls", {"rows": "x" * 9000})
    holding = speaking.open("a")
    fcntl.flock(holding.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        newest = output("ls", {"rows": "y" * 3000})
        assert newest.exists() and speaking.exists(), (
            "a log still being spoken into survives any budget"
        )
    finally:
        holding.close()

    newest = output("ls", {"rows": "z" * 3000})
    assert newest.exists() and not speaking.exists(), (
        "released, the same log is ordinary history and the budget takes it"
    )


def test_trim_survives_a_log_a_concurrent_run_already_dropped(tmp_path, monkeypatch):
    """Two readers trim the same verb's history at once, so a log listed here can be gone before this run weighs it."""
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    monkeypatch.setattr("locus.evidence.speak.OUTPUT_HISTORY_BYTES", 4096)

    older = [output("ls", {"rows": "x" * 3000}) for _ in range(3)]
    listing = Path.glob

    def raced(self, pattern):
        found = sorted(listing(self, pattern), reverse=True)
        for stale in found[1:]:
            stale.unlink(missing_ok=True)
        return iter(found)

    monkeypatch.setattr(Path, "glob", raced)
    newest = output("ls", {"rows": "y" * 3000})

    assert newest.exists(), "the run's own answer outlives the race"
    assert not [p for p in older if p.exists()]


def test_a_small_say_echoes_inline_and_a_large_one_stays_behind_the_address(
    tmp_path, capsys, monkeypatch
):
    """The echo is all or nothing on a byte budget: a small run's whole say rides stdout after the address, a large one leaves the address as the whole handoff, and the file is identical either way."""
    from locus.evidence.speak import ECHO_BYTES

    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    small = output("ae", {"n": 1})
    said = capsys.readouterr().out
    assert artifact(said) == small
    assert inline_echo(said) == small.read_text(), (
        "byte-exact — the echo is a transport copy, not a rendering"
    )

    from locus.analysis.cli import _speaking

    with _speaking("ls") as path:
        print(json.dumps({"scope": ""}))
        print(json.dumps({"headline": "0 slices"}))
    said = capsys.readouterr().out
    assert inline_echo(said) == path.read_text(), (
        "a multi-line say echoes whole, every line"
    )

    large = output("ae", {"rows": "x" * ECHO_BYTES})
    said = capsys.readouterr().out
    assert artifact(said) == large and inline_echo(said) == "", (
        "past the budget, nothing but the address"
    )
    assert json.loads(large.read_text())["rows"] == "x" * ECHO_BYTES, (
        "the file is whole regardless"
    )


def test_the_derivation_vintage_is_the_sources_own_identity(tmp_path, monkeypatch):
    # Derived from the code itself, never a hand-bumped constant: a forgotten bump would make the
    # detector affirm currency exactly when it lies.
    import locus.evidence.derive as derive_mod

    pkg = tmp_path / "distill"
    pkg.mkdir()
    (pkg / "distill.js").write_text("logic v1")
    (pkg / "flatten.js").write_text("flatten v1")
    py = tmp_path / "hydrate.py"
    py.write_text("canonical ts logic v1")
    sessions_py = tmp_path / "sessions.py"
    sessions_py.write_text("session logic v1")
    monkeypatch.setattr(
        derive_mod,
        "_stage_sources",
        lambda: {
            "timestamps": [py],
            "slices": [],
            "distill": sorted(pkg.glob("*.js")),
            "sessions": [sessions_py],
        },
    )
    definition = {
        "session": {"inactivity_minutes": 30},
        "engaged": {"min_seconds": 10, "min_pageviews": 2},
        "path": "wherever",
    }

    first = derive_mod.derivation_vintage(definition)
    assert first == derive_mod.derivation_vintage(definition), "stable across calls"

    (pkg / "flatten.js").write_text("flatten v2")
    second = derive_mod.derivation_vintage(definition)
    assert second != first, "a logic change changes the identity"

    # The path is wider than distillation: the Python side derives too — the canonical
    # timestamp, slice grouping, rescue eligibility — and a change there is the same staleness.
    py.write_text("canonical ts logic v2")
    third = derive_mod.derivation_vintage(definition)
    assert third != second

    # Zero-maintenance coverage: a module nobody registered anywhere still counts.
    (pkg / "new_stage.js").write_text("a whole new distillation stage")
    fourth = derive_mod.derivation_vintage(definition)
    assert fourth != third

    # The sessions stage derives from a declaration too: the operator's numbers move its
    # identity, and only its identity; where the file sits does not.
    adjusted = {**definition, "session": {"inactivity_minutes": 45}}
    assert derive_mod.derivation_vintage(adjusted) != fourth
    assert (
        derive_mod.stage_vintages(adjusted)["distill"]
        == (derive_mod.stage_vintages(definition)["distill"])
    )
    assert derive_mod.derivation_vintage({**definition, "path": "elsewhere"}) == fourth


def test_utc_ms_reads_iso_as_utc_and_refuses_epoch_seconds():
    from locus.analysis.ae import _utc_ms

    assert _utc_ms("2026-07-11") == 1783728000000
    assert _utc_ms("2026-07-11T22:17:55") == 1783808275000
    assert _utc_ms("2026-07-11T22:17:55+00:00") == 1783808275000
    assert _utc_ms("1783808275000") == 1783808275000
    with pytest.raises(SystemExit, match="epoch"):
        _utc_ms("1783808275")


def test_the_question_is_text_or_a_file_and_never_a_crash(tmp_path):
    from locus.analysis.cli import _inline_or_file

    held = tmp_path / "hypothesis.md"
    held.write_text("the visitor was hunting for pricing")
    assert _inline_or_file(str(held)) == "the visitor was hunting for pricing"

    prose = (
        "A focused question, long enough that no filesystem could hold it "
        "as a name, and carrying a newline besides.\n" + "x" * 300
    )
    assert _inline_or_file(prose) == prose, (
        "prose the filesystem cannot even be asked about is prose"
    )
    assert _inline_or_file(None) is None


def test_browse_open_passes_urls_and_pages_through_untouched(
    tmp_path, capsys, monkeypatch
):
    # A URL or an authored page is the browser's business — no db, no materialization;
    # the fragment a citation rides in survives the pass-through.
    from locus.analysis.cli import _resolve_open

    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    assert _resolve_open(["open", "https://x.test/a"]) == [
        "open",
        "https://x.test/a",
    ]
    page = tmp_path / "mine.html"
    page.write_text("<html></html>")
    target = f"{page}#t=123-456"
    # A leading window id names which window to open into, passed through untouched.
    assert _resolve_open(["open", "w2", target]) == [
        "open",
        "w2",
        target,
    ]
    assert capsys.readouterr().out == "", "pass-through writes nothing"


def test_browse_open_takes_a_slice_set_quoted_into_one_word(
    tmp_path, capsys, monkeypatch
):
    """No slice id holds a space, so a set quoted into one shell word is the same set as the unquoted form, and opens the same page."""
    from _support import click, env, full_snapshot, meta
    from locus.analysis.cli import _resolve_open
    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    conn = connect(str(data_home(tmp_path) / "events.db"))
    tab_a, tab_b = "00000000001000-taba", "00000000002000-tabb"
    hydrate(
        conn,
        "v1",
        [
            env(meta(1000), tab_a),
            env(full_snapshot(1001), tab_a),
            env(meta(2000), tab_b),
            env(full_snapshot(2001), tab_b),
            env(click(5000, 3), tab_a),
            env(click(5500, 3), tab_b),
        ],
    )
    materialize_slices(conn, "v1")
    conn.close()

    quoted = _resolve_open(["open", f"{tab_a} {tab_b}#t=5"])
    separate = _resolve_open(["open", tab_a, f"{tab_b}#t=5"])
    capsys.readouterr()
    assert quoted[1].endswith("#t=5")
    assert quoted[1] == separate[1]


def test_browse_open_passes_a_page_path_holding_a_space_through(tmp_path):
    """A page is named by whoever composed it, spaces included; an existing file is the target as-is, and only a token that would otherwise be read as a slice set is refused for holding one."""
    from locus.analysis.cli import _resolve_open

    page = tmp_path / "press kit.html"
    page.write_text("<!doctype html>")
    assert _resolve_open(["open", str(page)]) == ["open", str(page)]


def test_browse_open_mounts_concurrent_tabs_on_separate_players(
    tmp_path, capsys, monkeypatch
):
    # Two page contexts of one visitor recording at once: no single player can
    # honestly play their merged stream, so each lane gets its own mount on the
    # shared clock — concurrency shown as concurrency, never refused.
    from _support import click, env, full_snapshot, meta
    from locus.analysis.cli import _resolve_open
    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    conn = connect(str(data_home(tmp_path) / "events.db"))
    tab_a, tab_b = "00000000001000-taba", "00000000002000-tabb"
    hydrate(
        conn,
        "v1",
        [
            env(meta(1000), tab_a),
            env(full_snapshot(1001), tab_a),
            env(meta(2000), tab_b),
            env(full_snapshot(2001), tab_b),
            env(click(5000, 3), tab_a),
            env(click(5500, 3), tab_b),
        ],
    )
    materialize_slices(conn, "v1")
    conn.close()

    argv = _resolve_open(["open", tab_a, tab_b])
    capsys.readouterr()
    body = Path(argv[1]).read_text()
    assert body.count("<section") == 2, "one mount per concurrent lane"
    assert body.count('<script src="./') == 3, (
        "the component and one payload per lane — never one merged stream"
    )


def test_browse_open_groups_a_multi_visitor_set_one_mount_per_visitor(
    tmp_path, capsys, monkeypatch
):
    # A payload plays one visitor's timeline; a set spanning visitors composes as
    # one payload and one mount each — never interleaved into one stream.
    from _support import click, full_snapshot, meta, stamped
    from locus.analysis.cli import _resolve_open
    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    conn = connect(str(data_home(tmp_path) / "events.db"))
    for visitor, t0 in (("v1", 1000), ("v2", 60_000)):
        hydrate(
            conn,
            visitor,
            stamped([meta(t0), full_snapshot(t0 + 1), click(t0 + 500, 3)]),
        )
        materialize_slices(conn, visitor)
    slices = conn.execute(
        "SELECT recorder_slice FROM slices ORDER BY start_ts"
    ).fetchall()
    conn.close()

    argv = _resolve_open(
        ["open", slices[0]["recorder_slice"], slices[1]["recorder_slice"]]
    )
    capsys.readouterr()
    body = Path(argv[1]).read_text()
    assert body.count("<section") == 2, "one mount per visitor"
    assert body.count('<script src="./') == 3, (
        "the component and one payload script per visitor"
    )


def test_a_run_told_to_stop_leaves_a_last_word(capsys, tmp_path):
    """A SIGTERM mid-run lands in the run's log as its cause and ends the run through SystemExit, so the finally blocks close what the run holds; a log that just stops is a death with no cause. The line names the directory and whether an answer stands, never who sent the signal."""
    import os
    import signal

    from locus.analysis.cli import _last_word_on_termination

    previous = signal.getsignal(signal.SIGTERM)
    try:
        _last_word_on_termination(tmp_path / "run")
        with pytest.raises(SystemExit) as stopped:
            os.kill(os.getpid(), signal.SIGTERM)
            signal.pause() if hasattr(signal, "pause") else None
        assert stopped.value.code == 128 + signal.SIGTERM
    finally:
        signal.signal(signal.SIGTERM, previous)
    said = capsys.readouterr().err
    assert "stopped by SIGTERM" in said
    assert str(tmp_path / "run") in said and "no validated answer" in said
    assert "new run" in said


def test_a_local_price_refuses_a_card_without_a_base_url(monkeypatch, tmp_path):
    """A local card that declares no endpoint has no server to ask; the price says which card and what it lacks instead of dying on a missing key."""
    from locus.analysis.cli import _local_server

    cards = {"local": {"conversation": "openai-compatible", "model": "m"}}
    monkeypatch.setattr("locus.analysis.model.cards._cards", lambda: cards)
    with pytest.raises(SystemExit, match="card 'local' declares no base_url"):
        _local_server("local", 1000, tmp_path)


def test_a_local_price_states_the_servers_admission_and_its_measured_pace(
    monkeypatch, tmp_path
):
    """On a local card the price is also a wait: the report says the server's admission state and, from completed calls here, this machine's prefill rate as this payload's minutes before a first token, and what a call took. Without a server it says so; before any call it says there is nothing to measure by; a paid card carries neither."""
    from locus.analysis.cli import _local_server
    from locus.analysis.model import openai_compat

    cards = {
        "local": {
            "conversation": "openai-compatible",
            "model": "m",
            "base_url": "http://h/v1",
        },
        "paid": {"conversation": "gemini", "model": "g"},
    }
    monkeypatch.setattr("locus.analysis.model.cards._cards", lambda: cards)
    monkeypatch.setattr(
        openai_compat, "server_admission", lambda base_url: {"running": 1, "waiting": 3}
    )
    measured = {
        "calls": 4,
        "prompt_tokens_per_s": 400,
        "generated_tokens_per_s": 30,
        "generated_tokens_median": 6000,
        "served_s_median": 320,
        "since": "2026-09-04T10:00:00Z",
    }
    monkeypatch.setattr(
        openai_compat,
        "measured_serving",
        lambda analyses_dir, model_id, last=5: measured,
    )

    assert _local_server("paid", 48000, tmp_path) is None

    server = _local_server("local", 48000, tmp_path)
    assert server["admission"] == {"running": 1, "waiting": 3}
    assert server["prefill_estimate_s"] == 120
    assert server["line"] == (
        "; the server has 1 running, 3 waiting, served one at a time"
        "; the last 4 calls here prefilled at 400 tokens/s (this payload: about "
        "2.0 min before the first token) and generated 6000 tokens at 30 tokens/s — "
        "about 5 min served per call, plus the wait for admission"
    )

    monkeypatch.setattr(openai_compat, "server_admission", lambda base_url: None)
    monkeypatch.setattr(
        openai_compat, "measured_serving", lambda analyses_dir, model_id, last=5: None
    )
    server = _local_server("local", 48000, tmp_path)
    assert server["prefill_estimate_s"] is None
    assert server["line"] == (
        "; no server answers at http://h/v1 — a run would wait for one"
        "; no completed call here yet to measure this model's pace by"
    )
