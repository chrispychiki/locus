import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _analysis_support import (
    GEMINI,
    horizon_at,
    install_offline_tokenizer,
)
from _support import (
    DISTILL,
    FIXTURE,
    T0,
    VISITOR,
    FakeStore,
    chunk_at,
    chunk_blob,
    click,
    counted,
    full_snapshot,
    meta,
    read_recording,
)
from locus.analysis.ae import TELEMETRY_SCHEMA
from locus.analysis.cli import main
from locus.evidence.chunk import slice_date
from locus.evidence.clock import utc_stamp
from locus.evidence.db import connect


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    install_offline_tokenizer(monkeypatch)
    # The suite's deployment is the test's tmp_path, never this machine: no
    # banked key may leak in (a real key would let a Gemini card construct
    # and spend), and main()'s env self-serve walks cwd upward, which would
    # re-bank it from the real clone.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("locus.evidence.deployment.env_from_files", dict)


@pytest.fixture(autouse=True)
def local_default_card(monkeypatch, tmp_path):
    """The suite's deployment marks the local card `default = true`, so an analyze that names no model prices free and needs no key; a test of the Gemini declaration moves the mark itself."""
    from _analysis_support import remark_default
    from locus.analysis.model import cards

    local = cards.models_speaking(cards.OPENAI_COMPATIBLE)[0]
    remark_default(monkeypatch, tmp_path, local, dirname="suite-cards")


@pytest.fixture(autouse=True)
def shipped_definition(tmp_path):
    """Every derivation runs under a session definition and every load under a retention horizon, so the test deployment root carries the shipped definition and a horizon reaching the suite's recordings; a test of either declaration overwrites it."""
    _definitions_into(tmp_path)
    horizon_at(tmp_path)


def data_home(tmp_path: Path) -> Path:
    """The test deployment's accreted data home, as the CLI resolves it under the patched root."""
    home = tmp_path / "data"
    home.mkdir(exist_ok=True)
    return home


def artifact(out: str) -> Path:
    """The log a run announced, the way the agent reads it: `started <stamp>  <path>` is the whole of what reaches the caller."""
    opening = out.splitlines()[0].split()
    assert opening[0] == "started", out
    return Path(opening[2])


def read_output(capsys):
    """Follow the announced address to the run's result — its log's last line, so a tail delivers it whatever was said on the way."""
    return json.loads(artifact(capsys.readouterr().out).read_text().splitlines()[-1])


def read_lines(capsys) -> list[dict]:
    """A log its run filled as it went, one object per line — each line stands alone, so it reads the same half-written as finished."""
    return [
        json.loads(line)
        for line in artifact(capsys.readouterr().out).read_text().splitlines()
    ]


def bracket(out: str) -> list[str]:
    """What a run put in the caller's context, one word per line."""
    return [line.split()[0] for line in out.splitlines()]


def inline_echo(out: str) -> str:
    """What a run said inline past its opening line — a small say whole, or nothing."""
    _, _, echoed = out.partition("\n")
    return echoed


def test_cli_status_select_and_open_materialization(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    db = str(data_home(tmp_path) / "events.db")

    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    visitor_id, events = read_recording(FIXTURE)
    conn = connect(db)
    hydrate(conn, visitor_id, events)
    assert materialize_slices(conn, visitor_id)["replayable"] == 4
    conn.close()

    subprocess.run(["bun", str(DISTILL), db], check=True, capture_output=True)
    conn = connect(db)
    snippet = conn.execute("SELECT snippet FROM slices LIMIT 1").fetchone()["snippet"]
    an_hour_ago = int(datetime.now(timezone.utc).timestamp() * 1000) - 3_600_000
    conn.execute(
        "INSERT INTO loaded_chunks (key, etag, uploaded_ms) VALUES (?, 'e', ?)",
        (f"{snippet}/2026-01-01/{visitor_id}/x.gz", an_hour_ago),
    )
    conn.commit()
    conn.close()

    main(["status"])
    out = capsys.readouterr().out
    status = json.loads(artifact(out).read_text())
    assert status["sites"][0]["loaded_through"] == utc_stamp(an_hour_ago), (
        "each site says how far its loaded data reaches — the newest upload the db holds"
    )
    assert status["sites"][0]["behind"] == "1h 0m"
    assert "locus load" in status["sites_note"]
    assert inline_echo(out) == artifact(out).read_text(), (
        "a small say is repeated inline after the address, byte-exact"
    )
    assert status["headline"].startswith(f"{status['events']} events"), (
        "which leads with the shape of what follows"
    )
    assert status["distilled_events"] == 64
    assert status["slices"]["replayable"]["count"] == 4
    assert len(status["sites"]) == 1, "each site the corpus holds, by snippet id"
    assert status["sites"][0]["hosts"][0], (
        "and by the hosts its recordings opened on — the domains the operator knows it by"
    )
    assert "://" not in status["sites"][0]["hosts"][0]
    assert status["spend"]["entries"] == 0
    assert set(status["spend"]["periods"]) == {"day", "week", "month"}
    assert status["spend"]["periods"]["week"]["cap_usd"] is not None, (
        "a fresh deployment stands under the shipped default cap"
    )
    cov = status["coverage"]
    assert cov["analyzable_events"] + cov["dropped_events"] == status["events"]
    assert cov["dropped_events"] == (
        status["slices"].get("discarded", {}).get("events", 0)
        + cov["unmaterialized_events"]
    ), (
        "every dropped event is accounted to a discarded slice, whose row names the reason"
    )

    fill = (
        connect(db)
        .execute(
            "SELECT COUNT(*) total, SUM(url IS NOT NULL) attested, SUM(page_url IS NOT NULL) filled FROM events"
        )
        .fetchone()
    )
    assert fill["attested"] < fill["filled"] <= fill["total"], (
        "page_url must forward-fill beyond the attested url rows"
    )

    slices = [
        dict(row)
        for row in connect(db).execute(
            "SELECT recorder_slice, start_ts FROM slices WHERE status = 'replayable' ORDER BY start_ts LIMIT 2"
        )
    ]
    assert len(slices) == 2 and slices[0]["recorder_slice"]

    with pytest.raises(SystemExit, match="snippet"):
        main(["analyze", slices[0]["recorder_slice"], "what happened?"])
    assert capsys.readouterr().out.startswith("started "), (
        "a run that dies before producing anything was still addressed first"
    )

    connect(db).execute("UPDATE events SET snippet = 'demo'").connection.commit()
    (data_home(tmp_path) / "context").mkdir()
    (data_home(tmp_path) / "context" / "demo.md").write_text(
        "a demo site recorded for the test fixture"
    )
    main(
        [
            "analyze",
            slices[0]["recorder_slice"],
            slices[1]["recorder_slice"],
            "what happened?",
        ]
    )
    out = capsys.readouterr().out
    assert artifact(out).parent == data_home(tmp_path) / "outputs" and artifact(
        out
    ).name.endswith("_analyze.log"), (
        "the analysis leads by naming where it will say everything"
    )
    price = json.loads(artifact(out).read_text())
    turn1 = price["turn1"]
    assert (": fits — " in price["headline"]) is turn1["fits"]
    assert str(turn1["total_tokens"]) in price["headline"], (
        "the fit-and-cost headline leads the result it summarizes"
    )
    assert ("add --run to send it" in price["headline"]) is turn1["fits"], (
        "and a price that fits is where the spending step is learned"
    )
    assert price["n_activity_screenshots"] > 0 and turn1["text_tokens"] > 0
    assert 0 < turn1["context_pct"] <= 100
    assert len(price["per_slice"]) == 2
    assert turn1["verified"] is False, (
        "the local instrument is the exact one — nothing separate to verify"
    )

    main(
        [
            "analyze",
            slices[0]["recorder_slice"],
            "what happened?",
            "--screenshot-interval",
            "5000",
        ]
    )
    coarse = json.loads(artifact(capsys.readouterr().out).read_text())
    assert coarse["n_activity_screenshots"] <= price["n_activity_screenshots"]

    from locus.analysis.cli import _resolve_open

    argv = _resolve_open(["open", slices[0]["recorder_slice"]])
    narration = capsys.readouterr().out
    page = Path(argv[1])
    assert argv[0] == "open" and page.parent == data_home(tmp_path) / "pages"
    payload = data_home(tmp_path) / "pages" / f"{page.stem}.js"
    assert payload.stat().st_size > 1000
    component = data_home(tmp_path) / "pages" / "locus-replay.js"
    assert component.stat().st_size > 100_000, (
        "the component carries the whole player — nothing left to a CDN"
    )
    for path in (component, payload, page):
        assert str(path) in narration, "open prints every path it wrote"

    # A citation is made against a slice set, so the payload must carry the
    # whole set on one clock, or a multi-slice citation has nowhere to land.
    argv = _resolve_open(
        ["open", slices[0]["recorder_slice"], slices[1]["recorder_slice"]]
    )
    capsys.readouterr()
    window_payload = Path(argv[1]).with_suffix(".js")
    assert "_plus1_" in window_payload.name
    assert window_payload.stat().st_size > payload.stat().st_size


def test_a_load_says_everything_to_its_output_and_hands_over_an_address(
    tmp_path, capsys, monkeypatch
):
    """A load's heartbeats, per-chunk notices and outcome all land in its log, readable while they are still being said."""
    rid = f"0{T0}-aaaa"
    stream = [
        counted(e, s) for s, e in enumerate([meta(T0), full_snapshot(T0 + 1)], start=1)
    ]
    store = FakeStore(
        {
            f"snip01/{slice_date(rid)}/{VISITOR}/{rid}/{T0}000001.json.gz": chunk_blob(
                rid, stream
            ),
            # A loss reported from inside a GET worker, the deepest thing that speaks.
            f"snip01/{slice_date(rid)}/{VISITOR}/{rid}/{T0}000002.json.gz": b"not gzip at all",
        }
    )
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    midway = {}
    real_get = store.get

    def get(key):
        midway[key] = next((data_home(tmp_path) / "outputs").glob("*_load.log"))
        return real_get(key)

    store.get = get

    # Deriving is the stretch that yields nothing until it ends, so its pulse is
    # brought within the test's patience and its work pushed past one beat.
    import functools
    import time

    from locus.evidence import derive, slices

    monkeypatch.setattr(
        derive, "beating", functools.partial(derive.beating, every_s=0.02)
    )
    slice_visitor = slices.materialize_slices
    monkeypatch.setattr(
        slices,
        "materialize_slices",
        lambda *a, **k: (time.sleep(0.1), slice_visitor(*a, **k))[1],
    )

    main(["load"])
    said = capsys.readouterr()
    output = artifact(said.out)
    assert output.suffix == ".log" and output.parent == data_home(tmp_path) / "outputs"
    assert next(iter(midway.values())) == output, (
        "the output is open at the address the run named before the first fetch"
    )

    assert said.err == ""
    assert inline_echo(said.out) == output.read_text(), (
        "a small say is repeated inline whole — narration, losses, outcome"
    )

    written = output.read_text().splitlines()
    assert any(line.startswith("materializing slices:") for line in written), (
        "the derivation says it is alive while it produces nothing"
    )
    assert any("skipping corrupt chunk" in line for line in written), (
        "and a GET worker's loss lands there too, not on the way past it"
    )
    outcome = json.loads(written[-1])
    assert outcome["headline"] == (
        "loaded 2 new events across 1 visitors, 1 corrupt skipped; their slices now: 1 replayable, 0 discarded, 0 rescued"
    ), "the output ends on the outcome, so one that stops short is one that died"
    assert (outcome["inserted"], outcome["chunks"]["corrupt"]) == (2, 1), (
        "the outcome's fields carry the counts its headline reads out"
    )


def test_neither_documentation_nor_a_usage_error_opens_a_log(
    tmp_path, capsys, monkeypatch
):
    """Neither is a run: no verb executed, so there is nothing to address."""
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    for documentation in (["--help"], ["load", "--help"], ["browse", "help"]):
        with pytest.raises(SystemExit) as exit_:
            main(documentation)
        assert exit_.value.code == 0
        assert not [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("started ")
        ], documentation

    with pytest.raises(SystemExit) as exit_:
        main(["no-such-verb"])
    assert exit_.value.code
    said = capsys.readouterr()
    assert said.out == "" and "invalid choice" in said.err
    assert not (data_home(tmp_path) / "outputs").exists(), (
        "and no verb ran, so no log was opened for one"
    )


def test_a_refusal_reaches_the_caller_and_the_log(tmp_path, capsys, monkeypatch):
    """A refusal reaches both readers — the caller inline, and the log it was announced at."""
    _loaded_fixture_db(tmp_path, monkeypatch)

    with pytest.raises(SystemExit) as refusal:
        main(["analyze", "999-zzzz", "what happened?"])
    said = capsys.readouterr()
    assert "no slice 999-zzzz" in str(refusal.value)
    assert bracket(said.out) == ["started"], (
        "a dead run echoes nothing — the address and the raise are the handoff"
    )
    assert "no slice 999-zzzz" in artifact(said.out).read_text(), (
        "and the log says why it stopped rather than merely stopping"
    )


def test_cli_ls_inventory_and_precise_load(tmp_path, capsys, monkeypatch):
    rid_a = f"0{T0}-aaaa"
    rid_b = f"0{T0 + 90_000}-bbbb"
    stream = [
        counted(event, seq)
        for seq, event in enumerate(
            [
                meta(T0),
                full_snapshot(T0 + 1),
                meta(T0 + 90_000),
                full_snapshot(T0 + 90_001),
            ],
            start=1,
        )
    ]
    store = FakeStore(
        {
            f"snip01/{slice_date(rid_a)}/{VISITOR}/{rid_a}/{T0}000001.json.gz": chunk_blob(
                rid_a, stream[0:2]
            ),
            f"snip01/{slice_date(rid_b)}/{VISITOR}/{rid_b}/{T0}000003.json.gz": chunk_blob(
                rid_b, stream[2:4]
            ),
        }
    )
    resolved = []

    def fake_store_for(url):
        resolved.append(url)
        return store

    monkeypatch.setattr("locus.evidence.store.store_for", fake_store_for)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    prefix_a = f"snip01/{slice_date(rid_a)}/{VISITOR}/{rid_a}"
    main(["load", prefix_a])
    assert re.search(r"\b2\b[^\n]*event", artifact(capsys.readouterr().out).read_text())

    distilled = (
        connect(str(data_home(tmp_path) / "events.db"))
        .execute("SELECT COUNT(*) n FROM events WHERE type_str IS NOT NULL")
        .fetchone()["n"]
    )
    assert distilled == 2, "load must leave the db distilled"

    # The rows land in the output — the arrivals plane, joinable to a birth on
    # its key — as the listing finds them, under the scope and above the totals.
    main(["ls"])
    scope, *rest = read_lines(capsys)
    *rows, summary = rest
    assert scope["scope"] == "" and scope["loaded_known"] is True
    assert re.search(r"\b2\b", summary["headline"]) and re.search(
        r"\b1\b", summary["headline"]
    ), "the totals close the rows they count"
    assert {
        (s["snippet"], s["visitor_id"], s["slice_id"], s["loaded"]) for s in rows
    } == {
        ("snip01", VISITOR, rid_b, False),
        ("snip01", VISITOR, rid_a, True),
    }

    main(["load"])
    assert re.search(r"\b2\b[^\n]*event", artifact(capsys.readouterr().out).read_text())

    main(["ls"])
    assert re.search(r"\b0\b", read_lines(capsys)[-1]["headline"])

    assert resolved == ["s3://b"] * 4, (
        "every command resolves the store from the deployment — no address to pass or record"
    )

    with pytest.raises(
        SystemExit,
        match=r"no site context at data/context/<snippet>\.md for "
        r"\['snip01'\]",
    ):
        main(["analyze", rid_a, "what happened?"])
    capsys.readouterr()

    (data_home(tmp_path) / "context").mkdir()
    (data_home(tmp_path) / "context" / "snip01.md").write_text(
        "a test deployment; a good session reaches the second page"
    )
    main(["analyze", rid_a, "what happened?"])
    price = read_output(capsys)
    assert price["turn1"]["total_tokens"] > 0, (
        "context resolves by the slice set's snippet — nothing threaded"
    )


def test_each_read_run_owns_its_output(tmp_path, capsys, monkeypatch):
    """An output belongs to the run that made it, so two reads of a deployment never answer each other's question."""
    rid = f"0{T0}-aaaa"
    stream = [
        counted(e, s) for s, e in enumerate([meta(T0), full_snapshot(T0 + 1)], start=1)
    ]
    store = FakeStore(
        {
            f"snip01/{slice_date(rid)}/{VISITOR}/{rid}/{T0}000001.json.gz": chunk_blob(
                rid, stream
            )
        }
    )
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    main(["ls"])
    first = artifact(capsys.readouterr().out)
    main(["ls"])
    second = artifact(capsys.readouterr().out)
    assert first != second, (
        "each run owns its output — two readers at once never share a path"
    )
    assert {first.name, second.name} == {
        p.name for p in (data_home(tmp_path) / "outputs").iterdir()
    }
    assert json.loads(first.read_text().splitlines()[-1])["totals"]["slices"], (
        "the earlier run's answer survives the later one, whole"
    )


def output(name: str, payload: dict) -> Path:
    """One run of a verb, said into its own log."""
    from locus.analysis.cli import _speaking

    with _speaking(name) as path:
        print(json.dumps(payload))
    return path


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


def test_a_caller_that_took_the_address_and_left_declines_the_echo(tmp_path):
    """`locus <verb> | head -1` is the natural way to take just the address, and the reader is gone before a small say echoes. The write lands nowhere and the run still exits clean — through a real pipe and a real interpreter exit, because the failure modes are the raise in echo and the interpreter's own shutdown flush."""
    import os
    import subprocess
    import sys as _sys
    import textwrap

    script = textwrap.dedent("""
        import sys, time
        from pathlib import Path
        from locus.evidence.speak import speaking
        with speaking(Path(sys.argv[1]), "probe"):
            time.sleep(0.5)
            print("small result line")
    """)
    # bash -c consumes $0 from the first arg after the command string.
    run = subprocess.run(
        [
            "/bin/bash",
            "-c",
            f'"{_sys.executable}" -c "$SCRIPT" "$1" | head -1; exit "${{PIPESTATUS[0]}}"',
            "echo-probe",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "SCRIPT": script},
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.startswith("started "), "the address still reached head"
    assert "BrokenPipeError" not in run.stderr, (
        "neither the echo nor the exit flush complains about the gone reader"
    )


def test_reload_skips_unchanged_chunks_and_refetches_a_changed_one(
    tmp_path, capsys, monkeypatch
):
    rid = f"0{T0}-aaaa"
    stream = [
        counted(e, s) for s, e in enumerate([meta(T0), full_snapshot(T0 + 1)], start=1)
    ]
    key = f"snip01/{slice_date(rid)}/{VISITOR}/{rid}/{T0}000001.json.gz"
    store = FakeStore({key: chunk_blob(rid, stream)})
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    gets = []
    real_get = store.get
    store.get = lambda k: gets.append(k) or real_get(k)

    main(["load"])
    assert re.search(r"\b2\b[^\n]*event", artifact(capsys.readouterr().out).read_text())
    assert gets == [key], "the first load fetches the object"

    gets.clear()
    main(["load"])
    assert (
        "all 1 chunks already loaded" in artifact(capsys.readouterr().out).read_text()
    )
    assert gets == [], "an unchanged object is not fetched the second time"

    # Same slice and events, a new envelope (a late error) — a different body, so a different
    # ETag, so the object is fetched again and the late error lands.
    store.blobs[key] = chunk_blob(
        rid, stream, errors=("dropped a malformed buffered record",)
    )
    gets.clear()
    main(["load"])
    assert gets == [key], "a changed ETag forces the re-fetch"
    errors = (
        connect(str(data_home(tmp_path) / "events.db"))
        .execute("SELECT error FROM chunk_errors")
        .fetchall()
    )
    assert [r["error"] for r in errors] == ["dropped a malformed buffered record"]


def test_load_keeps_each_chunks_store_arrival_time(tmp_path, capsys, monkeypatch):
    # The store's copy of an arrival time dies with the object at the retention horizon; the load
    # is the one chance to keep it.
    rid = f"0{T0}-aaaa"
    stream = [
        counted(e, s) for s, e in enumerate([meta(T0), full_snapshot(T0 + 1)], start=1)
    ]
    key = f"snip01/{slice_date(rid)}/{VISITOR}/{rid}/{T0}000001.json.gz"
    uploaded = T0 + 86_400_000  # landed a day after the slice opened
    store = FakeStore({key: chunk_blob(rid, stream)}, uploaded={key: uploaded})
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    db = data_home(tmp_path) / "events.db"
    main(["load"])
    rows = connect(db).execute("SELECT key, uploaded_ms FROM loaded_chunks").fetchall()
    assert [(r["key"], r["uploaded_ms"]) for r in rows] == [(key, uploaded)]


def test_a_full_rederivation_is_said_on_the_outcome_line(tmp_path, capsys, monkeypatch):
    rid = f"0{T0}-aaaa"
    stream = [
        counted(e, s) for s, e in enumerate([meta(T0), full_snapshot(T0 + 1)], start=1)
    ]
    key = f"snip01/{slice_date(rid)}/{VISITOR}/{rid}/{T0}000001.json.gz"
    store = FakeStore({key: chunk_blob(rid, stream)})
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    main(["load"])
    assert "re-derived every visitor" not in capsys.readouterr().out

    db = data_home(tmp_path) / "events.db"
    conn = connect(db)
    conn.execute(
        "UPDATE meta SET value = 'stale' WHERE key IN ('derivation_vintage', 'vintage_slices')"
    )
    conn.commit()
    conn.close()
    main(["load"])
    outcome = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert outcome["headline"].startswith("loaded 0 new events")
    assert outcome["headline"].endswith(
        "re-derived every visitor from the slices stage on because the derivation changed"
    )
    assert outcome["rederived_from"] == "slices"


def test_a_load_that_dies_before_distillation_is_finished_by_the_next_load(
    tmp_path, capsys, monkeypatch
):
    import locus.evidence.derive as derive_mod

    rid = f"0{T0}-aaaa"
    stream = [
        counted(e, s) for s, e in enumerate([meta(T0), full_snapshot(T0 + 1)], start=1)
    ]
    key = f"snip01/{slice_date(rid)}/{VISITOR}/{rid}/{T0}000001.json.gz"
    store = FakeStore({key: chunk_blob(rid, stream)})
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    gets = []
    real_get = store.get
    store.get = lambda k: gets.append(k) or real_get(k)

    real_distill = derive_mod.run_distill
    fail = {"on": True}

    def flaky_distill(*a, **k):
        if fail["on"]:
            raise SystemExit(1)
        return real_distill(*a, **k)

    monkeypatch.setattr("locus.evidence.derive.run_distill", flaky_distill)

    with pytest.raises(SystemExit):
        main(["load"])
    capsys.readouterr()
    db = str(data_home(tmp_path) / "events.db")
    assert (
        connect(db)
        .execute("SELECT COUNT(*) FROM events WHERE type_str IS NULL")
        .fetchone()[0]
        == 2
    ), "the crash left the events hydrated but not distilled"
    assert (
        connect(db).execute("SELECT COUNT(*) FROM loaded_chunks").fetchone()[0] == 1
    ), "the clean chunk was recorded before the crash"

    fail["on"] = False
    gets.clear()
    main(["load"])
    said = artifact(capsys.readouterr().out).read_text()
    assert gets == [], "recovery re-uses the manifest — the chunk is not re-fetched"
    assert (
        connect(db)
        .execute("SELECT COUNT(*) FROM events WHERE type_str IS NOT NULL")
        .fetchone()[0]
        == 2
    ), "the stranded events are distilled by the next load"
    assert re.search(r"\b1\b.*visitor", said)


def test_the_derivation_vintage_is_the_sources_own_identity(tmp_path, monkeypatch):
    # Derived from the code itself, never a hand-bumped constant: a forgotten bump would make the
    # detector affirm currency exactly when it lies.
    import locus.evidence.derive as derive_mod

    pkg = tmp_path / "distill"
    pkg.mkdir()
    (pkg / "distill.js").write_text("logic v1")
    (pkg / "flatten.js").write_text("flatten v1")
    (pkg / "flatten.test.js").write_text("tests v1")
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
            "distill": [
                path
                for path in sorted(pkg.glob("*.js"))
                if not path.name.endswith(".test.js")
            ],
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

    (pkg / "flatten.test.js").write_text("tests v2")
    assert derive_mod.derivation_vintage(definition) == first, (
        "tests exercise the logic without being it"
    )

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


def test_a_load_after_an_distillation_code_change_re_derives_everything(
    tmp_path, capsys, monkeypatch
):
    import locus.evidence.derive as derive_mod

    rid_a = f"0{T0}-aaaa"
    rid_b = f"0{T0 + 90_000}-bbbb"
    stream = [
        counted(event, seq)
        for seq, event in enumerate(
            [
                meta(T0),
                full_snapshot(T0 + 1),
                meta(T0 + 90_000),
                full_snapshot(T0 + 90_001),
            ],
            start=1,
        )
    ]
    key_a = f"snip01/{slice_date(rid_a)}/{VISITOR}/{rid_a}/{T0}000001.json.gz"
    key_b = f"snip01/{slice_date(rid_b)}/{VISITOR}/{rid_b}/{T0}000003.json.gz"
    store = FakeStore({key_a: chunk_blob(rid_a, stream[0:2])})
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    passes = []
    real_distill = derive_mod.run_distill

    def recording_distill(db, full=False):
        passes.append(full)
        return real_distill(db, full)

    monkeypatch.setattr("locus.evidence.derive.run_distill", recording_distill)

    main(["load"])
    capsys.readouterr()
    assert passes == [False]

    # Same code, new chunk: incremental, no re-derivation of the untouched visitor's rows.
    store.blobs[key_b] = chunk_blob(rid_b, stream[2:4])
    main(["load"])
    said = artifact(capsys.readouterr().out).read_text()
    assert passes == [False, False]
    assert "derivation" not in said

    # The deriving code changes out from under the accreted db: the next load — even one that
    # fetches nothing — says so plainly and re-derives every visitor, so nothing ever reads
    # mixed-vintage columns.
    monkeypatch.setattr(
        "locus.evidence.derive.derivation_vintage", lambda definition: "a new vintage"
    )
    main(["load"])
    said = artifact(capsys.readouterr().out).read_text()
    assert "derivation" in said
    assert passes == [False, False, True]

    # The repair re-stamped: the next load is quiet and incremental again.
    main(["load"])
    said = artifact(capsys.readouterr().out).read_text()
    assert "derivation" not in said
    assert passes == [False, False, True]


def test_status_names_stale_derivations_and_doctor_heals_them(
    tmp_path, capsys, monkeypatch
):
    from test_doctor import healthy_surroundings

    healthy_surroundings(tmp_path, monkeypatch)
    db = str(data_home(tmp_path) / "events.db")

    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    visitor_id, events = read_recording(FIXTURE)
    conn = connect(db)
    hydrate(conn, visitor_id, events)
    materialize_slices(conn, visitor_id)
    conn.close()

    # Distilled rows with no recorded vintage are unknown-vintage rows: stale, not current.
    subprocess.run(["bun", str(DISTILL), db], check=True, capture_output=True)
    main(["status"])
    payload = read_output(capsys)
    assert payload["derivations"]["stale"] is True
    assert "locus doctor" in payload["derivations"]["note"]

    # Doctor re-derives and stamps; status is clean after it.
    main(["doctor"])
    capsys.readouterr()
    main(["status"])
    payload = read_output(capsys)
    assert payload["derivations"]["stale"] is False
    assert "note" not in payload["derivations"]


def test_an_undistilled_db_is_not_stale_merely_underived(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    db = str(data_home(tmp_path) / "events.db")

    from locus.evidence.hydrate import hydrate

    visitor_id, events = read_recording(FIXTURE)
    conn = connect(db)
    hydrate(conn, visitor_id, events)
    conn.close()

    main(["status"])
    payload = read_output(capsys)
    assert payload["derivations"]["stale"] is False


def test_a_lossy_object_is_re_fetched_every_load_while_a_clean_one_is_skipped(
    tmp_path, capsys, monkeypatch
):
    clean_rid = f"0{T0}-aaaa"
    clean_key = (
        f"snip01/{slice_date(clean_rid)}/{VISITOR}/{clean_rid}/{T0}000001.json.gz"
    )
    clean = [
        counted(e, s) for s, e in enumerate([meta(T0), full_snapshot(T0 + 1)], start=1)
    ]
    # A chunk whose only event carries no counter: the event is dropped at decode, so the object
    # is lossy and must never be recorded — its loss has to re-report on every load.
    bad_rid = f"0{T0 + 90_000}-bbbb"
    bad_key = f"snip01/{slice_date(bad_rid)}/{VISITOR}/{bad_rid}/{T0}000003.json.gz"
    store = FakeStore(
        {
            clean_key: chunk_blob(clean_rid, clean),
            bad_key: chunk_blob(bad_rid, [meta(T0 + 90_000)]),
        }
    )
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    gets = []
    real_get = store.get
    store.get = lambda k: gets.append(k) or real_get(k)

    main(["load"])
    capsys.readouterr()
    assert sorted(gets) == [clean_key, bad_key]
    assert [
        r["key"]
        for r in connect(str(data_home(tmp_path) / "events.db")).execute(
            "SELECT key FROM loaded_chunks"
        )
    ] == [clean_key], "only the clean chunk is recorded"

    gets.clear()
    main(["load"])
    assert gets == [bad_key], "the clean chunk is skipped, the lossy one re-fetched"


def test_a_prefix_no_key_can_begin_is_refused_before_the_store(tmp_path, monkeypatch):
    """A glob goes to the store as literal bytes and matches nothing — the same nothing an empty store returns — so both store verbs refuse it before any LIST, naming what a prefix is."""
    store = FakeStore({})
    listed = []
    real_objects = store.objects
    store.objects = lambda prefix="": listed.append(prefix) or real_objects(prefix)
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    for verb in ("ls", "load"):
        with pytest.raises(SystemExit, match="no wildcards") as refused:
            main([verb, "*/2026-08-22/"])
        assert "'*'" in str(refused.value)
    assert listed == [], "the store was never asked"


def test_a_load_counts_the_chunks_it_could_not_take_whole(
    tmp_path, capsys, monkeypatch
):
    """A corrupt or lossy chunk is named as it is skipped, and the summary carries the count whatever else the load did — it must never call a prefix empty because its only chunks were corrupt, nor drop the count because other chunks landed."""
    corrupt_rid = f"0{T0}-aaaa"
    corrupt_key = (
        f"snip01/{slice_date(corrupt_rid)}/{VISITOR}/{corrupt_rid}/{T0}000001.json.gz"
    )
    good_rid = f"0{T0 + 90_000}-bbbb"
    good_key = f"snip01/{slice_date(good_rid)}/{VISITOR}/{good_rid}/{T0}000003.json.gz"
    good = [
        counted(e, s)
        for s, e in enumerate([meta(T0 + 90_000), full_snapshot(T0 + 90_001)], start=1)
    ]
    blobs = {corrupt_key: b"not gzip at all"}
    store = FakeStore(blobs)
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    main(["load", "snip01/"])
    said = artifact(capsys.readouterr().out).read_text()
    assert corrupt_key in said and "corrupt" in said
    assert re.search(r"\b1\b[^\n]*corrupt", said), "the skip is counted"

    blobs[good_key] = chunk_blob(good_rid, good)
    main(["load", "snip01/"])
    said = artifact(capsys.readouterr().out).read_text()
    assert re.search(r"\b2\b[^\n]*event", said) and re.search(
        r"\b1\b[^\n]*corrupt", said
    )


def test_coverage_names_events_that_belong_to_no_slice(tmp_path, capsys, monkeypatch):
    # An event hydrated but its slice never materialized is a real state — a load that died between
    # the two. The discard reasons cannot speak for it, because it belongs to no slices row: unnamed, it surfaces
    # only as a coverage block that does not add up, which is exactly the silent loss `status` denies.
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    db = str(data_home(tmp_path) / "events.db")

    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    visitor_id, events = read_recording(FIXTURE)
    conn = connect(db)
    hydrate(conn, visitor_id, events)
    materialize_slices(conn, visitor_id)
    conn.execute(
        "UPDATE events SET slice_id = NULL WHERE id IN (SELECT id FROM events LIMIT 3)"
    )
    conn.commit()

    main(["status"])
    payload = read_output(capsys)
    cov = payload["coverage"]
    assert cov["unmaterialized_events"] == 3
    assert cov["analyzable_events"] + cov["dropped_events"] == payload["events"]
    assert cov["dropped_events"] == (
        payload["slices"].get("discarded", {}).get("events", 0)
        + cov["unmaterialized_events"]
    )


def test_load_without_a_deployment_fails_loud(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="wrangler.toml"):
        main(["load"])


def test_load_lands_the_recorder_error_log(tmp_path, capsys, monkeypatch):
    rid = f"0{T0}-aaaa"
    events = [counted(meta(T0), 1), counted(full_snapshot(T0 + 1), 2)]
    store = FakeStore(
        {
            f"snip01/{slice_date(rid)}/{VISITOR}/{rid}/{T0}000001.json.gz": chunk_blob(
                rid,
                events,
                errors=[
                    "dropped 2 malformed buffered records",
                    "resolved unreadable inflight batch 0x1",
                ],
            ),
        }
    )
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    main(["load"])
    rows = (
        connect(str(data_home(tmp_path) / "events.db"))
        .execute(
            "SELECT snippet, visitor_id, recorder_slice, chunk_key, error FROM chunk_errors ORDER BY error"
        )
        .fetchall()
    )
    assert [tuple(r) for r in rows] == [
        ("snip01", VISITOR, rid, f"{T0}000001", "dropped 2 malformed buffered records"),
        (
            "snip01",
            VISITOR,
            rid,
            f"{T0}000001",
            "resolved unreadable inflight batch 0x1",
        ),
    ]

    main(["load"])
    n = (
        connect(str(data_home(tmp_path) / "events.db"))
        .execute("SELECT COUNT(*) c FROM chunk_errors")
        .fetchone()["c"]
    )
    assert n == 2, "a re-load dedups, so nothing lands twice"


def test_ls_returns_every_slice_at_any_scale(tmp_path, capsys, monkeypatch):
    store = FakeStore(dict(chunk_at(f"v{i:04d}-visitor", T0 + i) for i in range(55)))
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    # Every slice, however large the store — the rows are what a completeness read
    # joins against, and there is no size at which they stop coming. They land
    # in the log; the caller's context stays one line at any scale.
    main(["ls"])
    _scope, *rest = read_lines(capsys)
    *rows, summary = rest
    assert len(rows) == 55 and summary["totals"]["slices"] == 55
    assert "v0003-visitor" in {s["visitor_id"] for s in rows}


def test_screenshot_pricing_follows_the_card_ceiling(tmp_path, capsys, monkeypatch):
    # What a screenshot costs on the wire is the model card's per-image token ceiling,
    # realized by the backend — not a per-run flag, so no flag exists to state it.
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    db = str(data_home(tmp_path) / "events.db")

    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    visitor_id, events = read_recording(FIXTURE)
    conn = connect(db)
    hydrate(conn, visitor_id, events)
    materialize_slices(conn, visitor_id)
    conn.close()
    subprocess.run(["bun", str(DISTILL), db], check=True, capture_output=True)

    connect(db).execute("UPDATE events SET snippet = 'demo'").connection.commit()
    (data_home(tmp_path) / "context").mkdir()
    (data_home(tmp_path) / "context" / "demo.md").write_text(
        "a demo site recorded for the test fixture"
    )
    slice_id = (
        connect(db)
        .execute(
            "SELECT recorder_slice FROM slices WHERE status = 'replayable' ORDER BY start_ts LIMIT 1"
        )
        .fetchone()["recorder_slice"]
    )

    price = ["analyze", slice_id, "what happened in this session?"]
    main(price)
    priced = read_output(capsys)

    from locus.analysis.model.cards import max_image_tokens

    ceiling = max_image_tokens(priced["model"])["activity_screenshots"]
    turn1 = priced["turn1"]
    assert (
        0 < turn1["screenshot_tokens"] <= ceiling * priced["n_activity_screenshots"]
    ), "every screenshot is priced at or under the card's per-image token ceiling"
    assert turn1["context_pct"] > 0, (
        "the price is stated against the card's own context"
    )
    with pytest.raises(SystemExit):
        main([*price, "--screenshot-longest-edge", "256"])


def test_analyze_model_flag_selects_over_the_roster(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    db = str(data_home(tmp_path) / "events.db")

    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    visitor_id, events = read_recording(FIXTURE)
    conn = connect(db)
    hydrate(conn, visitor_id, events)
    materialize_slices(conn, visitor_id)
    conn.close()
    subprocess.run(["bun", str(DISTILL), db], check=True, capture_output=True)

    connect(db).execute("UPDATE events SET snippet = 'demo'").connection.commit()
    (data_home(tmp_path) / "context").mkdir()
    (data_home(tmp_path) / "context" / "demo.md").write_text(
        "a demo site recorded for the test fixture"
    )
    slice_id = (
        connect(db)
        .execute(
            "SELECT recorder_slice FROM slices WHERE status = 'replayable' ORDER BY start_ts LIMIT 1"
        )
        .fetchone()["recorder_slice"]
    )

    # Under the shipped declaration the default mark sits on the Gemini card, so a local price under --model proves the flag selected over the default.
    from _analysis_support import remark_default
    from locus.analysis.model import cards

    remark_default(monkeypatch, tmp_path, GEMINI, dirname="shipped-cards")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-not-used")
    assert cards.default_card() == GEMINI
    local = cards.models_speaking(cards.OPENAI_COMPATIBLE)[0]
    assert local != GEMINI
    main(["analyze", slice_id, "what happened?", "--model", local])
    assert read_output(capsys)["model"] == local

    with pytest.raises(SystemExit, match="no card declares model"):
        main(["analyze", slice_id, "what happened?", "--model", "nope"])


def test_utc_ms_reads_iso_as_utc_and_refuses_epoch_seconds():
    from locus.analysis.ae import _utc_ms

    assert _utc_ms("2026-07-11") == 1783728000000
    assert _utc_ms("2026-07-11T22:17:55") == 1783808275000
    assert _utc_ms("2026-07-11T22:17:55+00:00") == 1783808275000
    assert _utc_ms("1783808275000") == 1783808275000
    with pytest.raises(SystemExit, match="epoch"):
        _utc_ms("1783808275")


def _real_schema_into(root: Path) -> None:
    """Place the repo's real layout declaration under a test deployment root, where _ae_layout reads it."""
    src = Path(__file__).parents[2] / TELEMETRY_SCHEMA
    dst = root / TELEMETRY_SCHEMA
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text())


def test_ae_reports_its_sampling_state(tmp_path, capsys, monkeypatch):
    responses = [
        {"data": [{"metric": "slice_started", "n": "2"}], "rows": 1},
        {"data": [{"m": 3}], "rows": 1},
    ]
    queries = []

    def fake_ae_query(sql):
        queries.append(sql)
        return responses[len(queries) - 1]

    monkeypatch.setattr("locus.evidence.analytics_engine.ae_query", fake_ae_query)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.ae._ae_dataset", lambda: "locus_capture")
    monkeypatch.setattr("locus.analysis.ae.deployment_root", lambda: tmp_path)
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    _real_schema_into(tmp_path)

    main(["ae", "SELECT 1"])
    out = capsys.readouterr().out
    envelope = json.loads(artifact(out).read_text())
    assert inline_echo(out) == artifact(out).read_text(), (
        "a small envelope is said inline too, byte-exact"
    )
    assert envelope["rows"] == [{"metric": "slice_started", "n": "2"}]
    assert envelope["sql"] == "SELECT 1", (
        "the result names the query it answers — self-describing on its own"
    )
    assert envelope["current_timestamp"].endswith("Z")
    assert re.search(r"\b3\b", envelope["sampling"])
    assert "sum(_sample_interval)" in envelope["sampling"]
    assert list(envelope).index("sampling") < list(envelope).index("rows"), (
        "the fact that judges the numbers is read before them"
    )
    assert str(tmp_path / TELEMETRY_SCHEMA) in envelope["schema"], (
        "the result points at the layout's owning file, never restates the layout"
    )
    assert "locus_capture" in queries[1]
    # The caller's query bounds its own window; the probe cannot know it, so it
    # must not bound itself either and then speak as if it had.
    assert "INTERVAL" not in queries[1] and "timestamp" not in queries[1]

    responses[:] = [
        {"data": [], "rows": 0},
        {"data": [{"latest": "2026-07-11 00:00:00"}], "rows": 1},
        {"data": [{"m": 1}], "rows": 1},
    ]
    queries.clear()
    main(["ae", "SELECT 1"])
    assert read_output(capsys)["sampling"].startswith("none")


def test_ae_zero_shaped_result_carries_its_diagnosis(tmp_path, capsys, monkeypatch):
    responses = []
    queries = []

    def fake_ae_query(sql):
        queries.append(sql)
        return responses[len(queries) - 1]

    monkeypatch.setattr("locus.evidence.analytics_engine.ae_query", fake_ae_query)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.ae._ae_dataset", lambda: "locus_capture")
    monkeypatch.setattr("locus.analysis.ae.deployment_root", lambda: tmp_path)
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    _real_schema_into(tmp_path)

    # The genesis shape: a well-formed row holding 0 is the same absence as an
    # empty set — a scoped count of an identifier that never existed.
    responses[:] = [
        {"data": [{"est": "0"}], "rows": 1},
        {"data": [{"latest": "2026-07-11 00:00:00"}], "rows": 1},
        {"data": [{"m": 1}], "rows": 1},
    ]
    main(["ae", "SELECT sum(_sample_interval) AS est FROM locus_capture"])
    out = capsys.readouterr().out
    envelope = json.loads(artifact(out).read_text())
    zero = envelope["zero"]
    assert "2026-07-11 00:00:00" in zero["latest_row"]
    assert any("blob2=visitor" in line for line in zero["layout"])
    assert any("recorder_fault" in line for line in zero["layout"])
    assert any('first_slice ["1" | "0"]' in line for line in zero["layout"]), (
        "a declared value domain rides its blob"
    )
    assert "locus usage" in zero["note"]
    assert "max(timestamp)" in queries[1], "the verb probes vitality itself"
    # The output leads with the diagnosis, then what would resolve it; the rows
    # it is a diagnosis about come last, so a head-read captures the critical half.
    keys = list(envelope)
    assert keys.index("headline") < keys.index("zero") < keys.index("rows")

    # Content rows are not zero-shaped: no diagnosis, no vitality probe.
    responses[:] = [
        {"data": [{"metric": "page_load", "est": "41"}], "rows": 1},
        {"data": [{"m": 1}], "rows": 1},
    ]
    queries.clear()
    main(["ae", "SELECT 1"])
    assert "zero" not in read_output(capsys)

    # A dataset with no rows at all says so rather than inventing a moment.
    responses[:] = [
        {"data": [], "rows": 0},
        {"data": [], "rows": 0},
        {"data": [], "rows": 0},
    ]
    queries.clear()
    main(["ae", "SELECT 1"])
    assert (
        "no rows anywhere in this dataset's retention"
        in read_output(capsys)["zero"]["latest_row"]
    )


def test_ae_expands_ms_datetimes_clientside(tmp_path, capsys, monkeypatch):
    queries = []

    def fake_ae_query(sql):
        queries.append(sql)
        return {"data": [], "rows": 0}

    monkeypatch.setattr("locus.evidence.analytics_engine.ae_query", fake_ae_query)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.ae._ae_dataset", lambda: None)
    monkeypatch.setattr("locus.analysis.ae.deployment_root", lambda: tmp_path)
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)

    main(
        [
            "ae",
            (
                "SELECT blob3 FROM t WHERE double1 >= ms('2026-07-12 00:18') AND double1 < ms(\"2026-07-13\")"
            ),
        ]
    )
    capsys.readouterr()
    assert queries == [
        (
            "SELECT blob3 FROM t WHERE double1 >= 1783815480000 AND double1 < 1783900800000"
        )
    ]

    with pytest.raises(SystemExit, match="not-a-datetime"):
        main(["ae", "SELECT 1 WHERE double1 >= ms('not-a-datetime')"])


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


def _loaded_fixture_db(tmp_path, monkeypatch):
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    db = str(data_home(tmp_path) / "events.db")
    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    visitor_id, events = read_recording(FIXTURE)
    conn = connect(db)
    hydrate(conn, visitor_id, events)
    materialize_slices(conn, visitor_id)
    conn.execute("UPDATE events SET snippet = 'demo'")
    conn.commit()
    conn.close()
    (data_home(tmp_path) / "context").mkdir(exist_ok=True)
    (data_home(tmp_path) / "context" / "demo.md").write_text("a demo site")
    subprocess.run(["bun", str(DISTILL), db], check=True, capture_output=True)
    conn = connect(db)
    rows = conn.execute(
        "SELECT recorder_slice, visitor_id FROM slices WHERE status='replayable' ORDER BY start_ts"
    ).fetchall()
    return db, conn, rows


def test_browse_open_materializes_beside_the_db_at_derived_names(
    tmp_path, capsys, monkeypatch
):
    from locus.analysis.cli import _resolve_open

    _db, _conn, rows = _loaded_fixture_db(tmp_path, monkeypatch)
    rid = rows[0]["recorder_slice"]

    argv = _resolve_open(["open", rid])
    capsys.readouterr()
    page = Path(argv[1])
    assert page.parent == data_home(tmp_path) / "pages", (
        "replay material is a derived family beside the db, never inside a wrapper dir"
    )
    assert page.name == f"{rows[0]['visitor_id']}_{rid}.html"
    payload = page.with_suffix(".js")
    assert payload.exists()
    assert f'<script src="./{payload.name}">' in page.read_text(), (
        "the default page embeds its own set's payload by relative src"
    )

    # Deterministic derivation, not evidence: re-opening refreshes in place.
    before = payload.read_text()
    assert _resolve_open(["open", rid])[1] == str(page)
    capsys.readouterr()
    assert payload.read_text() == before

    # Two sets sharing a first slice and a count are different payloads; the digest
    # keeps one from silently replacing the other under a page that embeds it.
    first_pair = _resolve_open(["open", rid, rows[1]["recorder_slice"]])[1]
    second_pair = _resolve_open(["open", rid, rows[2]["recorder_slice"]])[1]
    capsys.readouterr()
    assert first_pair != second_pair
    assert Path(first_pair).exists() and Path(second_pair).exists()


def test_browse_open_takes_a_slice_and_the_moment_to_see_it_at(
    tmp_path, capsys, monkeypatch
):
    # "This slice at this moment" is one target: the moment rides the slice name and lands
    # on the page composed from it, so opening a citation never costs a first open to learn
    # the composed page's derived name.
    from locus.analysis.cli import _resolve_open

    _db, _conn, rows = _loaded_fixture_db(tmp_path, monkeypatch)
    rid = rows[0]["recorder_slice"]
    page = _resolve_open(["open", rid])[1]
    capsys.readouterr()

    assert _resolve_open(["open", f"{rid}#t=123"])[1] == f"{page}#t=123"
    capsys.readouterr()

    # A set composes one page, so one moment addresses it, riding whichever name carries it.
    pair = _resolve_open(["open", rid, f"{rows[1]['recorder_slice']}#t=456"])
    capsys.readouterr()
    assert pair[1].endswith("#t=456")
    with pytest.raises(SystemExit, match="one moment"):
        _resolve_open(["open", f"{rid}#t=1", f"{rows[1]['recorder_slice']}#t=2"])


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


def test_browse_open_names_every_form_when_the_target_is_none_of_them(
    tmp_path, capsys, monkeypatch
):
    from locus.analysis.cli import _resolve_open

    _db, _conn, _rows = _loaded_fixture_db(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="URL.*analysis directory.*slice set"):
        _resolve_open(["open", "no-such-thing"])


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


def test_browse_open_composes_an_analysis_directory_from_its_slice_table(
    tmp_path, capsys, monkeypatch
):
    # The seam: window.json's slice table (as the engine writes it) drives
    # composition. Consecutive same-visitor slices that don't overlap share
    # one mount and play through as one stream, labeled by their range; a time
    # overlap is a concurrent lane and starts its own mount.
    from locus.analysis.cli import _resolve_open

    _db, _conn, rows = _loaded_fixture_db(tmp_path, monkeypatch)
    visitor = rows[0]["visitor_id"]

    def seg(label, row, start, end):
        # opening consumes label/visitor/slice; slice_id belongs to the writer
        return {
            "label": label,
            "visitor": visitor,
            "slice": row["recorder_slice"],
            "slice_id": 0,
            "start_ts": start,
            "end_ts": end,
        }

    analysis_dir = data_home(tmp_path) / "analyses" / "2026-07-18T00-00-00Z_demo-0001"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "window.json").write_text(
        json.dumps({"slices": [seg("S1", rows[0], 0, 10), seg("S2", rows[1], 20, 30)]})
    )

    argv = _resolve_open(["open", f"{analysis_dir}#t=1781254694465"])
    narration = capsys.readouterr().out
    page_ref, _, fragment = argv[1].partition("#")
    page = Path(page_ref)
    assert fragment == "t=1781254694465", "the citation fragment rides through"
    assert page.name == f"{analysis_dir.name}.html"
    body = page.read_text()
    assert body.count("<section") == 1, (
        "a sequential same-visitor window is one stream on one player"
    )
    assert 'data-label="S1–S2"' in body
    assert narration.count("materialized") >= 1 and str(page) in narration

    concurrent = data_home(tmp_path) / "analyses" / "2026-07-18T00-00-01Z_demo-0002"
    concurrent.mkdir()
    (concurrent / "window.json").write_text(
        json.dumps({"slices": [seg("S1", rows[0], 0, 10), seg("S2", rows[1], 5, 15)]})
    )
    argv = _resolve_open(["open", str(concurrent)])
    capsys.readouterr()
    body = Path(argv[1].partition("#")[0]).read_text()
    assert body.count("<section") == 2, "a concurrent lane gets its own mount"
    assert 'data-label="S1"' in body and 'data-label="S2"' in body

    empty = data_home(tmp_path) / "analyses" / "not-an-analysis"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no window.json"):
        _resolve_open(["open", str(empty)])


def test_browse_open_drives_the_deployment_browser_end_to_end(
    tmp_path, capsys, monkeypatch
):
    # The whole absorbed path, for real: `locus browse open <slice>` materializes,
    # composes, boots the headless deployment browser, and lands on the default
    # page with the player actually mounted.
    _db, _conn, rows = _loaded_fixture_db(tmp_path, monkeypatch)
    monkeypatch.setenv("LOCUS_BROWSE_HEADLESS", "1")
    rid = rows[0]["recorder_slice"]
    try:
        with pytest.raises(SystemExit) as leave:
            main(["browse", "open", rid])
        assert not leave.value.code, capsys.readouterr()
        out = capsys.readouterr().out
        page = data_home(tmp_path) / "pages" / f"{rows[0]['visitor_id']}_{rid}.html"
        assert page.resolve().as_uri() in out
        window = re.search(r"\[(w\d+)\]", out).group(1)

        import time

        mounted = False
        for _ in range(50):
            with pytest.raises(SystemExit):
                main(
                    [
                        "browse",
                        "eval",
                        window,
                        (
                            "document.querySelector('.rr-player iframe')?.contentDocument?.body?.childElementCount > 0"
                        ),
                    ]
                )
            if "true" in capsys.readouterr().out:
                mounted = True
                break
            time.sleep(0.2)
        assert mounted, "the player mounted for real"
    finally:
        with pytest.raises(SystemExit):
            main(["browse", "quit"])


def test_a_legal_slice_id_collision_is_addressable_visitor_qualified(
    tmp_path, capsys, monkeypatch
):
    # A slice id is its open millisecond plus a random 4-character draw, so two
    # visitors can legally coincide on one id. The bare id refuses with the
    # qualified addresses ready to paste; the qualified address resolves.
    from _support import env, full_snapshot, meta
    from locus.analysis.cli import _resolve, _resolve_address, _resolve_open
    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    conn = connect(str(data_home(tmp_path) / "events.db"))
    rid = "00000000001000-aaaa"
    for visitor in ("v1-alpha", "v2-beta"):
        hydrate(conn, visitor, [env(meta(1000), rid), env(full_snapshot(1001), rid)])
        materialize_slices(conn, visitor)

    with pytest.raises(SystemExit) as refusal:
        _resolve(conn, rid)
    assert f"v1-alpha/{rid}" in str(refusal.value) and f"v2-beta/{rid}" in str(
        refusal.value
    ), "the refusal ends with the qualified addresses, ready to paste"

    assert _resolve(conn, f"v2-beta/{rid}")["visitor_id"] == "v2-beta"

    with pytest.raises(SystemExit, match="no slice"):
        _resolve(conn, f"v3-nobody/{rid}")

    # The qualified base composes with a piece suffix: the '#' peels first,
    # so the address reaches the piece logic already resolved to one visitor.
    with pytest.raises(SystemExit, match="route"):
        _resolve_address(conn, f"v1-alpha/{rid}#1")
    conn.close()

    argv = _resolve_open(["open", f"v1-alpha/{rid}"])
    capsys.readouterr()
    assert Path(argv[1]).name == f"v1-alpha_{rid}.html", (
        "the exposed surfaces take the qualified form wherever a slice id goes"
    )

    with pytest.raises(SystemExit, match="is a slice id"):
        main(["analyze", f"v1-alpha/{rid}", f"v2-beta/{rid}"])


def test_analyze_derives_the_analysis_dir_from_occasion_and_subject(
    tmp_path, capsys, monkeypatch
):
    import re
    from datetime import datetime, timezone

    _db, _conn, rows = _loaded_fixture_db(tmp_path, monkeypatch)
    rid = rows[0]["recorder_slice"]

    seen = {}

    def fake_analysis(conn_, slice_ids, **kwargs):
        out = Path(kwargs["out_dir"])
        seen["out_dir"] = out
        return {"dir": str(out), "response_path": str(out / "2_answer_response.txt")}

    monkeypatch.setattr("locus.analysis.engine.run_analysis", fake_analysis)
    before = datetime.now(timezone.utc).replace(microsecond=0)
    main(["analyze", rid, "what happened?", "--run"])

    out = seen["out_dir"]
    said = capsys.readouterr().out
    assert artifact(said) == out, (
        "an analysis hands over the directory it made, not a log pointing at it"
    )
    assert (out / "analysis.log").exists(), (
        "and speaks into that directory, beside the evidence"
    )
    assert inline_echo(said) == (out / "analysis.log").read_text(), (
        "a small say echoes inline after the address"
    )
    assert out.parent == data_home(tmp_path) / "analyses", (
        "analyses are a derived family beside the db"
    )
    disambiguator = rid.rpartition("-")[2]
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)_"
        + re.escape(f"{rows[0]['visitor_id']}-{disambiguator}"),
        out.name,
    )
    assert match, f"occasion-first, subject after: {out.name}"
    stamp = datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%SZ").replace(
        tzinfo=timezone.utc
    )
    assert stamp >= before, "the occasion is minted UTC, at invocation"

    with pytest.raises(SystemExit):
        main(["analyze", rid, "what happened?", "--out", "somewhere"])


def test_a_capped_deployment_refuses_a_run_before_it_has_a_home(
    tmp_path, capsys, monkeypatch, spend_isolation
):
    """The wall is a ledger read, asked before the analysis directory is minted: a refused run leaves nothing in analyses/ for a later reader to take for a finding."""
    from _analysis_support import remark_default

    _db, _conn, rows = _loaded_fixture_db(tmp_path, monkeypatch)
    rid = rows[0]["recorder_slice"]
    remark_default(monkeypatch, tmp_path, GEMINI, dirname="shipped-cards")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-not-used")
    (spend_isolation / "config" / "spend.toml").write_text("week_usd = 0\n")

    with pytest.raises(SystemExit) as refusal:
        main(["analyze", rid, "what happened?", "--run"])
    assert "spend.toml" in str(refusal.value), (
        "the refusal is the wall's, naming its declaration"
    )
    said = capsys.readouterr()
    assert not [line for line in said.out.splitlines() if line.startswith("started ")]
    assert not (data_home(tmp_path) / "analyses").exists()


def test_a_multi_slice_analysis_subject_counts_its_extras(
    tmp_path, capsys, monkeypatch
):
    _db, _conn, rows = _loaded_fixture_db(tmp_path, monkeypatch)
    seen = {}

    def fake_analysis(conn_, slice_ids, **kwargs):
        out = Path(kwargs["out_dir"])
        seen["out_dir"] = out
        return {"dir": str(out), "response_path": str(out / "2_answer_response.txt")}

    monkeypatch.setattr("locus.analysis.engine.run_analysis", fake_analysis)
    main(
        [
            "analyze",
            rows[0]["recorder_slice"],
            rows[1]["recorder_slice"],
            "what happened?",
            "--run",
        ]
    )
    assert seen["out_dir"].name.endswith("_plus1")


def test_a_forgotten_question_is_caught_before_anything_runs(
    tmp_path, capsys, monkeypatch
):
    _db, _conn, rows = _loaded_fixture_db(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="question"):
        main(["analyze", rows[0]["recorder_slice"], rows[1]["recorder_slice"]])


def test_analyze_defaults_to_the_deployments_default_card(
    tmp_path, capsys, monkeypatch
):
    """An analysis runs on the deployment's default card, nothing else: the roster card marked `default = true`, read fresh by every analyze — a key in the environment never picks it."""
    _db, _conn, rows = _loaded_fixture_db(tmp_path, monkeypatch)
    seen = {}

    def fake_price(conn_, slice_ids, **kwargs):
        seen["model"] = kwargs["model"]
        return {
            "model": kwargs["model"],
            "turn1": {
                "fits": True,
                "total_tokens": 1,
                "text_tokens": 1,
                "screenshot_tokens": 0,
                "context_pct": 0.1,
            },
        }

    monkeypatch.setattr("locus.analysis.engine.price_analysis", fake_price)
    # The test's deployment is tmp_path, not this machine: main() self-serves
    # env from the .env files above cwd, which would re-bank the real key.
    monkeypatch.setattr("locus.evidence.deployment.env_from_files", dict)

    from locus.analysis.model import cards

    local = cards.default_card()
    assert local == cards.models_speaking(cards.OPENAI_COMPATIBLE)[0]
    monkeypatch.setenv("GOOGLE_API_KEY", "banked")
    main(["analyze", rows[0]["recorder_slice"], "what happened?"])
    capsys.readouterr()
    assert seen["model"] == local, "a banked key does not pick the model"

    from _analysis_support import remark_default

    remark_default(monkeypatch, tmp_path, GEMINI, dirname="gemini-marked-cards")
    monkeypatch.delenv("GOOGLE_API_KEY")
    main(["analyze", rows[0]["recorder_slice"], "what happened?"])
    capsys.readouterr()
    assert seen["model"] == GEMINI, "the declaration picks it; no key needed to pick"


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


def test_run_is_absent_from_help_but_parses(capsys):
    """The spend step is learned from a price report that fits, never from the flag list — a first-contact agent must not be able to compose a spending invocation before it has held a price."""
    with pytest.raises(SystemExit, match="0"):
        main(["analyze", "--help"])
    assert "--run" not in capsys.readouterr().out
    with pytest.raises(SystemExit, match="2"):
        main(["analyze", "--run"])
    err = capsys.readouterr().err
    assert "unrecognized" not in err, "the flag parses; only the help hides it"
    assert "required: slices, question" in err


def _definitions_into(root: Path, text: str | None = None) -> None:
    """The repo's shipped definitions.toml under a test deployment root, or a declaration of the test's own."""
    src = Path(__file__).parents[2] / "config" / "definitions.toml"
    (root / "config").mkdir(exist_ok=True)
    (root / "config" / "definitions.toml").write_text(
        text if text is not None else src.read_text()
    )


def test_a_load_derives_sessions_under_the_shipped_definition(
    tmp_path, capsys, monkeypatch
):
    """The sessions table is a derived surface like the slices: a load leaves it current, every user event stamped with its session, under the definition at the deployment root — and a definition that cannot shape it refuses loud before anything derives."""
    from locus.evidence.deployment import definitions

    # A head slice — minted before its Meta — whose page load is attested by its open.
    rid = f"0{T0 - 1}-aaaa"
    stream = [
        counted(event, seq)
        for seq, event in enumerate(
            [meta(T0), full_snapshot(T0 + 1), click(T0 + 15_000, 3)], start=1
        )
    ]
    key = f"snip01/{slice_date(rid)}/{VISITOR}/{rid}/{T0}000001.json.gz"
    store = FakeStore({key: chunk_blob(rid, stream)})
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    db = str(data_home(tmp_path) / "events.db")

    main(["load"])
    capsys.readouterr()
    conn = connect(db)
    (session,) = conn.execute("SELECT * FROM sessions").fetchall()
    assert (session["visitor_id"], session["snippet"]) == (VISITOR, "snip01")
    assert session["engaged"] == 1, "15 s between the first act and the last"
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE session_id IS NULL").fetchone()[
            0
        ]
        == 0
    )
    shipped = definitions(tmp_path)
    assert shipped["session"]["inactivity_minutes"] > 0
    assert shipped["engaged"]["min_pageviews"] >= 1
    conn.close()

    for text, complaint in [
        ("[session]\ninactivity_minutes = 30\n", "no \\[engaged\\] table"),
        (
            "[session]\ninactivity_minutes = 30\ngap_hours = 1\n[engaged]\nmin_seconds = 10\nmin_pageviews = 2\n",
            "unknown keys",
        ),
        (
            "[session]\ninactivity_minutes = 'thirty'\n[engaged]\nmin_seconds = 10\nmin_pageviews = 2\n",
            "non-negative number",
        ),
        (
            "[session]\ninactivity_minutes = 30\n[engaged]\nmin_seconds = 10\nmin_pageviews = 2\n[conversion]\nx = { url = '/buy' }\n",
            "unknown tables",
        ),
    ]:
        _definitions_into(tmp_path, text)
        with pytest.raises(SystemExit, match=complaint):
            main(["status"])


def test_a_load_holds_the_db_to_the_declared_retention_horizon(
    tmp_path, capsys, monkeypatch
):
    """The horizon governs every copy: a recording past it is never fetched, and one already in the db goes on the next load — said out loud, because these are recordings."""
    from locus.evidence.retention import CONFIG_FILE

    day = 86_400_000
    old, recent = f"0{T0 - 60 * day}-aaaa", f"0{T0}-bbbb"
    blobs = {}
    for rid, ms in ((old, T0 - 60 * day), (recent, T0)):
        stream = [
            counted(e, s)
            for s, e in enumerate([meta(ms), full_snapshot(ms + 1)], start=1)
        ]
        blobs[f"snip01/{slice_date(rid)}/{VISITOR}/{rid}/{ms}000001.json.gz"] = (
            chunk_blob(rid, stream)
        )
    store = FakeStore(blobs)
    monkeypatch.setattr("locus.evidence.store.store_for", lambda url: store)
    monkeypatch.setattr("locus.evidence.deployment.bucket", lambda: "b")
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    db = data_home(tmp_path) / "events.db"
    reach = (
        datetime.now(timezone.utc).date()
        - datetime.fromtimestamp((T0 - 60 * day) / 1000, timezone.utc).date()
    ).days + 1

    # First under a horizon that reaches both recordings, so both land.
    (tmp_path / CONFIG_FILE).write_text(f"retention_days = {reach}\n")
    main(["load"])
    capsys.readouterr()
    conn = connect(db)
    assert {r["recorder_slice"] for r in conn.execute("SELECT * FROM slices")} == {
        old,
        recent,
    }
    conn.close()

    # Then narrowed past the older recording: it is dropped from the db and never re-fetched.
    (tmp_path / CONFIG_FILE).write_text(f"retention_days = {reach - 40}\n")
    main(["load"])
    said = artifact(capsys.readouterr().out).read_text()
    assert re.search(r"retention[^\n]*\b1\b", said)
    conn = connect(db)
    assert [r["recorder_slice"] for r in conn.execute("SELECT * FROM slices")] == [
        recent
    ]
    assert conn.execute("SELECT COUNT(*) n FROM loaded_chunks").fetchone()["n"] == 1
    conn.close()

    main(["load"])
    assert re.search(
        r"\b1\b[^\n]*horizon", artifact(capsys.readouterr().out).read_text()
    ), "the object is still in the store, and every later load passes over it unfetched"


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
