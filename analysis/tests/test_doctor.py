"""locus doctor — the mechanical-integrity verb: verdicts, heals, and the names it gives what it cannot touch.

What "declared matches live" means belongs to store/scripts/doctor.js and is the store suite's to prove; here a stand-in script exercises this side's plumbing — the spawn, the verdict, the carried detail, the exit codes."""

import json
import shutil
from pathlib import Path

import pytest
from _analysis_support import GEMINI, GEMINI_CLIENT, horizon_at
from _support import (
    FIXTURE,
    read_recording,
)
from locus.analysis.cli import main
from locus.analysis.model.cards import models_speaking, pricing
from locus.evidence.db import connect
from locus.evidence.speak import ECHO_BYTES

STORE_OK = "console.log('  OK    everything matches'); process.exit(0);"
SHIPPED_DEFINITIONS = Path(__file__).parents[2] / "config" / "definitions.toml"


def definition_at(root: Path) -> dict:
    """The shipped definitions.toml placed under a test deployment root, if none is there, and read back — what the derivation path runs under there."""
    from locus.evidence.deployment import definitions

    target = root / "config" / "definitions.toml"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(SHIPPED_DEFINITIONS.read_text())
    return definitions(root)


def healthy_surroundings(tmp_path, monkeypatch, store_doctor=STORE_OK):
    """A deployment whose non-db checks pass: bun and the browser are the real ones on this machine, the store check answers from a stand-in doctor.js, credentials are present, and the shipped session definition and a horizon reaching the fixture corpus stand. cwd moves off the repo so no real .env leaks into the checks."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("locus.analysis.cli.deployment_root", lambda: tmp_path)
    definition_at(tmp_path)
    horizon_at(tmp_path)
    scripts = tmp_path / "store" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "doctor.js").write_text(store_doctor)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")


def run_doctor(capsys, clean=True):
    if clean:
        main(["doctor"])
    else:
        with pytest.raises(SystemExit, match="1"):
            main(["doctor"])
    opening, _, echoed = capsys.readouterr().out.partition("\n")
    assert opening.split()[0] == "started", (
        "the caller is handed an address first, the unhealthy run as much as the clean one"
    )
    body = Path(opening.split()[2]).read_text()
    small = len(body.encode()) <= ECHO_BYTES
    assert echoed == (body if clean and small else ""), (
        "a clean small say echoes inline; a failing run hands over only the address and its exit"
    )
    log = body.splitlines()
    report = json.loads(log[-1])
    assert report["clean"] is clean
    assert not clean or all(c["ok"] for c in report["checks"])
    return {c["check"]: c for c in report["checks"]}


def _hydrated_db(tmp_path, derived=False) -> str:
    from locus.evidence.derive import finish_derivations
    from locus.evidence.hydrate import hydrate

    (tmp_path / "data").mkdir(exist_ok=True)
    db = str(tmp_path / "data" / "events.db")
    visitor_id, events = read_recording(FIXTURE)
    conn = connect(db)
    hydrate(conn, visitor_id, events)
    if derived:
        finish_derivations(conn, db, definition_at(tmp_path))
    conn.close()
    return db


def test_a_clean_deployment_reports_clean(tmp_path, capsys, monkeypatch):
    healthy_surroundings(tmp_path, monkeypatch)

    # Before any load there is no db at all — that is a clean deployment, not a sick one.
    checks = run_doctor(capsys)
    assert "events.db" in checks["derivations"]["note"]
    assert checks["store"]["detail"], "the store doctor's own lines carry through"

    _hydrated_db(tmp_path, derived=True)
    checks = run_doctor(capsys)
    assert checks["derivations"] == {
        "check": "derivations",
        "ok": True,
        "stale_vintage": False,
        "stranded_events": 0,
    }


def test_doctor_heals_stranded_rows_through_the_load_path(
    tmp_path, capsys, monkeypatch
):
    # Hydrated, slices never materialized, nothing distilled — exactly what a load that died partway leaves.
    healthy_surroundings(tmp_path, monkeypatch)
    db = _hydrated_db(tmp_path)

    checks = run_doctor(capsys)
    found = checks["derivations"]["found"]
    assert found["stranded_events"] > 0 and found["stale_vintage"] is False
    assert "re-materialized" in checks["derivations"]["healed"]["re_derived"]

    conn = connect(db)
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE type_str IS NULL").fetchone()[0]
        == 0
    )
    assert conn.execute("SELECT COUNT(*) FROM slices").fetchone()[0] > 0, (
        "the heal is the load path whole — slice materialization included, or a later load would never revisit these rows"
    )
    assert (
        conn.execute(
            "SELECT value FROM meta WHERE key = 'derivation_vintage'"
        ).fetchone()
        is not None
    )

    checks = run_doctor(capsys)
    assert checks["derivations"]["stranded_events"] == 0


def test_the_heal_says_itself_into_the_run_log(tmp_path, capsys, monkeypatch):
    """A heal is a re-derivation of the corpus — minutes of slice materialization and a distillation pass that speaks from its own process. A caller who asked for a health report is handed an address, and every word of the heal is behind it."""
    healthy_surroundings(tmp_path, monkeypatch)
    _hydrated_db(tmp_path)

    main(["doctor"])
    said = capsys.readouterr()
    assert said.err == "", "the heal reaches the caller through no stream of its own"
    opening, _, echoed = said.out.partition("\n")
    body = Path(opening.split()[2]).read_text()
    assert echoed == (body if len(body.encode()) <= ECHO_BYTES else ""), (
        "past the address the caller sees the log's own body whole, or nothing"
    )

    log = body.splitlines()
    assert any("distilled" in line for line in log[:-1]), (
        "the distillation pass is a subprocess, and it lands there like everything else"
    )
    assert json.loads(log[-1])["clean"] is True, (
        "and the report is still the last line, behind however much was said"
    )


def test_doctor_heals_a_stale_vintage_by_full_re_derive(tmp_path, capsys, monkeypatch):
    healthy_surroundings(tmp_path, monkeypatch)
    db = _hydrated_db(tmp_path, derived=True)
    monkeypatch.setattr(
        "locus.evidence.derive.derivation_vintage", lambda definition: "a new vintage"
    )

    checks = run_doctor(capsys)
    assert checks["derivations"]["found"]["stale_vintage"] is True
    assert checks["derivations"]["healed"]["re_derived"] == (
        "every visitor, from the timestamps stage on — timestamps, slices, rescue, distillation, sessions"
    )
    assert (
        connect(db)
        .execute("SELECT value FROM meta WHERE key = 'derivation_vintage'")
        .fetchone()["value"]
        == "a new vintage"
    )

    checks = run_doctor(capsys)
    assert checks["derivations"]["stale_vintage"] is False


def test_the_stale_heal_walks_the_whole_derivation_path(tmp_path, capsys, monkeypatch):
    # Staleness means any deriving logic may have changed — the canonical timestamp and the
    # slice derivation included, not just the flat columns — so the heal must recompute all of
    # it from raw_json. Hand-corrupt one derived value of each kind, mark the vintage stale,
    # and the heal restores every one; healing only distillation would leave the corruption
    # standing under a fresh stamp that claims currency.
    healthy_surroundings(tmp_path, monkeypatch)
    db = _hydrated_db(tmp_path, derived=True)
    conn = connect(db)
    event = conn.execute(
        "SELECT id, timestamp FROM events ORDER BY id LIMIT 1"
    ).fetchone()
    conn.execute(
        "UPDATE events SET timestamp = timestamp + 12345 WHERE id = ?", (event["id"],)
    )
    a_slice = conn.execute(
        "SELECT id, status FROM slices WHERE status = 'replayable' LIMIT 1"
    ).fetchone()
    conn.execute(
        "UPDATE slices SET status = 'discarded', reason = 'corrupted by hand' WHERE id = ?",
        (a_slice["id"],),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        "locus.evidence.derive.derivation_vintage", lambda definition: "a new vintage"
    )

    run_doctor(capsys)
    conn = connect(db)
    assert (
        conn.execute(
            "SELECT timestamp FROM events WHERE id = ?", (event["id"],)
        ).fetchone()["timestamp"]
        == event["timestamp"]
    ), "the canonical timestamp re-derived from raw_json"
    assert (
        conn.execute(
            "SELECT status FROM slices WHERE id = ?", (a_slice["id"],)
        ).fetchone()["status"]
        == "replayable"
    ), "the slice derivation re-derived from raw_json"


def test_doctor_names_the_installs_it_cannot_do(tmp_path, capsys, monkeypatch):
    healthy_surroundings(tmp_path, monkeypatch)
    db = _hydrated_db(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    checks = run_doctor(capsys, clean=False)
    assert "bun.sh" in checks["bun"]["fix"]
    assert checks["derivations"]["ok"] is False
    assert "bun" in checks["derivations"]["fix"]
    assert (
        connect(db)
        .execute("SELECT COUNT(*) FROM events WHERE type_str IS NULL")
        .fetchone()[0]
        > 0
    ), "no runtime, no heal — the finding stands untouched"
    assert checks["store"]["ok"] is False and "bun" in checks["store"]["finding"]


def test_doctor_flags_a_missing_browser_and_names_the_install(
    tmp_path, capsys, monkeypatch
):
    healthy_surroundings(tmp_path, monkeypatch)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "empty"))

    checks = run_doctor(capsys, clean=False)
    assert checks["browser"]["ok"] is False
    assert "install.sh" in checks["browser"]["fix"]


def test_doctor_carries_the_store_doctors_drift_verdict(tmp_path, capsys, monkeypatch):
    healthy_surroundings(
        tmp_path,
        monkeypatch,
        store_doctor=(
            "console.log('  DRIFT lifecycle       7d, declared 30d');"
            "console.log('1 drifted setting — reconciles with `bun run provision`.');"
            "process.exit(1);"
        ),
    )

    checks = run_doctor(capsys, clean=False)
    assert checks["store"]["ok"] is False
    assert "does not match" in checks["store"]["finding"]
    assert any("DRIFT lifecycle" in line for line in checks["store"]["detail"])
    assert any("bun run provision" in line for line in checks["store"]["detail"]), (
        "the remedy is the store doctor's own words, carried verbatim"
    )


def test_doctor_reports_a_store_read_it_could_not_make(tmp_path, capsys, monkeypatch):
    healthy_surroundings(
        tmp_path,
        monkeypatch,
        store_doctor=(
            "console.error('doctor: bucket list failed (530)'); process.exit(2);"
        ),
    )

    checks = run_doctor(capsys, clean=False)
    assert checks["store"]["ok"] is False
    assert "could not be read" in checks["store"]["finding"]
    assert any("530" in line for line in checks["store"]["detail"])


def test_doctor_without_credentials_names_where_they_bank(
    tmp_path, capsys, monkeypatch
):
    healthy_surroundings(tmp_path, monkeypatch)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")

    checks = run_doctor(capsys, clean=False)
    assert checks["store"]["ok"] is False
    assert "store/.env" in checks["store"]["fix"]


REGISTRY = {
    f"gemini/{model}": {
        "input_cost_per_token": pricing(model)["input_per_mtok"] / 1e6,
        "cache_read_input_token_cost": pricing(model)["cached_input_per_mtok"] / 1e6,
        "output_cost_per_token": pricing(model)["output_per_mtok"] / 1e6,
        "output_cost_per_reasoning_token": pricing(model)["output_per_mtok"] / 1e6,
    }
    for model in models_speaking(GEMINI_CLIENT)
}


def test_doctor_verifies_declared_model_prices(tmp_path, capsys, monkeypatch):
    from locus.analysis import spend

    healthy_surroundings(tmp_path, monkeypatch)
    monkeypatch.setattr(spend, "_fetch_registry", lambda: REGISTRY)

    checks = run_doctor(capsys)
    card = next(c for c in checks["model_prices"]["cards"] if c["model"] == GEMINI)
    assert card["status"] == "verified"


def test_doctor_flags_price_drift_naming_both_numbers(tmp_path, capsys, monkeypatch):
    from locus.analysis import spend

    healthy_surroundings(tmp_path, monkeypatch)
    repriced = {
        f"gemini/{GEMINI}": {
            **REGISTRY[f"gemini/{GEMINI}"],
            "input_cost_per_token": 5e-07,
        }
    }
    monkeypatch.setattr(spend, "_fetch_registry", lambda: repriced)

    checks = run_doctor(capsys, clean=False)
    card = next(c for c in checks["model_prices"]["cards"] if c["model"] == GEMINI)
    assert card["status"] == "drift"
    (mismatch,) = card["mismatches"]
    assert mismatch["declared_usd_per_mtok"] == pricing(GEMINI)["input_per_mtok"]
    assert mismatch["registry_usd_per_mtok"] == pytest.approx(0.5)
    assert f"config/cards/{GEMINI}.toml" in card["fix"], (
        "the remedy is the declaration — the card is the price of record"
    )


def test_doctor_flags_a_gemini_card_with_no_pricing_block_as_unhealthy(
    tmp_path, capsys, monkeypatch
):
    """The undeclared card is the one state the registry sweep cannot reach — there is nothing to compare — and it is unhealthy outright: every paid call on it refuses at construction."""

    healthy_surroundings(tmp_path, monkeypatch)
    from _analysis_support import card_roster

    card_roster(
        monkeypatch,
        tmp_path,
        {"gemini-unpriced": 'conversation = "gemini"\ncontext_tokens = 1000\n'},
    )

    checks = run_doctor(capsys, clean=False)
    (card,) = checks["model_prices"]["cards"]
    assert checks["model_prices"]["ok"] is False
    assert card["model"] == "gemini-unpriced"
    assert card["status"] == "undeclared"
    assert "config/cards/gemini-unpriced.toml" in card["fix"]


def test_doctor_reports_unverifiable_prices_honestly(tmp_path, capsys, monkeypatch):
    healthy_surroundings(tmp_path, monkeypatch)

    # The registry stays offline (the suite's default): unverified, never
    # wrong, and never unhealthy.
    checks = run_doctor(capsys)
    card = next(c for c in checks["model_prices"]["cards"] if c["model"] == GEMINI)
    assert card["status"] == "unverified"
    assert "unreachable" in card["reason"]


def test_doctor_heals_rows_distilled_against_no_slice(tmp_path, capsys, monkeypatch):
    """A load killed between hydrating a visitor and placing its rows in slices can leave rows distillation later wrote against no slice at all — type_str set, slice_id NULL. That is stranded work too, and it shows only on the slice signal: the undistilled signal is clean, so a doctor reading that alone reports clean while the rows stay invisible to replay and to every slice-scoped query."""
    healthy_surroundings(tmp_path, monkeypatch)
    db = _hydrated_db(tmp_path, derived=True)
    conn = connect(db)
    conn.execute("UPDATE events SET slice_id = NULL")
    conn.execute("DELETE FROM slices")
    conn.commit()
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE type_str IS NULL").fetchone()[0]
        == 0
    )
    conn.close()

    checks = run_doctor(capsys)
    found = checks["derivations"]["found"]
    assert found["stranded_events"] > 0 and found["stale_vintage"] is False
    assert "re-materialized" in checks["derivations"]["healed"]["re_derived"]

    conn = connect(db)
    assert (
        conn.execute("SELECT COUNT(*) FROM events WHERE slice_id IS NULL").fetchone()[0]
        == 0
    )
    assert conn.execute("SELECT COUNT(*) FROM slices").fetchone()[0] > 0
    conn.close()
    assert run_doctor(capsys)["derivations"]["stranded_events"] == 0


def test_the_heal_starts_at_the_stage_whose_code_changed(tmp_path, capsys, monkeypatch):
    # Each stage derives from the one before, so a change to distillation alone has nothing
    # to say about slices or timestamps: the heal re-distills and leaves them standing, while
    # a change to the slice stage re-slices and re-distills but never touches a timestamp.
    from locus.evidence import derive as derive_mod

    healthy_surroundings(tmp_path, monkeypatch)
    db = _hydrated_db(tmp_path, derived=True)
    definition = definition_at(tmp_path)
    installed = derive_mod.stage_vintages(definition)

    conn = connect(db)
    # The stream's last event, moved later: still last, so the slice it ends keeps its
    # verdict — a corruption the slice stage can leave standing without being wrong.
    event = conn.execute(
        "SELECT id, timestamp FROM events ORDER BY timestamp DESC, id DESC LIMIT 1"
    ).fetchone()
    conn.execute(
        "UPDATE events SET timestamp = timestamp + 12345 WHERE id = ?", (event["id"],)
    )
    a_slice = conn.execute(
        "SELECT id FROM slices WHERE status = 'replayable' LIMIT 1"
    ).fetchone()
    conn.execute(
        "UPDATE slices SET status = 'discarded', reason = 'corrupted by hand' WHERE id = ?",
        (a_slice["id"],),
    )
    distilled = conn.execute(
        "SELECT id, type_str FROM events WHERE type_str IS NOT NULL ORDER BY id LIMIT 1"
    ).fetchone()
    conn.execute(
        "UPDATE events SET type_str = 'corrupted by hand' WHERE id = ?",
        (distilled["id"],),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        derive_mod,
        "stage_vintages",
        lambda definition: {**installed, "distill": "a new distill"},
    )
    checks = run_doctor(capsys)
    assert checks["derivations"]["healed"]["re_derived"].endswith(
        "on — distillation, sessions"
    )
    conn = connect(db)
    assert (
        conn.execute(
            "SELECT type_str FROM events WHERE id = ?", (distilled["id"],)
        ).fetchone()["type_str"]
        == distilled["type_str"]
    ), "distillation re-derived"
    assert (
        conn.execute(
            "SELECT status FROM slices WHERE id = ?", (a_slice["id"],)
        ).fetchone()["status"]
        == "discarded"
    ), "the slice stage was not re-derived: its code did not change"
    assert (
        conn.execute(
            "SELECT timestamp FROM events WHERE id = ?", (event["id"],)
        ).fetchone()["timestamp"]
        == event["timestamp"] + 12345
    ), "the timestamp stage was not re-derived either"
    conn.close()

    monkeypatch.setattr(
        derive_mod,
        "stage_vintages",
        lambda definition: {
            **installed,
            "slices": "a new slicing",
            "distill": "a new distill",
        },
    )
    checks = run_doctor(capsys)
    assert checks["derivations"]["healed"]["re_derived"].endswith(
        "on — slices, rescue, distillation, sessions"
    )
    conn = connect(db)
    assert (
        conn.execute(
            "SELECT status FROM slices WHERE id = ?", (a_slice["id"],)
        ).fetchone()["status"]
        == "replayable"
    ), "the slice stage re-derived"
    assert (
        conn.execute(
            "SELECT timestamp FROM events WHERE id = ?", (event["id"],)
        ).fetchone()["timestamp"]
        == event["timestamp"] + 12345
    ), "timestamps stand: their stage's code did not change"
    assert derive_mod.stale_stage(conn, definition) is None
    conn.close()


def test_a_definition_edit_re_derives_sessions_alone(tmp_path, capsys, monkeypatch):
    """The operator's numbers are part of the sessions stage's identity: an edit to definitions.toml is staleness that heals from that stage and no earlier one, and a comment edit is not."""
    from locus.evidence import derive as derive_mod

    healthy_surroundings(tmp_path, monkeypatch)
    db = _hydrated_db(tmp_path, derived=True)
    toml = tmp_path / "config" / "definitions.toml"
    before = toml.read_text()
    conn = connect(db)
    sessions_before = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert sessions_before > 0
    stamped_slices = conn.execute(
        "SELECT value FROM meta WHERE key = 'vintage_slices'"
    ).fetchone()[0]
    conn.close()

    toml.write_text("# reworded\n" + before)
    assert run_doctor(capsys)["derivations"]["stale_vintage"] is False

    toml.write_text(before.replace("inactivity_minutes = 30", "inactivity_minutes = 0"))
    checks = run_doctor(capsys)
    assert checks["derivations"]["found"]["stale_vintage"] is True
    assert checks["derivations"]["healed"]["re_derived"].endswith(
        "from the sessions stage on — sessions"
    )
    conn = connect(db)
    assert (
        conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] > sessions_before
    ), "a zero gap makes every act its own session"
    assert (
        conn.execute("SELECT value FROM meta WHERE key = 'vintage_slices'").fetchone()[
            0
        ]
        == stamped_slices
    )
    assert derive_mod.stale_stage(conn, definition_at(tmp_path)) is None
    conn.close()


def test_doctor_holds_the_db_to_the_horizon_and_says_what_went(
    tmp_path, capsys, monkeypatch
):
    """A deployment that is only ever read still honours the horizon: doctor makes the same drop a load does, and reports it — a recording deleted is never a silent housekeeping step."""
    from locus.evidence.retention import CONFIG_FILE

    healthy_surroundings(tmp_path, monkeypatch)
    db = _hydrated_db(tmp_path, derived=True)
    conn = connect(db)
    before = conn.execute("SELECT COUNT(*) FROM slices").fetchone()[0]
    conn.close()
    assert before

    checks = run_doctor(capsys)
    assert checks["retention"]["expired"]["slices"] == 0, (
        "the shipped-corpus horizon reaches every recording in it"
    )
    assert checks["retention"]["horizon_days"] >= 1

    (tmp_path / CONFIG_FILE).write_text("retention_days = 1\n")
    checks = run_doctor(capsys)
    assert checks["retention"]["expired"]["slices"] == before
    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    conn.close()

    (tmp_path / CONFIG_FILE).write_text("retention_days = 0\n")
    checks = run_doctor(capsys, clean=False)
    assert "at least" in checks["retention"]["finding"]
