"""What the CLI suites share: the autouse fixtures that give each test its own deployment, registered by import in each suite, and the helpers that build a deployment's data and read a run's output."""

import json
import subprocess
from pathlib import Path

import pytest
from _analysis_support import horizon_at, install_offline_count
from _support import DISTILL, FIXTURE, read_recording
from locus.evidence.db import connect

# The local card the suites' deployment marks default.
LOCAL_CARD = "qwen3.6-35b-a3b-6bit"


@pytest.fixture(autouse=True)
def offline_count(monkeypatch):
    install_offline_count(monkeypatch)
    # The suite's deployment is the test's tmp_path, never this machine: no
    # banked key may leak in (a real key would let a Gemini card construct
    # and spend), and main()'s env self-serve walks cwd upward, which would
    # re-bank it from the real clone.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("locus.evidence.deployment.env_from_files", dict)


@pytest.fixture(autouse=True)
def local_default_card(monkeypatch, tmp_path):
    """The suite's deployment marks LOCAL_CARD `default = true`, so an analyze that names no model prices free and needs no key; a test of the Gemini declaration moves the mark itself."""
    from _analysis_support import remark_default

    remark_default(monkeypatch, tmp_path, LOCAL_CARD, dirname="suite-cards")


@pytest.fixture(autouse=True)
def shipped_definition(tmp_path):
    """Every derivation runs under a session definition and every load under a retention horizon, so the test deployment root carries the shipped definition and a horizon reaching the suite's recordings; a test of either declaration overwrites it."""
    definitions_into(tmp_path)
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


def inline_echo(out: str) -> str:
    """What a run said inline past its opening line — a small say whole, or nothing."""
    _, _, echoed = out.partition("\n")
    return echoed


def output(name: str, payload: dict) -> Path:
    """One run of a verb, said into its own log."""
    from locus.analysis.cli import _speaking

    with _speaking(name) as path:
        print(json.dumps(payload))
    return path


def loaded_fixture_db(tmp_path, monkeypatch):
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


def definitions_into(root: Path, text: str | None = None) -> None:
    """The repo's shipped definitions.toml under a test deployment root, or a declaration of the test's own."""
    src = Path(__file__).parents[2] / "config" / "definitions.toml"
    (root / "config").mkdir(exist_ok=True)
    (root / "config" / "definitions.toml").write_text(
        text if text is not None else src.read_text()
    )
