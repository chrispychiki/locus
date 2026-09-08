"""Conftest for the live suite — the one that means the real deployment.

Collected only when this directory is named on the command line (`pytest tests_live`);
the hermetic suite's `testpaths = ["tests"]` never picks it up. Missing Cloudflare
credentials or a bucket declaration is a loud error naming the setup, never a skip:
anyone who invoked this suite asked for the live check.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib
from locus.evidence.deployment import env_from_files
from locus.evidence.slices import require_bun

EVIDENCE = Path(__file__).resolve().parents[1]
DEPLOYMENT = EVIDENCE.parent
RECORDER = DEPLOYMENT / "recorder"
# The shared browser helpers live in tests/; putting that directory on the
# import path lets live tests `from _harness import ...` the same way.
sys.path.insert(0, str(EVIDENCE / "tests"))


def _require_configured() -> None:
    """Fail loud when this clone is not a configured deployment.

    Reads without exporting: configure must not change the environment the tests
    then run in. The worker URL is not probed here — it is derived from these
    same credentials at test time.
    """
    declared = {**env_from_files(), **os.environ}
    problems = []
    missing = [
        k
        for k in ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID")
        if not declared.get(k)
    ]
    if missing:
        problems.append(
            f"missing {', '.join(missing)} — Cloudflare creds in store/.env"
        )
    try:
        wrangler = tomllib.loads((DEPLOYMENT / "store" / "wrangler.toml").read_text())
        if not (wrangler.get("r2_buckets") or [{}])[0].get("bucket_name"):
            problems.append("no r2_buckets[0].bucket_name in store/wrangler.toml")
    except OSError as exc:
        problems.append(f"cannot read store/wrangler.toml: {exc}")
    if problems:
        raise pytest.UsageError(
            "live suite requires a configured deployment: " + "; ".join(problems)
        )


def pytest_configure(config):
    """Once-only setup: refuse an unconfigured clone, then build the recorder
    library the drive serves. Under xdist only the controller runs this; workers
    inherit the built bundles on disk.
    """
    if hasattr(config, "workerinput"):
        return
    _require_configured()
    require_bun()
    if not (RECORDER / "node_modules").is_dir():
        subprocess.run(
            ["bun", "install", "--frozen-lockfile"], cwd=RECORDER, check=True
        )
    subprocess.run(
        ["bun", "run", "build:lib"], cwd=RECORDER, check=True, capture_output=True
    )


@pytest.fixture(scope="session")
def recorder_dist():
    """The recorder's built library bundle, ready to serve."""
    return RECORDER / "dist"


@pytest.fixture(autouse=True)
def at_deployment():
    """Every live test runs from the clone's deployment root, with its own
    environment restored after.

    Locus resolves the store and credentials by walking cwd upward, so the
    real store/wrangler.toml and store/.env are found exactly as the CLI finds
    them. os.environ is snapshotted and restored so what a test exports — or
    what deployment self-serve setdefaults — dies with it.
    """
    saved_cwd = Path.cwd()
    saved_env = dict(os.environ)
    os.chdir(DEPLOYMENT)
    try:
        yield
    finally:
        os.chdir(saved_cwd)
        os.environ.clear()
        os.environ.update(saved_env)
