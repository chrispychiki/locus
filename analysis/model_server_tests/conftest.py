"""The suite that needs the running model server: the bundled mlx server at the card's endpoint.

It runs only when this directory is named: `uv run pytest model_server_tests` from analysis/. Every test skips with a reason when mlx cannot run on the machine (it needs Apple silicon under macOS) or when the server is not up.
"""

import platform
import sys
import urllib.request
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parents[1]
# The analysis and evidence support modules, where the fixtures these tests run under live.
sys.path.insert(0, str(ANALYSIS / "tests"))
sys.path.insert(0, str(ANALYSIS.parent / "evidence" / "tests"))

from _analysis_support import (
    shipped_card_roster,  # noqa: F401 — autouse fixture
    spend_isolation,  # noqa: F401 — autouse fixture
)
from _support import (
    contained,  # noqa: F401 — autouse fixture, registered by import
    distilled_db,  # noqa: F401
)


def _unservable_platform() -> str | None:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return None
    return f"mlx runs only on Apple silicon under macOS; this machine is {platform.system()} on {platform.machine()}"


def _mlx_ready() -> bool:
    """Is the local server up? Asked at the address the backend actually dials, read from the card — the card is the one home for the endpoint, so a second copy here could go stale and skip (or run) these tests against a server that is not the one analysis talks to. An address that will not answer is the not-running verdict; anything else — a card whose base_url is not a URL — is a broken configuration and fails loud."""
    from locus.analysis.model.cards import OPENAI_COMPATIBLE, load_card, models_speaking

    models = models_speaking(OPENAI_COMPATIBLE)
    if not models:
        return False
    base = load_card(models[0])["base_url"]
    health = base.removesuffix("/v1") + "/health"
    try:
        with urllib.request.urlopen(health, timeout=2) as resp:
            return b"loaded_model" in resp.read()
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    reason = _unservable_platform()
    if reason is None and not _mlx_ready():
        reason = (
            "the bundled mlx server is not running — boot it with "
            "`uv run locus-mlx-serve qwen3.6-35b-a3b-6bit` (free, local weights)"
        )
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if Path(item.path).is_relative_to(Path(__file__).parent):
            item.add_marker(skip)
