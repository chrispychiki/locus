"""The suite that needs the running model server: the bundled mlx server at the local cards' endpoint.

It runs only when this directory is named: `uv run pytest model_server_tests` from analysis/. Every test runs against whichever local model the server holds, and skips with a reason when mlx cannot run on the machine (it needs Apple silicon under macOS) or when the server is not up; a test marked `family` runs only while the server holds a model of that family. The server records only its latest request's rendered prompt, so a test marked `reads_rendering` runs while no other test of the suite runs, on any worker.
"""

import fcntl
import json
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


def _served_card() -> str | None:
    """The card whose model the local server holds, or None when no server answers. Asked at the address every local card's base_url names (serve.sole_bind), so these tests run against the server analysis talks to. A server holding a model no card declares fails loud."""
    from locus.analysis.mlx.serve import sole_bind
    from locus.analysis.model.cards import OPENAI_COMPATIBLE, model_id, models_speaking

    host, port = sole_bind()
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as resp:
            loaded = json.loads(resp.read()).get("loaded_model")
    except OSError:
        return None
    if not loaded:
        return None
    for name in models_speaking(OPENAI_COMPATIBLE):
        if model_id(name) == loaded:
            return name
    raise RuntimeError(f"the local server holds {loaded}, which no card declares")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "family(name): runs only while the local server holds a model of that card family",
    )
    config.addinivalue_line(
        "markers",
        "reads_rendering: reads the server's record of the latest request's rendered prompt, "
        "so runs while no other test of the suite does",
    )


@pytest.fixture(autouse=True)
def rendering_record_held(request, tmp_path_factory):
    """Every test holds the lock shared and a test reading the rendering holds it alone, so no request of another test lands between its own request and its read. The lock file is in the parent of each worker's temp directory, which all workers share."""
    alone = request.node.get_closest_marker("reads_rendering") is not None
    with open(tmp_path_factory.getbasetemp().parent / "rendering.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if alone else fcntl.LOCK_SH)
        yield


@pytest.fixture(scope="session")
def served_card() -> str:
    """The name of the card whose model the local server holds."""
    return _served_card()


def pytest_collection_modifyitems(config, items):
    from locus.analysis.model.cards import OPENAI_COMPATIBLE, load_card, models_speaking

    here = [i for i in items if Path(i.path).is_relative_to(Path(__file__).parent)]
    reason = _unservable_platform()
    served = None
    if reason is None:
        served = _served_card()
        if served is None:
            reason = (
                "the bundled mlx server is not running — boot it on a local card "
                f"({', '.join(models_speaking(OPENAI_COMPATIBLE))}) with "
                "`uv run locus-mlx-serve <card name>` (free, local weights)"
            )
    for item in here:
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
            continue
        family = item.get_closest_marker("family")
        if family is not None and family.args[0] != load_card(served)["family"]:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"a {family.args[0]} test; the server holds {served}"
                )
            )
