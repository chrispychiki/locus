import sys
import urllib.request
from pathlib import Path

import pytest

# The event builders, db builders, and the fixtures both suites run under live once, in the
# evidence suite's support module; this suite depends on them and reaches them by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evidence" / "tests"))

from _analysis_support import spend_isolation  # noqa: F401 — autouse fixture
from _support import (
    contained,  # noqa: F401 — autouse fixture, registered by import
    distilled_db,  # noqa: F401
    two_visitor_db,  # noqa: F401
)


@pytest.fixture(autouse=True)
def shipped_card_roster(monkeypatch):
    """Tests chdir off the clone freely, and the card declarations resolve from the deployment root at cwd — pin them to the repo's shipped ones so a moved cwd never strands a card lookup.
    A test of the declarations themselves overrides with its own roster (card_roster in _analysis_support)."""
    from locus.analysis.model import cards

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(cards, "cards_dir", lambda: root / "config" / "cards")


def _mlx_ready() -> bool:
    """Is the local server up? Asked at the address the backend actually dials, read from the card — the card is the one home for the endpoint, so a second copy here could go stale and skip (or run) the model-gated tests against a server that is not the one analysis talks to. An address that will not answer is the not-running verdict; anything else — a card whose base_url is not a URL — is a broken configuration and fails loud."""
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


_GATES = {
    "needs_mlx": (
        _mlx_ready,
        (
            "the bundled mlx server is not running — boot it with "
            "`uv run locus-mlx-serve qwen3.6-35b-a3b-6bit` (free, local weights)"
        ),
    ),
}


def pytest_collection_modifyitems(config, items):
    ready = {name: gate() for name, (gate, _) in _GATES.items()}
    for item in items:
        for name, (_, reason) in _GATES.items():
            if name in item.keywords and not ready[name]:
                item.add_marker(pytest.mark.skip(reason=reason))
