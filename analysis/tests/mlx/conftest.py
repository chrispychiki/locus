"""Collection gate for the bundled server's own suite.

Its subject imports mlx, mlx-vlm and the rest of the `mlx` extra at module scope, so without the
extra installed this subtree cannot even be collected. Refusing collection here names the missing
extra. The reason rides a warning: collection runs in the xdist workers, whose stdout the run
discards while their warnings are relayed into the terminal summary.
"""

import importlib.util
import warnings

REQUIRED = ("mlx", "mlx_vlm")
_missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]

collect_ignore_glob = []
if _missing:
    warnings.warn(
        f"analysis/tests/mlx not collected: {', '.join(_missing)} not installed — "
        "install the extra with `uv sync --extra mlx` (macOS on Apple silicon only)",
        stacklevel=1,
    )
    collect_ignore_glob = ["*.py"]


if not _missing:
    from pathlib import Path

    import pytest

    @pytest.fixture(autouse=True)
    def shipped_serve_cards(monkeypatch):
        """The server reads its card from config/cards/ under the deployment root, resolved from cwd — and the suite runs chdir'd off the clone, so pin the read to the repo's shipped roster."""
        from locus.analysis.mlx import serve

        root = Path(__file__).resolve().parents[3]
        monkeypatch.setattr(
            serve, "card_file", lambda slug: root / "config" / "cards" / f"{slug}.toml"
        )
