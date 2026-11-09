"""Collection gate for the bundled server's unit suite.

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
        f"analysis/tests/unit/mlx not collected: {', '.join(_missing)} not installed — "
        "install the extra with `uv sync --extra mlx` (macOS on Apple silicon only)",
        stacklevel=1,
    )
    collect_ignore_glob = ["*.py"]
