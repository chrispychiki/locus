"""The local MLX-VLM server (OpenAI /v1/chat/completions) the analysis package's local cards call.

Its runtime dependencies are Apple-silicon-only and ride the `mlx` optional extra, so an install without them still carries the paid path. Every entry point calls require_extra() before touching them, so a missing extra names what to install.
"""

import importlib

EXTRA_HINT = (
    "install it with `uv sync --extra mlx` (macOS on Apple silicon only — "
    "the extra's dependencies are all sys_platform == 'darwin')"
)


def require_extra(*modules: str) -> None:
    """Refuse loudly when the `mlx` extra is not installed, naming what is missing and how to get it."""
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise SystemExit(
                f"the local mlx server needs {name!r}, which is not installed — {EXTRA_HINT}"
            ) from exc
