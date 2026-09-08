"""Where the server writes: the pidfile the boot guard reads, and the serve and memory logs.

The pidfile guards one resident model per machine, so it lives at one machine-level path (locus.evidence.deployment.machine_cache_dir) that every clone's boot guard reads — a per-clone pidfile would let two clones each boot a resident model past each other, the state this machine does not recover from.
The logs are state the deployment accretes, so they live under its data/ directory (deployment.py owns the name), resolved from cwd at the moment of use like every accreted family — the server opens its log at boot, so a boot outside a clone still fails loud there, and importing this module depends on no cwd.
"""

from pathlib import Path

from locus.evidence.deployment import DATA_DIR, deployment_root, machine_cache_dir

PIDFILE = machine_cache_dir() / "mlx-server.pid"


def mlx_data() -> Path:
    return deployment_root() / DATA_DIR / "mlx"


def serve_log() -> Path:
    return mlx_data() / "serve.log"


def memory_log() -> Path:
    return mlx_data() / "memory.jsonl"


def rendered_prompt() -> Path:
    """The last request's rendered prompt, as the server handed it to the tokenizer (serve.py's install_prompt_recording writes it, latest request winning)."""
    return mlx_data() / "rendered_prompt.txt"


# The telemetry samples at 1Hz for the life of every request, on a server that stays resident for days.
# The log rolls at this size, one generation back.
MEMORY_LOG_MAX_BYTES = 64 * 1024**2
