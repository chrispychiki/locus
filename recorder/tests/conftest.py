import sys
from pathlib import Path

# The event builders, harness, and shared fixtures live once, in the evidence
# suite's support modules; this suite depends on them and reaches them by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evidence" / "tests"))

from _support import (
    build_recorder,
    contained,  # noqa: F401 — autouse fixture, registered by import
    recorder_dist,  # noqa: F401
)


def pytest_configure(config):
    """The suite's once-only work, run in the one process that runs once: building the recorder writes into a shared directory under the clone, and anything session-scoped is per-worker under xdist, so the controller — the process whose config carries no `workerinput`, which a plain undistributed run also lacks — does it before a single worker spawns.

    The script-tag bundle is what the snippet-autostart suite serves; the library bundle is what the recorder-integration suite serves."""
    if hasattr(config, "workerinput"):
        return
    build_recorder("build", "build:lib")
