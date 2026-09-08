from pathlib import Path

from _support import (
    bun_install,
    contained,  # noqa: F401 — autouse fixture, registered by import
)

EVIDENCE = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    """The suite's once-only work, run in the one process that runs once.

    Installing bun packages writes into a shared directory under the clone, and the suites that
    shell into it read it right back. Anything session-scoped is per-worker under xdist, so N
    workers would do this N times over the same paths and tear each other's reads. The xdist
    controller — the process whose config carries no `workerinput` — does it before a single
    worker spawns, and a plain undistributed run has no `workerinput` either, so this is exactly
    once either way.
    """
    if hasattr(config, "workerinput"):
        return
    # distill/'s npm packages: the constants generator resolves @rrweb/types and rrweb out of them.
    bun_install(EVIDENCE / "distill")
