"""One policy for every upstream-keyed patch: verify the source you stand in for, or refuse to boot.

Each patch in this package rebinds a function inside the installed mlx-vlm and is correct only against the exact upstream source it was validated on. A mismatch means the installed mlx-vlm is one nobody here has judged — and the patch points are exactly the places judged critical enough to patch — so the server does not guess: it refuses to boot, names what drifted, and waits for the patch to be re-validated against the new source and re-pinned.

The lockfile resolves the mlx extra to the pinned upstream, so a mismatch is only ever reachable through a deliberate dependency upgrade; the refusal makes that upgrade a re-validation event instead of a silent behavior swap.
"""

import hashlib
import inspect


def refuse(patch: str, what: str, detail: str, owes: str) -> None:
    raise SystemExit(
        f"{patch}: upstream drift — {what} {detail} (mlx-vlm changed). The patch was validated "
        f"against the exact pinned source, so the server refuses to boot rather than run "
        f"unjudged behavior at a patched point. {owes}"
    )


def expect_source(patch: str, what: str, fn, expected_sha256: str, owes: str) -> None:
    """Refuse to boot unless fn's source hashes to the pinned value."""
    actual = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    if actual != expected_sha256:
        refuse(
            patch,
            what,
            f"source hash {actual[:12]} != pinned {expected_sha256[:12]}",
            owes,
        )
