"""One policy for every upstream-keyed patch: verify the source you stand in for, or refuse.

Each patch in this package rebinds a function inside the installed mlx-vlm and is correct only against the exact upstream source it was validated on. A mismatch means the installed mlx-vlm is one nobody here has judged — and the patch points are exactly the places judged critical enough to patch — so the server does not guess: it refuses to boot, names what drifted, and waits for the patch to be re-validated against the new source and re-pinned. The client-side count (count.py), which follows the steps of upstream's chat handler, is keyed to that handler the same way and refuses to count.

The lockfile resolves the mlx extra to the pinned upstream, so a mismatch is only ever reachable through a deliberate dependency upgrade; the refusal makes that upgrade a re-validation event instead of a silent behavior swap.
"""

import hashlib
import inspect


def refuse(patch: str, what: str, detail: str, owes: str) -> None:
    raise SystemExit(
        f"{patch}: upstream drift — {what} {detail} (mlx-vlm changed). This code was "
        f"validated against the exact pinned source, so it refuses rather than run "
        f"unjudged behavior at a pinned point. {owes}"
    )


def expect_source(patch: str, what: str, fn, expected_sha256: str, owes: str) -> None:
    """Refuse unless fn's source hashes to the pinned value."""
    actual = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    if actual != expected_sha256:
        refuse(
            patch,
            what,
            f"source hash {actual[:12]} != pinned {expected_sha256[:12]}",
            owes,
        )
