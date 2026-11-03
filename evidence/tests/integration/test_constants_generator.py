"""The checked-in rrweb constants files are what the generator (distill/extract_rrweb_constants.js) emits from the pinned `@rrweb/types`, checked by running the generator in its check mode."""

import subprocess

from _support import EVIDENCE


def test_generated_constants_are_what_the_generator_emits():
    proc = subprocess.run(
        ["bun", str(EVIDENCE / "distill" / "extract_rrweb_constants.js"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
