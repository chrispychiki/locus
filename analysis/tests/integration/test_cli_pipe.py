from _cli_support import (
    local_default_card,  # noqa: F401 — autouse fixture, registered by import
    offline_count,  # noqa: F401 — autouse fixture, registered by import
    shipped_definition,  # noqa: F401 — autouse fixture, registered by import
)


def test_a_caller_that_took_the_address_and_left_declines_the_echo(tmp_path):
    """`locus <verb> | head -1` is the natural way to take just the address, and the reader is gone before a small say echoes. The write lands nowhere and the run still exits clean — through a real pipe and a real interpreter exit, because the failure modes are the raise in echo and the interpreter's own shutdown flush."""
    import os
    import subprocess
    import sys as _sys
    import textwrap

    script = textwrap.dedent("""
        import sys, time
        from pathlib import Path
        from locus.evidence.speak import speaking
        with speaking(Path(sys.argv[1]), "probe"):
            time.sleep(0.5)
            print("small result line")
    """)
    # bash -c consumes $0 from the first arg after the command string.
    run = subprocess.run(
        [
            "/bin/bash",
            "-c",
            f'"{_sys.executable}" -c "$SCRIPT" "$1" | head -1; exit "${{PIPESTATUS[0]}}"',
            "echo-probe",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "SCRIPT": script},
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.startswith("started "), "the address still reached head"
    assert "BrokenPipeError" not in run.stderr, (
        "neither the echo nor the exit flush complains about the gone reader"
    )
