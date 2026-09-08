"""The engine run against a real local model — the one place the engine drives a
real model and a real browser in the suite, nothing stubbed: turn one with the
activity-screenshots and the caller's own question, an optional pull turn where the model
requests moments the renderer produces fresh, then the timestamp-cited answer,
its citations validated mechanically with a repair turn only on violation.
Measurement (the oracle) is a separate concern, so this verifies the engine's own
output — a cited answer, its screenshots, its provenance — and stops there.

Gated on the bundled mlx server (free, local weights, screenshots handed over as file
paths); the identical wiring runs on Gemini but spends money, so the free local run
is the one in the suite. Server down → the test skips with the boot command.
"""

import re
from pathlib import Path

import pytest
from _support import best_slice
from locus.analysis import engine
from locus.evidence.db import connect

MLX_MODEL = "qwen3.6-35b-a3b-6bit"
CONTEXT = "a demo session recorded for the test fixture."
QUESTION = "What was this visitor doing in this session, and did anything visibly go wrong for them? Cite the moments."


@pytest.mark.needs_mlx
def test_engine_against_local_model(distilled_db, tmp_path):
    conn = connect(distilled_db)
    slice_id = best_slice(conn)
    out = tmp_path / "run"

    from locus.analysis.window import resolve_window

    manifest = engine.run_analysis(
        conn,
        resolve_window(conn, [slice_id]),
        question=QUESTION,
        model=MLX_MODEL,
        site_contexts={"site": CONTEXT},
        out_dir=engine.home_analysis(out),
    )

    assert manifest["n_activity_screenshots"] >= 1, (
        "screenshots were rendered and pushed"
    )
    assert manifest["n_pulled_screenshots"] >= 0
    for screenshot, faults in manifest["render_faults"].items():
        assert faults and (out / "screenshots" / screenshot).exists(), (
            "a fault entry names a real screenshot file and says what failed"
        )

    answer = Path(manifest["response_path"]).read_text()
    assert re.search(r"\[\d{2}:\d{2}\.\d{3}", answer), (
        "the answer carries timestamp citations a replay page can open"
    )
    assert (out / "1_answer_input.txt").exists(), (
        "the transcript is the analysis's record, at its root"
    )
    assert all(not isinstance(v, (bytes, bytearray)) for v in manifest.values()), (
        "the manifest is paths and counts — never image bytes"
    )


@pytest.mark.needs_mlx
def test_engine_multi_slice_window(distilled_db, tmp_path):
    """A multi-slice window through a real model: two labeled slices
    on one shared clock, every pushed and pulled screenshot addressed by label, one
    answer out, citing by slice label and offset."""
    conn = connect(distilled_db)
    slice_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM slices WHERE status='replayable' ORDER BY start_ts LIMIT 2"
        ).fetchall()
    ]
    assert len(slice_ids) == 2, "the fixture carries a multi-slice visitor"
    out = tmp_path / "run"

    from locus.analysis.window import resolve_window

    manifest = engine.run_analysis(
        conn,
        resolve_window(conn, slice_ids),
        question=QUESTION,
        model=MLX_MODEL,
        site_contexts={"site": CONTEXT},
        out_dir=engine.home_analysis(out),
    )

    assert [s["slice_id"] for s in manifest["slices"]] == slice_ids
    assert manifest["n_activity_screenshots"] >= len(slice_ids), (
        "the pushed set spans both slices"
    )
    assert [s["label"] for s in manifest["slices"]] == ["S1", "S2"]
    answer = Path(manifest["response_path"]).read_text()
    assert answer.strip(), "the engine wrote an answer over the multi-slice window"
    assert re.search(r"\[S\d+ \d{2}:\d{2}\.\d{3}", answer), (
        "cited by slice label and offset"
    )
