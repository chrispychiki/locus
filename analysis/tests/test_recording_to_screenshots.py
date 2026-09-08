"""Hydrate → materialize slices → distill → select, run once over a real recording.

The seam, not the parts: each stage is pinned in its own suite, and this asserts they compose — a
recording goes in and a slice set with projections and screenshot moments comes out.
"""

import subprocess

from _support import (
    DISTILL,
    FIXTURE,
    read_recording,
)
from locus.analysis.select_screenshots import select_screenshots
from locus.evidence.db import connect
from locus.evidence.hydrate import hydrate
from locus.evidence.slices import materialize_slices


def test_recording_to_screenshots(tmp_path):
    visitor_id, events = read_recording(FIXTURE)
    db_path = tmp_path / "events.db"
    conn = connect(db_path)
    hydrate(conn, visitor_id, events)
    summary = materialize_slices(conn, visitor_id)
    assert summary["replayable"] == 4
    conn.close()

    subprocess.run(["bun", str(DISTILL), str(db_path)], check=True, capture_output=True)

    conn = connect(db_path)
    n_md = conn.execute(
        "SELECT COUNT(*) n FROM events WHERE md IS NOT NULL"
    ).fetchone()["n"]
    assert n_md == 4

    clicks = conn.execute(
        "SELECT COUNT(*) n FROM events WHERE type_str='Click' AND tag IS NOT NULL"
    ).fetchone()["n"]
    assert clicks > 0

    total_screenshots = 0
    for (slice_id,) in conn.execute("SELECT id FROM slices WHERE status='replayable'"):
        selection = select_screenshots(conn, slice_id)
        assert selection.timestamps
        assert selection.timestamps == sorted(set(selection.timestamps))
        total_screenshots += len(selection.timestamps)
    assert total_screenshots > 4
