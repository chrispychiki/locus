"""Retention — the declared horizon over the local copy of a recording."""

from _support import (
    DAY_MS,
    DEFINITION,
    VISITOR,
    FakeStore,
    recording,
)
from locus.evidence.chunk import slice_date
from locus.evidence.db import connect
from locus.evidence.derive import run_distill
from locus.evidence.load import LoadStats, load_chunks
from locus.evidence.retention import (
    expire,
)
from locus.evidence.sessions import materialize_sessions
from locus.evidence.slices import materialize_slices


def test_a_recording_past_the_horizon_leaves_with_everything_derived_from_it(tmp_path):
    """A slice goes whole — its events, its error log, its manifest rows, its sessions and the replay cache — and what the horizon still holds is untouched and still current."""
    now = 1_749_600_000_000
    old, recent = recording(now - 60 * DAY_MS), recording(now)
    store = FakeStore(dict([old, recent]))
    db = str(tmp_path / "events.db")
    conn = connect(db)
    load_chunks(conn, store, stats=LoadStats())
    materialize_slices(conn, VISITOR)
    run_distill(db)
    materialize_sessions(conn, VISITOR, DEFINITION)
    assert conn.execute("SELECT COUNT(*) n FROM sessions").fetchone()["n"] == 2
    assert conn.execute("SELECT COUNT(*) n FROM chunk_errors").fetchone()["n"] == 2

    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "locus-replay.js").write_text("cached material")

    cutoff = slice_date(f"{now - 30 * DAY_MS:014d}-aaaa")
    dropped = expire(conn, cutoff, DEFINITION, pages)

    assert (dropped["slices"], dropped["visitors"], dropped["chunks"]) == (1, 1, 1)
    assert dropped["events"] == 3 and dropped["pages_cleared"]
    assert list(pages.iterdir()) == []
    surviving = conn.execute("SELECT recorder_slice FROM slices").fetchall()
    assert [row["recorder_slice"] for row in surviving] == [f"{now:014d}-aaaa"]
    assert [row["key"] for row in conn.execute("SELECT key FROM loaded_chunks")] == [
        recent[0]
    ]
    assert conn.execute("SELECT COUNT(*) n FROM chunk_errors").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) n FROM sessions").fetchone()["n"] == 1, (
        "the surviving visitor's sessions are re-derived from what is left"
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) n FROM events WHERE session_id IS NULL"
        ).fetchone()["n"]
        == 0
    ), "no event is left pointing at a session that no longer exists"

    again = expire(conn, cutoff, DEFINITION, pages)
    assert (again["slices"], again["events"], again["chunks"]) == (0, 0, 0)


def test_a_visitor_whose_every_recording_expired_leaves_nothing_behind(tmp_path):
    now = 1_749_600_000_000
    db = str(tmp_path / "events.db")
    conn = connect(db)
    load_chunks(
        conn, FakeStore(dict([recording(now - 60 * DAY_MS)])), stats=LoadStats()
    )
    materialize_slices(conn, VISITOR)
    run_distill(db)
    materialize_sessions(conn, VISITOR, DEFINITION)

    expire(conn, slice_date(f"{now:014d}-aaaa"), DEFINITION)

    for table in ("events", "slices", "sessions", "loaded_chunks", "chunk_errors"):
        assert conn.execute(f"SELECT COUNT(*) n FROM {table}").fetchone()["n"] == 0
