"""Retention — the declared horizon over the local copy of a recording."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _support import (
    VISITOR,
    FakeStore,
    chunk_blob,
    click,
    counted,
    full_snapshot,
    meta,
)
from locus.evidence.chunk import slice_date
from locus.evidence.db import connect
from locus.evidence.derive import run_distill
from locus.evidence.load import LoadStats, load_chunks
from locus.evidence.retention import (
    CONFIG_FILE,
    cutoff_date,
    expire,
    expired_key,
    horizon_cutoff,
    horizon_days,
)
from locus.evidence.sessions import materialize_sessions
from locus.evidence.slices import materialize_slices

DEFINITION = {
    "session": {"inactivity_minutes": 30},
    "engaged": {"min_seconds": 10, "min_pageviews": 2},
}
DAY_MS = 86_400_000


def declare(root: Path, text: str) -> Path:
    (root / "store").mkdir(parents=True, exist_ok=True)
    (root / CONFIG_FILE).write_text(text)
    return root


def recording(ms: int, snippet: str = "snip01", visitor: str = VISITOR):
    """One page context's chunk, placed where the store's key layout puts it — the whole intake path a recording of that moment reaches the db through."""
    rid = f"{ms:014d}-aaaa"
    stream = [
        counted(event, seq)
        for seq, event in enumerate(
            [meta(ms), full_snapshot(ms + 1), click(ms + 15_000, 3)], start=1
        )
    ]
    key = f"{snippet}/{slice_date(rid)}/{visitor}/{rid}/{ms}000001.json.gz"
    return key, chunk_blob(rid, stream, visitor=visitor, errors=["a send failed"])


def test_the_shipped_declaration_is_the_one_both_runtimes_read():
    """The horizon Locus expires the db by is read from the same file the deploy converges the bucket by — one number, or the two copies drift."""
    root = Path(__file__).resolve().parents[2]
    assert horizon_days(root) >= 1
    assert horizon_cutoff(root) < datetime.now(timezone.utc).date().isoformat()


@pytest.mark.parametrize(
    "declared, complaint",
    [
        ("retention_days = 0\n", "at least"),
        ("retention_days = 30.5\n", "whole number"),
        ("retention_days = 'thirty'\n", "whole number"),
        ("# nothing declared\n", "whole number"),
    ],
)
def test_a_horizon_that_cannot_delete_by_is_refused_by_name(
    tmp_path, declared, complaint
):
    with pytest.raises(SystemExit, match=complaint):
        horizon_days(declare(tmp_path, declared))
    with pytest.raises(SystemExit, match="restore it"):
        horizon_days(tmp_path / "elsewhere")


def test_the_cutoff_keeps_the_horizon_and_drops_the_day_past_it():
    now = datetime(2026, 8, 27, 3, 0, tzinfo=timezone.utc)
    assert cutoff_date(30, now) == "2026-07-28"
    assert cutoff_date(1, now) == "2026-08-26"


def test_an_object_is_dated_by_the_slice_id_its_key_carries():
    ms = 1_749_600_000_000
    key, _ = recording(ms)
    day = slice_date(f"{ms:014d}-aaaa")
    assert expired_key(key, day) is False, "the horizon's own day is kept"
    assert expired_key(
        key, (datetime.fromisoformat(day) + timedelta(days=1)).date().isoformat()
    )
    assert expired_key("junk/not-a-chunk-key", "2999-01-01") is False, (
        "a key the layout does not describe dates to nothing"
    )


def test_the_load_never_fetches_what_the_horizon_has_dropped(tmp_path):
    """The intake and the sweep share one rule, so an expired object never lands: hydrating it only to delete it again would re-fetch it on every load."""
    now = 1_749_600_000_000
    old, recent = recording(now - 60 * DAY_MS), recording(now)
    store = FakeStore(dict([old, recent]))
    cutoff = slice_date(f"{now - 30 * DAY_MS:014d}-aaaa")

    conn = connect(tmp_path / "events.db")
    found = load_chunks(conn, store, stats=LoadStats(), cutoff=cutoff)

    assert found["expired_chunks"] == 1
    assert [row["key"] for row in conn.execute("SELECT key FROM loaded_chunks")] == [
        recent[0]
    ]
    assert (
        conn.execute("SELECT COUNT(DISTINCT recorder_slice) n FROM events").fetchone()[
            "n"
        ]
        == 1
    )


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
