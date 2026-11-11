"""Retention — the declared horizon over the local copy of a recording."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _support import (
    DAY_MS,
    DEFINITION,
    VISITOR,
    FakeStore,
    recording,
)
from locus.evidence.chunk import slice_date
from locus.evidence.db import connect
from locus.evidence.load import LoadStats, load_chunks
from locus.evidence.retention import (
    CONFIG_FILE,
    cutoff_date,
    expire,
    expired_key,
    horizon_cutoff,
    horizon_days,
)
from locus.evidence.slices import materialize_slices


def declare(root: Path, text: str) -> Path:
    (root / "store").mkdir(parents=True, exist_ok=True)
    (root / CONFIG_FILE).write_text(text)
    return root


def test_the_shipped_declaration_is_the_one_both_runtimes_read():
    """The horizon Locus expires the db by is read from the same file the deploy converges the bucket by — one number, or the two copies drift."""
    root = Path(__file__).resolve().parents[3]
    assert horizon_days(root) >= 1
    assert horizon_cutoff(root) < datetime.now(UTC).date().isoformat()


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
    now = datetime(2026, 8, 27, 3, 0, tzinfo=UTC)
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


def test_what_the_sweep_drops_leaves_the_disk_too(tmp_path):
    """The db is a cache in front of the store: a recording the horizon takes is gone from the file's size, not kept as free pages in a file that never shrinks."""
    now = 1_749_600_000_000
    old = [
        recording(now - 60 * DAY_MS - n * 1000, visitor=f"v{n:03d}") for n in range(60)
    ]
    db = tmp_path / "events.db"
    conn = connect(db)
    assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2, (
        "a db is born able to give space back"
    )
    load_chunks(conn, FakeStore(dict([*old, recording(now)])), stats=LoadStats())
    for n in range(60):
        materialize_slices(conn, f"v{n:03d}")
    materialize_slices(conn, VISITOR)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = db.stat().st_size

    dropped = expire(conn, slice_date(f"{now - 30 * DAY_MS:014d}-aaaa"), DEFINITION)

    assert dropped["slices"] == 60
    assert dropped["reclaimed_bytes"] > 0
    assert conn.execute("PRAGMA freelist_count").fetchone()[0] == 0
    assert db.stat().st_size == before - dropped["reclaimed_bytes"]
