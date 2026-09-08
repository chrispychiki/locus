"""The timestamp re-derivation decodes every row's raw_json, so it is cut into id ranges and run across cores; the split must change nothing about what moves."""

from pathlib import Path

from _support import read_recording
from locus.evidence import derive as derive_mod
from locus.evidence.db import connect
from locus.evidence.hydrate import hydrate

FIXTURE = Path(__file__).parent / "fixtures" / "demo_recording.json"


def _corrupted_db(tmp_path):
    db = str(tmp_path / "events.db")
    visitor_id, events = read_recording(FIXTURE)
    conn = connect(db)
    hydrate(conn, visitor_id, events)
    before = {
        row["id"]: row["timestamp"]
        for row in conn.execute("SELECT id, timestamp FROM events")
    }
    conn.execute("UPDATE events SET timestamp = timestamp + 777 WHERE id % 3 = 0")
    conn.commit()
    return db, conn, before


def test_timestamps_re_derive_the_same_across_ranges_as_in_one(tmp_path, monkeypatch):
    db, conn, before = _corrupted_db(tmp_path)
    moved_in_one = derive_mod.rederive_timestamps(conn, db)
    assert moved_in_one == sum(1 for row_id in before if row_id % 3 == 0)
    assert {
        row["id"]: row["timestamp"]
        for row in conn.execute("SELECT id, timestamp FROM events")
    } == before

    conn.execute("UPDATE events SET timestamp = timestamp + 777 WHERE id % 3 = 0")
    conn.commit()
    monkeypatch.setattr(derive_mod, "TIMESTAMP_RANGE", 7)
    ranges = []
    moved_across = derive_mod.rederive_timestamps(
        conn, db, on_range=lambda: ranges.append(1)
    )
    assert moved_across == moved_in_one
    assert len(ranges) > 1, "the split actually ran as several ranges"
    assert {
        row["id"]: row["timestamp"]
        for row in conn.execute("SELECT id, timestamp FROM events")
    } == before


def test_one_derivation_runs_per_db_and_a_second_waits_for_it(tmp_path, monkeypatch):
    """A caller that reaches the derivation while another process holds it waits instead of racing it, and runs once the holder is done."""
    import fcntl
    import threading
    import time

    db = str(tmp_path / "events.db")
    conn = connect(db)
    monkeypatch.setattr(
        derive_mod, "_finish_derivations", lambda conn, db, definition: {"ran": True}
    )
    result = {}

    def contender():
        result["value"] = derive_mod.finish_derivations(conn, db, {})

    with open(f"{db}.derive.lock", "w") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        thread = threading.Thread(target=contender)
        thread.start()
        time.sleep(0.3)
        assert thread.is_alive() and "value" not in result
        fcntl.flock(holder, fcntl.LOCK_UN)
        thread.join(timeout=5)
    assert result == {"value": {"ran": True}}
