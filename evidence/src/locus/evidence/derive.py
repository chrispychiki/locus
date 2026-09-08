"""Derivation — the path from hydrated raw events to the queryable derived surfaces, and the knowledge of when those surfaces are current.

Three planes of fact live in the db, and only one is this module's:
- **Identity** — raw_json and content_hash. raw_json is the source of truth; content_hash is the dedup identity every load leans on. Their canonical form is frozen (hydrate.py): a change there is not staleness this module can heal — it breaks dedup identity, and the fix is a rebuild, never a re-derive.
- **Derivations** — everything recomputable from raw_json: the canonical timestamp, the slice assignment and slices table, rescue verdicts, the flat columns and projections, the sessions table and the session stamp. This module owns their currency — a vintage per stage of the deriving code, and one heal that re-runs the path from the first stage whose code changed.
- **Testimony** — the envelope columns and load-time facts (device/os/browser through snippet, uploaded_ms, chunk_errors), captured at load from sources not retained — the userAgent, the object key, the store's LIST — so they are recorded once and never re-derived.

Two verbs drive the one derivation path here: `load` after it hydrates, and `doctor` as the heal for derivations left behind — a load that died partway, or deriving code that changed under the db.

The derivation pass stamps its own vintage: derived values drift silently from the logic that derives them, so a pass that leaves every derived value current records the deriving code's content-hash identity in db meta, and staleness is the recorded identity differing from the installed one.
The path is four stages, each deriving from the one before — timestamps from raw_json, slices from the timestamped stream, distillation from the sliced stream, sessions from the distilled columns — so the identity is kept per stage, and a change is healed from the stage it touched: a change to the session definition re-derives sessions alone, while a change to the canonical timestamp re-derives everything downstream of it.
The sessions stage derives from a declaration as well as from code — the operator's numbers in config/definitions.toml — so its identity carries the declared values too, and an edit there is staleness like any other.
"""

import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from .assets import script
from .speak import beating
from .user_activity import user_activity_clause

# The derivation path in order, each stage named with the sources whose change re-derives it —
# and, because each stage reads the one before, everything after it.
STAGES = ("timestamps", "slices", "distill", "sessions")


def _stage_sources() -> dict[str, list[Path]]:
    here = Path(__file__).parent
    return {
        "timestamps": [here / "hydrate.py"],
        "slices": [here / "slices.py"],
        "distill": [
            path
            for path in sorted(script("distill.js").parent.glob("*.js"))
            if not path.name.endswith(".test.js")
        ],
        "sessions": [here / "sessions.py", here / "user_activity.py"],
    }


def _declared(definition: dict) -> str:
    """The declared numbers the sessions stage derives by, as one canonical string: the session and engaged tables' values, key-sorted, so the identity moves with a value and never with the file's comments or formatting."""
    return json.dumps(
        {"session": definition["session"], "engaged": definition["engaged"]},
        sort_keys=True,
    )


def _digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def stage_vintages(definition: dict) -> dict[str, str]:
    """Each stage's identity: a hash over its own sources — hydrate.py for the canonical timestamp, slices.py for slice grouping and rescue eligibility, every non-test source in the distill package for flattening, projection, and the rescue gate, sessions.py and user_activity.py for the session derivation — so a stage's vintage moves exactly when its deriving logic does, and a new distillation module counts without registration. The sessions stage's identity also carries the operator's declared numbers, so a definition edit moves it the same way a code change does."""
    vintages = {stage: _digest(paths) for stage, paths in _stage_sources().items()}
    if "sessions" in vintages:
        vintages["sessions"] = _digest_strings(
            [vintages["sessions"], _declared(definition)]
        )
    return vintages


def derivation_vintage(definition: dict) -> str:
    """The whole path's identity — the stage vintages in path order, as one digest."""
    vintages = stage_vintages(definition)
    return _digest_strings(vintages[stage] for stage in STAGES if stage in vintages)


def _digest_strings(parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def stamped_vintage(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'derivation_vintage'"
    ).fetchone()
    return row["value"] if row else None


def derivations_stale(conn: sqlite3.Connection, definition: dict) -> bool:
    """Whether the derived surfaces were derived by code — or a declaration — that has since changed. Derived values drift silently from the logic that derives them, so distilled rows whose recorded vintage differs from the installed identity — or that carry no recorded vintage at all — are stale. A db with no distilled rows is not stale, merely underived; the incremental gate owns that state."""
    distilled = conn.execute(
        "SELECT 1 FROM events WHERE type_str IS NOT NULL LIMIT 1"
    ).fetchone()
    return bool(distilled) and stamped_vintage(conn) != derivation_vintage(definition)


def stale_stage(conn: sqlite3.Connection, definition: dict) -> str | None:
    """The first stage whose identity changed since the db's values were derived — where the heal starts, everything after it re-deriving too — or None when the path is current. A db stamped before the stages were recorded separately, or whose whole-path stamp disagrees with its stage stamps, is healed from the top: nothing narrower can claim currency."""
    if not derivations_stale(conn, definition):
        return None
    stamped = {
        row["key"].removeprefix("vintage_"): row["value"]
        for row in conn.execute(
            "SELECT key, value FROM meta WHERE key LIKE 'vintage_%'"
        )
    }
    installed = stage_vintages(definition)
    for stage in STAGES:
        if stage in installed and stamped.get(stage) != installed[stage]:
            return stage
    return STAGES[0]


def _stamp_vintages(conn: sqlite3.Connection, definition: dict) -> None:
    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [
            ("derivation_vintage", derivation_vintage(definition)),
            *(
                (f"vintage_{stage}", vintage)
                for stage, vintage in stage_vintages(definition).items()
            ),
        ],
    )
    conn.commit()


# Rows per timestamp-derivation task: one range is one worker's unit of raw_json decoding.
TIMESTAMP_RANGE = 250_000


def _moved_timestamps(db: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """One id range's rows whose canonical timestamp differs from the stored one, as (timestamp, id) — read off its own read-only connection, so it runs in a worker process."""
    from .hydrate import canonical_ts, raw_event

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [
            (ts, row_id)
            for row_id, stored, raw in conn.execute(
                "SELECT id, timestamp, raw_json FROM events WHERE id >= ? AND id < ?",
                (lo, hi),
            )
            if (ts := canonical_ts(raw_event(raw))) != stored
        ]
    finally:
        conn.close()


def rederive_timestamps(conn: sqlite3.Connection, db: str, on_range=None) -> int:
    """Recompute every event's canonical timestamp from raw_json — the hydrate-side derivation (a move batch lands at its last position's moment) applied in place, for when that logic changed under a standing db. Every row's raw_json is decoded, which is the whole cost, so the id space is cut into ranges and decoded across every core; a db that fits one range is decoded in-process. `on_range`, when given, is called as each range finishes. Returns how many rows moved."""
    bounds = conn.execute("SELECT MIN(id) lo, MAX(id) hi FROM events").fetchone()
    if bounds["lo"] is None:
        return 0
    ranges = [
        (lo, min(lo + TIMESTAMP_RANGE, bounds["hi"] + 1))
        for lo in range(bounds["lo"], bounds["hi"] + 1, TIMESTAMP_RANGE)
    ]
    changed: list[tuple[int, int]] = []
    if len(ranges) == 1:
        changed = _moved_timestamps(db, *ranges[0])
        if on_range is not None:
            on_range()
    else:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            tasks = [pool.submit(_moved_timestamps, db, lo, hi) for lo, hi in ranges]
            for task in as_completed(tasks):
                changed.extend(task.result())
                if on_range is not None:
                    on_range()
    if changed:
        conn.executemany("UPDATE events SET timestamp = ? WHERE id = ?", changed)
        conn.commit()
    return len(changed)


def _slice_every(conn: sqlite3.Connection, visitors: list[str]) -> None:
    """Each visitor's slices, materialized in one pass — minutes of it on a corpus-sized re-derivation, producing nothing until it ends."""
    from .slices import materialize_slices

    done = 0
    with beating("materializing slices", lambda: f"{done}/{len(visitors)} visitors"):
        for visitor_id in visitors:
            materialize_slices(conn, visitor_id)
            done += 1


def _session_every(
    conn: sqlite3.Connection, visitors: list[str], definition: dict
) -> None:
    """Each visitor's sessions, derived in one pass under the definition."""
    from .sessions import materialize_sessions

    done = 0
    with beating("deriving sessions", lambda: f"{done}/{len(visitors)} visitors"):
        for visitor_id in visitors:
            materialize_sessions(conn, visitor_id, definition)
            done += 1


def _every_visitor(conn: sqlite3.Connection) -> list[str]:
    return [
        row["visitor_id"]
        for row in conn.execute("SELECT DISTINCT visitor_id FROM events")
    ]


def run_distill(db: str, full: bool = False) -> None:
    """Run the distillation pass — the mechanism only; finish_derivations owns the vintage stamp, because only the whole path can claim currency. Incremental by default — it distills only visitors with undistilled rows. `full` re-distills every visitor, for when the deriving code itself changed.

    The pass inherits this process's descriptors and writes straight to them. This process's streams flush first, or buffered lines would surface after everything the subprocess said."""
    from .slices import require_bun

    require_bun()
    sys.stdout.flush()
    sys.stderr.flush()
    cmd = ["bun", str(script("distill.js")), db] + (["--all"] if full else [])
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _not_current() -> tuple[str, list]:
    """The rows the derivation path has not brought current, as one UNION over the three signals: distillation never wrote the row (`type_str IS NULL`), materialization never placed it (`slice_id IS NULL`), or it is user activity no session claimed (`session_id IS NULL` — every act is in a session once the sessions pass has run, so an unclaimed one is a pass that has not). Each is its own signal, because each pass can be the one that died — a load killed between hydrating a visitor and materializing its slices leaves rows distillation may later have written against no slice at all, and only the slice signal sees them. Every read comes off an index."""
    acts, params = user_activity_clause()
    sql = (
        "SELECT id, visitor_id FROM events WHERE type_str IS NULL "
        "UNION SELECT id, visitor_id FROM events WHERE slice_id IS NULL "
        "UNION SELECT id, visitor_id FROM events WHERE session_id IS NULL "
        f"AND ({acts})"
    )
    return sql, params


def dirty_visitors(conn: sqlite3.Connection) -> list[str]:
    """Visitors whose derived surfaces are not current (_not_current)."""
    sql, params = _not_current()
    return [
        row["visitor_id"]
        for row in conn.execute(f"SELECT DISTINCT visitor_id FROM ({sql})", params)
    ]


def stranded_events(conn: sqlite3.Connection) -> int:
    """How many rows the derivation path has not brought current, counted once each (_not_current)."""
    sql, params = _not_current()
    return conn.execute(f"SELECT COUNT(*) n FROM ({sql})", params).fetchone()["n"]


@contextmanager
def derivation_lock(db: str):
    """One derivation at a time per db. A pass holds the db's write lock for minutes, and a second caller that finds the path stale meanwhile would start its own full pass and race the first — two derivations rewriting the same rows until one dies on `database is locked`. So the pass is held under a file lock beside the db for its whole span; a caller that finds it taken waits for the holder, saying so, and re-reads what is left to do once the holder is done."""
    with open(f"{db}.derive.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            with beating("waiting for the derivation another process is running"):
                fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def finish_derivations(conn: sqlite3.Connection, db: str, definition: dict) -> dict:
    """Bring every derived surface current — the derivation half of a load, callable on its own — under the operator's session definition (deployment.definitions). Every pass says how far it has got as it goes, and only one runs per db at a time (derivation_lock).

    A visitor with undistilled, unplaced, or unsessioned rows is one a load just changed or one a prior interrupted load left unfinished; either needs the downstream passes — slice, rescue, distill, sessions — and nothing else does. Detected from db state, never from what was just fetched, so it finishes stranded work no matter which caller reaches it. A dirty visitor's distilled columns are re-derived whole — distillation runs a visitor against its slices, so columns written against a missing placement are wrong, not merely incomplete — and its rows are marked undistilled first so the pass selects them by its own signal. An identity that changed since the db's values were derived would leave a mixed-vintage db if only the dirty visitors were touched, so staleness widens the pass to every visitor, from the stage whose identity changed through everything derived after it: a timestamp change re-derives timestamps, slices, rescue, distillation, and sessions; a slice change starts at slices; a distillation change re-distills and re-derives sessions; a session change — this code, the user-activity definition, or the operator's numbers — re-derives sessions alone. A pass that leaves every derived value current stamps the installed identity; a run that cannot make that claim leaves the old stamp so the staleness stays detectable. Returns what it found and did: the dirty visitors, the stage the staleness started at (None when current), and the rescue counts."""
    with derivation_lock(db):
        return _finish_derivations(conn, db, definition)


def _finish_derivations(conn: sqlite3.Connection, db: str, definition: dict) -> dict:
    from .slices import rescue_orphan_slices

    dirty = dirty_visitors(conn)
    stage = stale_stage(conn, definition)
    rescue: dict = {}
    if dirty and stage is None:
        conn.executemany(
            "UPDATE events SET type_str = NULL WHERE visitor_id = ?",
            [(visitor_id,) for visitor_id in dirty],
        )
        conn.commit()
    if stage is not None:
        if stage == "timestamps":
            ranges_done = 0

            def one_range() -> None:
                nonlocal ranges_done
                ranges_done += 1

            with beating("re-deriving timestamps", lambda: f"{ranges_done} ranges"):
                rederive_timestamps(conn, db, on_range=one_range)
        if stage in ("timestamps", "slices"):
            _slice_every(conn, _every_visitor(conn))
            with beating("rescuing orphan slices"):
                rescue = rescue_orphan_slices(conn, db)
        if stage in ("timestamps", "slices", "distill"):
            with beating("distilling"):
                run_distill(db, full=True)
        _session_every(conn, _every_visitor(conn), definition)
    elif dirty:
        _slice_every(conn, dirty)
        with beating("rescuing orphan slices"):
            rescue = rescue_orphan_slices(conn, db, visitors=dirty)
        with beating("distilling"):
            run_distill(db, full=False)
        _session_every(conn, dirty, definition)
    if stage is not None or dirty:
        _stamp_vintages(conn, definition)
    return {"dirty": dirty, "stale": stage is not None, "from": stage, "rescue": rescue}
