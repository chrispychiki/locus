"""Keystone DB: one self-contained events.db, the local cache.

Hydration lands the raw rows; distillation materializes the flat columns later. What every column holds is stated in the schema itself, so `sqlite3 events.db '.schema'` is the whole answer for anyone querying — SQLite keeps those comments verbatim. What follows is only what a column comment cannot say.

The canonical stream is read back in CANONICAL_ORDER — declared below, the one statement of the order: counter is the recorder's emission-order stamp, the only same-ms order that survives out-of-order chunk arrival; it is minted per page context, so a visitor's concurrent page contexts can tie on it within one timestamp, and id keeps the order total.

`page_url`'s forward-fill is what keeps the selection plane flat: the page an event happened on rides its own row, so no aggregate query needs a join or a window function to reach it.

`hidden` applies project.js's own visibility criterion (isProjected) per event, so the boundary the projection draws and the boundary this column reports can never disagree.

Distillation always writes `type_str`, materialization always writes `slice_id`, and the session derivation stamps `session_id` on every act of user activity (user_activity.py), so `type_str IS NULL` marks an event hydration has landed but distillation has not yet reached, `slice_id IS NULL` one no slice has placed, and an act with `session_id IS NULL` one no session has claimed — the three incremental signals a load reuses to touch only the visitors it actually changed, each read off its own index (the partial index over undistilled rows keeps that probe constant-time as it empties; the slice and session indexes answer the others).

The envelope columns — device through snippet — are capture-time testimony, not derivations: what only the recording browser or the load itself could say, from sources not retained (the userAgent, the object key, the store's LIST), so they are recorded once and never re-derived.

The chunk errors are the recorder's own error log, shipped home inside its chunks and keyed by the chunk that carried them, so the recorder's testimony about its own trouble is queryable beside the events it rode in with. The log is every error a page context hit — a send failure whose batch was later delivered sits beside a genuine drop — capped on the device and landing only when a later chunk does, so it is best-effort testimony read record by record: the rows share no unit (one drop record can stand for a whole batch, a retry record for no loss at all), and no count over them measures anything.

The derivation vintage in `meta` is stamped when a pass leaves every derived surface current, so a later load or status can tell when the deriving code has changed out from under them.

A `loaded_chunks` key reappears with a new ETag when a later page context re-drains the same batch under a changed envelope; the differing ETag forces the re-fetch, so late-appended chunk errors still land. A lossy or unparseable object is never recorded at all, so a later load fetches it again and re-reports its loss.
"""

import re
import sqlite3
from pathlib import Path

CANONICAL_ORDER = "ORDER BY timestamp, counter, id"

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY,
    visitor_id   TEXT    NOT NULL,
    timestamp    INTEGER NOT NULL, -- epoch ms, UTC
    type         INTEGER NOT NULL, -- rrweb's numeric event type; type_str names it
    counter      TEXT,             -- emission order across chunk boundaries: the tiebreak within one timestamp
    raw_json     BLOB    NOT NULL, -- the original event, source of truth, zlib-compressed (read via hydrate.raw_event / distill/raw.js); timestamp's canonical form, slice_id, and type_str through diff re-derive from it
    content_hash TEXT    NOT NULL, -- SHA-256 of the uncompressed canonical bytes: the dedup identity, frozen (hydrate.py)
    slice_id     INTEGER,          -- slices.id, NULL until materialization places the event
    session_id   INTEGER,          -- sessions.id: the session whose inactivity timer was running when the event happened (a page context's opening events count to the session its page load starts); NULL until the session derivation runs, and after it for an event no session reaches
    type_str     TEXT,             -- the event kind by name (Click, Input, Mutation, ...), NULL until distillation
    url          TEXT,             -- attested: set only on events carrying a url of their own
    page_url     TEXT,             -- the page the event happened on: url forward-filled in canonical order within its slice (a slice is one page context; concurrent tabs interleave, so the fill never crosses slices)
    tag          TEXT,             -- the target element's tag name
    class        TEXT,             -- the target element's class attribute
    text         TEXT,             -- the target's visible text, or a Selection's selected text
    x            INTEGER,          -- viewport coordinates: a pointer's position, a Scroll's new offset
    y            INTEGER,
    input        TEXT,             -- an Input's value after the change, or 'true'/'false' when the field is checkable
    href         TEXT,             -- the link destination the target acts on: its own href, or the nearest enclosing anchor's
    title        TEXT,             -- the page title a PageLoad carries
    referrer     TEXT,             -- the referrer a PageLoad carries; empty string is a direct arrival
    pointer_type TEXT,             -- mouse, touch or pen: what the interaction was made with
    extra        TEXT,             -- JSON: every other scalar the event and its target element carried, attributes included, values capped
    hidden       INTEGER,          -- 1 when the target sat outside the page's projection, 0 when it was projected content, NULL when no target resolved (project.js names the projection's limits)
    md           TEXT,             -- a FullSnapshot's page content, projected to markdown
    diff         TEXT,             -- a run of mutations collapsed to a diff against the projection before it
    device       TEXT,             -- device class, from the userAgent, as are os and browser — capture-time testimony: the UA is not retained, so these never re-derive
    os           TEXT,
    browser      TEXT,
    language     TEXT,
    time_zone    TEXT,             -- the visitor's own zone: every timestamp here is UTC, so this is the only record of their local hour
    screen_width  INTEGER,         -- the display, not the viewport rrweb's Meta carries; the gap between them is a windowed browser
    screen_height INTEGER,
    script_version TEXT,           -- the recorder build that produced the event
    recorder_slice TEXT,           -- the recorder's own slice id, stamped on every event: slice identity, and what materialization keys on
    snippet      TEXT              -- the snippet id of the site this was recorded from, read from the object key
);
CREATE UNIQUE INDEX IF NOT EXISTS events_dedup ON events(visitor_id, content_hash);
CREATE INDEX IF NOT EXISTS events_canonical ON events(visitor_id, timestamp, counter, id);
CREATE INDEX IF NOT EXISTS events_slice ON events(slice_id);
CREATE INDEX IF NOT EXISTS events_undistilled ON events(visitor_id) WHERE type_str IS NULL;
CREATE INDEX IF NOT EXISTS events_site ON events(snippet, timestamp);

CREATE TABLE IF NOT EXISTS slices (
    id             INTEGER PRIMARY KEY, -- a private rowid, stable under incremental loads; recorder_slice is the agent-facing identity
    visitor_id     TEXT    NOT NULL,
    recorder_slice TEXT,
    start_ts       INTEGER NOT NULL,    -- epoch ms, UTC, as are end_ts and every timestamp here
    end_ts         INTEGER NOT NULL,
    n_events       INTEGER NOT NULL,
    status         TEXT    NOT NULL,    -- replayable (analysis reads these), rescued (appended onto the slice before it), or discarded
    reason         TEXT,                -- why a slice is not replayable, in its own words
    absorbed       TEXT,                -- JSON: the recorder_slice ids of the rescued slices appended onto this one
    snippet        TEXT,                -- the site's snippet id, as on every event of the slice — one page context records one site, one device, one browser, so these are the slice's own facts, lifted here so a per-site or per-device question never walks the events table
    url            TEXT,                -- the address the slice's Meta opened on, whole; the site is `snippet`, and the host the operator knows it by is a projection of this
    device         TEXT,                -- device class, os, browser, language, time_zone, screen_*: the envelope testimony the events carry, one value per slice
    os             TEXT,
    browser        TEXT,
    language       TEXT,
    time_zone      TEXT,
    screen_width   INTEGER,
    screen_height  INTEGER,
    script_version TEXT                 -- the recorder build that produced the slice
);
CREATE INDEX IF NOT EXISTS slices_visitor ON slices(visitor_id, start_ts);
CREATE UNIQUE INDEX IF NOT EXISTS slices_identity ON slices(visitor_id, recorder_slice);

-- the operator's definition of a session (config/definitions.toml) applied to the recording: derived, not testimony — the numbers it ran under are part of the derivation vintage, and doctor re-derives it when they change. A count here is of recorded sessions in what was loaded; a session whose recording never shipped, or whose earlier slices were not loaded, is not here whole
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY, -- a private rowid, re-minted whenever the visitor's sessions re-derive; events.session_id joins here, and a session is addressed by its visitor and start
    visitor_id  TEXT    NOT NULL,
    snippet     TEXT,                -- the site, as on the events
    start_ts    INTEGER NOT NULL,    -- the first user activity (epoch ms, UTC): the page load that opened the session
    end_ts      INTEGER NOT NULL,    -- the last user activity before the inactivity gap ran out
    duration_ms INTEGER NOT NULL,    -- end_ts - start_ts
    pageviews   INTEGER NOT NULL,    -- page loads in the session: every PageLoad event, plus a page context that opened and died before its first snapshot
    user_activity_ms INTEGER NOT NULL, -- time inside stretches of user activity — acts no further apart than the grain (user_activity.py) — summed; the replay skips everything outside them
    engaged     INTEGER NOT NULL     -- 1 when the definition's engaged clause held, 0 when it did not: a bounce
);
CREATE INDEX IF NOT EXISTS sessions_visitor ON sessions(visitor_id, start_ts);
CREATE INDEX IF NOT EXISTS sessions_site ON sessions(snippet, start_ts);

-- Locus's own facts about this db, the derivation vintage among them
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- every chunk object a load fetched and decoded cleanly
CREATE TABLE IF NOT EXISTS loaded_chunks (
    key  TEXT PRIMARY KEY,
    etag TEXT NOT NULL,       -- R2 stamps it with the body's MD5, so a later load skips the GET for a key whose ETag it holds
    uploaded_ms INTEGER       -- when the object reached the store (epoch ms, UTC): against the slice open time the key encodes, the record of delivery latency
);
CREATE INDEX IF NOT EXISTS loaded_chunks_uploaded ON loaded_chunks(uploaded_ms);

-- the recorder's own error log, shipped inside its chunks: retried failures sit beside genuine losses (which name themselves — a drop, an eviction), so read records — a row count measures nothing
CREATE TABLE IF NOT EXISTS chunk_errors (
    snippet        TEXT,
    visitor_id     TEXT NOT NULL,
    recorder_slice TEXT,
    chunk_key      TEXT NOT NULL,
    error          TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS chunk_errors_dedup
    ON chunk_errors(visitor_id, chunk_key, error);
"""


# The slice facts materialization lifts off the events: the site and the page's address, then the
# envelope testimony — named here in the order materialize_slices writes them; what each holds is
# the slices DDL above.
SLICE_FACTS = (
    "snippet",
    "url",
    "device",
    "os",
    "browser",
    "language",
    "time_zone",
    "screen_width",
    "screen_height",
    "script_version",
)

# Indexes over columns the widening below adds, created after it so a db from before the columns
# existed gets them too: per-site slice reads scope on the snippet first and the open time second;
# per-session reads over events group and join on the session stamp.
WIDENED_INDEXES = (
    "CREATE INDEX IF NOT EXISTS slices_site ON slices(snippet, start_ts)",
    "CREATE INDEX IF NOT EXISTS events_session ON events(session_id)",
)

_TABLE = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", re.DOTALL)
_COLUMN = re.compile(r"^\s+(\w+)\s+([A-Z][A-Z ]*[A-Z])\s*,?\s*(?:--\s*(.*?))?\s*$")


def declared_columns() -> dict[str, dict[str, str]]:
    """Every table's columns as SCHEMA declares them, name → the definition an ALTER adds it by: the type, and the column's comment as a block comment, since a line comment appended to a stored CREATE swallows the closing paren while a block comment survives inside it."""
    tables: dict[str, dict[str, str]] = {}
    for table, body in _TABLE.findall(SCHEMA):
        columns: dict[str, str] = {}
        for line in body.splitlines():
            match = _COLUMN.match(line)
            if match is None:
                continue
            name, kind, comment = match.groups()
            definition = f"{name} {' '.join(kind.split())}"
            if comment:
                assert "*/" not in comment, (
                    f"{table}.{name}: a column comment cannot close a block comment"
                )
                definition += f" /* {comment} */"
            columns[name] = definition
        tables[table] = columns
    return tables


def _widen(conn: sqlite3.Connection) -> None:
    """A table created before a column existed gets the column added, with the comment the schema above gives it, so a standing db reads by that schema — `.schema` included; the values land on the next derivation, which the derivation vintage forces because the deriving code changed. CREATE TABLE IF NOT EXISTS leaves an existing table as it was, so the widening is its own step."""
    for table, columns in declared_columns().items():
        live = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, definition in columns.items():
            if name in live:
                continue
            _addable(conn, table, name, definition)
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")
    for statement in WIDENED_INDEXES:
        conn.execute(statement)


_UNADDABLE = ("PRIMARY KEY", "UNIQUE")
_UNADDABLE_FILLED = ("NOT NULL",)


def _addable(conn: sqlite3.Connection, table: str, name: str, definition: str) -> None:
    """ALTER TABLE ADD COLUMN takes a plain column and refuses a constrained one — PRIMARY KEY and UNIQUE always, NOT NULL once the table holds rows — and SQLite's refusal names none of the table, the column, or the way out. A declaration the widening cannot honor is said here, before any ALTER runs, with the remedy: the db is a cache of the store, so it is rebuilt by dropping it and loading again."""
    kind = definition.split(" /*")[0].upper()
    refused = any(c in kind for c in _UNADDABLE)
    if not refused and any(c in kind for c in _UNADDABLE_FILLED):
        refused = conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None
    if refused:
        raise RuntimeError(
            f"{table}.{name} is declared `{definition.split(' /*')[0]}`, which ALTER TABLE "
            f"cannot add to a standing table{' holding rows' if 'NOT NULL' in kind else ''}; "
            "this db predates the column and cannot be widened to it — rebuild it: trash "
            "data/events.db and run `locus load` again (the store still holds everything "
            "within the retention horizon)"
        )


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    # WAL, persisted into the db file on first connect: a load writes for
    # minutes, and the flat-fact plane — SQL from any process, the sqlite3 CLI
    # included — must read the last commit meanwhile, not "database is locked".
    # The busy timeout covers writer-vs-writer (two loads racing) the same way.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    _widen(conn)
    conn.commit()
    return conn
