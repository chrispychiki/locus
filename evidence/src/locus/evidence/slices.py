"""Slices — everything that defines and owns a slice.

The recorder owns the boundary, not the reader. Every event it emits is stamped with the slice its own page context minted, so a slice is exactly the events sharing one recorder_slice — grouped by that key, never cut out of the merged stream by position. Position is not identity here: one visitor with two tabs open interleaves two page contexts into one time-ordered stream, and a positional cut lands across them, filing each context's events under the other's slice. An event that carries no slice id is a defect, not a source variant: the recorder stamps every event it emits, so a missing stamp means the stream was damaged upstream — the reader cannot recover the slicing, and inventing one would file the damage as a supported case.

A slice's head is its covering snapshot: rrweb emits Meta then FullSnapshot as one synchronous pair (takeFullSnapshot wrappedEmits the Meta then immediately the FullSnapshot: rrweb packages/rrweb/src/record/index.ts), so a replayable slice is one that *contains* that adjacent pair. It need not open on it — a page context opens its slice the moment it starts, before rrweb has a DOM to capture, so a marker can precede the Meta inside the slice it belongs to.

A slice that cannot keep the replayable promise is discarded whole with a counted reason. The four, and the whole set, are `discard_reason` — the criterion this module owns and every reader of a slice asks, so nothing downstream re-derives what replayable means:
- NO_SNAPSHOT_REASON — the slice holds no Meta at all: the page ended before rrweb ever captured its DOM. Nothing was lost in transit; there was never a snapshot. Never rescuable — the events belong to a document that no prior slice's DOM can cover.
- ORPHAN_REASON — a Meta whose FullSnapshot never arrived. The document is real and continues; only the snapshot is missing, which is what makes this class, and only this class, a rescue candidate.
- A Meta recording a zero-dimension viewport — hidden prerender contexts, pages no human ever saw.
- DAMAGED_SNAPSHOT_REASON — a FullSnapshot with no root node: the document it covers cannot be reconstructed.
All losses are counted — discard rates and reasons stay transparent.

materialize_slices stamps the derivation into the keystone DB: slice_id on each event row plus one summary row per slice, replayable and discarded alike, so the accounting is queryable, not just printed. The summary row also carries the slice's own facts (db.py SLICE_FACTS): the snippet and envelope testimony every one of its events repeats, and the address its Meta opened on — one page context is one site on one device in one browser, so a per-site or per-device question reads `slices` and never walks the events. Re-materialization always reflects the current canonical stream, but never renumbers: slices are keyed by recorder_slice, the recorder's own stamp. Late chunks land in their existing slice; new recorder ids become new slices. The integer primary key is a private rowid — stable, never agent-facing.

The rescue rule recovers severed slice tails: an ORPHAN_REASON slice whose events continue the immediately prior replayable slice (same visitor, same page URL, gap within RESCUE_MAX_GAP_MS) is appended to it. A rescued slice keeps its slices row (status 'rescued', reason naming the absorber); the absorber records the orphan's recorder_slice in its `absorbed` list — the primary id is identity, absorbed upstream ids are facts. A gated-out slice stays discarded with the gate's refusal appended to its reason.

Two walls stand before a rescue, and they are asked in that order because only one of them can be sure. First: does the orphan open a *new document*? A page context that reloads gets a fresh rrweb mirror whose node ids restart at 1, so its nodes coincide with the prior page's by number while being different nodes entirely — continuity and coincidence become the same answer, and no amount of DOM replay can separate them, because node ids carry no document identity. What separates them is the slice id (opens_a_new_document): a head slice's id is minted before its own Meta, a checkout's is minted from it. Second, for what survives that: the node-id continuity gate (distill/rescue_gate.js) replays the prior slice and proves every node id the orphan references resolves against its end state — which is what keeps the merged slice self-covering, replaying forward from its own head snapshot with no lookbacks. The gate remains necessary because one visitor's slices interleave across tabs: the slice before an orphan in time need not be from the orphan's own page context.
"""

import itertools
import json
import re
import shutil
import sqlite3
import subprocess

from .assets import script
from .db import CANONICAL_ORDER, SLICE_FACTS
from .hydrate import raw_event
from .rrweb_constants import EventType

ORPHAN_REASON = "no covering FullSnapshot after Meta"
NO_SNAPSHOT_REASON = "no snapshot: the page ended before rrweb captured its DOM"
ZERO_VIEWPORT_REASON = "zero-dimension viewport in Meta"
DAMAGED_SNAPSHOT_REASON = "damaged snapshot: the FullSnapshot carries no root node, so the document it covers cannot be reconstructed"
RESCUE_MAX_GAP_MS = 10_000

# The recorder's slice id: a zero-padded open-millisecond and a base36 disambiguator. Slice identity
# is owned here, so the shape a reader accepts is declared here too — the store's own path gate
# (store/src/keys.js) is the same grammar on the write side, and test_slices.py runs the two
# over one case list so they cannot drift apart silently. ASCII because JS \d is ASCII: without the
# flag Python's \d also takes Unicode digits, and the twins disagree. \Z because JS $ is \Z:
# Python's $ also matches before a trailing newline, and the twins disagree there too.
SLICE_ID_SHAPE = re.compile(r"^\d{14}-[a-z0-9]{4}\Z", re.ASCII)


def group_slices(
    events: list[dict], keys: list[str]
) -> list[tuple[str, list[int], str | None]]:
    """(recorder_slice, event indices, discard_reason) — one entry per slice, in open order.

    keys carries each event's stamped recorder_slice. discard_reason is None for a replayable slice."""
    grouped: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        if key is None:
            raise ValueError(
                f"event {i} (type {events[i]['type']}, ts "
                f"{events[i]['timestamp']}) carries no recorder_slice. The "
                f"recorder stamps every event it emits, so this stream was "
                f"damaged upstream — its slicing is unrecoverable here."
            )
        grouped.setdefault(key, []).append(i)

    return [
        (key, indices, discard_reason([events[i] for i in indices]))
        for key, indices in grouped.items()
    ]


def covering_pair(events: list) -> tuple | None:
    """One slice's (Meta, FullSnapshot), if the snapshot the Meta heralds actually arrived behind it. Only `type` is read, so a slice's events serve here either as raw events or as db rows carrying that column."""
    for a, b in itertools.pairwise(events):
        if a["type"] == EventType.Meta and b["type"] == EventType.FullSnapshot:
            return a, b
    return None


def covering_snapshot_ts(conn: sqlite3.Connection, slice_id: int) -> int:
    """When a slice's covering FullSnapshot captured its DOM — the first moment of the slice a rendered screenshot can testify to.

    A slice opens when its page context does, before rrweb has a DOM to capture, so its first instants hold no recorded page state at all: a replay seeked there has no page to show and refuses, naming the first capture, and a screenshot addressed at one of them would be stamped with a time the pixels never held. Those events are real testimony of *when* and keep their place in the event stream; it is the screenshot plane that opens here.

    Read from the type and timestamp columns alone — adjacency in canonical order is the whole of what the pair is, and a slice's payloads are large."""
    rows = conn.execute(
        f"SELECT timestamp, type FROM events WHERE slice_id = ? {CANONICAL_ORDER}",
        (slice_id,),
    ).fetchall()
    pair = covering_pair(rows)
    if pair is None:
        raise ValueError(
            f"slice {slice_id} holds no covering (Meta, FullSnapshot) pair, so "
            f"it has no screenshot plane — {discard_reason([dict(r) for r in rows])}"
        )
    return pair[1]["timestamp"]


def discard_reason(events: list[dict]) -> str | None:
    """Why one slice's events cannot keep the replayable promise — None when they can. The criterion every reader of a slice asks, so a slice is replayable to exactly one definition."""
    pair = covering_pair(events)
    if pair is None:
        if any(event["type"] == EventType.Meta for event in events):
            return ORPHAN_REASON
        return NO_SNAPSHOT_REASON
    meta_event, snapshot = pair
    meta = meta_event.get("data", {})
    if not meta.get("width") or not meta.get("height"):
        return f"{ZERO_VIEWPORT_REASON} ({meta.get('width')}x{meta.get('height')})"
    # A snapshot with no root node is not a snapshot the DOM can be rebuilt from. Projected, it
    # yields nothing — and nothing is indistinguishable from an empty page, which is what the model
    # would then be told the visitor was looking at, with every later mutation diffed against it.
    # There is no reading of this the slice survives, so it is discarded rather than projected.
    if (snapshot.get("data") or {}).get("node", {}).get("id") is None:
        return DAMAGED_SNAPSHOT_REASON
    return None


def slice_events(conn: sqlite3.Connection, slice_id: int) -> list[dict]:
    """A slice's raw events in canonical order — the replayable unit."""
    rows = conn.execute(
        f"SELECT raw_json FROM events WHERE slice_id = ? {CANONICAL_ORDER}",
        (slice_id,),
    ).fetchall()
    return [raw_event(row["raw_json"]) for row in rows]


def _slice_facts(events: list[dict], rows: list, indices: list[int]) -> tuple:
    """The slice's own facts, in SLICE_FACTS order: the snippet and envelope testimony read off its rows (every row carries them, so the first that does is the slice's), and the address its Meta opened on. A slice with no Meta has no page to name."""
    facts: dict[str, object] = {name: None for name in SLICE_FACTS}
    for i in indices:
        row = rows[i]
        for name in SLICE_FACTS:
            if name != "url" and facts[name] is None and row[name] is not None:
                facts[name] = row[name]
        if facts["url"] is None and events[i]["type"] == EventType.Meta:
            href = events[i].get("data", {}).get("href")
            facts["url"] = (
                _well_formed(href) if isinstance(href, str) and href else None
            )
    return tuple(facts[name] for name in SLICE_FACTS)


def _well_formed(text: str) -> str:
    """A flat column is well-formed UTF-8: a lone surrogate the recording carried (raw_json keeps it, as an escape) becomes U+FFFD here, the same rule distillation applies to every column it writes (distill_worker.js), so a strict reader never chokes on a column."""
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


_FACT_COLUMNS = ", ".join(name for name in SLICE_FACTS)
_FACT_ASSIGNMENTS = ", ".join(f"{name} = ?" for name in SLICE_FACTS)
# The facts the events carry; url is the one read off the Meta instead.
_TESTIMONY_COLUMNS = ", ".join(name for name in SLICE_FACTS if name != "url")


def materialize_slices(conn: sqlite3.Connection, visitor_id: str) -> dict:
    """One visitor's slices, derived from the canonical stream and written to `slices` and `events.slice_id`. What decides a slice — its events' types in canonical order, the Meta's viewport and address, the FullSnapshot's root — is read off the type and timestamp columns plus the raw_json of the Meta and FullSnapshot rows alone; the incremental rows, nearly all of a recording, are placed without decoding. Placement is one pass over the visitor's rows once every slice row stands: each row takes the slice its recorder_slice names, through the identity index."""
    rows = conn.execute(
        f"SELECT id, timestamp, type, recorder_slice, {_TESTIMONY_COLUMNS}, "
        f"CASE WHEN type IN (?, ?) THEN raw_json END AS raw_json "
        f"FROM events WHERE visitor_id = ? {CANONICAL_ORDER}",
        (EventType.Meta, EventType.FullSnapshot, visitor_id),
    ).fetchall()
    events = [
        {
            "type": row["type"],
            "timestamp": row["timestamp"],
            "data": raw_event(row["raw_json"]).get("data", {})
            if row["raw_json"] is not None
            else {},
        }
        for row in rows
    ]
    keys = [row["recorder_slice"] for row in rows]

    existing = {
        row["recorder_slice"]: row["id"]
        for row in conn.execute(
            "SELECT id, recorder_slice FROM slices WHERE visitor_id = ?", (visitor_id,)
        )
    }

    slices = group_slices(events, keys)
    summary = {"replayable": 0, "discarded": 0}
    live = []
    for recorder_slice, indices, reason in slices:
        status = "replayable" if reason is None else "discarded"
        fields = (
            events[indices[0]]["timestamp"],
            events[indices[-1]]["timestamp"],
            len(indices),
            status,
            reason,
        )
        facts = _slice_facts(events, rows, indices)

        slice_pk = existing.get(recorder_slice)
        if slice_pk is not None:
            conn.execute(
                "UPDATE slices SET recorder_slice = ?, start_ts = ?, "
                "end_ts = ?, n_events = ?, status = ?, reason = ?, "
                f"absorbed = NULL, {_FACT_ASSIGNMENTS} WHERE id = ?",
                (recorder_slice, *fields, *facts, slice_pk),
            )
        else:
            slice_pk = conn.execute(
                "INSERT INTO slices (visitor_id, recorder_slice, start_ts, "
                f"end_ts, n_events, status, reason, {_FACT_COLUMNS}) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, {', '.join('?' * len(SLICE_FACTS))})",
                (visitor_id, recorder_slice, *fields, *facts),
            ).lastrowid
        live.append(slice_pk)
        summary[status] += 1

    if live:
        conn.execute(
            f"DELETE FROM slices WHERE visitor_id = ? AND id NOT IN ({','.join('?' * len(live))})",
            (visitor_id, *live),
        )
    else:
        conn.execute("DELETE FROM slices WHERE visitor_id = ?", (visitor_id,))
    conn.execute(
        "UPDATE events SET slice_id = (SELECT s.id FROM slices s "
        "WHERE s.visitor_id = events.visitor_id AND s.recorder_slice = events.recorder_slice) "
        "WHERE visitor_id = ?",
        (visitor_id,),
    )
    conn.commit()
    return summary


def slice_open_ms(recorder_slice: str) -> int:
    """The millisecond a slice opened, read out of the id the recorder minted for it (`${padded-ms}-${disambiguator}`)."""
    return int(recorder_slice.split("-")[0])


def _opening_meta(conn: sqlite3.Connection, slice_id: int) -> dict | None:
    """The slice's Meta. Found by type, never by position: a slice opens when its page context does, so its first event can be a marker that beat rrweb's snapshot."""
    row = conn.execute(
        f"SELECT raw_json FROM events WHERE slice_id = ? AND type = ? {CANONICAL_ORDER} LIMIT 1",
        (slice_id, EventType.Meta),
    ).fetchone()
    return raw_event(row["raw_json"]) if row else None


def _meta_href(conn: sqlite3.Connection, slice_id: int) -> str | None:
    meta = _opening_meta(conn, slice_id)
    return meta.get("data", {}).get("href") if meta else None


def opens_a_new_document(conn: sqlite3.Connection, slice_row) -> bool:
    """Whether this slice is a page context's *head* — a fresh document, which nothing before it can continue.

    The recorder answers this for free, and it has been answering it all along. A page context mints its head slice id the instant it starts, before rrweb exists — so the id's millisecond predates every event the context will ever emit, including its own record-start Meta. A checkout slice is minted *from* its Meta's timestamp, so its id's millisecond is that Meta's, exactly. An id that opened strictly before its own Meta is therefore a new document, and a reload cannot close that gap: the buffer open and rrweb's startup sit between the two.

    Equality is the safe answer, not the confident one — a slice that reads as a checkout still has to face the node-id gate."""
    meta = _opening_meta(conn, slice_row["id"])
    if meta is None:
        return False
    return slice_open_ms(slice_row["recorder_slice"]) < meta["timestamp"]


def opened_a_page(conn: sqlite3.Connection, slice_row) -> bool:
    """Whether this slice is a page context opening in front of a person — a page load, whatever became of the recording after. A head by opens_a_new_document is one; so is a slice with no Meta at all, because a checkout is minted from its Meta and a slice without one is a page that died before rrweb captured its DOM. A prerendered context (a zero-dimension viewport) is a page nobody saw, and is not."""
    if slice_row["reason"] == NO_SNAPSHOT_REASON:
        return True
    if (slice_row["reason"] or "").startswith(ZERO_VIEWPORT_REASON):
        return False
    return opens_a_new_document(conn, slice_row)


def require_bun() -> None:
    if shutil.which("bun") is None:
        raise SystemExit(
            "no bun on PATH: the distillation pass, the rescue gate, and the "
            "store integrity check are bun scripts. Install it from "
            "https://bun.sh"
        )


def _run_gate(db_path: str, prior_id: int, orphan_id: int) -> dict:
    """The gate runs per candidate: a rescue rewrites the prior slice's events and end time, and the next orphan's verdict is taken against that new state — chained tails are the point, so each verdict must see the writes of the one before it."""
    require_bun()
    proc = subprocess.run(
        ["bun", str(script("rescue_gate.js")), db_path, str(prior_id), str(orphan_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"the rescue gate failed on slices {prior_id}/{orphan_id} "
            f"(exit {proc.returncode}): {proc.stderr.strip() or 'no stderr'}"
        )
    return json.loads(proc.stdout)


def rescue_orphan_slices(
    conn: sqlite3.Connection,
    db_path: str,
    visitors: list[str] | None = None,
) -> dict:
    """Rescue severed tails. A rescue depends only on the orphan's own visitor's slices, so a load scopes this to the visitors it changed (`visitors`); None walks every orphan in the db, for a full rebuild."""
    summary = {"rescued": 0, "gated_out": 0, "ineligible": 0}
    scope = (
        ""
        if visitors is None
        else f" AND visitor_id IN ({','.join('?' * len(visitors))})"
    )
    orphans = conn.execute(
        f"SELECT * FROM slices WHERE status = 'discarded' AND reason = ?{scope} ORDER BY visitor_id, start_ts",
        (ORPHAN_REASON, *(visitors or [])),
    ).fetchall()

    for orphan in orphans:
        prior = conn.execute(
            "SELECT * FROM slices WHERE visitor_id = ? AND start_ts < ? "
            "AND status != 'rescued' ORDER BY start_ts DESC LIMIT 1",
            (orphan["visitor_id"], orphan["start_ts"]),
        ).fetchone()
        if (
            prior is None
            or prior["status"] != "replayable"
            or orphan["start_ts"] - prior["end_ts"] > RESCUE_MAX_GAP_MS
            or _meta_href(conn, orphan["id"]) != _meta_href(conn, prior["id"])
            or opens_a_new_document(conn, orphan)
        ):
            summary["ineligible"] += 1
            continue

        verdict = _run_gate(db_path, prior["id"], orphan["id"])
        if verdict["rescued"]:
            conn.execute(
                "UPDATE events SET slice_id = ? WHERE slice_id = ?",
                (prior["id"], orphan["id"]),
            )
            absorbed = json.loads(prior["absorbed"] or "[]")
            absorbed.append(orphan["recorder_slice"])
            conn.execute(
                "UPDATE slices SET end_ts = ?, n_events = n_events + ?, absorbed = ? WHERE id = ?",
                (
                    orphan["end_ts"],
                    orphan["n_events"],
                    json.dumps(absorbed),
                    prior["id"],
                ),
            )
            conn.execute(
                "UPDATE slices SET status = 'rescued', reason = ? WHERE id = ?",
                (
                    (
                        f"id-continuity verified; events appended to slice {prior['recorder_slice']}"
                    ),
                    orphan["id"],
                ),
            )
            summary["rescued"] += 1
        else:
            conn.execute(
                "UPDATE slices SET reason = ? WHERE id = ?",
                (f"{ORPHAN_REASON}; rescue gated: {verdict['reason']}", orphan["id"]),
            )
            summary["gated_out"] += 1
        conn.commit()

    return summary
