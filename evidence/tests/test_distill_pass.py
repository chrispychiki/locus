"""The distillation pass as it actually runs: `bun distill/distill.js <db>`.

The unit behavior of each stage is pinned bun-side (distill/*.test.js). What only the real pass can
show is the part that writes: that every distilled value lands in the column it belongs to, and that a
visitor set is sharded across workers without losing anyone.
"""

import json
import subprocess
from pathlib import Path

from _support import stamped
from locus.evidence.db import connect
from locus.evidence.hydrate import hydrate_stream, raw_event
from locus.evidence.rrweb_constants import (
    EVENTTYPE_NAMES,
    EventType,
    IncrementalSource,
    MouseInteractions,
    NodeType,
    PointerTypes,
)
from locus.evidence.slices import materialize_slices

DISTILL = Path(__file__).parent.parent / "distill" / "distill.js"
T0 = 1_700_000_000_000

# A code the generated schema does not contain — derived from it, so it stays foreign
# when rrweb adds a type rather than becoming a real one behind the test's back.
FOREIGN_TYPE = max(EVENTTYPE_NAMES) + 1


def distill(db_path) -> str:
    done = subprocess.run(
        ["bun", str(DISTILL), str(db_path)], check=True, capture_output=True, text=True
    )
    return done.stdout


def meta(ts, href="https://shop.test/cart"):
    return {
        "type": EventType.Meta,
        "timestamp": ts,
        "data": {"href": href, "width": 1280, "height": 800},
    }


def snapshot(ts, link_text="Buy now"):
    """A DOM with one anchor the later events can address by id."""
    return {
        "type": EventType.FullSnapshot,
        "timestamp": ts,
        "data": {
            "initialOffset": {"left": 0, "top": 0},
            "node": {
                "type": NodeType.Document,
                "id": 1,
                "childNodes": [
                    {
                        "type": NodeType.Element,
                        "tagName": "html",
                        "id": 2,
                        "attributes": {},
                        "childNodes": [
                            {
                                "type": NodeType.Element,
                                "tagName": "body",
                                "id": 3,
                                "attributes": {},
                                "childNodes": [
                                    {
                                        "type": NodeType.Element,
                                        "tagName": "a",
                                        "id": 4,
                                        "attributes": {
                                            "class": "cta",
                                            "href": "/checkout",
                                            "data-track": "hero",
                                        },
                                        "childNodes": [
                                            {
                                                "type": NodeType.Text,
                                                "id": 5,
                                                "textContent": link_text,
                                            }
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        },
    }


def click(ts, node_id=4):
    return {
        "type": EventType.IncrementalSnapshot,
        "timestamp": ts,
        "data": {
            "source": IncrementalSource.MouseInteraction,
            "type": MouseInteractions.Click,
            "id": node_id,
            "x": 137,
            "y": 42,
            "pointerType": PointerTypes.Mouse,
        },
    }


def typing(ts, text="hello"):
    return {
        "type": EventType.IncrementalSnapshot,
        "timestamp": ts,
        "data": {
            "source": IncrementalSource.Input,
            "id": 4,
            "text": text,
            "isChecked": False,
        },
    }


def page_load(ts):
    return {
        "type": EventType.PageLoad,
        "timestamp": ts,
        "data": {
            "url": "https://shop.test/cart",
            "title": "Your cart",
            "referrer": "https://google.test/",
        },
    }


def retext(ts, value):
    return {
        "type": EventType.IncrementalSnapshot,
        "timestamp": ts,
        "data": {
            "source": IncrementalSource.Mutation,
            "adds": [],
            "removes": [],
            "attributes": [],
            "texts": [{"id": 5, "value": value}],
        },
    }


def build(tmp_path, visitors: dict):
    db_path = tmp_path / "events.db"
    conn = connect(db_path)
    for visitor, events in visitors.items():
        hydrate_stream(conn, ((visitor, e) for e in stamped(events)))
        materialize_slices(conn, visitor)
    conn.close()
    return db_path


def test_every_distilled_value_lands_in_the_column_it_belongs_to(tmp_path):
    # The worker binds every distilled column positionally in one UPDATE. A transposed pair would put the
    # link text in `class`, or the referrer in `title`, and every downstream query would read a
    # confident, wrong answer. Each value below is distinct, so a crossed binding cannot pass.
    db_path = build(
        tmp_path,
        {
            "v1": [
                meta(T0),
                snapshot(T0 + 1),
                page_load(T0 + 2),
                click(T0 + 100),
                typing(T0 + 200, "hello"),
                retext(T0 + 300, "Sold out"),
                retext(T0 + 400, "Back in stock"),
            ]
        },
    )
    distill(db_path)

    conn = connect(db_path)
    rows = {
        r["type_str"]: r
        for r in conn.execute("SELECT * FROM events ORDER BY timestamp")
    }

    assert rows["Meta"]["url"] == "https://shop.test/cart"

    load = rows["PageLoad"]
    assert load["url"] == "https://shop.test/cart"
    assert load["title"] == "Your cart"
    assert load["referrer"] == "https://google.test/"

    tap = rows["Click"]
    assert tap["tag"] == "a"
    assert tap["class"] == "cta"
    assert tap["text"] == "Buy now"
    assert tap["href"] == "/checkout"
    assert (tap["x"], tap["y"]) == (137, 42)
    assert tap["pointer_type"] == "Mouse"
    assert json.loads(tap["extra"])["data-track"] == "hero"

    assert rows["Input"]["input"] == "hello"
    assert rows["FullSnapshot"]["md"] and "Buy now" in rows["FullSnapshot"]["md"]

    # Every row carries the page it happened on, forward-filled from the last attested url.
    urls = [
        r["page_url"]
        for r in conn.execute("SELECT page_url FROM events ORDER BY timestamp")
    ]
    assert urls == ["https://shop.test/cart"] * len(urls)


def test_a_mutation_run_writes_one_diff_on_the_event_that_closed_it(tmp_path):
    db_path = build(
        tmp_path,
        {
            "v1": [
                meta(T0),
                snapshot(T0 + 1),
                retext(T0 + 100, "Sold out"),
                retext(T0 + 200, "Back in stock"),
                click(T0 + 300),
            ]
        },
    )
    distill(db_path)

    conn = connect(db_path)
    diffs = conn.execute(
        "SELECT timestamp, diff FROM events WHERE diff IS NOT NULL"
    ).fetchall()
    assert len(diffs) == 1, "the run collapses to one delta, not one per mutation"
    assert diffs[0]["timestamp"] == T0 + 200, "carried on the event that closed the run"
    assert "Back in stock" in diffs[0]["diff"]
    assert "Sold out" not in diffs[0]["diff"], (
        "the diff is against the projection before the run, not each step inside it"
    )


def test_no_visitor_is_lost_when_the_pass_shards(tmp_path):
    # Visitors are dealt round-robin across workers. A sharding bug drops a whole visitor silently —
    # the pass still reports success, and the missing events look like a visitor whose chunks never arrived.
    visitors = {
        f"v{i}": [meta(T0), snapshot(T0 + 1), click(T0 + 100)] for i in range(9)
    }
    db_path = build(tmp_path, visitors)
    out = distill(db_path)

    assert "across 9 visitors" in out
    conn = connect(db_path)
    distilled = conn.execute(
        "SELECT visitor_id, COUNT(*) n FROM events WHERE type_str IS NOT NULL GROUP BY visitor_id"
    ).fetchall()
    assert {r["visitor_id"] for r in distilled} == set(visitors)
    assert all(r["n"] == 3 for r in distilled)


def test_an_event_type_outside_the_generated_schema_is_counted_and_named(tmp_path):
    # The store is open-write, so a foreign or future event type can land. The pass must not
    # silently coin a type_str the kind vocabulary doesn't contain — it says how many.
    alien = {"type": FOREIGN_TYPE, "timestamp": T0 + 500, "data": {}}
    db_path = build(tmp_path, {"v1": [meta(T0), snapshot(T0 + 1), alien]})
    out = distill(db_path)

    assert "1 events carry type codes outside the generated schema" in out
    conn = connect(db_path)
    assert conn.execute(
        "SELECT type_str FROM events WHERE type = ?", (FOREIGN_TYPE,)
    ).fetchone()["type_str"] == str(FOREIGN_TYPE)


def test_mutation_residue_is_reported_never_swallowed(tmp_path):
    # A mutation the replayer would abort or drop is not data loss — the screenshots show the same thing —
    # but it is a mechanical fact about the recording, and it must reach the operator.
    ghost = {
        "type": EventType.IncrementalSnapshot,
        "timestamp": T0 + 100,
        "data": {
            "source": IncrementalSource.Mutation,
            "adds": [],
            "removes": [{"parentId": 3, "id": 999}],
            "texts": [],
            "attributes": [],
        },
    }
    db_path = build(tmp_path, {"v1": [meta(T0), snapshot(T0 + 1), ghost]})
    out = distill(db_path)

    assert "mutation residue" in out
    assert "1 ops on unresolvable ids" in out


def test_a_wire_value_its_column_cannot_hold_is_dropped_loudly_not_bound(tmp_path):
    # Binding an object into a TEXT column would kill the whole pass. The value drops to null,
    # raw_json keeps it, and the count is reported — one bad field costs itself.
    bad = {
        "type": EventType.Meta,
        "timestamp": T0 + 500,
        "data": {"href": {"not": "a string"}, "width": 1, "height": 1},
    }
    db_path = build(tmp_path, {"v1": [meta(T0), snapshot(T0 + 1), bad]})
    out = distill(db_path)

    assert "WARNING: dropped 1 wire values" in out
    conn = connect(db_path)
    row = conn.execute(
        "SELECT url, raw_json FROM events WHERE timestamp = ?", (T0 + 500,)
    ).fetchone()
    assert row["url"] is None
    assert raw_event(row["raw_json"])["data"]["href"] == {"not": "a string"}


def test_the_pass_is_rerunnable(tmp_path):
    # Distillation is not a migration — the agent re-runs it after a load, over rows that already
    # carry columns: one NULL type_str row marks its whole visitor for re-distillation, so the pass
    # re-steps every row the visitor holds. The re-run must land the same values on the rows that
    # already carried them — not double a diff or drop an md — and distill the newly landed event.
    events = [
        meta(T0),
        snapshot(T0 + 1),
        retext(T0 + 100, "Sold out"),
        click(T0 + 200),
    ]
    db_path = build(tmp_path, {"v1": events})
    distill(db_path)
    conn = connect(db_path)
    first = [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY id")]
    conn.close()

    # A later load lands one more event for the same visitor and slice; content-hash dedup
    # makes the old events no-ops and the new row arrives with type_str NULL.
    conn = connect(db_path)
    hydrate_stream(conn, (("v1", e) for e in stamped([*events, click(T0 + 300)])))
    materialize_slices(conn, "v1")
    conn.close()

    out = distill(db_path)
    assert "across 1 visitors" in out, (
        "the undistilled row re-triggers its visitor — a no-op pass would prove nothing"
    )
    conn = connect(db_path)
    second = [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY id")]
    assert second[: len(first)] == first, (
        "re-distillation lands the same values on the rows that already carried them"
    )
    (new_row,) = second[len(first) :]
    assert new_row["type_str"] == "Click", "and the newly landed event is distilled"
