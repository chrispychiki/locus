"""Lone surrogates survive the whole boundary deliberately, not by accident.

Recorded text can carry unpaired UTF-16 surrogates (a truncated emoji in an
input field). The wire ships them as \\uXXXX escapes (well-formed
JSON.stringify); json.loads turns them into real lone surrogates in Python
strings. Two guarantees hold from there: raw_json stays faithful (the escape
round-trips, hydrate._canonical_json's ensure_ascii), and every flat column
distillation writes is well-formed UTF-8 (lone surrogates → U+FFFD at the
distill_worker write chokepoint) so a strict reader never chokes on a column.
"""

import gzip
import json
import sqlite3
import subprocess
import zlib

from _support import DISTILL, chunk_blob

LONE_HIGH = "\ud83d"  # unpaired high surrogate
LONE_LOW = "\udc00"  # unpaired low surrogate


def test_lone_surrogates_survive_decode_hydrate_distill_query(tmp_path):
    from locus.evidence.chunk import decode_or_skip
    from locus.evidence.db import connect
    from locus.evidence.hydrate import hydrate_stream
    from locus.evidence.rrweb_constants import EventType, IncrementalSource, NodeType
    from locus.evidence.slices import materialize_slices

    dirty = f"pasted{LONE_HIGH}text{LONE_LOW}end"
    t0 = 1700000000000
    events = [
        {
            "type": EventType.Meta,
            "timestamp": t0,
            "counter": f"{t0}000001",
            "data": {
                "href": f"https://x.test/?q={LONE_HIGH}",
                "width": 800,
                "height": 600,
            },
        },
        {
            "type": EventType.FullSnapshot,
            "timestamp": t0 + 1,
            "counter": f"{t0}000002",
            "data": {
                "node": {"type": NodeType.Document, "id": 1, "childNodes": []},
                "initialOffset": {"left": 0, "top": 0},
            },
        },
        {
            "type": EventType.IncrementalSnapshot,
            "timestamp": t0 + 100,
            "counter": f"{t0}000003",
            "data": {
                "source": IncrementalSource.Input,
                "id": 1,
                "text": dirty,
                "isChecked": False,
            },
        },
    ]
    blob = chunk_blob(f"0{t0}-surr", events, visitor="v-surrogate")

    # the wire is what a well-formed JSON.stringify emits: escapes, pure ASCII
    assert b"\\ud83d" in gzip.decompress(blob)

    pairs = decode_or_skip(blob, f"snip/2026-07-11/v-surrogate/0{t0}-surr/k")
    assert len(pairs) == 3, "escaped lone surrogates decode, never skip"

    db = str(tmp_path / "events.db")
    conn = connect(db)
    assert hydrate_stream(conn, iter(pairs)) == 3
    materialize_slices(conn, "v-surrogate")

    stored = conn.execute(
        "SELECT raw_json FROM events WHERE type = ?",
        (int(EventType.IncrementalSnapshot),),
    ).fetchone()["raw_json"]
    canonical = zlib.decompress(stored).decode()
    assert json.loads(canonical)["data"]["text"] == dirty, (
        "raw_json is the source of truth — the surrogate round-trips intact"
    )
    assert "\\ud83d" in canonical, "stored as an ASCII escape, never a raw surrogate"
    conn.close()

    subprocess.run(["bun", str(DISTILL), db], check=True, capture_output=True)

    strict = sqlite3.connect(db)
    strict.row_factory = sqlite3.Row
    row = strict.execute(
        "SELECT input, url FROM events WHERE input IS NOT NULL"
    ).fetchone()
    assert row is not None, "the input event was distilled"
    assert row["input"] == "pasted�text�end", (
        "flat columns are well-formed: lone surrogates became U+FFFD"
    )
    url = strict.execute(
        "SELECT url FROM events WHERE url IS NOT NULL LIMIT 1"
    ).fetchone()["url"]
    assert "�" in url and url.encode("utf-8"), (
        "url column sanitized and strictly encodable"
    )
    slice_url = strict.execute("SELECT url FROM slices").fetchone()["url"]
    assert slice_url == url, "the slice's address is sanitized by the same rule"
