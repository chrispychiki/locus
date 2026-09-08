"""The embeddable replay component, a slice set's payload, and the default page over them.

Replay is a component a page embeds, not a page this module emits: two scripts — the self-contained player component and a slice set's events payload — that any sibling HTML composes into whatever the moment needs (a plain replay page, a side-by-side comparison, a summary page with an embedded moment). Both travel as scripts because a composed page must open from a local file, where nothing can fetch; loaded together, `LocusReplay.mount(element, LOCUS_REPLAYS[0])` puts a player on the page. `locus browse open` drives this module: it materializes the scripts beneath `pages/` beside the db, composes the default page over them, and opens it.

The seek/fragment contract and the scripting drive surface live in the component's own header (`replay_component.js`), stated once there.
"""

import hashlib
import html
import itertools
import json
import sqlite3
from pathlib import Path

from .assets import rrweb_constants_js, vendored
from .clock import utc_stamp
from .rrweb_constants import EventType
from .slices import covering_pair, slice_events

COMPONENT_NAME = "locus-replay.js"


def component_js() -> str:
    """The component, assembled fresh from its three sources: the seek module (replay_seek.js, shared with the screenshot renderer), the contract in replay_component.js — its header is the contract a composer reads — then the vendored rrweb-player (JS, and CSS injected at load), which the contract touches only at mount time. One file, no external references — a single script tag and it works from file://."""
    css = vendored("rrweb-player-*.min.css").read_text()
    inject = (
        "\n;(() => { const s = document.createElement('style'); "
        f"s.textContent = {json.dumps(css)}; "
        "document.head.appendChild(s); })();\n"
    )
    here = Path(__file__)
    seek = here.with_name("replay_seek.js").read_text()
    contract = here.with_name("replay_component.js").read_text()
    return (
        rrweb_constants_js()
        + seek
        + contract
        + vendored("rrweb-player-*.min.js").read_text()
        + inject
    )


def payload_js(conn: sqlite3.Connection, slice_ids: list[int]) -> str:
    """One slice set's replay payload: a script registering {visitor, slices, viewport, url, events} on window.LOCUS_REPLAYS — `slices` one {id, start_ts, end_ts, snapshot_ts} per slice in time order, the covering snapshot being the first moment of each the player can paint. The set is one visitor's and time-disjoint, each slice carrying its own Meta+FullSnapshot (the player replays them as ordinary checkouts), so the whole set plays through on one clock; a set the component could not honestly play — two visitors interleaved, time-overlapping slices (concurrent page contexts merged into one stream), a player sized off no viewport — is refused here rather than rendered plausible and wrong."""
    rows = [
        conn.execute(
            "SELECT id, recorder_slice, visitor_id, n_events, start_ts, end_ts FROM slices WHERE id = ?",
            (sid,),
        ).fetchone()
        for sid in slice_ids
    ]
    if any(r is None for r in rows):
        raise ValueError(f"unknown slice in set {slice_ids}")
    visitors = {r["visitor_id"] for r in rows}
    if len(visitors) > 1:
        raise ValueError(
            f"a slice set is one visitor; got {sorted(visitors)} for {slice_ids}"
        )
    rows.sort(key=lambda r: r["start_ts"])
    for prev, nxt in itertools.pairwise(rows):
        if nxt["start_ts"] < prev["end_ts"]:
            raise ValueError(
                f"slices {prev['recorder_slice']} and {nxt['recorder_slice']} "
                f"overlap in time — concurrent page contexts (tabs) interleave "
                f"into one stream no single player can honestly play; mount "
                f"each on its own player"
            )

    events: list[dict] = []
    slices: list[dict] = []
    for r in rows:
        ev = slice_events(conn, r["id"])
        if not ev:
            raise ValueError(f"slice {r['id']} has no events")
        pair = covering_pair(ev)
        if pair is None:
            raise ValueError(
                f"slice {r['recorder_slice']} holds no covering (Meta, FullSnapshot) pair — nothing a player could paint for it"
            )
        events.extend(ev)
        slices.append(
            {
                "id": r["recorder_slice"],
                "start_ts": ev[0]["timestamp"],
                "end_ts": ev[-1]["timestamp"],
                "snapshot_ts": pair[1]["timestamp"],
            }
        )
    events.sort(key=lambda e: e["timestamp"])

    meta = next(
        (e.get("data", {}) for e in events if e.get("type") == EventType.Meta), None
    )
    if not meta or not meta.get("width") or not meta.get("height"):
        raise ValueError(
            f"slice set {slice_ids} carries no Meta viewport (width/height) — nothing to size the player with"
        )

    url = conn.execute(
        "SELECT url FROM events WHERE slice_id = ? AND url IS NOT NULL ORDER BY timestamp LIMIT 1",
        (rows[0]["id"],),
    ).fetchone()
    record = {
        "visitor": rows[0]["visitor_id"],
        "slices": slices,
        "viewport": {"width": meta["width"], "height": meta["height"]},
        "url": url["url"] if url else None,
        "events": events,
    }
    # `</` is escaped so the payload stays inert even when a composer inlines it
    # into a <script> block instead of loading it by src.
    body = json.dumps(record).replace("</", "<\\/")
    return f"(window.LOCUS_REPLAYS ??= []).push({body});\n"


def set_name(rows) -> str:
    """A slice set's derived name, the address its replay artifacts hang off: the earliest slice's visitor and id, then `_plusN_<digest>` when the set holds more. The digest keys the whole set — two sets sharing a first slice and a count are different payloads, and _plusN alone would let one silently replace the other under a page that still embeds it."""
    rows = sorted(rows, key=lambda r: r["start_ts"])
    first = rows[0]
    name = f"{first['visitor_id']}_{first['recorder_slice']}"
    if len(rows) > 1:
        ids = "+".join(sorted(f"{r['visitor_id']}:{r['recorder_slice']}" for r in rows))
        name += f"_plus{len(rows) - 1}_{hashlib.sha1(ids.encode()).hexdigest()[:6]}"
    return name


def materialize(conn: sqlite3.Connection, pages: Path, rows) -> dict:
    """Write the component and one slice set's payload beneath pages/ at their derived names, refreshed in place — deterministic derivation, never evidence. Returns the two paths."""
    pages.mkdir(parents=True, exist_ok=True)
    component = pages / COMPONENT_NAME
    js = component_js()
    if not component.exists() or component.read_text() != js:
        component.write_text(js)
    payload = pages / f"{set_name(rows)}.js"
    payload.write_text(payload_js(conn, [r["id"] for r in rows]))
    return {"component": component, "payload": payload}


def analysis_slices(analysis_dir: Path) -> list[dict]:
    """An analysis directory's slice table, read from window.json exactly as the engine writes it: a list of rows in window order, each {label, snippet, visitor, slice, slice_id, start_ts, end_ts, screenshot_start_ts}. Opening a replay consumes label, visitor, and slice; the intervals belong to citation resolution and to reading what the analysis's screenshots could cover."""
    manifest_path = analysis_dir / "window.json"
    if not manifest_path.exists():
        raise ValueError(
            f"{analysis_dir} is not an analysis directory — no window.json"
        )
    manifest = json.loads(manifest_path.read_text())
    table = manifest.get("slices")
    if not table:
        raise ValueError(
            f"{manifest_path} carries no slice table (`slices`) — nothing names the visitors and slices to replay"
        )
    return [dict(row) for row in table]


def disjoint_lanes(rows) -> list[list]:
    """One visitor's slices split into the lanes one player can honestly play: in start order, each slice joins the first lane whose every slice has already ended, else opens a new lane. Time-overlapping slices — concurrent page contexts — land in separate lanes, one mount each, so a payload composed from any single lane is time-disjoint by construction."""
    lanes: list[list] = []
    ends: list[int] = []
    for row in sorted(rows, key=lambda r: r["start_ts"]):
        for i, end in enumerate(ends):
            if row["start_ts"] >= end:
                lanes[i].append(row)
                ends[i] = row["end_ts"]
                break
        else:
            lanes.append([row])
            ends.append(row["end_ts"])
    return lanes


def default_page(pages: Path, name: str, mounts: list[tuple[Path, str | None]]) -> Path:
    """The default replay page over materialized payloads: one mount per payload — a sequential stream plays through on one player, concurrent lanes and other visitors mount beside it — every mount on the shared absolute clock, so the page's `#t=` fragment drives them all, and each named in its topbar (an analysis window's slice label, else m1, m2, …) so `&m=` can address it alone. Composed by `locus browse open` at the derived name and refreshed on every open; a page of the composer's own gets its own name beside it (the contract is the component's file header)."""
    scripts = "\n".join(
        f'    <script src="./{path.name}"></script>' for path, _ in mounts
    )
    sections = "\n".join(
        '    <section class="replay"><div class="topbar"'
        + (f' data-label="{html.escape(label, quote=True)}"' if label else "")
        + '></div><div class="mount"></div></section>'
        for _, label in mounts
    )
    page = pages / f"{name}.html"
    page.write_text(f"""<!DOCTYPE html>
<!-- The default replay page, composed {utc_stamp()} by `locus browse open` and
     refreshed on every open — edits here do not survive. To author a page of your
     own, write it under another name beside the material it embeds; the
     composition contract is the header of ./locus-replay.js. -->
<html>
  <head>
    <meta charset="utf-8">
    <title>Locus replay — {html.escape(name)}</title>
    <script src="./locus-replay.js"></script>
{scripts}
    <style>
      body {{ margin: 0; font-family: system-ui; background: #1c1c1e; }}
      .replay {{ height: 100vh; box-sizing: border-box; }}
      .topbar {{ height: 34px; box-sizing: border-box; display: flex; align-items: center; padding: 0 14px; color: #8e8e93; font-size: 12px; letter-spacing: .02em; }}
      .mount {{ height: calc(100vh - 34px); display: flex; justify-content: center; }}
      .mount .rr-player {{ box-shadow: 0 10px 50px rgba(0,0,0,.55); border-radius: 6px; overflow: hidden; }}
    </style>
  </head>
  <body>
{sections}
    <script>
      document.querySelectorAll('.replay').forEach((section, i) => {{
        const replay = LOCUS_REPLAYS[i];
        const slices = replay.slices.length === 1
          ? `slice ${{replay.slices[0].id}}`
          : `slices ${{replay.slices[0].id}} +${{replay.slices.length - 1}}`;
        const u = replay.url && new URL(replay.url);
        const bar = section.querySelector('.topbar');
        const mounted = LocusReplay.mount(section.querySelector('.mount'), replay,
          bar.dataset.label ? {{ name: bar.dataset.label }} : {{}});
        bar.textContent = [
          mounted.name, slices, `visitor ${{replay.visitor}}`,
          `${{replay.events.length}} events`,
          u ? u.host + u.pathname + u.hash : '(no url)'].join(' \\u00b7 ');
      }});
    </script>
  </body>
</html>
""")
    return page
