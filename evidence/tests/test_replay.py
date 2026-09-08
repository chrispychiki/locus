import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from _support import (
    URL,
    click,
    env,
    full_snapshot,
    meta,
    read_recording,
    stamped,
    text_mutation,
)
from locus.evidence.db import connect
from locus.evidence.hydrate import hydrate, pack_raw
from locus.evidence.replay import (
    analysis_slices,
    component_js,
    default_page,
    disjoint_lanes,
    materialize,
    payload_js,
    set_name,
)
from locus.evidence.rrweb_constants import EventType, IncrementalSource, NodeType
from locus.evidence.slices import materialize_slices

FIXTURE = Path(__file__).parent / "fixtures" / "demo_recording.json"


def players_mounted(n):
    """The player mounts the recording into an iframe asynchronously. Waiting on that condition — the iframe having children — is what "the replay is up" actually means; a fixed sleep is a guess that is either slower than it needs to be or, on a loaded machine, a flake."""
    return f"""
      () => [...document.querySelectorAll('.rr-player iframe')]
        .filter((f) => (f.contentDocument?.body?.childElementCount ?? 0) > 0)
        .length >= {n}
    """


def replayable_slices(tmp_path):
    visitor_id, events = read_recording(FIXTURE)
    conn = connect(tmp_path / "events.db")
    hydrate(conn, visitor_id, events)
    materialize_slices(conn, visitor_id)
    return conn


def rows_for(conn, slice_ids):
    return [
        conn.execute("SELECT * FROM slices WHERE id = ?", (sid,)).fetchone()
        for sid in slice_ids
    ]


def compose(tmp_path, conn, slice_ids):
    """A page composed the way `browse open` composes one: material materialized at derived names, the default page over it."""
    rows = rows_for(conn, slice_ids)
    written = materialize(conn, tmp_path, rows)
    return default_page(tmp_path, set_name(rows), [(written["payload"], None)])


@contextmanager
def opened(page_path, fragment="", players=1):
    """The composed page, driven to the point where the replay is actually mounted."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(
            page_path.resolve().as_uri() + fragment, wait_until="domcontentloaded"
        )
        page.wait_for_selector(".rr-controller", timeout=10_000)
        page.wait_for_function(players_mounted(players), timeout=10_000)
        try:
            yield page, errors
        finally:
            browser.close()


def test_a_citation_fragment_opens_the_composed_page_at_its_moment(tmp_path):
    conn = replayable_slices(tmp_path)
    slice_row = conn.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()
    cited = (slice_row["start_ts"] + slice_row["end_ts"]) // 2

    page_path = compose(tmp_path, conn, [slice_row["id"]])

    with opened(page_path, f"#t={cited}") as (page, errors):
        timer = page.evaluate(
            "() => document.querySelector('.rr-timeline__time')?.textContent"
        )

    assert not errors
    assert timer not in (None, "00:00"), (
        "the player pre-seeks to the cited moment, so a citation opens where it points"
    )


def test_the_next_citation_moves_the_replay_without_a_reload(tmp_path):
    # Walking an answer's citations points the fragment at one moment after another on the
    # page already open. Changing only the fragment is a same-document navigation, so the
    # component's own hashchange seek is the whole mechanism — the document never reloads,
    # and a replay left standing at the previous moment is the failure.
    conn = replayable_slices(tmp_path)
    slice_row = conn.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()
    span = slice_row["end_ts"] - slice_row["start_ts"]
    first, second = (
        slice_row["start_ts"] + span // 5,
        slice_row["start_ts"] + span * 4 // 5,
    )

    page_path = compose(tmp_path, conn, [slice_row["id"]])

    offset = "() => window.locus.player.getReplayer().getCurrentTime()"
    with opened(page_path, f"#t={first}") as (page, errors):
        at_first = page.evaluate(offset)
        page.evaluate("() => { window.__survived = true; }")
        page.evaluate(f"() => {{ location.hash = '#t={second}'; }}")
        page.wait_for_function(f"{offset} === {span * 4 // 5 + 1}", timeout=5_000)
        at_second = page.evaluate(offset)
        survived = page.evaluate("() => window.__survived === true")

    assert not errors
    # rrweb's clock at B holds the events stamped before B, so a moment's own events sit
    # under the reading one millisecond past it.
    assert at_first == span // 5 + 1
    assert at_second == span * 4 // 5 + 1
    assert survived, "the document never reloaded — the seek is the component's own"


def test_a_range_fragment_marks_the_timeline_and_stops_play_at_its_end(tmp_path):
    conn = replayable_slices(tmp_path)
    slice_row = conn.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()
    start = slice_row["start_ts"]
    seg_lo, seg_hi = start + 200, start + 900

    page_path = compose(tmp_path, conn, [slice_row["id"]])

    with opened(page_path, f"#t={seg_lo}-{seg_hi}") as (page, errors):
        band = page.evaluate(
            """() => { const b = document.querySelector('.locus-range');
                       if (!b) return null;
                       return { left: b.style.left, width: b.style.width }; }"""
        )
        assert band is not None, "the range is marked on the timeline"
        assert band["width"] not in ("", "0%")

        page.evaluate(
            """() => { window.__paused = false;
                       window.locus.player.getReplayer().on('pause', () => { window.__paused = true; });
                       window.locus.player.play(); }"""
        )
        page.wait_for_function("() => window.__paused", timeout=10_000)
        current = page.evaluate(
            "() => document.querySelector('.rr-timeline__time').textContent"
        )

    assert not errors
    # A 700ms range: playback pauses itself at the range's end rather than running on —
    # the timer reads 00:00 (sub-second, floored) or 00:01, never further into the replay.
    assert current in ("00:00", "00:01"), (
        f"playback stopped at the range end (timer read {current!r})"
    )


def test_one_player_plays_a_multi_slice_set_through_its_snapshots(tmp_path):
    # The substrate the default page stands on: a time-disjoint set's later
    # Meta+FullSnapshots replay as ordinary checkouts on one mount — the DOM must
    # actually transition, seeked into directly or played across the boundary.
    conn = replayable_slices(tmp_path)
    rows = conn.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY start_ts LIMIT 2"
    ).fetchall()
    assert len(rows) == 2, "the fixture carries a multi-slice visitor"

    page_path = compose(tmp_path, conn, [r["id"] for r in rows])

    title = "() => document.querySelector('.rr-player iframe').contentDocument.title"
    with opened(page_path) as (page, errors):
        assert page.evaluate("() => window.LOCUS_REPLAYS[0].slices.length") == 2
        first_title = page.evaluate(title)

        mid_b = (rows[1]["start_ts"] + rows[1]["end_ts"]) // 2
        page.evaluate(f"() => window.locus.seek({mid_b})")
        page.wait_for_function(f"{title} !== {json.dumps(first_title)}", timeout=5_000)
        seeked_title = page.evaluate(title)

        page.evaluate(f"() => window.locus.seek({rows[0]['end_ts'] - 500})")
        page.wait_for_function(f"{title} === {json.dumps(first_title)}", timeout=5_000)
        page.evaluate(
            "() => { window.locus.player.setSpeed(8); window.locus.player.play(); }"
        )
        page.wait_for_function(
            f"{title} === {json.dumps(seeked_title)}", timeout=15_000
        )

    assert not errors
    assert seeked_title != first_title, (
        "the fixture's adjacent slices are different pages, or this proves nothing"
    )


def test_the_default_page_mounts_one_attributed_player_per_payload(tmp_path):
    conn = replayable_slices(tmp_path)
    rows = conn.execute(
        "SELECT * FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 2"
    ).fetchall()
    assert len(rows) == 2

    a = materialize(conn, tmp_path, [rows[0]])
    b = materialize(conn, tmp_path, [rows[1]])
    page_path = default_page(
        tmp_path,
        "two-up",
        [(a["payload"], "A: first look"), (b["payload"], "B: second look")],
    )

    cited = (rows[0]["start_ts"] + rows[0]["end_ts"]) // 2
    timers = (
        "() => [...document.querySelectorAll('.rr-player')]"
        ".map((p) => p.querySelector('.rr-timeline__time').textContent)"
    )
    with opened(page_path, players=2) as (page, errors):
        page.evaluate(
            "() => addEventListener('hashchange', () => { window.__hashed = true; })"
        )
        assert page.evaluate("() => LocusReplay.mounts.length") == 2
        bars = page.evaluate(
            "() => [...document.querySelectorAll('.topbar')].map((b) => b.textContent)"
        )
        # The mounts are two different slices, so no one moment is on both: the
        # page-wide seek is refused whole, and neither mount moves.
        page.evaluate(f"() => {{ location.hash = '#t={cited}'; }}")
        page.wait_for_function("() => window.__hashed", timeout=5_000)
        after_refusal = page.evaluate(timers)
        refused = list(errors)
        # Addressed by its name, the cited mount seeks alone.
        page.evaluate(
            f"() => {{ location.hash = '#t={cited}&m=A%3A%20first%20look'; }}"
        )
        page.wait_for_function(f"{timers}[0] !== '00:00'", timeout=5_000)
        after_named = page.evaluate(timers)

    assert len(bars) == 2
    assert bars[0].startswith("A: first look") and "visitor" in bars[0]
    assert bars[1].startswith("B: second look")
    assert after_refusal == ["00:00", "00:00"]
    assert len(refused) == 1 and "B: second look cannot show" in refused[0]
    assert after_named[0] != "00:00" and after_named[1] == "00:00"
    assert errors == refused, "the named seek raised nothing"


def test_two_mounts_compose_side_by_side_and_one_fragment_is_one_request_across_them(
    tmp_path,
):
    conn = replayable_slices(tmp_path)
    rows = conn.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 2"
    ).fetchall()
    assert len(rows) == 2

    (tmp_path / "locus-replay.js").write_text(component_js())
    (tmp_path / "a.js").write_text(payload_js(conn, [rows[0]["id"]]))
    (tmp_path / "b.js").write_text(payload_js(conn, [rows[1]["id"]]))
    page_path = tmp_path / "side-by-side.html"
    page_path.write_text("""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <script src="./locus-replay.js"></script>
    <script src="./a.js"></script>
    <script src="./b.js"></script>
    <style>body { margin: 0; display: flex; } .pane { width: 50vw; height: 100vh; }</style>
  </head>
  <body>
    <div id="a" class="pane"></div>
    <div id="b" class="pane"></div>
    <script>
      LocusReplay.mount(document.getElementById('a'), LOCUS_REPLAYS[0]);
      LocusReplay.mount(document.getElementById('b'), LOCUS_REPLAYS[1]);
    </script>
  </body>
</html>
""")

    cited = (rows[0]["start_ts"] + rows[0]["end_ts"]) // 2
    timers = (
        "() => [...document.querySelectorAll('.pane')].map((p) => "
        "p.querySelector('.rr-timeline__time').textContent)"
    )
    # Opened at a moment only the first mount holds: the page still builds both
    # players, the opening seek is refused as one error naming the mount that cannot
    # show it, and nothing moves.
    with opened(page_path, f"#t={cited}", players=2) as (page, errors):
        names = page.evaluate("() => LocusReplay.mounts.map((h) => h.name)")
        page.evaluate("() => window.locus.moved")
        after_refusal = page.evaluate(timers)
        opening = list(errors)
        page.evaluate(f"() => {{ location.hash = '#t={cited}&m=m1'; }}")
        page.wait_for_function(f"{timers}[0] !== '00:00'", timeout=5_000)
        after_named = page.evaluate(timers)
        # The scripting surface refuses the same way, and names an unknown mount.
        thrown = page.evaluate(
            f"""() => {{
              const out = [];
              for (const names of [null, ['m2'], ['m9']]) {{
                try {{ window.locus.seek({cited}, null, names); out.push(null); }}
                catch (e) {{ out.push(String(e)); }}
              }}
              return out;
            }}"""
        )

    assert names == ["m1", "m2"]
    assert after_refusal == ["00:00", "00:00"]
    assert len(opening) == 1 and "m2" in opening[0] and "m1" not in opening[0]
    assert after_named[0] != "00:00" and after_named[1] == "00:00"
    assert errors == opening
    assert "m2" in thrown[0] and "m2" in thrown[1]
    assert "m9" in thrown[2] and "m1" in thrown[2] and "m2" in thrown[2]


def test_the_payload_is_an_inert_registration_script(tmp_path):
    conn = replayable_slices(tmp_path)
    sid = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' LIMIT 1"
    ).fetchone()["id"]

    script = payload_js(conn, [sid])
    assert script.startswith("(window.LOCUS_REPLAYS ??= []).push(")
    assert "</" not in script, (
        "escaped so an inlined payload cannot close a <script> block"
    )


# The payload's refusals. A citation resolves to a slice set, and a payload the player
# cannot honestly play is worse than one that refuses: an interleaved two-visitor replay
# looks like a session and is not, concurrent tabs merged onto one clock replay as
# neither tab, and a player sized off nothing shows a plausible wrong page.


def test_a_set_naming_a_slice_the_db_does_not_hold_is_refused(tmp_path):
    conn = replayable_slices(tmp_path)
    real = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' LIMIT 1"
    ).fetchone()["id"]

    with pytest.raises(ValueError, match="unknown slice"):
        payload_js(conn, [real, 9999])


def test_a_set_spanning_two_visitors_is_refused(tmp_path):
    conn = connect(tmp_path / "events.db")
    for visitor in ("v1", "v2"):
        hydrate(
            conn, visitor, stamped([meta(1000), full_snapshot(1001), click(1500, 3)])
        )
        materialize_slices(conn, visitor)
    slice_ids = [r["id"] for r in conn.execute("SELECT id FROM slices")]
    assert len(slice_ids) == 2

    with pytest.raises(ValueError, match="visitor"):
        payload_js(conn, slice_ids)


def test_a_time_overlapping_set_is_refused(tmp_path):
    # Two page contexts (tabs) recording at once: each slice is honest alone, but their
    # merged stream puts tab B's snapshot mid-way through tab A's events, and one player
    # rebuilds on it — from there A's timeline is gone. Refused, not rendered.
    conn = connect(tmp_path / "events.db")
    tab_a, tab_b = "00000000001000-taba", "00000000002000-tabb"
    hydrate(
        conn,
        "v1",
        [
            env(meta(1000), tab_a),
            env(full_snapshot(1001), tab_a),
            env(meta(2000), tab_b),
            env(full_snapshot(2001), tab_b),
            env(click(5000, 3), tab_a),
            env(click(5500, 3), tab_b),
        ],
    )
    materialize_slices(conn, "v1")
    rows = conn.execute("SELECT id, status FROM slices ORDER BY start_ts").fetchall()
    assert [r["status"] for r in rows] == ["replayable", "replayable"]

    with pytest.raises(ValueError, match="overlap"):
        payload_js(conn, [r["id"] for r in rows])


def test_a_set_with_no_meta_viewport_is_refused(tmp_path):
    conn = connect(tmp_path / "events.db")
    hydrate(conn, "v1", stamped([meta(1000), full_snapshot(1001)]))
    materialize_slices(conn, "v1")
    dimensionless = {"type": EventType.Meta, "timestamp": 1000, "data": {"href": URL}}
    conn.execute(
        "UPDATE events SET raw_json = ? WHERE type = ?",
        (pack_raw(json.dumps(dimensionless)), EventType.Meta),
    )
    conn.commit()
    [sid] = [r["id"] for r in conn.execute("SELECT id FROM slices")]

    with pytest.raises(ValueError, match="viewport"):
        payload_js(conn, [sid])


# The analysis-directory seam: window.json's slice table is the citation→replay
# resolution contract — rows of {label, visitor, slice, slice_id, start_ts,
# end_ts} in window order, exactly as the engine writes it. Opening consumes
# label, visitor, and slice, and fails loud when the table is not there.


def test_analysis_slices_reads_the_slice_table_in_window_order(tmp_path):
    (tmp_path / "window.json").write_text(
        json.dumps(
            {
                "slices": [
                    {
                        "label": "S1",
                        "visitor": "a1b2c3d4-full",
                        "slice": "s1",
                        "slice_id": 1,
                        "start_ts": 1000,
                        "end_ts": 2000,
                    },
                    {
                        "label": "S2",
                        "visitor": "e5f6a7b8-full",
                        "slice": "s3",
                        "slice_id": 3,
                        "start_ts": 2000,
                        "end_ts": 3000,
                    },
                ]
            }
        )
    )
    table = analysis_slices(tmp_path)
    assert [s["label"] for s in table] == ["S1", "S2"]
    assert table[0]["visitor"] == "a1b2c3d4-full"
    assert table[0]["slice"] == "s1"


def test_disjoint_lanes_split_only_where_time_overlaps():
    a, b, c, d = (
        {"start_ts": 0, "end_ts": 100},
        {"start_ts": 10, "end_ts": 20},
        {"start_ts": 30, "end_ts": 40},
        {"start_ts": 150, "end_ts": 160},
    )
    assert disjoint_lanes([a, d]) == [[a, d]], (
        "a visitor's sequential slices are one lane — one player plays them through"
    )
    assert disjoint_lanes([d, b, c, a]) == [[a, d], [b, c]], (
        "concurrent lanes split; each lane stays time-disjoint and start-ordered"
    )


def test_a_directory_without_a_manifest_is_not_an_analysis(tmp_path):
    with pytest.raises(ValueError, match="window.json"):
        analysis_slices(tmp_path)


def test_a_manifest_without_a_slice_table_fails_loud(tmp_path):
    (tmp_path / "window.json").write_text(json.dumps({"slice_ids": [1]}))
    with pytest.raises(ValueError, match="table"):
        analysis_slices(tmp_path)


def test_materialized_names_derive_from_the_set(tmp_path):
    conn = replayable_slices(tmp_path)
    rows = conn.execute(
        "SELECT * FROM slices WHERE status='replayable' ORDER BY start_ts"
    ).fetchall()

    single = set_name([rows[0]])
    assert single == f"{rows[0]['visitor_id']}_{rows[0]['recorder_slice']}"

    # The digest keys the whole set: two sets sharing a first slice and a count are
    # different payloads, and _plusN alone would let one silently replace the other
    # under a page that still embeds it.
    first_pair = set_name([rows[0], rows[1]])
    second_pair = set_name([rows[0], rows[2]])
    assert first_pair.startswith(f"{single}_plus1_")
    assert first_pair != second_pair

    # Deterministic derivation, not evidence: re-materializing refreshes in place.
    written = materialize(conn, tmp_path / "pages", [rows[0]])
    before = written["payload"].read_text()
    again = materialize(conn, tmp_path / "pages", [rows[0]])
    assert again["payload"] == written["payload"]
    assert again["payload"].read_text() == before
    assert written["component"].name == "locus-replay.js"
    assert written["component"].stat().st_size > 100_000, (
        "the component carries the whole player — nothing left to a CDN"
    )


def synthetic_page(tmp_path, events, visitor="v1"):
    conn = connect(tmp_path / "events.db")
    hydrate(conn, visitor, stamped(events))
    materialize_slices(conn, visitor)
    ids = [r["id"] for r in conn.execute("SELECT id FROM slices ORDER BY start_ts")]
    return conn, ids


def page_snapshot(ts, css=None):
    """A snapshot whose body holds an element — the mount counts a replay as up once the frame's body has element children — with the text node 5 inside it."""
    from test_render import _page_snapshot

    paragraph = {
        "type": NodeType.Element,
        "tagName": "p",
        "id": 4,
        "attributes": {},
        "childNodes": [{"type": NodeType.Text, "id": 5, "textContent": "hello"}],
    }
    return _page_snapshot(ts, [paragraph], css=css)


BODY = (
    "() => document.querySelector('.rr-player iframe').contentDocument.body.textContent"
)


def test_a_moment_shows_its_own_events(tmp_path):
    # A citation names the moment something happened; the frame at it must hold that
    # something. rrweb's own seek stops one event short of the offset it is given.
    conn, ids = synthetic_page(
        tmp_path,
        [
            meta(1000),
            page_snapshot(1001),
            text_mutation(2000, 5, "goodbye"),
            click(3000, 4),
        ],
    )
    page_path = compose(tmp_path, conn, ids)

    with opened(page_path, "#t=2000") as (page, errors):
        page.wait_for_function(f"{BODY} === 'goodbye'", timeout=5_000)
        at_change = page.evaluate(BODY)
        page.evaluate("() => window.locus.seek(1999)")
        page.wait_for_function(f"{BODY} === 'hello'", timeout=5_000)
        before = page.evaluate(BODY)

    assert not errors
    assert (at_change, before) == ("goodbye", "hello")


def test_a_moment_the_mount_holds_no_page_at_is_refused_and_nothing_moves(tmp_path):
    # Three moments a recording holds no page at: a slice's lead (opened, page not yet
    # captured), the gap between two slices (no context recording), and outside the set.
    # The mount refuses each by name, and the frame stays where it was.
    conn, ids = synthetic_page(
        tmp_path,
        [
            {"type": EventType.DomContentLoaded, "timestamp": 1000, "data": {}},
            meta(5000),
            page_snapshot(5001),
            text_mutation(7000, 5, "goodbye"),
            meta(9000),
            page_snapshot(9001),
            text_mutation(9500, 5, "again"),
        ],
    )
    first, second = [
        r["recorder_slice"]
        for r in conn.execute("SELECT recorder_slice FROM slices ORDER BY start_ts")
    ]
    page_path = compose(tmp_path, conn, ids)

    attempts = """() => {
      const out = {};
      for (const t of [1000, 4999, 8000, 9000, 999, 9501, 7000.5, 'x']) {
        try { window.locus.seek(t); out[t] = null; } catch (e) { out[t] = String(e); }
      }
      return out;
    }"""
    with opened(page_path, "#t=7000") as (page, errors):
        page.wait_for_function(f"{BODY} === 'goodbye'", timeout=5_000)
        before = page.evaluate(
            "() => window.locus.player.getReplayer().getCurrentTime()"
        )
        thrown = page.evaluate(attempts)
        page.evaluate("() => window.locus.moved")
        body = page.evaluate(BODY)
        after = page.evaluate(
            "() => window.locus.player.getReplayer().getCurrentTime()"
        )
        # A showable moment still seeks after every refusal.
        page.evaluate("() => window.locus.seek(9500)")
        page.wait_for_function(f"{BODY} === 'again'", timeout=5_000)

    assert not errors
    assert (body, after) == ("goodbye", before)
    lead = f"nothing captured at t={{}}; slice {first} opened at t=1000 and its page was first captured at t=5001"
    assert (
        thrown["1000"]
        == f"Error: LocusReplay: m1 cannot show t=1000 — {lead.format(1000)}"
    )
    assert (
        thrown["4999"]
        == f"Error: LocusReplay: m1 cannot show t=4999 — {lead.format(4999)}"
    )
    assert thrown["8000"] == (
        f"Error: LocusReplay: m1 cannot show t=8000 — nothing recorded at t=8000; "
        f"slice {first} ended at t=7000 and slice {second} opened at t=9000"
    )
    assert thrown["9000"] == (
        f"Error: LocusReplay: m1 cannot show t=9000 — nothing captured at t=9000; "
        f"slice {second} opened at t=9000 and its page was first captured at t=9001"
    )
    span = "nothing recorded at t={}; the recording spans t=1000 to t=9500"
    assert (
        thrown["999"]
        == f"Error: LocusReplay: m1 cannot show t=999 — {span.format(999)}"
    )
    assert (
        thrown["9501"]
        == f"Error: LocusReplay: m1 cannot show t=9501 — {span.format(9501)}"
    )
    assert (
        thrown["7000.5"] == "Error: LocusReplay: m1: t=7000.5 is not an epoch-ms moment"
    )
    assert thrown["x"] == "Error: LocusReplay: m1: t=x is not an epoch-ms moment"


def test_an_opening_fragment_the_page_cannot_show_is_one_error_over_a_built_page(
    tmp_path,
):
    # The fragment is applied once the document has parsed, after every mount is up: a
    # refusal is a page error naming the mount, the players all exist at 00:00, and the
    # next fragment that can be shown seeks them.
    conn, ids = synthetic_page(
        tmp_path,
        [
            {"type": EventType.DomContentLoaded, "timestamp": 1000, "data": {}},
            meta(5000),
            page_snapshot(5001),
            text_mutation(7000, 5, "goodbye"),
        ],
    )
    page_path = compose(tmp_path, conn, ids)

    with opened(page_path, "#t=3000") as (page, errors):
        page.evaluate("() => window.locus.moved")
        opening = list(errors)
        timer = page.evaluate(
            "() => document.querySelector('.rr-timeline__time').textContent"
        )
        page.evaluate("() => { location.hash = '#t=7000'; }")
        page.wait_for_function(f"{BODY} === 'goodbye'", timeout=5_000)

    assert timer == "00:00"
    assert (
        len(opening) == 1
        and "m1 cannot show t=3000 — nothing captured at t=3000" in opening[0]
    )
    assert errors == opening


def test_a_range_the_mount_cannot_hold_is_refused(tmp_path):
    conn, ids = synthetic_page(
        tmp_path, [meta(1000), page_snapshot(1001), text_mutation(2000, 5, "goodbye")]
    )
    page_path = compose(tmp_path, conn, ids)

    attempts = """() => {
      const out = [];
      for (const [s, e] of [[1500, 1200], [1500, 2001], [1500, 2000]]) {
        try { window.locus.seek(s, e); out.push(null); } catch (err) { out.push(String(err)); }
      }
      return out;
    }"""
    with opened(page_path) as (page, errors):
        thrown = page.evaluate(attempts)
        band = page.evaluate("() => document.querySelector('.locus-range') !== null")

    assert not errors
    assert (
        thrown[0]
        == "Error: LocusReplay: m1: the range ends at t=1200, before it starts at t=1500"
    )
    assert (
        thrown[1]
        == "Error: LocusReplay: m1: the range ends at t=2001, past its recording's end at t=2000"
    )
    assert thrown[2] is None and band


def test_seeked_scrolls_land_instantly_whatever_the_page_declares(tmp_path):
    # A seeked scroll defers to the page's CSS scroll-behavior; a page declaring smooth
    # animates it, and a frame read mid-animation holds an earlier scroll position. The
    # mount's injected rule overrides the declaration inside the replayed document.
    from test_render import _page_snapshot

    tall = {
        "type": NodeType.Element,
        "tagName": "div",
        "id": 4,
        "attributes": {"style": "height: 5000px"},
        "childNodes": [],
    }
    conn, ids = synthetic_page(
        tmp_path,
        [
            meta(1000),
            _page_snapshot(1001, [tall], css="html { scroll-behavior: smooth }"),
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 2000,
                "data": {"source": IncrementalSource.Scroll, "id": 1, "x": 0, "y": 700},
            },
        ],
    )
    page_path = compose(tmp_path, conn, ids)

    behavior = (
        "() => { const d = document.querySelector('.rr-player iframe').contentDocument;"
        " return getComputedStyle(d.documentElement).scrollBehavior; }"
    )
    with opened(page_path, "#t=2000") as (page, errors):
        declared = page.evaluate(behavior)
        scrolled = page.evaluate(
            "() => document.querySelector('.rr-player iframe').contentWindow.scrollY"
        )

    assert not errors
    assert declared == "auto"
    assert scrolled == 700


def test_a_seek_lands_on_a_transitions_end_state(tmp_path):
    """A rebuilt document restarts every CSS transition — a drawer that slid in at the recorded moment is mid-slide again on every seek, at a different offset each time. The seek finishes finite animations before it reports moved, so the frame holds the end state the visitor saw."""
    from test_render import _page_snapshot

    drawer = {
        "type": NodeType.Element,
        "tagName": "div",
        "id": 4,
        "attributes": {"id": "drawer"},
        "childNodes": [],
    }
    conn, ids = synthetic_page(
        tmp_path,
        [
            meta(1000),
            _page_snapshot(
                1001,
                [drawer],
                css="#drawer { transform: translateX(300px); transition: transform 60s linear }"
                " #drawer.open { transform: translateX(0) }",
            ),
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 2000,
                "data": {
                    "source": IncrementalSource.Mutation,
                    "texts": [],
                    "attributes": [{"id": 4, "attributes": {"class": "open"}}],
                    "removes": [],
                    "adds": [],
                },
            },
        ],
    )
    page_path = compose(tmp_path, conn, ids)

    transform = (
        "() => { const d = document.querySelector('.rr-player iframe').contentDocument;"
        " return getComputedStyle(d.getElementById('drawer')).transform; }"
    )
    with opened(page_path, "#t=2000") as (page, errors):
        landed = page.evaluate(transform)

    assert not errors
    assert landed in ("none", "matrix(1, 0, 0, 1, 0, 0)")


def test_a_re_seek_shows_the_moment_not_the_route_to_it(tmp_path):
    # rrweb carries state across seeks on one replayer: its virtual document keeps the
    # last document scroll through every rebuild and re-applies it to any later batch that
    # mutates without scrolling. A mount opens a fresh player at every seek, so the frame
    # is the moment's alone.
    from test_render import _page_snapshot

    tall = {
        "type": NodeType.Element,
        "tagName": "div",
        "id": 4,
        "attributes": {"style": "height: 5000px"},
        "childNodes": [{"type": NodeType.Text, "id": 5, "textContent": "hello"}],
    }
    conn, ids = synthetic_page(
        tmp_path,
        [
            meta(1000),
            _page_snapshot(1001, [tall]),
            text_mutation(1200, 5, "early"),
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 2000,
                "data": {"source": IncrementalSource.Scroll, "id": 1, "x": 0, "y": 700},
            },
            text_mutation(2001, 5, "scrolled"),
        ],
    )
    page_path = compose(tmp_path, conn, ids)
    state = (
        "() => { const f = document.querySelector('.rr-player iframe');"
        " return [f.contentDocument.body.textContent.trim(), f.contentWindow.scrollY]; }"
    )

    with opened(page_path, "#t=2001") as (page, errors):
        page.wait_for_function(f"({state})()[1] === 700", timeout=5_000)
        later = page.evaluate(state)
        page.evaluate("() => window.locus.seek(1500)")
        page.wait_for_function(f"({state})()[0] === 'early'", timeout=5_000)
        earlier = page.evaluate(state)
    with opened(page_path, "#t=1500") as (page, errors2):
        page.wait_for_function(f"({state})()[0] === 'early'", timeout=5_000)
        fresh = page.evaluate(state)

    assert not errors and not errors2
    assert later == ["scrolled", 700]
    assert earlier == fresh == ["early", 0]


def test_only_a_moved_player_is_replaced_by_a_seek(tmp_path):
    # An untouched player shows a moment exactly as a fresh one would, so the first seek
    # keeps it; once anything has moved it — a seek, the controller, playback — the next
    # seek opens a fresh one, and the handle's player is whichever is current.
    conn, ids = synthetic_page(
        tmp_path, [meta(1000), page_snapshot(1001), text_mutation(2000, 5, "goodbye")]
    )
    page_path = compose(tmp_path, conn, ids)
    mark = (
        "(tag) => { document.querySelector('.rr-player iframe').dataset.mark = tag; }"
    )
    marked = "() => document.querySelector('.rr-player iframe').dataset.mark ?? null"
    same_player = "() => window.locus.player === LocusReplay.mounts[0].player"

    with opened(page_path) as (page, errors):
        page.evaluate(mark, "untouched")
        page.evaluate("() => window.locus.seek(2000)")
        page.wait_for_function(f"{BODY} === 'goodbye'", timeout=5_000)
        after_first = page.evaluate(marked)
        page.evaluate(mark, "moved")
        page.evaluate("() => window.locus.seek(1500)")
        page.wait_for_function(f"{BODY} === 'hello'", timeout=5_000)
        after_second = page.evaluate(marked)
        page.evaluate(mark, "seeked")
        page.evaluate("() => window.locus.player.goto(0, false)")
        page.evaluate("() => window.locus.seek(2000)")
        page.wait_for_function(f"{BODY} === 'goodbye'", timeout=5_000)
        after_scrub = page.evaluate(marked)
        current = page.evaluate(same_player)
        players = page.evaluate("() => document.querySelectorAll('.rr-player').length")

    assert not errors
    assert (after_first, after_second, after_scrub) == ("untouched", None, None)
    assert current and players == 1


@pytest.mark.parametrize(
    "fragment, why",
    [
        ("#t=abc", "not an epoch-ms moment"),
        ("#t=2000.7", "not an epoch-ms moment"),
        ("#t=2000-", "not an epoch-ms moment"),
        ("#t=", "not an epoch-ms moment"),
        ("#t=2000&m=", "empty m"),
        ("#m=m1", "names no t"),
        ("#t=2000&t=2001", "more than once"),
        ("#t=2000&mount=m1", "names mount, which is not a key"),
    ],
)
def test_a_malformed_fragment_is_refused_not_guessed_at(tmp_path, fragment, why):
    # #t=2000.7 seeking to 2000, #t=abc seeking nowhere, and &m= with no name seeking every
    # mount would each answer a request the page never understood.
    conn, ids = synthetic_page(
        tmp_path, [meta(1000), page_snapshot(1001), text_mutation(2000, 5, "goodbye")]
    )
    page_path = compose(tmp_path, conn, ids)
    offset = "() => window.locus.player.getReplayer().getCurrentTime()"

    with opened(page_path, fragment) as (page, errors):
        at = page.evaluate(offset)
        page.evaluate("() => { location.hash = '#t=2000'; }")
        page.wait_for_function(f"{BODY} === 'goodbye'", timeout=5_000)
        thrown = page.evaluate(
            "() => { try { location.hash = '#t=1500&m='; return null; } catch (e) { return String(e); } }"
        )

    assert len(errors) >= 1 and why in errors[0] and "expected #t=" in errors[0]
    assert at <= 0, "an untouched player: rrweb's clock reads no offset before any seek"
    assert thrown is None and len(errors) == 2 and "empty m" in errors[1]


def test_a_fragment_that_is_not_a_seek_is_left_to_the_page(tmp_path):
    conn, ids = synthetic_page(
        tmp_path, [meta(1000), page_snapshot(1001), text_mutation(2000, 5, "goodbye")]
    )
    page_path = compose(tmp_path, conn, ids)

    with opened(page_path, "#summary") as (page, errors):
        at = page.evaluate("() => window.locus.player.getReplayer().getCurrentTime()")

    assert not errors and at <= 0


def tall_png(path, height):
    """A 4×height PNG, tall enough that an <img> without dimensions reflows the page when it loads."""
    import struct
    import zlib

    raw = b"".join(b"\x00" + b"\x80\x80\x80\xff" * 4 for _ in range(height))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return path


def scroll(ts, node_id, y):
    return {
        "type": EventType.IncrementalSnapshot,
        "timestamp": ts,
        "data": {"source": IncrementalSource.Scroll, "id": node_id, "x": 0, "y": y},
    }


def test_a_recorded_scroll_holds_while_the_pages_resources_arrive(tmp_path):
    # rrweb applies the recorded scroll the instant it seeks; the page's images and fonts
    # arrive after, reflow the document, and the browser's scroll anchoring then moves the
    # scroll to keep the same content in view — a position no event recorded. The mount
    # turns anchoring off in the replayed document, so the scroll stays what the events said.
    from test_render import _page_snapshot

    img = {
        "type": NodeType.Element,
        "tagName": "img",
        "id": 4,
        "attributes": {"src": tall_png(tmp_path / "tall.png", 2000).as_uri()},
        "childNodes": [],
    }
    filler = {
        "type": NodeType.Element,
        "tagName": "div",
        "id": 6,
        "attributes": {"style": "height: 5000px"},
        "childNodes": [{"type": NodeType.Text, "id": 7, "textContent": "hello"}],
    }
    conn, ids = synthetic_page(
        tmp_path,
        [meta(1000), _page_snapshot(1001, [img, filler]), scroll(2000, 1, 500)],
    )
    page_path = compose(tmp_path, conn, ids)
    loaded = (
        "() => document.querySelector('.rr-player iframe').contentDocument"
        ".querySelector('img').naturalHeight === 2000"
    )

    with opened(page_path, "#t=2000") as (page, errors):
        page.wait_for_function(loaded, timeout=5_000)
        page.evaluate("() => window.locus.moved")
        scrolled = page.evaluate(
            "() => document.querySelector('.rr-player iframe').contentWindow.scrollY"
        )

    assert not errors
    assert scrolled == 500


def test_a_player_never_writes_into_the_payload(tmp_path):
    # rrweb edits the events it replays: a mutation it applies a second time — its seek
    # resumes from a MouseMove's first sampled position, re-applying what came after —
    # keeps only the removes still in its mirror, on the event object itself. Every
    # player replays its own copy, so a scrub on one never changes what a later one shows.
    from test_render import _page_snapshot

    kept = {
        "type": NodeType.Element,
        "tagName": "p",
        "id": 4,
        "attributes": {},
        "childNodes": [{"type": NodeType.Text, "id": 5, "textContent": "hello"}],
    }
    doomed = {
        "type": NodeType.Element,
        "tagName": "p",
        "id": 6,
        "attributes": {},
        "childNodes": [{"type": NodeType.Text, "id": 7, "textContent": "bye"}],
    }
    removal = {
        "type": EventType.IncrementalSnapshot,
        "timestamp": 2000,
        "data": {
            "source": IncrementalSource.Mutation,
            "texts": [],
            "attributes": [],
            "removes": [{"parentId": 3, "id": 6}],
            "adds": [],
        },
    }
    move = {
        "type": EventType.IncrementalSnapshot,
        "timestamp": 2500,
        "data": {
            "source": IncrementalSource.MouseMove,
            "positions": [{"x": 5, "y": 5, "id": 4, "timeOffset": -600}],
        },
    }
    conn, ids = synthetic_page(
        tmp_path,
        [
            meta(1000),
            _page_snapshot(1001, [kept, doomed]),
            removal,
            move,
            click(3000, 4),
        ],
    )
    page_path = compose(tmp_path, conn, ids)
    removes = (
        "() => LOCUS_REPLAYS[0].events.find((e) => e.data && e.data.removes)"
        ".data.removes.length"
    )

    with opened(page_path, "#t=2600") as (page, errors):
        page.wait_for_function(f"{BODY} === 'hello'", timeout=5_000)
        page.evaluate("() => window.locus.player.goto(1600, false)")
        page.wait_for_function(
            "() => window.locus.player.getReplayer().getCurrentTime() === 1600",
            timeout=5_000,
        )
        page.evaluate("() => window.locus.seek(2600)")
        after = page.evaluate(BODY)
        intact = page.evaluate(removes)

    assert not errors
    assert (after, intact) == ("hello", 1)


def test_mount_names_that_are_not_a_list_are_refused_in_words(tmp_path):
    conn, ids = synthetic_page(tmp_path, [meta(1000), page_snapshot(1001)])
    page_path = compose(tmp_path, conn, ids)

    with opened(page_path) as (page, errors):
        thrown = page.evaluate(
            "() => { try { window.locus.seek(1001, null, 'm1'); return null; }"
            " catch (e) { return String(e); } }"
        )

    assert not errors
    assert '"m1"' in thrown


def test_an_opening_seek_waits_for_the_page_to_lay_out_before_its_scrolls_apply(
    tmp_path,
):
    # A scroll applied while the page's images are still arriving lands where the
    # content's height at that instant allows and stays there. The player is built, its
    # snapshot page's fonts and images are waited for, and only then is it moved.
    from test_render import _page_snapshot

    img = {
        "type": NodeType.Element,
        "tagName": "img",
        "id": 4,
        "attributes": {"src": tall_png(tmp_path / "tall.png", 2000).as_uri()},
        "childNodes": [],
    }
    scroller = {
        "type": NodeType.Element,
        "tagName": "div",
        "id": 6,
        "attributes": {"style": "height: 300px; overflow: auto"},
        "childNodes": [img, {"type": NodeType.Text, "id": 7, "textContent": "hello"}],
    }
    conn, ids = synthetic_page(
        tmp_path, [meta(1000), _page_snapshot(1001, [scroller]), scroll(2000, 6, 1500)]
    )
    page_path = compose(tmp_path, conn, ids)
    scrolled = (
        "() => document.querySelector('.rr-player iframe').contentDocument"
        ".querySelector('div').scrollTop"
    )

    with opened(page_path, "#t=2000") as (page, errors):
        page.wait_for_function(f"({scrolled})() > 0", timeout=5_000)
        first = page.evaluate(scrolled)

    assert not errors
    assert first == 1500
