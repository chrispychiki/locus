from pathlib import Path

import pytest
from _support import read_recording
from locus.evidence.db import connect
from locus.evidence.render import RenderSession
from locus.evidence.rrweb_constants import EventType, IncrementalSource, NodeType
from locus.evidence.slices import (
    ORPHAN_REASON,
    covering_snapshot_ts,
    materialize_slices,
)

FIXTURE = Path(__file__).parent / "fixtures" / "demo_recording.json"


@pytest.fixture
def corpus(tmp_path):
    visitor_id, events = read_recording(FIXTURE)
    conn = connect(tmp_path / "events.db")
    hydrate_into(conn, visitor_id, events)
    return conn


def hydrate_into(conn, visitor_id, events):
    from locus.evidence.hydrate import hydrate

    hydrate(conn, visitor_id, events)
    materialize_slices(conn, visitor_id)


def test_render_real_slice(corpus, tmp_path):
    slice_row = corpus.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()
    first = covering_snapshot_ts(corpus, slice_row["id"])
    span = slice_row["end_ts"] - first
    timestamps = [first + span * i // 3 for i in range(4)]

    with RenderSession(
        corpus, {"S1": slice_row["id"]}, tmp_path / "screenshots"
    ) as session:
        screenshots = [session.capture("S1", ts) for ts in timestamps]

    assert len(screenshots) == len(timestamps)
    for screenshot in screenshots:
        png = Path(screenshot.path)
        assert png.exists() and png.stat().st_size > 1000
        assert png.name == f"S1_screenshot_{screenshot.timestamp}.png", (
            "a screenshot's file name is its (label, moment) identity"
        )
        assert set(screenshot.faults) <= {
            "failedImages",
            "pendingImages",
            "fontsTimedOut",
            "networkBusy",
        }
        for key in ("failedImages", "pendingImages"):
            for url in screenshot.faults.get(key, []):
                assert url.startswith("http"), (
                    "a fault names the network resource that did not arrive"
                )


def test_faults_name_the_failed_resource_and_ignore_structural_stubs(corpus, tmp_path):
    """Faults exist to disambiguate a replay artifact from what the visitor really saw, and that
    takes the failed resource's identity. An img that never had a real network source — a
    lazy-load stub, a data: URI — fails or succeeds identically for visitor and replay, so it
    is not a fault."""
    from locus.evidence.render import READINESS_BUDGET_MS, WAIT_READY_JS

    slice_row = corpus.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()
    dead = "https://locus-render-test.invalid/gone.png"

    with RenderSession(
        corpus, {"S1": slice_row["id"]}, tmp_path / "screenshots"
    ) as session:
        session.capture("S1", (slice_row["start_ts"] + slice_row["end_ts"]) // 2)
        session.page.evaluate(
            """(dead) => {
                const doc = document.querySelector('#replayer iframe').contentDocument;
                const gif = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP//'
                          + '/yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
                for (const src of [dead, '', gif]) {
                  const img = doc.createElement('img');
                  if (src) img.src = src;
                  doc.body.appendChild(img);
                }
            }""",
            dead,
        )
        faults = session.page.evaluate(WAIT_READY_JS, READINESS_BUDGET_MS)

    assert faults.get("failedImages") == [dead]


def test_render_session_captures_by_label_across_a_window(corpus, tmp_path):
    rows = corpus.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY start_ts LIMIT 2"
    ).fetchall()
    assert len(rows) == 2, "the fixture carries a multi-slice visitor"
    slices = {f"S{i}": r["id"] for i, r in enumerate(rows, 1)}
    mids = [(r["start_ts"] + r["end_ts"]) // 2 for r in rows]

    with RenderSession(corpus, slices, tmp_path / "screenshots") as session:
        assert len(session.slices) == 2
        # capture out of label order, so the session must re-open S1 after S2
        second = session.capture("S2", mids[1])
        first = session.capture("S1", mids[0])
        # The live replay now holds exactly the addressed slice's events —
        # correctness never depends on capture order.
        injected = session.page.evaluate("() => window.__events.length")
        assert (
            injected
            == corpus.execute(
                "SELECT COUNT(*) n FROM events WHERE slice_id = ?", (rows[0]["id"],)
            ).fetchone()["n"]
        ), "the replay behind the screenshot is the addressed slice"

    for screenshot, row in ((first, rows[0]), (second, rows[1])):
        png = Path(screenshot.path)
        assert png.exists() and png.stat().st_size > 1000
        assert row["start_ts"] <= screenshot.timestamp <= row["end_ts"]

    with (
        RenderSession(corpus, slices, tmp_path / "screenshots2") as session,
        pytest.raises(ValueError, match="unknown label"),
    ):
        session.capture("S9", mids[0])


def test_the_browser_is_launched_by_a_capture_and_released_between_rounds(
    corpus, tmp_path
):
    """A session holds no browser until something is captured, gives it up on release(), and launches it again for the next capture — so a caller that waits on something else between rounds waits holding nothing."""
    slice_row = corpus.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()
    mid = (slice_row["start_ts"] + slice_row["end_ts"]) // 2

    with RenderSession(
        corpus, {"S1": slice_row["id"]}, tmp_path / "screenshots"
    ) as session:
        assert session._browser is None and session.page is None
        first = session.capture("S1", mid)
        assert session._browser is not None
        session.release()
        assert session._browser is None and session.page is None
        second = session.capture("S1", mid + 1)
        assert session._browser is not None
    assert session._browser is None
    for screenshot in (first, second):
        assert Path(screenshot.path).stat().st_size > 1000


# The renderer's refusals are the replayability reject rule at the render boundary: what it cannot
# render faithfully it must refuse, never capture a wrong screenshot from. Each is a construction-time
# refusal — before a browser is ever launched.


def _synthetic(conn, events, visitor="v1"):
    from _support import stamped
    from locus.evidence.hydrate import hydrate

    hydrate(conn, visitor, stamped(events))
    return conn


def _meta(ts, width=1280, height=800):
    return {
        "type": EventType.Meta,
        "timestamp": ts,
        "data": {"href": "https://x.test/", "width": width, "height": height},
    }


def _snapshot(ts):
    return {
        "type": EventType.FullSnapshot,
        "timestamp": ts,
        "data": {"node": {}, "initialOffset": {"left": 0, "top": 0}},
    }


def _dom_content_loaded(ts):
    return {"type": EventType.DomContentLoaded, "timestamp": ts, "data": {}}


def _page_snapshot(ts, body_children, css=None):
    """A FullSnapshot over a real page skeleton: html > head (optionally carrying a stylesheet, in the `_cssText` shape rrweb-snapshot records an inlined stylesheet as) + body > the given children."""
    style = {
        "type": NodeType.Element,
        "tagName": "style",
        "id": 11,
        "attributes": {"_cssText": css},
        "childNodes": [],
    }
    return {
        "type": EventType.FullSnapshot,
        "timestamp": ts,
        "data": {
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
                                "tagName": "head",
                                "id": 10,
                                "attributes": {},
                                "childNodes": [style] if css else [],
                            },
                            {
                                "type": NodeType.Element,
                                "tagName": "body",
                                "id": 3,
                                "attributes": {},
                                "childNodes": list(body_children),
                            },
                        ],
                    }
                ],
            },
            "initialOffset": {"left": 0, "top": 0},
        },
    }


def test_a_slice_that_opens_before_its_snapshot_renders_on_its_own_clock(tmp_path):
    """A page context opens its slice the moment it starts, before rrweb has a DOM to capture, so
    events precede the Meta+FullSnapshot that covers them — materialize_slices calls such a slice replayable
    because it *contains* the pair, and the renderer honors exactly that.

    The replay clock is the slice's own: rrweb measures every offset from its first event, so a
    capture addressed to an absolute moment after the covering snapshot must land there and show the
    DOM as of that moment, not rewind by the head's lead."""
    from _support import full_snapshot, text_mutation

    conn = connect(tmp_path / "events.db")
    _synthetic(
        conn,
        [
            _dom_content_loaded(1000),
            _meta(6000),
            full_snapshot(6001),
            text_mutation(7000, 4, "goodbye"),
            text_mutation(9000, 4, "later still"),
        ],
    )
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id, status FROM slices").fetchone()
    assert row["status"] == "replayable", (
        "materialize_slices owns the criterion: containing the pair is replayable"
    )

    with RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session:
        screenshot = session.capture("S1", 7500)
        rendered = session.page.evaluate(
            "() => document.querySelector('#replayer iframe').contentDocument.body.textContent"
        )

    assert "goodbye" in rendered, (
        "the screenshot is the DOM as of the absolute moment it was addressed to"
    )
    assert "later still" not in rendered
    assert Path(screenshot.path).exists()


def test_a_screenshot_with_no_replayer_is_refused_rather_than_captured_blank(
    corpus, tmp_path
):
    """A readiness wait that answered "nothing pending, nothing failed, no timeout" when the
    replayer's document isn't there at all would be a clean bill of health for a page that does not
    exist — and capture() would screenshot on it. A blank screenshot, certified faithful, is the worst
    thing this module can produce: the model grounds its claims on exactly these pixels."""
    slice_row = corpus.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()
    ts = (slice_row["start_ts"] + slice_row["end_ts"]) // 2

    with RenderSession(
        corpus, {"S1": slice_row["id"]}, tmp_path / "screenshots"
    ) as session:
        session.capture("S1", ts)  # a real screenshot, from a real replayer
        session.page.evaluate(
            """() => {
                document.querySelector('#replayer iframe').remove();
                rrwebReplay.Replayer = class {
                    on(_, built) { setTimeout(built, 0); } pause() {} destroy() {}
                };
            }"""
        )

        with pytest.raises(Exception, match="rendered"):
            session.capture("S1", ts)


def test_an_empty_window_is_refused(corpus, tmp_path):
    with pytest.raises(ValueError, match="slice"):
        RenderSession(corpus, {}, tmp_path / "screenshots")


def test_an_unknown_slice_is_refused(corpus, tmp_path):
    with pytest.raises(ValueError, match="events"):
        RenderSession(corpus, {"S1": 9999}, tmp_path / "screenshots")


def test_a_slice_without_its_own_snapshot_is_refused(tmp_path):
    conn = connect(tmp_path / "events.db")
    _synthetic(conn, [_meta(1000)])
    materialize_slices(conn, "v1")
    slice_id = conn.execute("SELECT id FROM slices").fetchone()["id"]
    # A lone Meta is exactly the slice the guard exists for, and it is refused by the reason
    # materialize_slices discarded it with — one criterion, quoted, never a second one re-derived here.
    with pytest.raises(ValueError, match=ORPHAN_REASON):
        RenderSession(conn, {"S1": slice_id}, tmp_path / "screenshots")


def test_a_zero_viewport_slice_is_refused(tmp_path):
    """The viewport is read off the covering Meta, so a slice that opened before its snapshot is
    still judged on the Meta that sizes the render — never on whatever event happens to be first."""
    conn = connect(tmp_path / "events.db")
    _synthetic(
        conn,
        [_dom_content_loaded(999), _meta(1000, width=0, height=0), _snapshot(1001)],
    )
    materialize_slices(conn, "v1")
    slice_id = conn.execute("SELECT id FROM slices").fetchone()["id"]
    with pytest.raises(ValueError, match="viewport"):
        RenderSession(conn, {"S1": slice_id}, tmp_path / "screenshots")


def test_labels_covering_the_same_instant_render_apart(corpus, tmp_path):
    """Two slices over the very same instants — a second tab, a concurrent
    visitor — are two labeled slices: the same timestamp captured under each label
    yields two distinct screenshots, each from its own replay."""
    slice_row = corpus.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()
    mid = (slice_row["start_ts"] + slice_row["end_ts"]) // 2
    slices = {"S1": slice_row["id"], "S2": slice_row["id"]}

    with RenderSession(corpus, slices, tmp_path / "screenshots") as session:
        first = session.capture("S1", mid)
        second = session.capture("S2", mid)

    assert first.path != second.path, (
        "one instant, two labels, two screenshot identities"
    )
    for screenshot in (first, second):
        png = Path(screenshot.path)
        assert png.exists() and png.stat().st_size > 1000


# The module header's two fidelity requirements are pixel behavior, not configuration spelling: a
# renderer that loses either still hands back a screenshot — silently unfaithful, the module's worst
# failure mode — so each is asserted on the rendered pixels themselves.


def test_a_text_selection_paints_into_the_screenshot(tmp_path):
    """The native ::selection highlight exists only as paint — the DOM is byte-identical with and without it — so it reaches the screenshot only through a browser context that actually paints (the header's first fidelity requirement). The assertion is the pixels: the same document captured before and after the visitor's selection must differ, and the highlight is the only thing between the two screenshots."""
    from PIL import Image, ImageChops

    text = "The visitor selected this entire sentence, and the highlight is the proof."
    conn = connect(tmp_path / "events.db")
    _synthetic(
        conn,
        [
            _meta(1000),
            _page_snapshot(
                1001, [{"type": NodeType.Text, "id": 4, "textContent": text}]
            ),
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 3000,
                "data": {
                    "source": IncrementalSource.Selection,
                    "ranges": [
                        {
                            "start": 4,
                            "startOffset": 0,
                            "end": 4,
                            "endOffset": len(text),
                        }
                    ],
                },
            },
        ],
    )
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id FROM slices").fetchone()

    with RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session:
        before = session.capture("S1", 2000)
        after = session.capture("S1", 3000)

    diff = ImageChops.difference(
        Image.open(before.path).convert("RGB"), Image.open(after.path).convert("RGB")
    )
    changed = sum(1 for px in diff.get_flattened_data() if any(c > 8 for c in px))
    assert changed > 100, (
        "the selection highlight paints pixels the unselected screenshot does not have"
    )


def test_content_revealed_by_an_animation_renders_revealed(tmp_path):
    """Content captured at opacity:0 with a reveal animation reaches the visitor only because the animation advances (the header's second fidelity requirement: pauseAnimation off). A replay whose animations are frozen holds the reveal at its first screenshot forever and renders blank where the visitor saw content, so the assertion is the revealed content's own pixels. No sleep is needed: the capture's readiness wait (network quiet alone is 500ms of wall clock) outlasts the 100ms reveal."""
    from PIL import Image

    css = (
        "#reveal { opacity: 0; animation: locus-reveal 100ms forwards; "
        "background: rgb(255, 0, 0); width: 200px; height: 200px; } "
        "@keyframes locus-reveal { to { opacity: 1; } }"
    )
    conn = connect(tmp_path / "events.db")
    _synthetic(
        conn,
        [
            _meta(1000),
            _page_snapshot(
                1001,
                [
                    {
                        "type": NodeType.Element,
                        "tagName": "div",
                        "id": 4,
                        "attributes": {"id": "reveal"},
                        "childNodes": [],
                    }
                ],
                css=css,
            ),
        ],
    )
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id FROM slices").fetchone()

    with RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session:
        screenshot = session.capture("S1", 1001)

    img = Image.open(screenshot.path).convert("RGB")
    revealed = sum(
        1 for r, g, b in img.get_flattened_data() if r > 200 and g < 80 and b < 80
    )
    assert revealed > 1000, (
        "the reveal ran to its end state — the revealed content's pixels are in the screenshot"
    )


def test_a_capture_holds_the_moment_s_own_events_in_any_seek_order(tmp_path):
    """A screenshot at a moment is the DOM after every event stamped at or before it, whatever was
    captured before it — every capture is its own fresh replayer. Each capture below is judged on
    its own DOM."""
    from _support import full_snapshot, text_mutation

    conn = connect(tmp_path / "events.db")
    _synthetic(
        conn,
        [
            _meta(1000),
            full_snapshot(1001),
            text_mutation(2000, 4, "goodbye"),
            text_mutation(3000, 4, "later still"),
        ],
    )
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id FROM slices").fetchone()
    body = "() => document.querySelector('#replayer iframe').contentDocument.body.textContent"

    seen = {}
    with RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session:
        for ts in (2000, 3000, 1001, 2000, 1999):
            session.capture("S1", ts)
            seen.setdefault(ts, []).append(session.page.evaluate(body))

    assert seen == {
        2000: ["goodbye", "goodbye"],
        3000: ["later still"],
        1001: ["hello"],
        1999: ["hello"],
    }


def test_a_capture_ahead_of_the_covering_snapshot_is_refused(tmp_path):
    """The lead is recorded time with no recorded page: a file stamped with such a moment would
    testify to pixels that never existed. Refused by the address; nothing is captured."""
    from _support import full_snapshot

    conn = connect(tmp_path / "events.db")
    _synthetic(conn, [_dom_content_loaded(1000), _meta(6000), full_snapshot(6001)])
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id FROM slices").fetchone()

    with (
        RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session,
        pytest.raises(
            ValueError,
            match="t=3000.*t=6001",
        ),
    ):
        session.capture("S1", 3000)
    with (
        RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session,
        pytest.raises(ValueError, match="t=6002.*t=1000.*t=6001"),
    ):
        session.capture("S1", 6002)
    assert not list((tmp_path / "screenshots").glob("*.png"))


def test_seeked_scrolls_land_instantly_whatever_the_page_declares(tmp_path):
    """A seeked scroll defers to the page's CSS scroll-behavior; a page declaring smooth animates
    it, and a screenshot mid-animation holds an earlier scroll position. The renderer's injected
    rule overrides the declaration inside the replayed document."""
    tall = {
        "type": NodeType.Element,
        "tagName": "div",
        "id": 4,
        "attributes": {"style": "height: 5000px"},
        "childNodes": [],
    }
    conn = connect(tmp_path / "events.db")
    _synthetic(
        conn,
        [
            _meta(1000),
            _page_snapshot(1001, [tall], css="html { scroll-behavior: smooth }"),
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 2000,
                "data": {"source": IncrementalSource.Scroll, "id": 1, "x": 0, "y": 700},
            },
        ],
    )
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id FROM slices").fetchone()

    with RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session:
        session.capture("S1", 2000)
        declared = session.page.evaluate(
            "() => getComputedStyle(document.querySelector('#replayer iframe').contentDocument.documentElement).scrollBehavior"
        )
        scrolled = session.page.evaluate(
            "() => document.querySelector('#replayer iframe').contentWindow.scrollY"
        )

    assert declared == "auto"
    assert scrolled == 700


def test_a_capture_shows_the_moment_not_the_captures_before_it(tmp_path):
    """rrweb carries state across seeks on one replayer: its virtual document keeps the last
    document scroll through every rebuild and re-applies it to any later batch that mutates
    without scrolling. A capture is a fresh replayer, so the frame is the moment's alone."""
    from _support import text_mutation

    tall = {
        "type": NodeType.Element,
        "tagName": "div",
        "id": 4,
        "attributes": {"style": "height: 5000px"},
        "childNodes": [{"type": NodeType.Text, "id": 5, "textContent": "hello"}],
    }
    conn = connect(tmp_path / "events.db")
    _synthetic(
        conn,
        [
            _meta(1000),
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
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id FROM slices").fetchone()
    state = (
        "() => { const f = document.querySelector('#replayer iframe');"
        " return [f.contentDocument.body.textContent.trim(), f.contentWindow.scrollY]; }"
    )

    seen = []
    with RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session:
        for ts in (2001, 1500):
            session.capture("S1", ts)
            seen.append(session.page.evaluate(state))

    assert seen == [["scrolled", 700], ["early", 0]]


def test_a_capture_lays_out_complete_before_its_scrolls_apply(
    tmp_path,
):
    """A scroll applied while the page's images and fonts are still arriving lands where the
    content's height at that instant allows and stays there. A capture builds the snapshot
    page, waits for its fonts and images, and only then pauses at the moment."""
    from test_replay import scroll, tall_png

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
    conn = connect(tmp_path / "events.db")
    _synthetic(
        conn, [_meta(1000), _page_snapshot(1001, [scroller]), scroll(2000, 6, 1500)]
    )
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id FROM slices").fetchone()
    scrolled = (
        "() => document.querySelector('#replayer iframe').contentDocument"
        ".querySelector('div').scrollTop"
    )

    def slowly(route):
        import time

        time.sleep(0.3)
        route.fulfill(path=str(tmp_path / "tall.png"), content_type="image/png")

    with RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session:
        session._ensure_browser()
        session.page.route("**/tall.png", slowly)
        session.capture("S1", 2000)
        first = session.page.evaluate(scrolled)
        digest = "() => JSON.stringify(window.__events).length"
        before = session.page.evaluate(digest)
        session.capture("S1", 1001)
        session.capture("S1", 2000)
        after = session.page.evaluate(digest)

    assert first == 1500
    assert before == after, "the injected events are never written into"
