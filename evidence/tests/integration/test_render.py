from pathlib import Path

import pytest
from _support import (
    dom_content_loaded,
    sized_meta,
    synthetic,
)
from locus.evidence.db import connect
from locus.evidence.render import RenderSession
from locus.evidence.rrweb_constants import EventType, IncrementalSource, NodeType
from locus.evidence.slices import (
    covering_snapshot_ts,
    materialize_slices,
)


def native(width, height):
    """The viewport's own pixels: a screenshot drawn at device scale 1."""
    return width, height


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
        screenshots = [
            session.capture("S1", ts, kind="activity", size=native) for ts in timestamps
        ]

    assert len(screenshots) == len(timestamps)
    for screenshot in screenshots:
        png = Path(screenshot.path)
        assert png.exists() and png.stat().st_size > 1000
        assert png.name == f"S1_activity_screenshot_{screenshot.timestamp}.png", (
            "a screenshot's file name is its (label, kind, moment) identity"
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
        session.capture(
            "S1",
            (slice_row["start_ts"] + slice_row["end_ts"]) // 2,
            kind="activity",
            size=native,
        )
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
        second = session.capture("S2", mids[1], kind="activity", size=native)
        first = session.capture("S1", mids[0], kind="activity", size=native)
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
        session.capture("S9", mids[0], kind="activity", size=native)


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
        first = session.capture("S1", mid, kind="activity", size=native)
        assert session._browser is not None
        session.release()
        assert session._browser is None and session.page is None
        second = session.capture("S1", mid + 1, kind="activity", size=native)
        assert session._browser is not None
    assert session._browser is None
    for screenshot in (first, second):
        assert Path(screenshot.path).stat().st_size > 1000


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
    synthetic(
        conn,
        [
            dom_content_loaded(1000),
            sized_meta(6000),
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
        screenshot = session.capture("S1", 7500, kind="activity", size=native)
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
    thing this module can produce: the model grounds their claims on exactly these pixels."""
    slice_row = corpus.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()
    ts = (slice_row["start_ts"] + slice_row["end_ts"]) // 2

    with RenderSession(
        corpus, {"S1": slice_row["id"]}, tmp_path / "screenshots"
    ) as session:
        session.capture(
            "S1", ts, kind="activity", size=native
        )  # a real screenshot, from a real replayer
        session.page.evaluate(
            """() => {
                document.querySelector('#replayer iframe').remove();
                rrwebReplay.Replayer = class {
                    on(_, built) { setTimeout(built, 0); } pause() {} destroy() {}
                };
            }"""
        )

        with pytest.raises(Exception, match="rendered"):
            session.capture("S1", ts, kind="activity", size=native)


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
        first = session.capture("S1", mid, kind="activity", size=native)
        second = session.capture("S2", mid, kind="activity", size=native)

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
    synthetic(
        conn,
        [
            sized_meta(1000),
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
        before = session.capture("S1", 2000, kind="activity", size=native)
        after = session.capture("S1", 3000, kind="activity", size=native)

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
    synthetic(
        conn,
        [
            sized_meta(1000),
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
        screenshot = session.capture("S1", 1001, kind="activity", size=native)

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
    synthetic(
        conn,
        [
            sized_meta(1000),
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
            session.capture("S1", ts, kind="activity", size=native)
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
    synthetic(conn, [dom_content_loaded(1000), sized_meta(6000), full_snapshot(6001)])
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id FROM slices").fetchone()

    with (
        RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session,
        pytest.raises(ValueError, match="t=3000.*t=6001"),
    ):
        session.capture("S1", 3000, kind="activity", size=native)
    with (
        RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session,
        pytest.raises(ValueError, match="t=6002.*t=1000.*t=6001"),
    ):
        session.capture("S1", 6002, kind="activity", size=native)
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
    synthetic(
        conn,
        [
            sized_meta(1000),
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
        session.capture("S1", 2000, kind="activity", size=native)
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
    synthetic(
        conn,
        [
            sized_meta(1000),
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
            session.capture("S1", ts, kind="activity", size=native)
            seen.append(session.page.evaluate(state))

    assert seen == [["scrolled", 700], ["early", 0]]


def _resizing_slice(tmp_path):
    conn = connect(tmp_path / "events.db")
    body = {
        "type": NodeType.Element,
        "tagName": "div",
        "id": 4,
        "attributes": {},
        "childNodes": [{"type": NodeType.Text, "id": 5, "textContent": "hello"}],
    }
    synthetic(
        conn,
        [
            sized_meta(1000),
            _page_snapshot(1001, [body]),
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 2000,
                "data": {
                    "source": IncrementalSource.ViewportResize,
                    "width": 390,
                    "height": 663,
                },
            },
            {
                "type": EventType.IncrementalSnapshot,
                "timestamp": 3000,
                "data": {"source": IncrementalSource.Scroll, "id": 1, "x": 0, "y": 0},
            },
        ],
    )
    materialize_slices(conn, "v1")
    return conn, conn.execute("SELECT id FROM slices").fetchone()["id"]


def test_a_screenshot_is_the_size_of_its_moments_viewport(tmp_path):
    """A visitor who resizes mid-slice resizes every screenshot after — the size pricing counts each one at (slices.viewport_at) — in whatever order the moments are captured."""
    from PIL import Image

    conn, slice_id = _resizing_slice(tmp_path)
    sizes = []
    with RenderSession(conn, {"S1": slice_id}, tmp_path / "screenshots") as session:
        for ts in (2500, 1500, 2000):
            with Image.open(
                session.capture("S1", ts, kind="activity", size=native).path
            ) as png:
                sizes.append(png.size)
    assert sizes == [(390, 663), (1280, 800), (390, 663)]


def test_a_screenshot_is_drawn_at_the_size_asked_for_its_moments_viewport(tmp_path):
    """Each moment's screenshot is drawn at the size `size` gives its own viewport, larger or smaller and in different proportions, whatever order the moments are captured in."""
    from PIL import Image

    asked = {(1280, 800): (640, 400), (390, 663): (780, 1312)}
    conn, slice_id = _resizing_slice(tmp_path)
    sizes = []
    with RenderSession(conn, {"S1": slice_id}, tmp_path / "screenshots") as session:
        for ts in (2500, 1500, 2000):
            shot = session.capture(
                "S1", ts, kind="activity", size=lambda w, h: asked[(w, h)]
            )
            with Image.open(shot.path) as png:
                sizes.append(png.size)
    assert sizes == [(780, 1312), (640, 400), (780, 1312)]


def test_the_screenshot_is_the_replayed_page_edge_to_edge(tmp_path):
    """One-pixel black rules fixed to the replayed page's four edges land on the screenshot's four edges: the harness adds no border or offset and cuts nothing off."""
    from PIL import Image

    def rule(i, place):
        return {
            "type": NodeType.Element,
            "tagName": "div",
            "id": 4 + i,
            "attributes": {"style": f"position: fixed; background: #000; {place}"},
            "childNodes": [],
        }

    rules = [
        rule(0, "left: 0; top: 0; width: 1px; height: 100%"),
        rule(1, "right: 0; top: 0; width: 1px; height: 100%"),
        rule(2, "left: 0; top: 0; height: 1px; width: 100%"),
        rule(3, "left: 0; bottom: 0; height: 1px; width: 100%"),
    ]
    conn = connect(tmp_path / "events.db")
    synthetic(
        conn,
        [
            sized_meta(1000, width=640, height=400),
            _page_snapshot(1001, rules, css="body { background: #fff }"),
        ],
    )
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id FROM slices").fetchone()

    with RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session:
        shot = session.capture("S1", 1001, kind="activity", size=native)

    with Image.open(shot.path) as png:
        gray = png.convert("L")
        # Sampled away from the top-left corner, where the replayer draws its cursor.
        middle_rows, middle_cols = range(100, 300, 20), range(100, 540, 20)
        assert {gray.getpixel((0, y)) for y in middle_rows} == {0}
        assert {gray.getpixel((639, y)) for y in middle_rows} == {0}
        assert {gray.getpixel((x, 0)) for x in middle_cols} == {0}
        assert {gray.getpixel((x, 399)) for x in middle_cols} == {0}
        assert {gray.getpixel((1, y)) for y in middle_rows} == {255}
        assert {gray.getpixel((x, 1)) for x in middle_cols} == {255}


def test_detail_at_a_larger_size_is_rendered_at_that_density(tmp_path):
    """A one-pixel black rule on white, at twice the viewport's pixels, is two device pixels of pure black with pure white beside them, which only a drawing at that scale produces."""
    from PIL import Image

    rule = {
        "type": NodeType.Element,
        "tagName": "div",
        "id": 4,
        "attributes": {
            "style": "position: absolute; left: 100px; top: 0; width: 1px; height: 300px; background: #000"
        },
        "childNodes": [],
    }
    conn = connect(tmp_path / "events.db")
    synthetic(
        conn,
        [
            sized_meta(1000),
            _page_snapshot(1001, [rule], css="body { background: #fff }"),
        ],
    )
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id FROM slices").fetchone()

    with RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session:
        shot = session.capture(
            "S1", 1001, kind="pulled", size=lambda w, h: (w * 2, h * 2)
        )

    with Image.open(shot.path) as png:
        assert png.size == (2560, 1600)
        gray = png.convert("L")
        inked = {
            x: gray.getpixel((x, 100))
            for x in range(100, 300)
            if gray.getpixel((x, 100)) < 255
        }
    (x, _), _ = inked.items()
    assert inked == {x: 0, x + 1: 0}, (
        "the rule is two pure-black device pixels wide, with nothing gray beside it"
    )


def test_a_capture_at_a_new_scale_is_drawn_at_that_scale(tmp_path):
    """A capture at twice the viewport's pixels, after one at its own pixels, is drawn at twice them: the one-pixel rule is two pure-black device pixels with nothing gray beside it."""
    from PIL import Image

    rule = {
        "type": NodeType.Element,
        "tagName": "div",
        "id": 4,
        "attributes": {
            "style": "position: absolute; left: 100px; top: 0; width: 1px; height: 300px; background: #000"
        },
        "childNodes": [],
    }
    conn = connect(tmp_path / "events.db")
    synthetic(
        conn,
        [
            sized_meta(1000),
            _page_snapshot(1001, [rule], css="body { background: #fff }"),
        ],
    )
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id FROM slices").fetchone()

    with RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session:
        session.capture("S1", 1001, kind="activity", size=native)
        shot = session.capture(
            "S1", 1001, kind="pulled", size=lambda w, h: (w * 2, h * 2)
        )

    with Image.open(shot.path) as png:
        assert png.size == (2560, 1600)
        gray = png.convert("L")
        inked = {
            x: gray.getpixel((x, 100))
            for x in range(100, 300)
            if gray.getpixel((x, 100)) < 255
        }
    (x, _), _ = inked.items()
    assert inked == {x: 0, x + 1: 0}, (
        "the rule is two pure-black device pixels wide, with nothing gray beside it"
    )


def test_a_size_off_the_viewports_proportions_is_squeezed_along_one_side(tmp_path):
    """Asked for 2560x1568 of a 1280x800 viewport, the page is drawn at scale 2, exact across and 32 rows too tall, then resampled onto the size: a vertical one-pixel rule keeps its two pure-black device pixels, since nothing is resampled horizontally; a horizontal one is squeezed into grays; and the page's bottom edge is still the screenshot's last row."""
    from PIL import Image

    def rule(i, style):
        return {
            "type": NodeType.Element,
            "tagName": "div",
            "id": 4 + i,
            "attributes": {"style": f"position: absolute; background: #000; {style}"},
            "childNodes": [],
        }

    conn = connect(tmp_path / "events.db")
    synthetic(
        conn,
        [
            sized_meta(1000),
            _page_snapshot(
                1001,
                [
                    rule(0, "left: 100px; top: 0; width: 1px; height: 300px"),
                    rule(1, "left: 400px; top: 200px; width: 300px; height: 1px"),
                    rule(2, "left: 800px; bottom: 0; width: 300px; height: 1px"),
                ],
                css="body { background: #fff; margin: 0; height: 100vh; position: relative }",
            ),
        ],
    )
    materialize_slices(conn, "v1")
    row = conn.execute("SELECT id FROM slices").fetchone()

    with RenderSession(conn, {"S1": row["id"]}, tmp_path / "screenshots") as session:
        shot = session.capture(
            "S1", 1001, kind="pulled", size=lambda w, h: (2560, 1568)
        )

    with Image.open(shot.path) as png:
        assert png.size == (2560, 1568)
        gray = png.convert("L")
        across = {
            x: gray.getpixel((x, 100))
            for x in range(100, 300)
            if gray.getpixel((x, 100)) < 255
        }
        down = [gray.getpixel((1100, y)) for y in range(370, 410)]
        bottom = gray.getpixel((1800, 1567))
    (x, _), _ = across.items()
    assert across == {x: 0, x + 1: 0}, "horizontally, the pixels are the browser's"
    assert any(0 < v < 255 for v in down), "vertically, the pixels are resampled"
    assert min(down) < 128, "the squeezed rule is still visible"
    assert bottom < 128, "the page's bottom edge is the screenshot's last row"


def test_one_moment_captured_as_both_kinds_is_two_files(corpus, tmp_path):
    from PIL import Image

    slice_row = corpus.execute(
        "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()
    mid = (slice_row["start_ts"] + slice_row["end_ts"]) // 2

    with RenderSession(
        corpus, {"S1": slice_row["id"]}, tmp_path / "screenshots"
    ) as session:
        activity = session.capture(
            "S1", mid, kind="activity", size=lambda w, h: (w // 2, h // 2)
        )
        pulled = session.capture("S1", mid, kind="pulled", size=native)
        with pytest.raises(ValueError, match="kind"):
            session.capture("S1", mid, kind="pushed", size=native)

    from locus.evidence.slices import slice_events, viewport_at

    width, height = viewport_at(slice_events(corpus, slice_row["id"]), mid)
    assert Path(activity.path).name == f"S1_activity_screenshot_{mid}.png"
    assert Path(pulled.path).name == f"S1_pulled_screenshot_{mid}.png"
    with Image.open(activity.path) as small, Image.open(pulled.path) as large:
        assert small.size == (width // 2, height // 2)
        assert large.size == (width, height)


def test_a_frame_that_disagrees_with_the_moments_viewport_is_refused(
    tmp_path, monkeypatch
):
    from locus.evidence import render

    conn, slice_id = _resizing_slice(tmp_path)
    monkeypatch.setattr(render, "viewport_at", lambda events, ts: (1000, 700))
    with (
        RenderSession(conn, {"S1": slice_id}, tmp_path / "screenshots") as session,
        pytest.raises(RuntimeError, match="sized its frame 390x663"),
    ):
        session.capture("S1", 2500, kind="activity", size=native)


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
    synthetic(
        conn,
        [sized_meta(1000), _page_snapshot(1001, [scroller]), scroll(2000, 6, 1500)],
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
        session._ensure_page("S1", 1.0)
        page = session.page
        session.page.route("**/tall.png", slowly)
        session.capture("S1", 2000, kind="activity", size=native)
        first = session.page.evaluate(scrolled)
        digest = "() => JSON.stringify(window.__events).length"
        before = session.page.evaluate(digest)
        session.capture("S1", 1001, kind="activity", size=native)
        session.capture("S1", 2000, kind="activity", size=native)
        after = session.page.evaluate(digest)
        assert session.page is page, "the captures did not rebuild the page"

    assert first == 1500
    assert before == after, "the injected events are never written into"
