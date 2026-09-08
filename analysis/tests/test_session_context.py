import subprocess

import pytest
from _support import (
    DISTILL,
    URL,
    click,
    full_snapshot,
    hidden,
    meta,
    stamped,
)
from locus.analysis.session_context import (
    INCOMPLETE,
    arrival,
    session_context_block,
    window_pages,
)
from locus.evidence.db import connect
from locus.evidence.hydrate import hydrate
from locus.evidence.rrweb_constants import EventType
from locus.evidence.slices import materialize_slices

PAGE_B = "https://x.test/checkout"
EXIT_PAGE = "https://x.test/bye"


def page_load(ts, url=URL, referrer=""):
    """The recorder fires PageLoad on the page's first FullSnapshot — inside the slice that snapshot covers — and again in place on each SPA route change."""
    return {
        "type": EventType.PageLoad,
        "timestamp": ts,
        "data": {"url": url, "title": "t", "referrer": referrer},
    }


def load_visitor(conn, visitor, events):
    hydrate(conn, visitor, stamped(events))
    materialize_slices(conn, visitor)


@pytest.fixture()
def context_db(tmp_path):
    conn = connect(tmp_path / "events.db")
    load_visitor(
        conn,
        "v1",
        [
            meta(1_000),
            full_snapshot(1_001),
            page_load(1_002, URL, referrer="https://google.com/"),
            click(60_000, 3),
            meta(100_000, href=PAGE_B),
            full_snapshot(100_001),
            page_load(100_002, PAGE_B, referrer=URL),
            click(150_000, 3),
            meta(200_000, href=PAGE_B),
            full_snapshot(200_001),  # periodic checkout
            click(210_000, 3),
            meta(300_000, href=EXIT_PAGE),
            hidden(300_500, EXIT_PAGE),
        ],
    )
    load_visitor(
        conn,
        "v2",
        [
            meta(500_000),
            full_snapshot(500_001),
            click(510_000, 3),
            meta(600_000),
            full_snapshot(600_001),
            click(610_000, 3),
        ],
    )
    subprocess.run(
        ["bun", str(DISTILL), str(tmp_path / "events.db")],
        check=True,
        capture_output=True,
    )
    return conn


def v1_slices(conn):
    return [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM slices WHERE visitor_id='v1' AND status='replayable' ORDER BY start_ts"
        )
    ]


def test_block_states_prior_page_referrer_and_next_page_and_nothing_else(context_db):
    slices = v1_slices(context_db)
    block = session_context_block(context_db, [slices[1]])
    prior, referrer, following = block.splitlines()[1:-1]
    assert prior.startswith("Prior page") and URL in prior and "40s" in prior
    assert referrer.startswith("Referrer") and URL in referrer
    assert (
        following.startswith("Next page") and PAGE_B in following and "50s" in following
    )


def test_entry_windows_state_referrer_and_bare_absence(context_db):
    slices = v1_slices(context_db)
    block = session_context_block(context_db, [slices[0]])
    prior = next(l for l in block.splitlines() if l.startswith("Prior page"))
    assert "http" not in prior, "no earlier recording: the line names no page"
    assert "https://google.com/" in block


def test_departure_to_an_unrecorded_page_is_stated(context_db):
    slices = v1_slices(context_db)
    block = session_context_block(context_db, [slices[1], slices[2]])
    following = next(l for l in block.splitlines() if l.startswith("Next page"))
    assert EXIT_PAGE in following and "1m30s" in following and INCOMPLETE in following


def test_mid_page_windows_state_the_page_and_its_age(context_db):
    slices = v1_slices(context_db)
    checkout = session_context_block(context_db, [slices[2]])
    opening, prior = checkout.splitlines()[1:3]
    assert PAGE_B in opening and "1m40s" in opening
    assert prior.startswith("Prior page") and URL in prior and "2m20s" in prior
    assert "Referrer" not in checkout, (
        "a checkout head attests no page-load, so no referrer"
    )

    fragment = session_context_block(context_db, [slices[1]], window_start=150_000)
    opening = fragment.splitlines()[1]
    assert PAGE_B in opening and "50s" in opening


def test_last_window_states_bare_absence(context_db):
    v2_last = context_db.execute(
        "SELECT id FROM slices WHERE visitor_id='v2' ORDER BY start_ts DESC LIMIT 1"
    ).fetchone()["id"]
    block = session_context_block(context_db, [v2_last])
    following = next(l for l in block.splitlines() if l.startswith("Next page"))
    assert "http" not in following, "no later recording: the line names no page"


def test_attestation_matches_on_the_route_not_the_full_url(tmp_path):
    """Query strings churn between a PageLoad and the snapshot behind it on the same physical page — an ad-click landing's Meta carries gclid/wbraid params the PageLoad never saw. Exact-URL matching would orphan exactly those arrivals' traffic-source referrers, so attestation drops the query and keeps origin+path+fragment."""
    conn = connect(tmp_path / "events.db")
    load_visitor(
        conn,
        "v3",
        [
            meta(1_000, href=EXIT_PAGE + "?gclid=x"),
            full_snapshot(1_001),
            page_load(1_002, EXIT_PAGE, referrer="https://ads.example/"),
            click(5_000, 3),
            meta(100_000, href=EXIT_PAGE),
            full_snapshot(100_001),  # checkout, no PageLoad
            click(110_000, 3),
        ],
    )
    subprocess.run(
        ["bun", str(DISTILL), str(tmp_path / "events.db")],
        check=True,
        capture_output=True,
    )
    slices = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM slices WHERE visitor_id='v3' ORDER BY start_ts"
        )
    ]

    assert arrival(conn, slices[0]) == (
        "page-load",
        EXIT_PAGE + "?gclid=x",
        "https://ads.example/",
    )
    assert arrival(conn, slices[1]) == ("unattested", EXIT_PAGE, None)


SPA = "https://app.test"


def test_spa_routes_are_in_slice_pageloads(tmp_path):
    conn = connect(tmp_path / "events.db")
    load_visitor(
        conn,
        "spa",
        [
            meta(1_000, href=f"{SPA}/#/home"),
            full_snapshot(1_001),
            page_load(1_002, f"{SPA}/#/home", referrer="https://google.com/"),
            click(5_000, 3),
            page_load(10_000, f"{SPA}/#/courses"),
            click(15_000, 3),
            page_load(20_000, f"{SPA}/#/students"),
            click(25_000, 3),
            # periodic checkout fires on the current route — a new slice, no PageLoad
            meta(30_000, href=f"{SPA}/#/students"),
            full_snapshot(30_001),
            click(35_000, 3),
            page_load(40_000, f"{SPA}/#/reports"),
            click(45_000, 3),
        ],
    )
    subprocess.run(
        ["bun", str(DISTILL), str(tmp_path / "events.db")],
        check=True,
        capture_output=True,
    )
    slices = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM slices WHERE visitor_id='spa' AND status='replayable' ORDER BY start_ts"
        )
    ]
    assert len(slices) == 2

    # the page-load slice is attested; the checkout slice carrying a *later* route
    # PageLoad — same origin+path on a hash SPA — is NOT mis-attested as a page-load.
    assert arrival(conn, slices[0]) == (
        "page-load",
        f"{SPA}/#/home",
        "https://google.com/",
    )
    assert arrival(conn, slices[1]) == ("unattested", f"{SPA}/#/students", None)

    # window_pages enumerates each slice head plus its in-place route PageLoads, in
    # order, each arrival carrying its timestamp (route timestamps are the natural
    # in-slice cut points).
    pages = window_pages(conn, slices)
    assert [(url, kind) for url, kind, _ in pages] == [
        (f"{SPA}/#/home", "page-load"),
        (f"{SPA}/#/courses", "page-load"),
        (f"{SPA}/#/students", "page-load"),
        (f"{SPA}/#/students", "unattested"),
        (f"{SPA}/#/reports", "page-load"),
    ]
    timestamps = [ts for _, _, ts in pages]
    assert all(ts is not None for ts in timestamps)
    assert timestamps == sorted(timestamps)

    # "Before this recording" is the route the prior slice ended on, not its head.
    block = session_context_block(conn, [slices[1]])
    prior = next(l for l in block.splitlines() if l.startswith("Prior page"))
    assert f"{SPA}/#/students" in prior


def test_a_concurrent_lane_left_out_is_not_a_gap(tmp_path):
    """One visitor, two tabs: a slice running alongside an included one is a
    deliberately-excluded concurrent lane, not a hole in the window, so
    it earns no gap statement."""
    conn = connect(tmp_path / "events.db")
    hydrate(
        conn,
        "v1",
        stamped(
            [
                meta(1_000),
                full_snapshot(1_001),
                click(5_000, 3),
                meta(10_000),
                full_snapshot(10_001),
                click(15_000, 3),
            ]
        ),
    )
    hydrate(conn, "v1", stamped([meta(2_000), full_snapshot(2_001), click(8_000, 3)]))
    materialize_slices(conn, "v1")
    subprocess.run(
        ["bun", str(DISTILL), str(tmp_path / "events.db")],
        check=True,
        capture_output=True,
    )
    a1, a2 = (
        conn.execute("SELECT id FROM slices WHERE start_ts = ?", (ts,)).fetchone()["id"]
        for ts in (1_000, 10_000)
    )

    block = session_context_block(conn, [a1, a2])
    assert "Between" not in block, (
        "the tab running alongside a1 is a concurrent lane, not a gap"
    )


def test_an_interior_gap_is_stated_like_an_edge(context_db):
    """Dropping middle slices is normal whittling; the seam gets what the
    window's edges get: the time elapsed and the pages of the recording
    left out."""
    slices = v1_slices(context_db)
    labeled = session_context_block(
        context_db, [slices[0], slices[2]], labels={slices[0]: "S1", slices[2]: "S2"}
    )
    gap = next(l for l in labeled.splitlines() if l.startswith("Between"))
    assert "S1" in gap and "S2" in gap and "2m20s" in gap and PAGE_B in gap

    bare = session_context_block(context_db, [slices[0], slices[2]])
    gap = next(l for l in bare.splitlines() if l.startswith("Between"))
    assert "2m20s" in gap and "S1" not in gap

    contiguous = session_context_block(
        context_db, [slices[0], slices[1]], labels={slices[0]: "S1", slices[1]: "S2"}
    )
    assert "Between" not in contiguous, (
        "adjacent slices have no recording between them to state"
    )


def test_a_discarded_slice_in_an_interior_gap_testifies(tmp_path):
    """A discarded slice between the window's named slices still testifies its
    mechanical facts. Incompleteness attaches to the page whose recording is
    incomplete — the way the After edge attaches it to the page it names —
    while a fully recorded excluded page in the same gap carries no mark."""
    conn = connect(tmp_path / "events.db")
    load_visitor(
        conn,
        "v1",
        [
            meta(1_000),
            full_snapshot(1_001),
            click(5_000, 3),
            meta(30_000, href=EXIT_PAGE),
            full_snapshot(30_001),
            click(35_000, 3),
            meta(50_000, href=PAGE_B),
            click(55_000, 3),  # no snapshot: discarded
            meta(100_000),
            full_snapshot(100_001),
            click(105_000, 3),
        ],
    )
    subprocess.run(
        ["bun", str(DISTILL), str(tmp_path / "events.db")],
        check=True,
        capture_output=True,
    )
    replayable = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM slices WHERE status='replayable' ORDER BY start_ts"
        )
    ]
    a, b = replayable[0], replayable[-1]

    block = session_context_block(conn, [a, b], labels={a: "S1", b: "S2"})
    gap = next(l for l in block.splitlines() if l.startswith("Between"))
    assert "1m35s" in gap
    assert gap.index(EXIT_PAGE) < gap.index(PAGE_B), "pages in the order they were on"
    assert gap.index(INCOMPLETE) > gap.index(PAGE_B), (
        "incompleteness attaches to the page whose recording is incomplete"
    )
    assert gap.count(INCOMPLETE) == 1, "a fully recorded excluded page carries no mark"


def test_a_mixed_visitor_window_fails_loud(context_db):
    slices = v1_slices(context_db)
    v2 = context_db.execute(
        "SELECT id FROM slices WHERE visitor_id='v2' LIMIT 1"
    ).fetchone()["id"]
    with pytest.raises(ValueError, match="one visitor"):
        session_context_block(context_db, [slices[0], v2])
