"""The window context block: the visitor's prior recorded page and how long before this window it was left, the window's page's referrer as the browser reported it, the next recorded page and how long after — and, for any gap between the window's named slices whose recording was left out, the time elapsed and the pages recorded there. Each is one named fact, stated as what the recording holds; reconciling them (a referrer that survives a reload, a prior page that is this same page) is the model's reading.

Mechanical facts read from the flat columns, never a summary, never a trajectory — session reconstruction is what running the ordered slices is for, and anything semantic belongs in the caller's own question. One extra fact answers "where were they" for a boundary that lies inside a page: a window that opens mid-page (periodic snapshot, route-boundary piece) states the page and how much earlier it loaded. Absence is bare — "none recorded" — with no apparatus-level explanation of why: the model's frame is this visitor's recording, and recording infrastructure does not exist in it.
"""

import itertools
import sqlite3

from locus.evidence.db import CANONICAL_ORDER
from locus.evidence.kinds import Kind
from locus.evidence.text import describe_url

INCOMPLETE = "recording there is incomplete"


def _duration(ms: int) -> str:
    """Coarse by design: these durations describe boundary gaps, and a sub-10-second gap is page-turnover time — navigation, load, snapshot latency — not something the visitor did, so stating it as "2.1s" invites arithmetic about behavior the number does not carry. "seconds" states the one fact that matters at that scale: the session was continuous."""
    seconds = ms / 1000
    if seconds < 10:
        return "seconds"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s"


def _head_event(conn: sqlite3.Connection, slice_id: int) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT id, timestamp, counter, url FROM events WHERE slice_id = ? {CANONICAL_ORDER} LIMIT 1",
        (slice_id,),
    ).fetchone()


def _route_url(url: str | None) -> str | None:
    """origin+path+fragment, query dropped — the route key the recorder dedups navigations on. Query churns between a PageLoad and its Meta on the same physical page (gclid/wbraid on ad landings), so it cannot be identity; the fragment is kept because it carries the route on a hash-routed SPA, so two hash routes in one slice stay distinct and a checkout slice's later route PageLoads do not attest its head."""
    if url is None:
        return None
    base, _, frag = url.partition("#")
    base = base.split("?")[0]
    return f"{base}#{frag}" if frag else base


def arrival(
    conn: sqlite3.Connection, slice_id: int
) -> tuple[str, str | None, str | None]:
    """(boundary_kind, url, referrer): how this slice's page came to be.

    A slice head is a page-load boundary iff a PageLoad matching its URL attests it. The recorder fires PageLoad on the page's first FullSnapshot, inside the slice the snapshot covers, so a PageLoad attests exactly the slice that contains it — attestation is read from the row, never inferred from timing.

    Attestation matches on the route (_route_url), never the full URL: exact matching would orphan a page-load whose query churned between the load and the snapshot — losing exactly those arrivals' traffic-source referrers.

    A checkout head carries no PageLoad of its own — it is at least checkoutEveryNms from its page's — and is "unattested".
    """
    head = _head_event(conn, slice_id)
    if head is None:
        return "unknown", None, None
    url = head["url"]
    route = _route_url(url)

    for row in conn.execute(
        "SELECT url, referrer FROM events WHERE slice_id = :s AND type_str = :page_load",
        {"s": slice_id, "page_load": Kind.PAGE_LOAD},
    ):
        if _route_url(row["url"]) != route:
            continue
        return "page-load", url, row["referrer"] or None

    return "unattested", url, None


def _tail_url(conn: sqlite3.Connection, slice_id: int) -> str | None:
    """The route the visitor was on when the slice ended — its last in-place page arrival, or its head if it held no later route."""
    pages = window_pages(conn, [slice_id])
    return pages[-1][0] if pages else None


def window_pages(
    conn: sqlite3.Connection,
    slice_ids: list[int],
    window_start: int | None = None,
    window_end: int | None = None,
) -> list[tuple[str | None, str, int | None]]:
    """Ordered (url, kind, timestamp) page arrivals across the window's slices, in time order. A head arrival's timestamp is the slice's first event; an in-slice route's is its PageLoad's — the natural cut points when a window must split inside a slice, since a route boundary is where the page content just turned over wholesale.

    Each slice contributes its head arrival (arrival()), plus any in-slice route PageLoads: an SPA navigation fires a PageLoad in place rather than a checkout, so one slice can hold several pages. kind is 'page-load' for a real arrival, otherwise the head's boundary kind. A windowed split bounds the in-slice routes to [window_start, window_end); the slice head stays as the window's opening page context."""
    pages: list[tuple[str | None, str, int | None]] = []
    for slice_id in slice_ids:
        kind, url, _ = arrival(conn, slice_id)
        head = _head_event(conn, slice_id)
        if head is None:
            continue
        pages.append((url, kind, head["timestamp"]))
        head_route = _route_url(url)
        loads = conn.execute(
            f"SELECT url, timestamp FROM events WHERE slice_id = :s AND type_str = :pl {CANONICAL_ORDER}",
            {"s": slice_id, "pl": Kind.PAGE_LOAD},
        ).fetchall()
        for i, row in enumerate(loads):
            if i == 0 and _route_url(row["url"]) == head_route:
                continue  # the slice's own page-load, already counted as the head
            if window_start is not None and row["timestamp"] < window_start:
                continue
            if window_end is not None and row["timestamp"] >= window_end:
                continue
            pages.append((row["url"], "page-load", row["timestamp"]))
    return pages


def _page_loaded_at(
    conn: sqlite3.Connection, timeline: list[sqlite3.Row], position: int
) -> tuple[str | None, int]:
    """The window's page URL and when it loaded, walking back through same-page snapshot continuations to the page-load boundary."""
    while position > 0:
        row = timeline[position]
        kind, url, _ = arrival(conn, row["id"])
        if kind != "unattested":
            break
        prior = timeline[position - 1]
        prior_head = _head_event(conn, prior["id"])
        if prior_head is None or prior_head["url"] != url:
            break
        position -= 1
    row = timeline[position]
    head = _head_event(conn, row["id"])
    return (head["url"] if head else None), row["start_ts"]


def session_context_block(
    conn: sqlite3.Connection,
    slice_ids: list[int],
    window_start: int | None = None,
    window_end: int | None = None,
    labels: dict[int, str] | None = None,
) -> str:
    """`labels` maps slice_id → the window's label for it, so a gap between named slices is stated between the labels the citations use; without labels the gap is stated between the window's slices unnamed."""
    placeholders = ",".join("?" * len(slice_ids))
    visitors = [
        row["visitor_id"]
        for row in conn.execute(
            f"SELECT DISTINCT visitor_id FROM slices WHERE id IN ({placeholders})",
            slice_ids,
        )
    ]
    if len(visitors) != 1:
        raise ValueError(
            f"a session-context block is one visitor's story, got {visitors} — "
            f"the composer builds one block per visitor group"
        )
    visitor_id = visitors[0]

    timeline = conn.execute(
        "SELECT * FROM slices WHERE visitor_id = ? AND status != 'rescued' ORDER BY start_ts",
        (visitor_id,),
    ).fetchall()
    positions = {row["id"]: i for i, row in enumerate(timeline)}
    window_positions = sorted(positions[slice_id] for slice_id in slice_ids)
    first_in, last_in = window_positions[0], window_positions[-1]
    included = [timeline[i] for i in window_positions]

    def concurrent(row) -> bool:
        """A slice running alongside an included one — a second tab's lane. Leaving it out is a deliberate choice about the other lane, not a gap in this window, so it earns no gap statement."""
        return any(
            row["start_ts"] <= m["end_ts"] and m["start_ts"] <= row["end_ts"]
            for m in included
        )

    in_window = set(window_positions)
    gap_lines = []
    for a_pos, b_pos in itertools.pairwise(window_positions):
        left_out = [
            timeline[i]
            for i in range(a_pos + 1, b_pos)
            if i not in in_window and not concurrent(timeline[i])
        ]
        if not left_out:
            continue
        gap = _duration(timeline[b_pos]["start_ts"] - timeline[a_pos]["end_ts"])
        entries: list[str] = []
        positions_by_url: dict[str, int] = {}
        unplaced_incomplete = False
        for r in left_out:
            marked = f" ({INCOMPLETE})" if r["status"] == "discarded" else ""
            pages = [u for u, _, _ in window_pages(conn, [r["id"]]) if u]
            if not pages and marked:
                unplaced_incomplete = True
            for url in map(describe_url, pages):
                if url in positions_by_url:
                    if marked:
                        entries[positions_by_url[url]] = url + marked
                else:
                    positions_by_url[url] = len(entries)
                    entries.append(url + marked)
        where = ": " + ", ".join(entries) if entries else ""
        tail = f"; {INCOMPLETE}" if unplaced_incomplete else ""
        seam = (
            f"Between {labels[timeline[a_pos]['id']]} and {labels[timeline[b_pos]['id']]}"
            if labels
            else "Between this recording's slices"
        )
        gap_lines.append(
            f"{seam}: {gap} pass; what was recorded there is not part of this recording{where}{tail}."
        )

    window_lo = (
        window_start if window_start is not None else timeline[first_in]["start_ts"]
    )
    lines = []

    kind, _, referrer = arrival(conn, timeline[first_in]["id"])
    opens_mid_page = (
        kind == "unattested"
        and first_in > 0
        and (head := _head_event(conn, timeline[first_in - 1]["id"]))
        and head["url"] == _head_event(conn, timeline[first_in]["id"])["url"]
    ) or (window_start is not None and window_start > timeline[first_in]["start_ts"])

    if opens_mid_page:
        url, loaded_at = _page_loaded_at(conn, timeline, first_in)
        lines.append(
            f"This recording opens mid-page: {describe_url(url)}, loaded {_duration(window_lo - loaded_at)} earlier."
        )
        before_position = next(
            (
                i
                for i in range(first_in - 1, -1, -1)
                if (h := _head_event(conn, timeline[i]["id"])) and h["url"] != url
            ),
            None,
        )
    else:
        before_position = first_in - 1 if first_in > 0 else None

    if before_position is not None:
        before = timeline[before_position]
        gap = window_lo - before["end_ts"]
        lines.append(
            f"Prior page: {describe_url(_tail_url(conn, before['id']))}, left {_duration(gap)} before this recording."
        )
    elif not opens_mid_page:
        lines.append("Prior page: none recorded.")
    if kind == "page-load":
        lines.append(f"Referrer: {describe_url(referrer) if referrer else 'none'}.")
    lines.extend(gap_lines)

    if last_in + 1 < len(timeline):
        after = timeline[last_in + 1]
        window_hi = (
            window_end if window_end is not None else timeline[last_in]["end_ts"]
        )
        gap = after["start_ts"] - window_hi
        url = (_head_event(conn, after["id"]) or {"url": None})["url"]
        where = f"Next page: {describe_url(url)}, {_duration(max(gap, 0))} after this recording"
        if after["status"] == "discarded":
            where += f" ({INCOMPLETE})"
        lines.append(where + ".")
    else:
        lines.append("Next page: none recorded.")

    return "<SESSION_CONTEXT>\n" + "\n".join(lines) + "\n</SESSION_CONTEXT>"
