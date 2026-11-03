import pytest
from _analysis_support import ScriptedConversation as FakeConversation
from locus.analysis.analyze import compose_window, screenshot_index
from locus.analysis.select_screenshots import select_screenshots
from locus.analysis.window import format_offset, resolve_window
from locus.evidence.db import connect


def test_distilled_kinds_are_names_never_numeric_codes(distilled_db):
    conn = connect(distilled_db)
    kinds = [
        r["type_str"] for r in conn.execute("SELECT DISTINCT type_str FROM events")
    ]
    assert kinds
    assert all(kind and not kind.isdigit() for kind in kinds)


def test_compose_window_shape(distilled_db, tmp_path):
    conn = connect(distilled_db)
    ordered = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM slices WHERE status='replayable' ORDER BY start_ts"
        )
    ]
    biggest = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]
    at = ordered.index(biggest)
    slice_ids = (
        ordered[at : at + 2] if at + 1 < len(ordered) else ordered[at - 1 : at + 1]
    )

    selection = select_screenshots(conn, slice_ids[0])
    screenshot_dir = tmp_path / "screenshots"
    screenshot_dir.mkdir()
    for ts in selection.timestamps:
        (screenshot_dir / f"S1_activity_screenshot_{ts}.png").write_bytes(
            b"\x89PNG fake"
        )

    conversation = FakeConversation()
    manifest = compose_window(
        conn,
        conversation,
        resolve_window(conn, slice_ids),
        screenshot_index([screenshot_dir]),
        site_contexts={"site": "An example site."},
    )

    kinds = [kind for kind, _ in conversation.parts]
    texts = [t for k, t in conversation.parts if k == "text"]
    assert any(t.startswith("<WEBSITE>") for t in texts)
    recording = next(t for t in texts if t.startswith("<SUMMARY>"))
    assert "2026-06-12T08:58:14Z" in recording and "2026-06-12T08:58:26Z" in recording
    assert "Visitor:" not in recording, (
        "one visitor's window needs no attribution labels"
    )
    slices_line = next(l for l in recording.splitlines() if l.startswith("Slices:"))
    assert "S1" in slices_line and "S2" in slices_line, (
        "a multi-slice group declares its labels and their bounds"
    )

    single = FakeConversation()
    single_manifest = compose_window(
        conn,
        single,
        resolve_window(conn, slice_ids[:1]),
        screenshot_index([screenshot_dir]),
        site_contexts={"site": "An example site."},
    )
    single_recording = next(
        t for k, t in single.parts if k == "text" and t.startswith("<SUMMARY>")
    )
    assert "Slices:" not in single_recording, (
        "a single-slice window carries no label apparatus"
    )
    assert [s.label for s in single_manifest.slices] == ["S1"]
    assert any(t.startswith("<SESSION_CONTEXT>") for t in texts)
    assert not any("<QUESTION>" in text for _, text in conversation.parts), (
        "the question is never composed into the window; it is sent last, "
        "on the turn that writes the answer, and never recorded"
    )
    assert kinds.count("image") == len(selection.timestamps) == manifest.n_screenshots

    stream_lines = "\n".join(
        t
        for t in texts
        if not t.startswith(("<WEBSITE", "<SUMMARY>", "<SESSION_CONTEXT>"))
    )
    assert "[S1 00:00.000]" in stream_lines, (
        "a multi-slice window stamps every stream line with its slice label"
    )
    assert "\n[00:" not in stream_lines, "no line is left bare-stamped"
    assert "FullSnapshot" in stream_lines

    single_stream = "\n".join(
        t
        for k, t in single.parts
        if k == "text"
        and not t.startswith(("<WEBSITE", "<SUMMARY>", "<SESSION_CONTEXT>"))
    )
    assert "[00:00.000]" in single_stream and "[S1 " not in single_stream, (
        "a single-slice window stamps bare offsets"
    )

    assert [s.label for s in manifest.slices] == ["S1", "S2"]
    assert [s.slice_id for s in manifest.slices] == slice_ids, (
        "labels run S1..Sn in time order"
    )
    placeholders = ",".join("?" * len(slice_ids))
    expected_events = conn.execute(
        f"SELECT SUM(n_events) total FROM slices WHERE id IN ({placeholders})",
        slice_ids,
    ).fetchone()["total"]
    assert manifest.n_events == expected_events
    first_event_ts = conn.execute(
        "SELECT MIN(timestamp) t FROM events WHERE slice_id = ?", (slice_ids[0],)
    ).fetchone()["t"]
    assert manifest.window_start_ts == first_event_ts


def test_the_screenshot_plane_opens_at_the_covering_snapshot(distilled_db, tmp_path):
    """A slice opens before rrweb has a DOM to capture, so its first instants hold no
    recorded page state — a screenshot there would be stamped with a time whose pixels were
    never recorded. The window's screenshot-addressable bounds open at the covering
    snapshot; the events ahead of it keep their place in the stream."""
    from locus.analysis.window import slice_table
    from locus.evidence.rrweb_constants import EventType

    conn = connect(distilled_db)
    slice_id = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]
    snapshot_ts = conn.execute(
        "SELECT MIN(timestamp) t FROM events WHERE slice_id = ? AND type = ?",
        (slice_id, EventType.FullSnapshot),
    ).fetchone()["t"]

    row = slice_table(conn, [slice_id])[0]
    assert (
        row.snippet
        == conn.execute(
            "SELECT snippet FROM slices WHERE id = ?", (slice_id,)
        ).fetchone()["snippet"]
    ), "the table names the site each slice was recorded on"
    assert row.start_ts < snapshot_ts, (
        "this recording really does open before its snapshot"
    )
    assert row.screenshot_start_ts == snapshot_ts
    assert row.end_ts > snapshot_ts

    screenshot_dir = tmp_path / "screenshots"
    screenshot_dir.mkdir()
    (screenshot_dir / f"S1_activity_screenshot_{row.start_ts}.png").write_bytes(
        b"\x89PNG fake"
    )
    with pytest.raises(ValueError, match="screenshot-addressable"):
        compose_window(
            conn,
            FakeConversation(),
            resolve_window(conn, [slice_id]),
            screenshot_index([screenshot_dir]),
            site_contexts={"site": "x"},
        )

    events_before = conn.execute(
        "SELECT COUNT(*) n FROM events WHERE slice_id = ? AND timestamp < ?",
        (slice_id, snapshot_ts),
    ).fetchone()["n"]
    conversation = FakeConversation()
    manifest = compose_window(
        conn,
        conversation,
        resolve_window(conn, [slice_id]),
        {},
        site_contexts={"site": "x"},
    )
    assert manifest.n_events > events_before > 0, (
        "the events ahead of the snapshot still ride the stream"
    )


def test_a_window_holding_only_a_slices_lead_is_refused(distilled_db):
    from locus.analysis.window import slice_table
    from locus.evidence.rrweb_constants import EventType

    conn = connect(distilled_db)
    slice_id = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]
    snapshot_ts = conn.execute(
        "SELECT MIN(timestamp) t FROM events WHERE slice_id = ? AND type = ?",
        (slice_id, EventType.FullSnapshot),
    ).fetchone()["t"]
    with pytest.raises(ValueError, match="ends before slice"):
        slice_table(conn, [slice_id], None, snapshot_ts)


def test_every_screenshot_has_a_text_anchor_at_its_timestamp(distilled_db, tmp_path):
    # Every screenshot's moment is stated in the stream's own clock, on its own
    # Screenshot line — never inferred from adjacency to whatever event happens
    # to share its timestamp, and owed even at a moment whose event prints no
    # line of its own.
    conn = connect(distilled_db)
    suppressed = conn.execute(
        "SELECT e.slice_id s, e.timestamp t FROM events e "
        "JOIN slices ON slices.id = e.slice_id "
        "WHERE e.type_str = 'Focus' AND slices.status = 'replayable' "
        "LIMIT 1"
    ).fetchone()
    screenshot_dir = tmp_path / "screenshots"
    screenshot_dir.mkdir()
    (screenshot_dir / f"S1_activity_screenshot_{suppressed['t']}.png").write_bytes(
        b"\x89PNG fake"
    )

    conversation = FakeConversation()
    manifest = compose_window(
        conn,
        conversation,
        resolve_window(conn, [suppressed["s"]]),
        screenshot_index([screenshot_dir]),
        site_contexts={"site": "x"},
    )

    image_at = next(
        i for i, (kind, _) in enumerate(conversation.parts) if kind == "image"
    )
    anchor = conversation.parts[image_at - 1][1].splitlines()[-1]
    offset = format_offset(suppressed["t"], manifest.window_start_ts)
    assert anchor == f"{offset} Screenshot"


def test_compose_prefetches_the_window_screenshots_before_adding_any(
    distilled_db, tmp_path
):
    conn = connect(distilled_db)
    slice_id = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]
    selection = select_screenshots(conn, slice_id)
    screenshot_dir = tmp_path / "screenshots"
    screenshot_dir.mkdir()
    for ts in selection.timestamps:
        (screenshot_dir / f"S1_activity_screenshot_{ts}.png").write_bytes(
            b"\x89PNG fake"
        )

    class PrefetchSpy(FakeConversation):
        def __init__(self):
            super().__init__()
            self.prefetched = []

        def prefetch_images(self, image_paths):
            assert not self.images, "prefetch must precede every image add"
            self.prefetched.append(sorted(str(p) for p in image_paths))

    conversation = PrefetchSpy()
    compose_window(
        conn,
        conversation,
        resolve_window(conn, [slice_id]),
        screenshot_index([screenshot_dir]),
        site_contexts={"site": "x"},
    )

    assert conversation.prefetched == [
        sorted(
            str(screenshot_dir / f"S1_activity_screenshot_{ts}.png")
            for ts in selection.timestamps
        )
    ]
    assert conversation.prefetched[0] == sorted(conversation.images), (
        "exactly the screenshots the window sends, no more"
    )


def test_window_bounds_split_a_slice_into_pieces(distilled_db, tmp_path):
    conn = connect(distilled_db)
    slice_id = conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]
    bounds = conn.execute(
        "SELECT MIN(timestamp) lo, MAX(timestamp) hi, COUNT(*) n FROM events WHERE slice_id = ?",
        (slice_id,),
    ).fetchone()
    midpoint = (bounds["lo"] + bounds["hi"]) // 2

    first = FakeConversation()
    head = compose_window(
        conn,
        first,
        resolve_window(conn, [slice_id], None, midpoint),
        {},
        site_contexts={"site": "x"},
    )
    second = FakeConversation()
    tail = compose_window(
        conn,
        second,
        resolve_window(conn, [slice_id], midpoint, None),
        {},
        site_contexts={"site": "x"},
    )

    assert head.n_events + tail.n_events == bounds["n"]
    assert head.window_start_ts == bounds["lo"]
    assert tail.window_start_ts >= midpoint

    head_stream = "\n".join(t for k, t in first.parts if k == "text")
    tail_stream = "\n".join(t for k, t in second.parts if k == "text")
    assert "FullSnapshot" in head_stream
    assert "FullSnapshot" not in tail_stream
    assert "\n[00:00." in tail_stream

    with pytest.raises(ValueError, match="holds no events"):
        resolve_window(conn, [slice_id], None, bounds["lo"])


def test_an_identify_rides_the_stream_and_displaces_no_slice_page(tmp_path):
    """The operator's user id reaches the model as a stream line of its own, raw. An Identify ahead of a slice's snapshot names no page, so every slice still opens on the page it opened on."""
    import subprocess

    from _support import DISTILL, FIXTURE, identify, read_recording
    from locus.analysis.session_context import window_pages
    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    visitor_id, events = read_recording(FIXTURE)
    by_slice: dict[str, list[dict]] = {}
    for event in events:
        by_slice.setdefault(event["_envelope"]["recorder_slice"], []).append(event)
    identified = list(events)
    for recorder_slice, members in by_slice.items():
        ts = min(e["timestamp"] for e in members) - 5
        identified.append(
            {
                **identify(ts, "user-42@example.com"),
                "counter": f"{ts}000999",
                "_envelope": {"recorder_slice": recorder_slice},
            }
        )

    def loaded(name, recording):
        path = tmp_path / f"{name}.db"
        conn = connect(path)
        hydrate(conn, visitor_id, recording)
        materialize_slices(conn, visitor_id)
        subprocess.run(
            ["bun", str(DISTILL), str(path)], check=True, capture_output=True
        )
        slice_ids = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM slices WHERE status = 'replayable' ORDER BY start_ts"
            )
        ]
        return conn, slice_ids

    baseline, baseline_ids = loaded("baseline", events)
    conn, slice_ids = loaded("identified", identified)
    assert [(url, kind) for url, kind, _ in window_pages(conn, slice_ids)] == [
        (url, kind) for url, kind, _ in window_pages(baseline, baseline_ids)
    ]

    conversation = FakeConversation()
    compose_window(
        conn,
        conversation,
        resolve_window(conn, slice_ids),
        {},
        site_contexts={"site": "x"},
    )
    text = "\n".join(conversation.texts)
    assert text.count("Identify user_id='user-42@example.com'") == len(slice_ids)


def _overlap_db(tmp_path):
    """The same recording under two visitors — two slices occupying the very same instants, the concurrency case the window clock must carry honestly."""
    import subprocess

    from _support import FIXTURE, read_recording
    from locus.evidence.hydrate import hydrate
    from locus.evidence.slices import materialize_slices

    visitor_id, events = read_recording(FIXTURE)
    conn = connect(tmp_path / "overlap.db")
    hydrate(conn, visitor_id, events)
    materialize_slices(conn, visitor_id)
    hydrate(conn, "concurrent-visitor", events)
    materialize_slices(conn, "concurrent-visitor")
    from _support import DISTILL as distill

    subprocess.run(
        ["bun", str(distill), str(tmp_path / "overlap.db")],
        check=True,
        capture_output=True,
    )
    first = conn.execute(
        "SELECT id FROM slices WHERE visitor_id = ? AND status='replayable' ORDER BY start_ts LIMIT 1",
        (visitor_id,),
    ).fetchone()["id"]
    second = conn.execute(
        "SELECT id FROM slices WHERE visitor_id = 'concurrent-visitor' "
        "AND status='replayable' ORDER BY start_ts LIMIT 1"
    ).fetchone()["id"]
    return conn, first, second


def test_overlapping_recordings_compose_on_one_shared_clock(tmp_path):
    """Two slices covering the same instants compose as two labeled slices
    whose offsets genuinely overlap — the concurrency lives in the numbers,
    never serialized apart."""
    conn, first, second = _overlap_db(tmp_path)

    conversation = FakeConversation()
    manifest = compose_window(
        conn,
        conversation,
        resolve_window(conn, [first, second]),
        {},
        site_contexts={"site": "x"},
    )

    assert [s.label for s in manifest.slices] == ["S1", "S2"]
    starts = [s.start_ts for s in manifest.slices]
    ends = [s.end_ts for s in manifest.slices]
    assert starts[1] <= ends[0], "the fixture's slices truly overlap"
    assert manifest.window_start_ts == min(starts)

    stream_lines = "\n".join(t for k, t in conversation.parts if k == "text")
    assert "[S1 00:00.000]" in stream_lines and "[S2 00:00.000]" in stream_lines, (
        "both slices open at the same window offset — concurrency is embodied in the numbers"
    )

    recordings = [
        t for k, t in conversation.parts if k == "text" and t.startswith("<SUMMARY>")
    ]
    assert len(recordings) == 2, "concurrent visitors stay delimited, attributed groups"


def test_a_multi_site_window_attributes_each_sites_frame(tmp_path):
    """A window spanning sites carries each site's own operator frame beside
    its slices — attributed WEBSITE blocks, and each SUMMARY naming its
    site — so no site's evidence reads under another site's frame."""
    conn, first, second = _overlap_db(tmp_path)
    conn.execute("UPDATE events SET snippet = 'site-a' WHERE slice_id = ?", (first,))
    conn.execute("UPDATE events SET snippet = 'site-b' WHERE slice_id = ?", (second,))
    conn.commit()

    conversation = FakeConversation()
    compose_window(
        conn,
        conversation,
        resolve_window(conn, [first, second]),
        {},
        site_contexts={"site-a": "Frame A.", "site-b": "Frame B."},
    )
    texts = [t for k, t in conversation.parts if k == "text"]
    websites = [t for t in texts if t.startswith("<WEBSITE")]
    assert websites == [
        '<WEBSITE site="site-a">\nFrame A.\n</WEBSITE>',
        '<WEBSITE site="site-b">\nFrame B.\n</WEBSITE>',
    ], "one attributed block per site, in the order the groups appear"
    recordings = [t for t in texts if t.startswith("<SUMMARY>")]
    assert "Site: site-a" in recordings[0] and "Site: site-b" in recordings[1]

    with pytest.raises(ValueError, match="site_contexts"):
        compose_window(
            conn,
            FakeConversation(),
            resolve_window(conn, [first, second]),
            {},
            site_contexts={"site-a": "Frame A.", "other": "x"},
        )


def test_a_group_window_attributes_each_visitors_slices(two_visitor_db):
    conn = connect(two_visitor_db)
    picks = conn.execute(
        "SELECT s.id, s.visitor_id, s.start_ts FROM slices s "
        "JOIN (SELECT visitor_id, MIN(start_ts) first_ts FROM slices "
        "      WHERE status='replayable' GROUP BY visitor_id) heads "
        "ON heads.visitor_id = s.visitor_id AND heads.first_ts = s.start_ts "
        "ORDER BY s.start_ts"
    ).fetchall()
    assert len(picks) == 2, "the fixture carries two visitors"

    conversation = FakeConversation()
    manifest = compose_window(
        conn,
        conversation,
        resolve_window(conn, [p["id"] for p in picks]),
        {},
        site_contexts={"site": "x"},
    )

    texts = [t for k, t in conversation.parts if k == "text"]
    recordings = [t for t in texts if t.startswith("<SUMMARY>")]
    assert len(recordings) == 2, (
        "each visitor's slices open with their own SUMMARY block"
    )
    for pick, block in zip(picks, recordings):
        assert f"Visitor: {pick['visitor_id']}" in block, (
            "the evidence itself attributes each slice to its visitor"
        )
    assert len([t for t in texts if t.startswith("<SESSION_CONTEXT>")]) == 2
    assert manifest.window_start_ts == min(p["start_ts"] for p in picks), (
        "one clock: offsets anchor at the earliest event across the group"
    )
    assert [(s.label, s.visitor) for s in manifest.slices] == [
        ("S1", picks[0]["visitor_id"]),
        ("S2", picks[1]["visitor_id"]),
    ], "the slice table names each label's visitor and slice"
