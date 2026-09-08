import gzip
import json
import zlib

from _support import (
    SNIPPET,
    T0,
    UA_DESKTOP,
    VISITOR,
    chunk_blob,
    click,
    counted,
    full_snapshot,
    meta,
)
from locus.evidence.chunk import decode_or_skip, parse_user_agent, prefix_skip_reason
from locus.evidence.db import connect
from locus.evidence.hydrate import canonical_events, hydrate_stream
from locus.evidence.rrweb_constants import EventType, IncrementalSource
from locus.evidence.slices import materialize_slices

META = counted(meta(T0), 1)


def test_user_agent_facts_are_uap_core_families():
    assert parse_user_agent(UA_DESKTOP) == {
        "device": "desktop",
        "os": "Windows",
        "browser": "Chrome",
    }
    assert parse_user_agent(
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
        "Mobile/15E148 Safari/604.1"
    ) == {"device": "mobile", "os": "iOS", "browser": "Mobile Safari"}
    assert parse_user_agent(
        "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
    ) == {"device": "mobile", "os": "Android", "browser": "Chrome Mobile"}
    assert parse_user_agent(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.1 Safari/605.1.15"
    ) == {"device": "desktop", "os": "Mac OS X", "browser": "Safari"}
    assert (
        parse_user_agent(
            "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
        )["device"]
        == "tablet"
    )
    assert (
        parse_user_agent(
            "Mozilla/5.0 (Linux; Android 14; SM-T510) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )["device"]
        == "tablet"
    )
    assert parse_user_agent(None) == {}
    assert parse_user_agent("gibberish") == {"device": "desktop"}


def test_touch_points_unmask_the_ipad_desktop_mode_ua():
    """iPadOS desktop mode sends macOS Safari's exact UA; maxTouchPoints is what contradicts it."""
    ipad_desktop_mode = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.1 Safari/605.1.15"
    )
    assert parse_user_agent(ipad_desktop_mode, 5)["device"] == "tablet"
    # A real Mac reports 0 touch points and stays a desktop.
    assert parse_user_agent(ipad_desktop_mode, 0)["device"] == "desktop"
    # Chunks from recorders that predate the field carry no touch points and keep the UA's lie.
    assert parse_user_agent(ipad_desktop_mode, None)["device"] == "desktop"
    # The override is the iPad shape only: a touch-screen laptop is honestly a desktop.
    assert parse_user_agent(UA_DESKTOP, 10)["device"] == "desktop"
    # An open-write store can hand junk in the field; anything but an int is no evidence.
    assert parse_user_agent(ipad_desktop_mode, True)["device"] == "desktop"
    assert parse_user_agent(ipad_desktop_mode, "5")["device"] == "desktop"


def test_screen_shape_splits_desktop_mode_iphones_from_ipads():
    """An iPhone with Request Desktop Website on sends the same Mac UA and touch points as an iPad; the screen's aspect ratio is what differs."""
    mac_ua = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.1 Safari/605.1.15"
    )
    # iPhone panels, portrait and landscape alike.
    assert parse_user_agent(mac_ua, 5, 390, 844)["device"] == "mobile"
    assert parse_user_agent(mac_ua, 5, 932, 430)["device"] == "mobile"
    # iPad panels.
    assert parse_user_agent(mac_ua, 5, 820, 1180)["device"] == "tablet"
    assert parse_user_agent(mac_ua, 5, 1366, 1024)["device"] == "tablet"
    # A window-sized screen (Stage Manager, Split View) is still tablet-shaped.
    assert parse_user_agent(mac_ua, 5, 600, 702)["device"] == "tablet"
    # No usable screen leaves the verdict at tablet.
    assert parse_user_agent(mac_ua, 5)["device"] == "tablet"
    assert parse_user_agent(mac_ua, 5, 0, 844)["device"] == "tablet"
    assert parse_user_agent(mac_ua, 5, "390", 844)["device"] == "tablet"
    # The screen never speaks outside the Mac-with-touch shape.
    assert parse_user_agent(mac_ua, 0, 390, 844)["device"] == "desktop"
    assert parse_user_agent(UA_DESKTOP, 10, 390, 844)["device"] == "desktop"


def test_a_pattern_in_a_prefix_is_named_impossible():
    """The store matches a prefix literally, so a glob would return the same nothing an empty store does; any other byte can sit in a real key of an open-write bucket and passes."""
    assert prefix_skip_reason("") is None
    assert prefix_skip_reason(f"{SNIPPET}/") is None
    assert prefix_skip_reason(f"{SNIPPET}/2026-09-01/{VISITOR}/0{T0}-aaaa/") is None
    assert prefix_skip_reason("backup 2026.tar.gz") is None
    assert prefix_skip_reason(f"{SNIPPET}/notes:é/") is None
    for bad in ("*/2026-09-01/", f"{SNIPPET}/2026-09-?1/", "snip[01]/"):
        reason = prefix_skip_reason(bad)
        assert reason is not None and "no wildcards" in reason, bad
    assert "'*'" in prefix_skip_reason("*/2026-09-01/")


def test_chromium_skins_are_their_own_families_not_chrome():
    """Every Chromium skin's UA also carries Chrome/; the family is the skin, not the engine."""
    assert (
        parse_user_agent(
            "Mozilla/5.0 (Linux; Android 14; SM-S911B) AppleWebKit/537.36 "
            "(KHTML, like Gecko) SamsungBrowser/23.0 Chrome/115.0.0.0 "
            "Mobile Safari/537.36"
        )["browser"]
        == "Samsung Internet"
    )
    assert (
        parse_user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0"
        )["browser"]
        == "Opera"
    )
    assert (
        parse_user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0"
        )["browser"]
        == "Edge"
    )


def test_a_ua_naming_no_browser_still_yields_the_engine_family():
    """A bare iOS in-app WKWebView names no browser, but the engine is nameable and named."""
    assert parse_user_agent(
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
    ) == {"device": "mobile", "os": "iOS", "browser": "Mobile Safari UI/WKWebView"}


def test_decode_maps_provenance_into_the_envelope():
    key = f"{SNIPPET}/2026-06-11/{VISITOR}/0{T0}-ab12/{T0}000001.json.gz"
    pairs = decode_or_skip(chunk_blob(f"0{T0}-ab12", [META]), key, SNIPPET)
    assert pairs == [
        (
            VISITOR,
            {
                **META,
                "_envelope": {
                    "device": "desktop",
                    "os": "Windows",
                    "browser": "Chrome",
                    "language": None,
                    "time_zone": None,
                    "screen_width": None,
                    "screen_height": None,
                    "snippet": SNIPPET,
                    "script_version": "locus-recorder/0.1.0",
                    "recorder_slice": f"0{T0}-ab12",
                },
            },
        )
    ]


def test_the_device_facts_only_the_browser_knew_reach_the_db():
    """Language, time zone and screen are not in any rrweb event and cannot be derived from one.
    A visitor reading a page not in their language, at 3am their time, in a small window is three
    facts about the session — and each is lost forever if the load drops it on the floor."""
    blob = chunk_blob(
        f"0{T0}-ab12",
        [META],
        envelope={
            "language": "fr-FR",
            "timeZone": "Europe/Paris",
            "screen": {"width": 2560, "height": 1440},
        },
    )
    [(_, event)] = decode_or_skip(blob, "snip/2026-06-11/v/slice/k", SNIPPET)
    envelope = event["_envelope"]
    assert envelope["language"] == "fr-FR"
    assert envelope["time_zone"] == "Europe/Paris"
    assert (envelope["screen_width"], envelope["screen_height"]) == (2560, 1440)


def test_a_transcoded_body_decodes_whole_from_under_the_gzip_wrapper():
    """The worker stores a non-gzip arrival byte-exact under a gzip wrapper. Ungzipped, the wire
    bytes may be the JSON payload itself (a transcoding proxy decompressed the upload) or a
    deflate stream in either framing (a broken CompressionStream emitted the wrong format) —
    each decodes to the same events, nothing lost."""
    wire = gzip.decompress(chunk_blob(f"0{T0}-ab12", [META]))
    for arrived in [
        wire,
        zlib.compress(wire),
        zlib.compress(wire)[2:-4],
    ]:  # raw deflate: zlib minus header/checksum
        assert len(decode_or_skip(gzip.compress(arrived), "k", SNIPPET)) == 1


def test_corrupt_chunk_is_named_and_skipped_not_a_load_wedge(capsys):
    good = chunk_blob(f"0{T0}-ab12", [META])
    assert len(decode_or_skip(good, "good-key")) == 1

    bad = bytearray(good)
    bad[-1] ^= 0xFF
    assert decode_or_skip(bytes(bad), "snip/2026-06-11/v/slice/bad-key") == []
    err = capsys.readouterr().err
    assert "corrupt chunk" in err and "bad-key" in err


def test_decodable_bytes_that_are_not_a_chunk_skip_named(capsys):
    for payload in [
        [1, 2, 3],
        {"visitorId": 5, "events": []},
        {"visitorId": VISITOR, "events": "not-a-list"},
        {"visitorId": VISITOR, "events": [], "envelope": "not-an-object"},
        {"visitorId": VISITOR, "events": [], "errors": "not-a-list"},
        {"visitorId": VISITOR, "events": [], "sliceId": 12345},
        {
            "visitorId": VISITOR,
            "sliceId": f"0{T0}-ab12",
            "recorderVersion": "locus-recorder/0.1.0",
            "envelope": {"userAgent": 42},
            "events": [],
        },
    ]:
        blob = gzip.compress(json.dumps(payload).encode())
        assert decode_or_skip(blob, "snip/2026-06-11/v/slice/k") == []
    err = capsys.readouterr().err
    assert err.count("skipping corrupt chunk snip/2026-06-11/v/slice/k") == 7


def test_a_chunk_with_no_slice_id_costs_itself_and_not_the_whole_load(capsys):
    """The bucket is open-write, so an object that isn't a recorder chunk can land in it. Admitting
    one because its slice id merely *might* be absent does not make the load resilient — it moves the
    failure to slice materialization, which cannot recover a slicing and stops the entire load. The loss is
    named here, by the key that carried it, and the rest of the corpus loads."""
    for payload in [
        {
            "visitorId": VISITOR,
            "recorderVersion": "locus-recorder/0.1.0",
            "envelope": {"userAgent": UA_DESKTOP},
            "events": [META],
        },  # no sliceId at all
        {
            "visitorId": VISITOR,
            "sliceId": "not-a-slice-id",
            "recorderVersion": "locus-recorder/0.1.0",
            "events": [META],
        },  # not the recorder's shape
        {
            "visitorId": VISITOR,
            "sliceId": f"0{T0}-ab12",
            "events": [META],
        },  # no recorderVersion
    ]:
        blob = gzip.compress(json.dumps(payload).encode())
        assert decode_or_skip(blob, "snip/2026-06-11/v/slice/k", SNIPPET) == []
    assert capsys.readouterr().err.count("skipping corrupt chunk") == 3


def test_an_event_with_no_counter_is_dropped_rather_than_ordered_by_luck(capsys):
    """The canonical order (db.py's CANONICAL_ORDER) leans on counter. The recorder stamps one on every event it emits, so a
    chunk's event without one is damaged — and the row id it would fall back to is, for a store whose
    chunks are fetched concurrently, simply which GET returned first. That is not an order."""
    uncounted = {
        "type": EventType.IncrementalSnapshot,
        "timestamp": T0,
        "data": {"source": IncrementalSource.Scroll, "id": 4, "x": 0, "y": 12},
    }
    blob = chunk_blob(f"0{T0}-ab12", [META, uncounted])

    pairs = decode_or_skip(blob, "snip/2026-06-11/v/slice/k", SNIPPET)

    assert len(pairs) == 1  # the Meta, which is counted
    assert "no counter" in capsys.readouterr().err


def test_malformed_event_costs_itself_never_the_chunk(capsys):
    good = counted(
        {
            "type": EventType.IncrementalSnapshot,
            "timestamp": T0,
            "data": {"source": IncrementalSource.Scroll, "id": 4, "x": 0, "y": 12},
        },
        1,
    )
    blob = gzip.compress(
        json.dumps(
            {
                "visitorId": VISITOR,
                "sliceId": f"0{T0}-ab12",
                "recorderVersion": "locus-recorder/0.1.0",
                "envelope": {"userAgent": UA_DESKTOP},
                "events": [
                    good,
                    42,
                    {"type": str(EventType.Meta), "timestamp": T0, "data": {}},
                    {"type": EventType.Meta, "data": {}},
                    {"type": EventType.Meta, "timestamp": "soon", "data": {}},
                    {"type": EventType.Meta, "timestamp": T0, "data": {}, "counter": 7},
                    {"type": EventType.Meta, "timestamp": T0, "data": "not-an-object"},
                    {
                        "type": EventType.IncrementalSnapshot,
                        "timestamp": T0,
                        "data": {
                            "source": IncrementalSource.MouseMove,
                            "positions": [{"x": 1, "y": 2}],
                        },
                    },
                ],
                "errors": [],
            }
        ).encode()
    )
    pairs = decode_or_skip(blob, "snip/2026-06-11/v/slice/k")
    assert [event["counter"] for _, event in pairs] == [good["counter"]]
    err = capsys.readouterr().err
    assert "skipping 7 malformed events in chunk snip/2026-06-11/v/slice/k" in err
    assert "timestamp is not a number" in err


def test_payload_errors_reach_the_on_errors_hook():
    seen = []

    def hook(key, payload):
        seen.append((key, payload["errors"]))

    loud = chunk_blob(
        f"0{T0}-ab12", [META], errors=["dropped 2 malformed buffered records"]
    )
    assert len(decode_or_skip(loud, "k1", SNIPPET, on_errors=hook)) == 1
    assert seen == [("k1", ["dropped 2 malformed buffered records"])]

    quiet = chunk_blob(f"0{T0}-ab12", [META])
    assert len(decode_or_skip(quiet, "k2", SNIPPET, on_errors=hook)) == 1

    corrupt = bytearray(loud)
    corrupt[-1] ^= 0xFF
    assert decode_or_skip(bytes(corrupt), "k3", SNIPPET, on_errors=hook) == []
    assert seen == [("k1", ["dropped 2 malformed buffered records"])], (
        "only decodable chunks testify; corrupt ones already print their skip"
    )


def test_hydrated_stream_is_invariant_to_chunking_order_and_duplication(tmp_path):
    slice_a = f"0{T0}-aaaa"
    slice_b = f"0{T0 + 90_000}-bbbb"
    stream = [
        counted(event, seq)
        for seq, event in enumerate(
            [
                meta(T0),
                full_snapshot(T0 + 1),
                click(T0 + 5_000, 10),
                click(T0 + 6_000, 11),
                click(T0 + 6_000, 12),
                meta(T0 + 90_000),
                full_snapshot(T0 + 90_001),
                click(T0 + 95_000, 13),
            ],
            start=1,
        )
    ]
    chunks = [
        chunk_blob(slice_a, stream[0:2]),
        chunk_blob(slice_a, stream[2:5]),
        chunk_blob(slice_b, stream[5:8]),
    ]
    orders = [
        chunks,
        chunks[::-1],
        [chunks[1], chunks[2], chunks[0], chunks[1], chunks[1]],
    ]

    outcomes = []
    for i, order in enumerate(orders):
        conn = connect(tmp_path / f"events_{i}.db")
        for blob in order:
            hydrate_stream(conn, decode_or_skip(blob, "key"))
        materialize_slices(conn, VISITOR)
        slices = conn.execute(
            "SELECT start_ts, end_ts, n_events, status FROM slices ORDER BY start_ts"
        ).fetchall()
        outcomes.append(
            (canonical_events(conn, VISITOR), [tuple(row) for row in slices])
        )

    assert outcomes[0] == outcomes[1] == outcomes[2]
    events, slices = outcomes[0]
    assert [event["counter"] for event in events] == [
        event["counter"] for event in stream
    ]
    assert len(slices) == 2

    row = (
        connect(tmp_path / "events_0.db")
        .execute("SELECT device, os, browser, script_version FROM events LIMIT 1")
        .fetchone()
    )
    assert tuple(row) == ("desktop", "Windows", "Chrome", "locus-recorder/0.1.0")

    keyed = (
        connect(tmp_path / "events_0.db")
        .execute("SELECT recorder_slice FROM slices ORDER BY start_ts")
        .fetchall()
    )
    assert [row["recorder_slice"] for row in keyed] == [slice_a, slice_b]
