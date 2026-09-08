"""The recorder's wiring, exercised in a real Chromium: the built bundle recording a real page, draining real chunks to a collecting sink, and the captured stream fed back through evidence — hydrate, materialize slices, distill — into analysis's arrival, closing the loop the unit suites cannot: recorder, evidence, and analysis agreeing across the chunk boundary."""

import gzip
import json
import shutil
import subprocess

import pytest
from _harness import serve_dir
from _support import DISTILL
from locus.analysis.session_context import arrival
from locus.evidence.db import connect
from locus.evidence.hydrate import hydrate
from locus.evidence.rrweb_constants import EventType
from locus.evidence.slices import materialize_slices

PAGE_A = """<!DOCTYPE html>
<html><head><title>Front Door</title></head>
<body><a id="go" href="/page_b.html">enter the shop</a></body></html>
"""

PAGE_B = """<!DOCTYPE html>
<html>
  <head><title>Demo Shop</title></head>
  <body>
    <h1>Demo Shop</h1>
    <input id="name" type="text">
    <input id="pw" type="password">
    <button id="buy">Buy now</button>
    <script type="module">
      import * as LocusRecorder from "/locus-recorder.lib.js";
      window.LocusRecorder = LocusRecorder;
    </script>
  </body>
</html>
"""

START_JS = """
async (config) => {
  window.__chunks = [];
  window.__beacons = [];
  window.__sendAttempts = 0;
  window.__failPuts = config.failPuts ?? 0;
  const sink = {
    send: async (bytes, descriptor) => {
      window.__sendAttempts += 1;
      if (window.__failPuts > 0) {
        window.__failPuts -= 1;
        throw new Error("sink down");
      }
      window.__chunks.push({ bytes: Array.from(bytes), descriptor });
    },
    beacon: (bytes, descriptor) => {
      window.__beacons.push({ bytes: Array.from(bytes), descriptor });
      return true;
    },
  };
  window.__recorder = await LocusRecorder.start({ sink, ...config.start });
  return window.__recorder !== null;
}
"""

FLUSH_JS = """
async () => {
  while (true) {
    const before = window.__chunks.length;
    await window.__recorder.flush();
    if (window.__chunks.length === before) return window.__chunks;
  }
}
"""

HIDE_JS = """
() => {
  Object.defineProperty(document, "visibilityState",
                        { value: "hidden", configurable: true });
  document.dispatchEvent(new Event("visibilitychange"));
}
"""

IDB_COUNTS_JS = """
async () => {
  const db = await new Promise((resolve, reject) => {
    const open = indexedDB.open("locus-recorder");
    open.onsuccess = () => resolve(open.result);
    open.onerror = () => reject(open.error);
  });
  const count = (store) => new Promise((resolve, reject) => {
    const request = db.transaction(store).objectStore(store).count();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  return { outbox: await count("outbox"), inflight: await count("inflight") };
}
"""


@pytest.fixture(scope="session")
def site(tmp_path_factory, recorder_dist):
    # The subject is the library build — the artifact deployments vend — driven
    # through its real API so the test can inject a collecting sink. (The
    # autostart bundle wires its own HTTP sink from its script tag and exposes
    # no start(), so it cannot be driven this way.)
    root = tmp_path_factory.mktemp("site")
    (root / "page_a.html").write_text(PAGE_A)
    (root / "page_b.html").write_text(PAGE_B)
    shutil.copy(recorder_dist / "locus-recorder.lib.js", root / "locus-recorder.lib.js")
    server, origin = serve_dir(root)
    yield origin
    server.shutdown()


@pytest.fixture()
def page():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser.new_page()
        browser.close()


def start_recorder(page, failPuts=0, **options):
    page.wait_for_function("typeof LocusRecorder !== 'undefined'")
    options.setdefault("recordLocalEnvironment", True)
    options.setdefault("botDetection", False)
    options.setdefault("intervalMs", 60_000)
    return page.evaluate(START_JS, {"start": options, "failPuts": failPuts})


def decode(chunk):
    return json.loads(gzip.decompress(bytes(chunk["bytes"])))


def stream_of(payloads):
    records = [
        (payload["sliceId"], event)
        for payload in payloads
        for event in payload["events"]
    ]
    return sorted(records, key=lambda record: record[1]["counter"])


def test_capture_then_evidence_round_trip(page, site, tmp_path):
    page.goto(f"{site}/page_a.html")
    page.click("#go")
    assert start_recorder(page)
    page.click("#buy")
    page.fill("#name", "chris")
    page.fill("#pw", "s3cret-hunter2")
    payloads = [decode(chunk) for chunk in page.evaluate(FLUSH_JS)]

    page_b = f"{site}/page_b.html"
    for payload in payloads:
        # The stamp's presence and producer prefix are the contract; the number is the
        # recorder's own and moves with its releases.
        assert payload["recorderVersion"].startswith("locus-recorder/")
        assert "HeadlessChrome" in payload["envelope"]["userAgent"]
        blob = json.dumps(payload)
        assert "s3cret-hunter2" not in blob, "passwords are always masked"
        assert "chris" in blob, (
            "recordings are verbatim except passwords — typed content is analysis signal"
        )

    records = stream_of(payloads)
    slice_ids = {slice_id for slice_id, _ in records}
    assert len(slice_ids) == 1
    events = [event for _, event in records]
    assert events[0]["type"] == EventType.Meta
    assert events[0]["data"]["href"] == page_b
    assert events[1]["type"] == EventType.FullSnapshot
    assert events[2]["type"] == EventType.PageLoad
    assert events[2]["data"] == {
        "url": page_b,
        "title": "Demo Shop",
        "referrer": f"{site}/page_a.html",
    }
    assert any(event["type"] == EventType.IncrementalSnapshot for event in events)
    sequences = [int(event["counter"][-6:]) for event in events]
    assert sequences == list(range(1, len(events) + 1))

    visitor = payloads[0]["visitorId"]
    conn = connect(tmp_path / "events.db")
    for payload in payloads:
        assert payload["visitorId"] == visitor
        for event in payload["events"]:
            event["_envelope"] = {
                "script_version": payload["recorderVersion"],
                "recorder_slice": payload["sliceId"],
            }
        hydrate(conn, visitor, payload["events"])
    materialize_slices(conn, visitor)
    subprocess.run(
        ["bun", str(DISTILL), str(tmp_path / "events.db")],
        check=True,
        capture_output=True,
    )
    row = conn.execute("SELECT id, status FROM slices").fetchone()
    assert row["status"] == "replayable"
    assert arrival(conn, row["id"]) == ("page-load", page_b, f"{site}/page_a.html")


def test_checkout_opens_new_slice_without_new_pageload(page, site):
    page.goto(f"{site}/page_b.html")
    assert start_recorder(
        page, rrwebRules=[{"pattern": "*", "options": {"checkoutEveryNms": 300}}]
    )
    page.click("#buy")
    page.wait_for_timeout(400)
    page.click("#buy")
    payloads = [decode(chunk) for chunk in page.evaluate(FLUSH_JS)]

    records = stream_of(payloads)
    slice_ids = [slice_id for slice_id, _ in records]
    assert len(set(slice_ids)) == 2
    second = [event for slice_id, event in records if slice_id == slice_ids[-1]]
    assert second[0]["type"] == EventType.Meta
    assert second[1]["type"] == EventType.FullSnapshot
    page_loads = [event for _, event in records if event["type"] == EventType.PageLoad]
    assert len(page_loads) == 1


def test_hidden_marker_rides_the_beacon_alone(page, site):
    page.goto(f"{site}/page_b.html")
    assert start_recorder(page)
    page.evaluate(HIDE_JS)

    beacons = page.evaluate("() => window.__beacons")
    assert len(beacons) == 1
    payload = decode(beacons[0])
    [event] = payload["events"]
    assert event["type"] == EventType.PageHidden
    assert event["data"] == {"url": f"{site}/page_b.html", "referrer": ""}
    assert payload["errors"] == []

    payloads = [decode(chunk) for chunk in page.evaluate(FLUSH_JS)]
    assert any(
        event["type"] == EventType.PageHidden for _, event in stream_of(payloads)
    )


def test_local_environment_gate(page, site):
    page.goto(f"{site}/page_b.html")
    page.wait_for_function("typeof LocusRecorder !== 'undefined'")
    result = page.evaluate(
        "async () => await LocusRecorder.start({ sink: { send: async () => {} } })"
    )
    assert result is None


def test_botd_gates_automated_browsers(page, site):
    page.goto(f"{site}/page_b.html")
    result = start_recorder(page, botDetection=True)
    assert result is False


def test_failed_send_retries_the_same_batch(page, site):
    page.goto(f"{site}/page_b.html")
    assert start_recorder(page, intervalMs=100, failPuts=2)
    page.wait_for_function("window.__chunks.length > 0", timeout=10_000)

    assert page.evaluate("() => window.__sendAttempts") >= 3
    chunks = page.evaluate("() => window.__chunks")
    types = [event["type"] for event in decode(chunks[0])["events"]]
    assert types[:3] == [EventType.Meta, EventType.FullSnapshot, EventType.PageLoad]


def test_failed_uploads_never_self_terminate(page, site):
    """A sink that fails every send must never self-terminate the recorder.
    Delivery failures do not count toward errorThreshold — only internal
    host-harm errors do — and this sink's rejection carries no refusal from the
    store, so the batch is an outage's, retried at the cadence for as long as it
    lasts, with the recorder recording throughout."""
    page.goto(f"{site}/page_b.html")
    assert start_recorder(page, intervalMs=50, errorThreshold=2, failPuts=1_000_000)

    # attempt after attempt, well past errorThreshold=2 — never stopping at it
    page.wait_for_function("window.__sendAttempts >= 5", timeout=10_000)

    # still alive after a long stretch of failures: fresh activity keeps being
    # captured and attempted, which a terminated recorder never would
    before = page.evaluate("() => window.__sendAttempts")
    page.click("#buy")
    page.fill("#name", "still-here")
    page.wait_for_function(f"window.__sendAttempts > {before}", timeout=10_000)
    assert page.evaluate(IDB_COUNTS_JS)["inflight"] >= 1, (
        "the undelivered batch is preserved while it retries"
    )
