"""Regenerate demo_recording.json — the synthetic test fixture.

The fixture is a real recorder capture of a synthetic site (tests/fixtures/site, the Meridian Coffee demo): the actual recorder bundle drives the actual wire path (stamped events → gzipped chunks → HTTP PUT sink), so the fixture carries everything the recorder really emits — Meta/FullSnapshot slice openings, synthetic PageLoad/PageHidden markers, counters, the envelope — with no real visitor's data in it. No part of it is hand-written.

Run from evidence/: uv run python tests/fixtures/generate.py

Regeneration is not idempotent (timestamps, visitor id, and event ids change every run), so after regenerating, every count-asserting test and the distillation projection expectations must be re-derived against the new artifact. The committed fixture is the frozen reference; this script exists so it can be reproduced and evolved, not so it churns.
"""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from _harness import chromium_page, serve_dir

FIXTURES = Path(__file__).parent
SITE = FIXTURES / "site"
BUNDLE = FIXTURES.parent.parent.parent / "recorder" / "dist" / "locus-recorder.min.js"
OUT = FIXTURES / "demo_recording.json"

SNIPPET = """(function () {
  var s = document.createElement("script");
  s.async = true;
  s.src = "/locus-recorder.min.js";
  s.onload = function () {
    LocusRecorder.start({
      sink: LocusRecorder.httpSink({
        url: function (d) {
          return "/chunks/" + d.visitorId + "/" + d.sliceId + "/" + d.chunkKey;
        },
      }),
      recordLocalEnvironment: true,
      botDetection: false,
      intervalMs: 400,
    }).then(function (recorder) {
      window.__locusRecorder = recorder;
    });
  };
  document.head.appendChild(s);
})();
"""


def drive(origin: str) -> None:
    with chromium_page() as page:
        page.goto(f"{origin}/index.html")
        page.wait_for_function("() => window.__locusRecorder")
        page.wait_for_timeout(1500)
        page.mouse.move(400, 300)
        page.mouse.wheel(0, 900)
        page.wait_for_timeout(900)
        page.mouse.move(640, 500)
        page.mouse.wheel(0, 1200)
        page.wait_for_timeout(900)
        page.mouse.wheel(0, 1500)
        page.wait_for_timeout(900)
        page.mouse.wheel(0, -1600)
        page.wait_for_timeout(900)
        page.mouse.move(300, 420)
        page.wait_for_timeout(900)

        page.click("#view-1")
        page.wait_for_function("() => window.__locusRecorder")
        page.wait_for_timeout(1500)
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(900)
        page.fill("#qty", "2")
        page.click("#notes-toggle")
        page.fill("#notes", "medium-fine for AeroPress")
        page.click("#add")
        page.wait_for_timeout(2200)
        page.fill("#qty", "3")
        page.click("#add")
        page.wait_for_timeout(2200)
        page.mouse.wheel(0, 600)
        page.wait_for_timeout(900)

        page.goto(f"{origin}/index.html")
        page.wait_for_function("() => window.__locusRecorder")
        page.wait_for_timeout(1500)
        page.mouse.wheel(0, 2400)
        page.wait_for_timeout(900)
        page.mouse.wheel(0, 1200)
        page.wait_for_timeout(900)
        page.mouse.move(900, 360)
        page.wait_for_timeout(900)

        page.click("#view-4")
        page.wait_for_function("() => window.__locusRecorder")
        page.wait_for_timeout(1500)
        page.mouse.wheel(0, 400)
        page.wait_for_timeout(900)
        page.mouse.wheel(0, 700)
        page.wait_for_timeout(900)
        page.mouse.wheel(0, -500)
        page.wait_for_timeout(2500)
        for _ in range(5):
            page.evaluate("() => window.__locusRecorder.flush()")
            page.wait_for_timeout(300)


def main() -> None:
    chunks: dict[str, bytes] = {}
    server, origin = serve_dir(
        SITE,
        routes={
            "/snippet.js": (SNIPPET.encode(), "text/javascript"),
            "/locus-recorder.min.js": (BUNDLE.read_bytes(), "text/javascript"),
        },
        on_put=chunks.__setitem__,
    )
    try:
        drive(origin)
    finally:
        server.shutdown()

    visitors = {path.split("/")[2] for path in chunks}
    assert len(visitors) == 1, f"one synthetic visitor expected: {visitors}"
    events = []
    for path in sorted(chunks):
        payload = json.loads(gzip.decompress(chunks[path]))
        events.extend(payload["events"])
    events.sort(key=lambda event: (event["timestamp"], event.get("counter", "")))

    OUT.write_text(json.dumps({"events": events}, indent=1))
    slices = {path.split("/")[3] for path in chunks}
    print(
        f"{OUT.name}: {len(events)} events, "
        f"{len(chunks)} chunks, {len(slices)} slices, "
        f"{OUT.stat().st_size / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    main()
