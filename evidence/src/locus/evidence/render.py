"""Local screenshot renderer — a slice's events to PNGs via a painting browser.

The rendered screenshots ARE the visual evidence the model grounds on, so a silently-unfaithful screenshot is the worst failure mode: it feeds the model evidence that doesn't match what the visitor saw, with no error surfaced. Two fidelity requirements follow:
- Render in a PAINTING browser. The native ::selection highlight only paints in a painting context (https://www.w3.org/TR/css-pseudo-4/#highlight-painting), so a non-painting context silently drops every text selection from the screenshot. Chromium launches with --headless=new — the real Chrome browser — which paints without a display; truly-headed also works but needs one.
- Replay with pauseAnimation off, and finish every finite animation before the screenshot. A reveal animation frozen at its start — content captured at opacity:0 with a reveal transition — renders blank; finished, it renders its revealed end state. A repeating animation, a spinner, has no end state and keeps running. (pauseAnimation injects animation-play-state: paused via the rrweb-paused class: rrweb packages/rrweb/src/replay/index.ts.)

Opens the slice in the vendored rrweb replayer at each requested timestamp and screenshots. rrweb inlines stylesheets but not images or fonts (rrweb-snapshot defaults inlineStylesheet true, inlineImages false: rrweb packages/rrweb-snapshot/src/snapshot.ts), so those fetch live at replay; each capture builds the snapshot page and waits on its fonts and images before seeking, then waits again on what the moment added (image decode, fonts, a capped settle), so a slow connection costs latency, not blank screenshots or short-landed scrolls. What waiting can't recover — resources 404'd or changed since recording — rides on the screenshot as its faults, failed resources named by URL, never silently captured over.

Assets behind session cookies or referer checks 403 at replay and surface as failed-image URLs in the screenshot's faults; record-time inlining is the only true fix, shared with any renderer.
"""

import json
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from .assets import rrweb_constants_js, vendored
from .rrweb_constants import EventType, IncrementalSource, MouseInteractions
from .slices import covering_pair, discard_reason, slice_events

READINESS_BUDGET_MS = 10_000
# A follow-on request — a stylesheet's font, an image a script inserts — is issued within a
# frame of the response that triggered it being parsed, so a chain shows itself as a new
# request inside a frame of the last completion; two frames of grace after the last completion
# catches a chain, and anything that takes longer to start is not a chain.
FRAME_MS = 1000 / 60
NETWORK_GRACE_MS = 2 * FRAME_MS
NETWORK_BACKSTOP_MS = 5_000
INJECT_BATCH_BYTES = 15_000_000

# Fidelity flags for replaying a recording out of a local file: the harness page's origin is opaque, so
# every cross-origin webfont fetch would fail CORS and silently render a fallback face (fonts.ready cannot
# flag a failed font), and scrollbar chrome is a render-environment artifact, not page content. They belong
# to that situation — a page fetched from its own live origin needs neither and must not be given them.
FIDELITY_ARGS = ["--disable-web-security", "--hide-scrollbars"]


def launch_chromium(playwright):
    """A PAINTING Chromium (see the fidelity note above — `--headless=new` is the real browser, which paints without a display; Playwright's own `headless=True` is the non-painting one, so it stays False and the flag carries the mode)."""
    return playwright.chromium.launch(
        headless=False, args=["--headless=new", *FIDELITY_ARGS]
    )


HARNESS = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <style>
      body, html {{ margin: 0; padding: 0; }}
      .replayer-mouse.mouse-down::after {{
        background: #b30713 !important;
        border: 2px solid #8a050f !important;
      }}
    </style>
    <script src="{replay_js}"></script>
    <script>{constants_js}</script>
    <script src="{seek_js}"></script>
    <link rel="stylesheet" href="{replay_css}">
  </head>
  <body><div id="replayer"></div></body>
</html>
"""

WAIT_READY_JS = """
async (budgetMs) => {
  const iframe = document.querySelector('#replayer iframe');
  const doc = iframe && iframe.contentDocument;
  // No replayer document means there is nothing to be ready. Reporting a clean wait here would be
  // a clean bill of health for a page that does not exist — and the caller screenshots on it,
  // producing a blank screenshot the model then grounds on.
  if (!doc) {
    throw new Error(
      'locus: the replayer has no document — nothing was rendered, so there is nothing to '
      + 'capture and no readiness to await');
  }

  const faults = {};
  const deadline = Date.now() + budgetMs;
  const remaining = () => Math.max(0, deadline - Date.now());
  const withDeadline = (promise) => Promise.race([
    promise,
    new Promise((resolve) => setTimeout(() => resolve('timeout'), remaining())),
  ]);

  if (doc.fonts) {
    if (window.__fontsGaveUp) {
      // The wait gave up permanently, but the fault is only real while the fonts
      // still aren't in: a screenshot rendered after they finally arrived is clean.
      if (doc.fonts.status !== 'loaded') faults.fontsTimedOut = true;
    } else if ((await withDeadline(doc.fonts.ready)) === 'timeout') {
      faults.fontsTimedOut = true;
      window.__fontsGaveUp = true;
    }
  }

  // Only network-fetched resources can render differently than the visitor saw them — a fault
  // is a live fetch failing (403, 404, changed since recording). An img with no real network
  // source (a lazy-load stub, a data: URI) fails or succeeds identically for visitor and
  // replay: recording truth, never a fault.
  const images = Array.from(doc.querySelectorAll('img'))
    .map((img) => [img, img.currentSrc || img.src])
    .filter(([, url]) => /^https?:/.test(url));
  const results = await Promise.all(images.map(([img, url]) =>
    withDeadline(img.decode().then(() => 'ok', () => 'failed'))
      .then((state) => [state, url])));
  const failed = results.filter(([s]) => s === 'failed').map(([, url]) => url);
  const pending = results.filter(([s]) => s === 'timeout').map(([, url]) => url);
  if (failed.length) faults.failedImages = failed;
  if (pending.length) faults.pendingImages = pending;

  LocusSeek.finishAnimations(doc);

  await new Promise((resolve) => requestAnimationFrame(resolve));
  return faults;
}
"""


def _await_network_quiet(page, inflight: dict, done: dict) -> bool:
    """Backstop for fetches img.decode() can't see (CSS backgrounds, XHR): wait until no *recent* request is in flight and NETWORK_GRACE_MS have passed since the last completion, capped at NETWORK_BACKSTOP_MS. A request older than the backstop is treated as hung and excluded, so one dead resource taxes one screenshot, not every screenshot. Returns whether quiet was reached."""
    import time

    deadline = time.time() + NETWORK_BACKSTOP_MS / 1000
    while time.time() < deadline:
        now = time.time()
        recent = sum(
            1
            for started in inflight.values()
            if now - started < NETWORK_BACKSTOP_MS / 1000
        )
        if recent == 0 and now - done.get("at", 0) >= NETWORK_GRACE_MS / 1000:
            return True
        page.wait_for_timeout(NETWORK_GRACE_MS)
    return False


def _mouse_pressed_at(events: list[dict], timestamps: list[int]) -> dict[int, bool]:
    """Mouse-button state at each capture timestamp."""
    transitions = [
        (e["timestamp"], e["data"]["type"] == MouseInteractions.MouseDown)
        for e in events
        if e["type"] == EventType.IncrementalSnapshot
        and e.get("data", {}).get("source") == IncrementalSource.MouseInteraction
        and e["data"].get("type")
        in (MouseInteractions.MouseUp, MouseInteractions.MouseDown)
    ]
    states = {}
    for ts in timestamps:
        pressed = False
        for t, value in transitions:
            if t > ts:
                break
            pressed = value
        states[ts] = pressed
    return states


@dataclass
class ScreenshotResult:
    timestamp: int
    path: str
    # What the readiness wait could not deliver, failed resources named by URL;
    # empty when the screenshot rendered clean.
    faults: dict


@dataclass
class _CoveredSlice:
    slice_id: int
    events: list[dict]
    start_ts: int
    snapshot_ts: int
    viewport: dict


def load_slice(
    page,
    harness_uri: str,
    events: list[dict],
    slice_id: int,
    facts: dict,
) -> None:
    """Navigate the page and inject the slice's events, leaving them live as window.__events
    beside the slice's seek facts (`facts`: {id, start_ts, end_ts, snapshot_ts}) as
    window.__slice, which every capture reads through the shared seek module (replay_seek.js)."""
    page.goto(harness_uri)
    page.wait_for_function(
        "typeof rrwebReplay !== 'undefined' && typeof LocusSeek !== 'undefined'"
    )
    page.evaluate("(facts) => { window.__slice = facts; }", facts)

    page.evaluate("() => { window.__events = []; }")
    batch: list[dict] = []
    batch_bytes = 0
    for event in [*events, None]:
        if event is not None:
            batch.append(event)
            batch_bytes += len(json.dumps(event))
        if batch and (event is None or batch_bytes >= INJECT_BATCH_BYTES):
            page.evaluate(
                "(batch) => { for (const e of batch) window.__events.push(e); }", batch
            )
            batch, batch_bytes = [], 0
    injected = page.evaluate("() => window.__events.length")
    if injected != len(events):
        raise RuntimeError(
            f"slice {slice_id}: injected {injected} of {len(events)} events — refusing to replay a partial slice"
        )


# A capture is a fresh replayer opened at its moment, over its own copy of the events: rrweb
# carries state across seeks on one replayer and writes into the events it is given. Three
# steps: construct with the renderer's exact config, which builds the snapshot page; wait for
# that page's fonts and images; then pause at the moment, so the recorded scrolls land on a
# laid-out page.
BUILD_JS = """async () => {
    if (window.replayer instanceof rrwebReplay.Replayer) window.replayer.destroy();
    window.replayer = new rrwebReplay.Replayer(structuredClone(window.__events), {
        root: document.getElementById('replayer'),
        mouseTail: false,
        skipInactive: false,
        showWarning: false,
        pauseAnimation: false,
        useVirtualDom: true,
        insertStyleRules: LocusSeek.STYLE_RULES,
    });
    await new Promise((built) => window.replayer.on('fullsnapshot-rebuilded', built));
}"""

PAUSE_AT_JS = (
    "(ts) => window.replayer.pause(LocusSeek.offsetOf(ts, window.__slice.start_ts))"
)


class RenderSession:
    """Per-slice isolated replayers over one window — the render service the engine calls.

    A window is a slice table: label → slice (analysis's window.slice_table mints it), any time relation
    between slices legal — concurrent tabs and overlapping visitors included, which is why every
    capture is addressed by its slice's label, never routed by timestamp. Each slice is self-covering
    (its own Meta+FullSnapshot), so it is replayed ALONE, never concatenated: feeding several slices
    into one replayer would ask rrweb to reset DOM, viewport, base href, adopted stylesheets, and the
    node-id mirror cleanly across page boundaries, and a screenshot captured through that bleed is silently
    wrong evidence.

    The browser is a service opened on demand: the first capture launches it, captures arriving
    slice by slice reuse the loaded events, a capture landing on an earlier slice reloads them —
    correctness never depends on capture order — and release() closes it again the moment a
    caller is done capturing, so a session that goes on to wait on something else holds no
    Chromium while it waits. The next capture launches it again.

    Self-covering is containment, not position: slices.py owns what replayable means, and a slice
    opens when its page context does — before rrweb has a DOM to capture — so events precede the
    covering Meta+FullSnapshot inside the slice they belong to. A slice slices.py would discard is
    refused here too, by its reason, and the whole window with it — the replayability reject rule at
    the render boundary."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        slices: dict[str, int],
        out_dir: str | Path,
    ):
        if not slices:
            raise ValueError("a render window needs at least one slice")
        self.slices: dict[str, _CoveredSlice] = {}
        for label, slice_id in slices.items():
            events = slice_events(conn, slice_id)
            if not events:
                raise ValueError(f"slice {slice_id} has no events")
            reason = discard_reason(events)
            if reason is not None:
                raise ValueError(
                    f"slice {slice_id} is not replayable — {reason}; refused (replayability reject rule)"
                )
            pair = covering_pair(events)
            meta = pair[0]["data"]
            self.slices[label] = _CoveredSlice(
                slice_id=slice_id,
                events=events,
                # The replay clock's origin is the slice's first event, whatever it is —
                # rrweb measures every offset from its own events[0], so a slice that opened
                # before its snapshot would be seeked short by that lead if the Meta anchored it.
                start_ts=events[0]["timestamp"],
                snapshot_ts=pair[1]["timestamp"],
                viewport={"width": meta["width"], "height": meta["height"]},
            )
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: str | None = None
        self._viewport: dict | None = None
        self._inflight: dict = {}
        self._done: dict = {}
        self._pw = self._browser = self.page = self._harness_path = None

    def __enter__(self) -> Self:
        harness = HARNESS.format(
            replay_js=vendored("rrweb-replay-*.min.js").as_uri(),
            constants_js=rrweb_constants_js(),
            seek_js=Path(__file__).with_name("replay_seek.js").as_uri(),
            replay_css=vendored("rrweb-replay-*.min.css").as_uri(),
        )
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
            f.write(harness)
            self._harness_path = f.name
        return self

    def _ensure_browser(self) -> None:
        if self._browser is not None:
            return
        import time

        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = launch_chromium(self._pw)
        self._viewport = dict(next(iter(self.slices.values())).viewport)
        self.page = self._browser.new_page(viewport=self._viewport)
        self.page.on("request", lambda r: self._inflight.__setitem__(r, time.time()))
        for settled in ("requestfinished", "requestfailed"):
            self.page.on(settled, self._settled)

    def _settled(self, request) -> None:
        import time

        self._inflight.pop(request, None)
        self._done["at"] = time.time()

    def release(self) -> None:
        """Close the browser; the session's slices stay addressable and the next capture launches it again."""
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._pw is not None:
            self._pw.stop()
            self._pw = None
        self.page = None
        self._loaded = None
        self._inflight.clear()

    def _ensure_loaded(self, label: str) -> None:
        """Make the captured slice's events the live ones — exactly that slice, never a
        concatenation. In slice-by-slice capture order this fires once per slice; a capture that
        lands on an already-passed slice reloads it."""
        if self._loaded == label:
            return
        sl = self.slices[label]
        if sl.viewport != self._viewport:
            self.page.set_viewport_size(sl.viewport)
            self._viewport = dict(sl.viewport)
        self._inflight.clear()
        load_slice(
            self.page,
            Path(self._harness_path).as_uri(),
            sl.events,
            sl.slice_id,
            {
                "id": sl.slice_id,
                "start_ts": sl.start_ts,
                "end_ts": sl.events[-1]["timestamp"],
                "snapshot_ts": sl.snapshot_ts,
            },
        )
        self._loaded = label

    def capture(self, label: str, ts: int) -> ScreenshotResult:
        """Capture the slice labeled `label` at absolute ms `ts` — the DOM after every event stamped at or before it. The label is the address — a timestamp alone routes nowhere, because two slices may cover the same instant. The screenshot file's name is its identity: {label}_screenshot_{ts}.png, the vocabulary the window's citations use. A moment the slice holds no page at — outside its recording, or in its lead ahead of the covering snapshot — is refused: a file stamped with a time the pixels never held would read as evidence of it."""
        if label not in self.slices:
            raise ValueError(
                f"unknown label {label!r} — this window renders {sorted(self.slices)}"
            )
        self._ensure_browser()
        self._ensure_loaded(label)
        sl = self.slices[label]
        page = self.page
        gone = page.evaluate("(ts) => LocusSeek.unshowable([window.__slice], ts)", ts)
        if gone is not None:
            raise ValueError(f"{label} cannot be captured at {ts} — {gone}")
        page.evaluate(BUILD_JS)
        page.evaluate(WAIT_READY_JS, READINESS_BUDGET_MS)
        page.evaluate(PAUSE_AT_JS, ts)
        dims = page.evaluate(
            """() => {
                const iframeEl = document.querySelector('#replayer iframe');
                if (!iframeEl) return null;
                const width = Number(iframeEl.width) || iframeEl.clientWidth;
                const height = Number(iframeEl.height) || iframeEl.clientHeight;
                return width && height ? {width, height} : null;
            }"""
        )
        if dims and dims != self._viewport:
            page.set_viewport_size(dims)
            self._viewport = dims
        page.evaluate(
            """(pressed) => {
                const cursor = document.querySelector('.replayer-mouse');
                if (cursor) cursor.classList.toggle('mouse-down', pressed);
            }""",
            _mouse_pressed_at(sl.events, [ts])[ts],
        )
        faults = page.evaluate(WAIT_READY_JS, READINESS_BUDGET_MS)
        if not _await_network_quiet(page, self._inflight, self._done):
            faults["networkBusy"] = True
        screenshot_path = self.out_dir / f"{label}_screenshot_{ts}.png"
        page.screenshot(path=str(screenshot_path))
        return ScreenshotResult(timestamp=ts, path=str(screenshot_path), faults=faults)

    def __exit__(self, *exc) -> None:
        self.release()
        if self._harness_path is not None:
            Path(self._harness_path).unlink(missing_ok=True)
