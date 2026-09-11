"""The deployment's browser — the daemon that drives the windows.

One persistent Chromium (Locus's own Playwright build, its own profile under `data/browse/profile`, so logins survive across commands), headed so the operator can watch the same windows the agent drives (`LOCUS_BROWSE_HEADLESS=1` at start is the test suite's hook for running it headless on a display-less runner). The browser is its own process: the daemon starts it detached on a debugging port and connects over CDP, so the browser and every window in it outlive the daemon. The daemon serves one command at a time over a unix socket; the `locus browse` client boots it on demand — an exclusive lock under the home admits one daemon per home, so racing first commands converge on a single boot — and `quit` shuts every window, the browser, and the daemon down together. The daemon runs the code it booted with, and every command carries the digest of the driver on disk: a mismatch is answered by the daemon shutting down, the client booting the code on disk under that same command, and the new daemon reconnecting to the same browser with every window still there under its same id — the window table is kept beside the profile, and a page's refs live in the page.

Instances are windows: each is its own OS window, addressed `w1`, `w2`, …, and every command names the window it acts on. There is no shared active window, so any number of agents drive their own windows through the one daemon without colliding, and an agent building in one window never disturbs a window the operator is looking at. `open` with no id makes a window for the page; when some window already holds that page, it refuses and names that window, because which window an agent means is the agent's to say. `show` is the only command that brings one to the front. `read` snapshots a window's page — ref-tagged interactables (`e1`, `e2`, …) plus visible text — and refs stay resolvable until that page navigates; acting on a stale ref fails loud and says to read again. The snapshot walk pierces open shadow roots; closed shadow content and cross-origin frames are not in it.

Every reply opens with the window id it acted on, and an action that navigated says where it landed — the agent should never have to guess which page it is now talking to.
"""

import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# The snapshot walk's shadow-piercing rule — what "the whole page" means to every command
# that scans it. Spliced into each snippet (__WALK__) so read and find can never disagree
# about what is visible through a shadow root.
WALK_JS = """function* walk(root) {
    for (const el of root.querySelectorAll('*')) {
      yield el;
      if (el.shadowRoot) yield* walk(el.shadowRoot);
    }
  }"""

# Where an element's click point is: scrolled to center first, then the rect's center.
# Spliced as __CENTER__; leaves cx/cy in scope for the snippet's own return.
CENTER_JS = """el.scrollIntoView({ block: 'center', inline: 'center' });
  const r = el.getBoundingClientRect();
  const cx = Math.round(r.left + r.width / 2);
  const cy = Math.round(r.top + r.height / 2);"""


def _splice(js: str) -> str:
    return js.replace("__WALK__", WALK_JS).replace("__CENTER__", CENTER_JS)


READ_JS = _splice("""() => {
  const refs = [];
  __WALK__
  const interactable = (el) => {
    const tag = el.tagName;
    if (['A', 'BUTTON', 'SELECT', 'TEXTAREA'].includes(tag)) return true;
    if (tag === 'INPUT') return el.type !== 'hidden';
    const role = el.getAttribute('role');
    if (['button', 'link', 'tab', 'menuitem', 'checkbox', 'radio', 'combobox',
         'option', 'slider', 'switch'].includes(role)) return true;
    if (el.hasAttribute('onclick') || el.isContentEditable) return true;
    return el.tabIndex >= 0 && !['BODY', 'HTML'].includes(tag);
  };
  const labelOf = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria;
    if (el.id) {
      const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (l) return l.innerText;
    }
    const wrap = el.closest && el.closest('label');
    if (wrap) return wrap.innerText;
    return el.innerText || el.value || el.placeholder || el.title
      || el.getAttribute('name') || '';
  };
  const lines = [];
  let below = 0, above = 0;
  const vw = innerWidth, vh = innerHeight;
  const seen = new Set();
  for (const el of walk(document)) {
    if (!interactable(el)) continue;
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) continue;
    const cx = Math.round(r.left + r.width / 2);
    const cy = Math.round(r.top + r.height / 2);
    let tag = el.tagName.toLowerCase();
    if (tag === 'input' && el.type) tag += `[${el.type}]`;
    const label = String(labelOf(el)).replace(/\\s+/g, ' ').trim().slice(0, 80);
    const key = `${tag}|${label}|${cx},${cy}`;
    if (seen.has(key)) continue;
    seen.add(key);
    if (cy < 0) { above++; continue; }
    if (cy > vh || cx < 0 || cx > vw) { below++; continue; }
    refs.push(el);
    lines.push(`e${refs.length}`.padEnd(5) + tag.padEnd(16) + ' '
      + JSON.stringify(label) + `  (${cx},${cy})`);
  }
  window.__locusRefs = refs;
  const text = (document.body ? document.body.innerText : '')
    .replace(/[ \\t]+/g, ' ');
  return { url: location.href, title: document.title, lines, below, above,
           text };
}""")

REF_CENTER_JS = _splice("""(i) => {
  const el = window.__locusRefs && window.__locusRefs[i];
  if (!el || !el.isConnected) return null;
  __CENTER__
  return { x: cx, y: cy,
           desc: el.tagName.toLowerCase()
             + ' ' + JSON.stringify((el.innerText || el.value || '').trim().slice(0, 40)) };
}""")

FIND_BY_TEXT_JS = _splice("""(q) => {
  __WALK__
  const ok = (el) => ['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName)
    || ['button', 'link', 'tab', 'menuitem', 'checkbox', 'radio'].includes(el.getAttribute('role'))
    || el.hasAttribute('onclick') || (el.tabIndex != null && el.tabIndex >= 0);
  const text = (el) => (el.innerText || el.value || el.getAttribute('aria-label')
    || el.getAttribute('title') || '').replace(/\\s+/g, ' ').trim();
  let exact = null, partial = null;
  const ql = q.toLowerCase();
  for (const el of walk(document)) {
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) continue;
    if (!ok(el)) continue;
    const t = text(el);
    if (!t) continue;
    if (t === q) { exact = el; break; }
    if (!partial && t.toLowerCase().includes(ql)) partial = el;
  }
  const el = exact || partial;
  if (!el) return null;
  __CENTER__
  return { x: cx, y: cy,
           desc: el.tagName.toLowerCase() + ' ' + JSON.stringify(text(el).slice(0, 40)) };
}""")

FIND_BY_CSS_JS = _splice("""(sel) => {
  let el;
  try { el = document.querySelector(sel); }
  catch (e) { return { err: 'bad selector: ' + e.message }; }
  if (!el) return null;
  __CENTER__
  if (r.width <= 0 || r.height <= 0)
    return { err: 'matched ' + sel + ' but it has zero size / is hidden' };
  return { x: cx, y: cy,
           desc: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') };
}""")

FOCUS_JS = """(spec) => {
  let el;
  if (spec.ref != null) {
    el = window.__locusRefs && window.__locusRefs[spec.ref];
    if (!el || !el.isConnected) return null;
  } else if (spec.css != null) {
    try { el = document.querySelector(spec.css); }
    catch (e) { return { err: 'bad selector: ' + e.message }; }
    if (!el) return null;
  } else {
    el = document.activeElement;
    if (!el || el === document.body) return { err: 'nothing is focused — name a ref or --css' };
  }
  el.scrollIntoView({ block: 'center', inline: 'center' });
  el.focus && el.focus();
  if (typeof el.select === 'function') el.select();
  else if (el.isContentEditable) {
    const r = document.createRange();
    r.selectNodeContents(el);
    const s = getSelection();
    s.removeAllRanges();
    s.addRange(r);
  }
  return { desc: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') };
}"""

SELECT_JS = """(spec) => {
  let el;
  if (spec.ref != null) {
    el = window.__locusRefs && window.__locusRefs[spec.ref];
    if (!el || !el.isConnected) return null;
  } else {
    try { el = document.querySelector(spec.css); }
    catch (e) { return { err: 'bad selector: ' + e.message }; }
    if (!el) return null;
  }
  if (el.tagName !== 'SELECT')
    return { err: 'that element is not a <select>' };
  const opts = [].slice.call(el.options);
  let i = opts.findIndex((o) => o.value === spec.want || o.label === spec.want
    || o.text.trim() === spec.want);
  if (i < 0) i = opts.findIndex(
    (o) => o.text.trim().toLowerCase() === String(spec.want).toLowerCase());
  if (i < 0) return { err: 'no option matching ' + JSON.stringify(spec.want)
    + ' — options: ' + opts.map((o) => JSON.stringify(o.text.trim())).join(', ') };
  el.scrollIntoView({ block: 'center', inline: 'center' });
  el.selectedIndex = i;
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
  return { picked: opts[i].text.trim() };
}"""

MODS = {
    "ctrl": "Control",
    "control": "Control",
    "cmd": "Meta",
    "meta": "Meta",
    "alt": "Alt",
    "opt": "Alt",
    "shift": "Shift",
}


def chord(token: str) -> str:
    """`ctrl+Enter` → Playwright's `Control+Enter`; bare keys pass through."""
    parts = token.split("+")
    return "+".join([MODS.get(p.lower(), p) for p in parts[:-1]] + [parts[-1]])


def target_url(target: str) -> str:
    """A URL passes through; a path becomes a file URI, failing loud if it does not exist. The fragment survives (`page.html#t=...` is how a citation is opened), so the existence check strips it first."""
    from urllib.parse import urlparse

    if urlparse(target).scheme in ("http", "https", "file", "data", "about"):
        return target
    path_part, _, fragment = target.partition("#")
    path = Path(path_part)
    if not path.exists():
        raise FileNotFoundError(f"not a URL and no such file: {target}")
    return path.resolve().as_uri() + (f"#{fragment}" if fragment else "")


def _flag(rest: list[str], name: str) -> str | None:
    """Pop `--name value` out of rest, returning the value."""
    if name in rest:
        i = rest.index(name)
        if i + 1 >= len(rest):
            raise ValueError(f"{name} needs a value")
        value = rest[i + 1]
        del rest[i : i + 2]
        return value
    return None


# What the browser is started with: the switches Playwright's own launcher passes that
# matter to a browser started by hand. The keychain and password-store pair is the one
# that bites first on macOS — without it every start asks the operator for Keychain
# access in a modal dialog. The throttling trio keeps a window the operator has covered,
# or a headless one, running at full speed. --headless=new (added when headless) is the
# renderer's painting-headless trick (render.py): the real browser painting without a
# display. Only that trick is shared — the renderer's FIDELITY_ARGS stay off, by
# render.py's own rule: they belong to file:// replay harnesses, and these windows drive
# live origins.
# How long an open waits for a replay page's opening seek to settle before failing it:
# the wait is the snapshot page's fonts and images arriving, so it is bounded the way
# every other navigation wait in this driver is, at the browser's own default patience.
LANDING_MS = 30_000

CHROME_ARGS = [
    # No window at start: a used profile would otherwise come up restoring its last
    # session as placeholder tabs that never load until shown, and a connect that
    # attaches to them waits on them forever. The first `open` makes the first window.
    "--no-startup-window",
    "--use-mock-keychain",
    "--password-store=basic",
    "--no-first-run",
    "--no-default-browser-check",
    "--no-service-autorun",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-default-apps",
    "--disable-search-engine-choice-screen",
    "--disable-hang-monitor",
    "--disable-prompt-on-repost",
    "--disable-popup-blocking",
    "--disable-breakpad",
    "--metrics-recording-only",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    # Frames come off a timer, never the display's vsync: a sleeping display drives no
    # vsync, and a browser waiting on one paints nothing — every screenshot capture and
    # every animation frame then waits forever, on a machine left running unattended.
    "--disable-gpu-vsync",
    "--disable-frame-rate-limit",
]


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def _free_port() -> int:
    """A free loopback port, released for the browser to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _running(pid: int) -> bool:
    """Whether the process is alive. The daemon that started the browser is its parent, and a child that has exited stays a zombie that signal 0 still reaches until it is reaped — so a child is asked through waitpid, which reaps it, and only a process that is not ours is asked with the signal."""
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        return reaped == 0
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class Browser:
    """The Chromium process itself, outliving the daemon. Started detached on a debugging port with the profile under the home, its endpoint and pid written to `browser.json` beside the profile; a later daemon reads that file and connects to the same browser, so a daemon reboot loses no window. `quit` and the last window closing end it, verified by process death."""

    def __init__(self, playwright, home: Path):
        self._pw = playwright
        self.home = home
        self.record = home / "browser.json"

    def _alive(self) -> dict | None:
        if not self.record.exists():
            return None
        info = json.loads(self.record.read_text())
        if not _running(info["pid"]):
            return None
        return info if _http_ok(info["endpoint"] + "/json/version") else None

    def connect(self, headless: bool):
        """A connected Playwright browser over the running Chromium — started here if none is running — with whether it runs headless (decided when it started, whatever this daemon was told) and whether this call started it."""
        info = self._alive()
        fresh = info is None
        if fresh:
            info = self._start(headless)
        return (
            self._pw.chromium.connect_over_cdp(info["endpoint"], timeout=30000),
            info["headless"],
            fresh,
        )

    def _start(self, headless: bool) -> dict:
        profile = self.home / "profile"
        profile.mkdir(parents=True, exist_ok=True)
        # A second Chromium on a profile another still holds does not start; it puts up a
        # dialog and waits. The profile's own lock names its holder, so a live holder is
        # a refusal here, never a second launch.
        lock = profile / "SingletonLock"
        if lock.is_symlink():
            holder = os.readlink(lock).rpartition("-")[2]
            if holder.isdigit():
                if not _running(int(holder)):
                    lock.unlink(missing_ok=True)
                else:
                    raise RuntimeError(
                        f"a browser (pid {holder}) still holds {profile} but is not the "
                        f"one recorded in {self.record} — `locus browse quit` ends it"
                    )
        # A headed browser restores the last session's windows the moment its first
        # window is made — old windows nobody asked for, adopted as instances. The
        # restore reads the profile's saved sessions, so those are removed before every
        # start; logins and everything else in the profile stay.
        import shutil

        shutil.rmtree(profile / "Default" / "Sessions", ignore_errors=True)
        # A fixed port, chosen here: with `--remote-debugging-port=0` Chromium reports
        # `navigator.webdriver` true in every page and bot gates, the recorder's included,
        # refuse the window; under a fixed port the same browser reports false.
        port = _free_port()
        log = (self.home / "browser.log").open("a")
        args = [
            self._pw.chromium.executable_path,
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={port}",
            *CHROME_ARGS,
            *(["--headless=new", "--window-size=1280,800"] if headless else []),
        ]
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        info = {
            "pid": proc.pid,
            "endpoint": f"http://127.0.0.1:{port}",
            "headless": headless,
        }
        # Readiness is the endpoint answering: the socket opens before the server behind
        # it does, and a connect in that gap hangs on the half-open server.
        deadline = time.time() + 60
        while not _http_ok(info["endpoint"] + "/json/version"):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"the browser exited during start (code {proc.returncode}) — see {self.home / 'browser.log'}"
                )
            if time.time() > deadline:
                proc.kill()
                raise RuntimeError(
                    f"the browser never answered on its debugging port — see {self.home / 'browser.log'}"
                )
            time.sleep(0.1)
        self.record.write_text(json.dumps(info))
        return info

    def stop(self) -> None:
        """End the browser process, verified by its death: a close over CDP, then SIGTERM, then SIGKILL — a browser still holding its profile lock would refuse the next start."""
        info = self._alive() if self.record.exists() else None
        if info is None:
            self.record.unlink(missing_ok=True)
            return
        pid = info["pid"]
        try:
            browser = self._pw.chromium.connect_over_cdp(info["endpoint"])
            browser.new_browser_cdp_session().send("Browser.close")
        except Exception:  # noqa: BLE001, S110 — the signals below are the guarantee
            pass

        def dead_within(seconds: float) -> bool:
            deadline = time.time() + seconds
            while time.time() < deadline:
                if not _running(pid):
                    return True
                time.sleep(0.1)
            return False

        for sig, grace in ((None, 5), (signal.SIGTERM, 5), (signal.SIGKILL, 10)):
            if sig is not None:
                try:
                    os.kill(pid, sig)
                except OSError:
                    break
            if dead_within(grace):
                break
        self.record.unlink(missing_ok=True)


class Driver:
    """Instances are windows. Each is its own OS window, addressed w1, w2, …, and every command names the window it acts on — there is no shared active window for a second agent's command to move, so any number of agents drive their own windows through the one daemon without colliding. `open` with no id makes a window; when a window already holds that page it refuses and names the window. Every other verb takes an id. `show` is the only command that brings a window to the front, so an agent building in one window never steals the screen from a window the operator is looking at.

    Windows are genuine separate OS windows (a chrome target created with newWindow through the browser-level CDP session, in the background so it opens behind whatever is up, and bring_to_front raises one over the others). Every window is an instance: a browser with no windows has no reason to exist, and ends.

    The window table — id to chrome target — lives in `windows.json` beside the profile, so a daemon that reconnects to a running browser hands back the same ids for the same windows."""

    def __init__(self, playwright, home: Path, headless: bool):
        self._pw = playwright
        self.home = home
        self.headless = headless
        self.browser = None
        self.ctx = None
        self.session = None
        self.windows: dict[str, object] = {}
        self.targets: dict[str, str] = {}
        self.dialog_mode: dict[str, str] = {}
        self.page_errors: dict[int, list[str]] = {}
        self._watched: set[int] = set()
        self._counter = 0
        self.wants_exit = False

    def _answer_dialog(self, wid: str, dialog) -> None:
        mode = self.dialog_mode.get(wid, "dismiss")
        (dialog.accept if mode == "accept" else dialog.dismiss)()
        self.dialog_mode[wid] = "dismiss"

    def _target_id(self, page) -> str:
        session = self.ctx.new_cdp_session(page)
        try:
            return session.send("Target.getTargetInfo")["targetInfo"]["targetId"]
        finally:
            session.detach()

    def _table_path(self) -> Path:
        return self.home / "windows.json"

    def _save_table(self) -> None:
        self._table_path().write_text(
            json.dumps({"counter": self._counter, "windows": self.targets})
        )

    def _hold(self, wid: str, page) -> None:
        self.windows[wid] = page
        self.dialog_mode[wid] = "dismiss"
        page.on("dialog", lambda d, w=wid: self._answer_dialog(w, d))
        self._watch(page)

    def _watch(self, page) -> None:
        """Keep every error the page throws, from before its first navigation lands — an error thrown while the opening document runs is the one an open most needs to answer with."""
        if id(page) in self._watched:
            return
        self._watched.add(id(page))
        page.on(
            "pageerror",
            lambda e: self.page_errors.setdefault(id(page), []).append(str(e)),
        )

    def _adopt(self, page) -> str:
        self._counter += 1
        wid = f"w{self._counter}"
        self.targets[wid] = self._target_id(page)
        self._hold(wid, page)
        self._save_table()
        return wid

    def _ensure(self) -> None:
        if self.ctx is not None:
            return
        self.browser, self.headless, fresh = Browser(self._pw, self.home).connect(
            self.headless
        )
        self.ctx = self.browser.contexts[0]
        # The browser-level session windows are created and placed through.
        self.session = self.browser.new_browser_cdp_session()
        if fresh:
            # Started with no window: a fresh browser has none by definition, so nothing
            # it might have come up with is kept.
            for page in list(self.ctx.pages):
                page.close()
        # The windows a previous daemon held, by the same ids: the table names each
        # window's chrome target, and the targets still open take their ids back.
        if self._table_path().exists():
            table = json.loads(self._table_path().read_text())
            self._counter = table["counter"]
            by_target = {
                self._target_id(p): p for p in self.ctx.pages if not p.is_closed()
            }
            for wid, target in table["windows"].items():
                page = by_target.get(target)
                if page is not None:
                    self.targets[wid] = target
                    self._hold(wid, page)
        self._reconcile()

    def _window_bounds(self, page) -> dict:
        """Where a new window goes: most of the operator's display, centered, so a page reads at the size it was composed for and nothing is clipped — a fresh chrome window otherwise opens at whatever size the profile last remembered, which on a large display is a small box. The display is read through the window itself. Headless has no display, so nothing to size."""
        screen = page.evaluate(
            "({left: screen.availLeft, top: screen.availTop,"
            " width: screen.availWidth, height: screen.availHeight})"
        )
        width = round(screen["width"] * 0.8)
        height = round(screen["height"] * 0.85)
        return {
            "left": screen["left"] + (screen["width"] - width) // 2,
            "top": screen["top"] + (screen["height"] - height) // 2,
            "width": width,
            "height": height,
        }

    def _new_window(self, url: str):
        """A genuine separate OS window: a chrome target created with newWindow (which window.open cannot reliably do for a data:/file: URL) through the browser-level session, adopted as a Playwright page, sized and placed on the operator's display, then navigated. Created in the background, so it lands behind every existing window; `show` is the only thing that brings a window forward. Serialized command handling means the only page event in flight is this one."""
        with self.ctx.expect_event("page", timeout=30000) as info:
            created = self.session.send(
                "Target.createTarget",
                {"url": "about:blank", "newWindow": True, "background": True},
            )
        page = info.value
        self._watch(page)
        if not self.headless:
            window_id = self.session.send(
                "Browser.getWindowForTarget", {"targetId": created["targetId"]}
            )["windowId"]
            self.session.send(
                "Browser.setWindowBounds",
                {"windowId": window_id, "bounds": self._window_bounds(page)},
            )
        page.goto(url, wait_until="domcontentloaded")
        return page

    def _reconcile(self) -> None:
        """Drop closed windows and adopt any a site opened itself — a link that spawned its own window becomes an addressable instance rather than an orphan."""
        changed = False
        for wid in [w for w, p in self.windows.items() if p.is_closed()]:
            del self.windows[wid]
            del self.targets[wid]
            self.dialog_mode.pop(wid, None)
            changed = True
        known = set(self.windows.values())
        for page in self.ctx.pages:
            if page not in known and not page.is_closed():
                self._adopt(page)
        if changed:
            self._save_table()

    def _win(self, rest: list[str]):
        """The window a command names — its leading `w<id>`, required, popped off rest."""
        self._ensure()
        self._reconcile()
        if not rest or not (rest[0].startswith("w") and rest[0][1:].isdigit()):
            raise ValueError(
                "name the window, like w1 — `status` lists this browser's windows"
            )
        wid = rest.pop(0)
        if wid not in self.windows:
            raise ValueError(f"no window {wid} — `status` lists this browser's windows")
        return wid, self.windows[wid]

    def _settled_url(self, page) -> str | None:
        """The address of the page's own settled document, or None if it never settles.

        The question is put to the page, and Playwright holds any evaluation until the frame has no navigation in flight, so asking is itself the wait: a navigation the caller just started is waited out however slowly its server answers, and the answer comes from the document it landed on. `location.href` is the document's own truth, so a move within one document — a fragment, a pushState — is in the answer too. A plain `evaluate` asks the same question but nothing bounds it, and a navigation that never lands would hold the daemon forever; this waits as long as the browser itself waits for a navigation and then admits it does not know. Polling on an interval rather than on frames, because a headed window the operator has covered may paint no frames at all.
        """
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        try:
            return page.wait_for_function(
                "() => location.href", polling=50
            ).json_value()
        except PlaywrightTimeout:
            return None

    def _act(self, page, act) -> str:
        """Run an action and report where it left the page, so the reply names the page the agent is now talking to."""
        going: list[str] = []
        on_req = lambda r: (
            r.is_navigation_request()
            and r.frame == page.main_frame
            and going.append(r.url)
        )
        page.on("request", on_req)
        try:
            before = self._settled_url(page)
            act()
            after = self._settled_url(page)
        finally:
            page.remove_listener("request", on_req)
        if after is None:
            where = f" to {going[-1]}" if going else ""
            return f"  → still navigating{where} — the tab has not landed"
        return f"  → now at {after}" if after != before else ""

    def _found(self, page, rest: list[str]):
        """Resolve a click target — ref, --text, --css, or x y — to a viewport point."""
        text = _flag(rest, "--text")
        css = _flag(rest, "--css")
        if text is not None:
            hit = page.evaluate(FIND_BY_TEXT_JS, text)
            miss = f"nothing clickable matching {text!r}"
        elif css is not None:
            hit = page.evaluate(FIND_BY_CSS_JS, css)
            miss = f"nothing matching {css!r}"
        elif len(rest) == 2 and all(a.lstrip("-").isdigit() for a in rest):
            return int(rest[0]), int(rest[1]), f"({rest[0]},{rest[1]})"
        elif len(rest) == 1 and rest[0].startswith("e") and rest[0][1:].isdigit():
            hit = page.evaluate(REF_CENTER_JS, int(rest[0][1:]) - 1)
            miss = f"no such ref {rest[0]} on this page (refs die on navigation) — read again"
        else:
            raise ValueError(
                'name a target: a ref from `read` (e3), --text "label", --css "selector", or x y'
            )
        if hit is None:
            raise ValueError(miss)
        if "err" in hit:
            raise ValueError(hit["err"])
        return hit["x"], hit["y"], hit["desc"]

    def cmd_open(self, rest: list[str]) -> str:
        self._ensure()
        self._reconcile()
        wid = None
        if rest and rest[0].startswith("w") and rest[0][1:].isdigit():
            wid = rest.pop(0)
            if wid not in self.windows:
                raise ValueError(
                    f"no window {wid} — `status` lists this browser's windows"
                )
        if not rest:
            raise ValueError("open needs a URL or a path")
        url = target_url(rest[0])
        # domcontentloaded, never load: a replay page fetches the recording's images and
        # fonts live, and one dead resource would hold the load event past any timeout.
        if wid is None:
            # A window already holds this page. Which window the agent means — that one,
            # a second one, or none — is the agent's to say and nothing here can know it,
            # so the open is refused with the window named and the commands that would
            # say it. The address match strips the fragment: a replay parked at another
            # moment is the same page.
            address = url.partition("#")[0]
            held = [
                w for w, p in self.windows.items() if p.url.partition("#")[0] == address
            ]
            if held:
                names = ", ".join(held)
                raise ValueError(
                    f"{rest[0]} is already open as {names} — `show {held[0]}` puts it on "
                    f"the operator's screen, `open {held[0]} {rest[0]}` reopens it there"
                )
        if wid is None:
            page = self._new_window(url)
            self._landed(page)
            wid = self._adopt(page)
            # A new window lands behind everything the operator has up, so the reply says
            # what makes it theirs to see — an agent otherwise builds a page nobody was shown.
            return (
                f"[{wid}] {page.url} — {page.title()}\n"
                f"  → behind the other windows; `show {wid}` puts it on the operator's screen"
            )
        else:
            page = self.windows[wid]
            if url == page.url:
                # Opening the address the window already shows. A fragment-only navigation
                # to an unchanged fragment is one the browser drops entirely, so the page
                # would keep whatever state it had drifted into — a replay scrubbed away
                # from the moment its citation names — while this reported the open as
                # landed. Re-opening an address means putting the page back at it.
                page.reload(wait_until="domcontentloaded")
            else:
                page.goto(url, wait_until="domcontentloaded")
            self._landed(page)
        return f"[{wid}] {page.url} — {page.title()}"

    def _landed(self, page) -> None:
        """A replay page's opening seek moves once its players' pages are laid out, after the document has parsed; the page exposes that movement as `window.locus.moved`, and an open reports landed only once it has — a screenshot taken on the reply would otherwise show the frame from before the seek. Any other page has nothing pending. The wait is bounded: commands are served one at a time, so a seek that never settles would hold every agent's browser; past the bound the open fails naming it, the window left as the page stands."""
        unsettled = page.evaluate(
            "(ms) => (window.locus && window.locus.moved) ? Promise.race(["
            "  window.locus.moved.then(() => false),"
            "  new Promise((r) => setTimeout(() => r(true), ms)),"
            "]) : false",
            LANDING_MS,
        )
        if unsettled:
            raise ValueError(
                f"the replay's opening seek did not settle within {LANDING_MS // 1000}s — "
                "its snapshot page is still waiting on fonts or images; the window is open "
                "and shows the page as far as it has loaded"
            )

    def cmd_read(self, rest: list[str]) -> str:
        tab, page = self._win(rest)
        r = page.evaluate(READ_JS)
        more = []
        if r["below"]:
            more.append(f"{r['below']} below fold / off-screen")
        if r["above"]:
            more.append(f"{r['above']} above")
        head = [
            f"[{tab}] url:   {r['url']}",
            f"title: {r['title']}",
            "",
            f"[interactables in view] {len(r['lines'])}"
            + (f"   (+{', '.join(more)} — scroll, then read again)" if more else ""),
        ]
        text_lines = [s.strip() for s in r["text"].split("\n") if s.strip()]
        return "\n".join(head + r["lines"] + ["", "[page text]"] + text_lines)

    def cmd_click(self, rest: list[str]) -> str:
        tab, page = self._win(rest)
        x, y, desc = self._found(page, rest)
        note = self._act(page, lambda: page.mouse.click(x, y))
        return f"[{tab}] clicked {desc}{note}"

    def cmd_type(self, rest: list[str]) -> str:
        tab, page = self._win(rest)
        css = _flag(rest, "--css")
        spec: dict = {"css": css}
        if css is None and rest and rest[0].startswith("e") and rest[0][1:].isdigit():
            spec = {"ref": int(rest[0][1:]) - 1}
            rest = rest[1:]
        text = " ".join(rest)
        hit = page.evaluate(FOCUS_JS, spec)
        if hit is None:
            raise ValueError("no such field (refs die on navigation — read again)")
        if "err" in hit:
            raise ValueError(hit["err"])
        # The field's prior content is selected, so typing replaces and empty text clears.
        note = self._act(
            page,
            lambda: page.keyboard.type(text) if text else page.keyboard.press("Delete"),
        )
        return f"[{tab}] typed into {hit['desc']}: {text!r}{note}"

    def cmd_press(self, rest: list[str]) -> str:
        tab, page = self._win(rest)
        if not rest:
            raise ValueError("press needs at least one key or chord")
        note = self._act(page, lambda: [page.keyboard.press(chord(t)) for t in rest])
        return f"[{tab}] pressed {' '.join(rest)}{note}"

    def cmd_scroll(self, rest: list[str]) -> str:
        tab, page = self._win(rest)
        return f"[{tab}] {self._scroll(page, rest, 'down')}"

    def cmd_select(self, rest: list[str]) -> str:
        tab, page = self._win(rest)
        css = _flag(rest, "--css")
        spec: dict = {"css": css}
        if css is None:
            if not (rest and rest[0].startswith("e") and rest[0][1:].isdigit()):
                raise ValueError("select takes a ref or --css, then the option")
            spec = {"ref": int(rest[0][1:]) - 1}
            rest = rest[1:]
        if not rest:
            raise ValueError("select needs the option to pick")
        spec["want"] = " ".join(rest)
        hit = page.evaluate(SELECT_JS, spec)
        if hit is None:
            raise ValueError("no such <select> (refs die on navigation — read again)")
        if "err" in hit:
            raise ValueError(hit["err"])
        return f"[{tab}] selected {hit['picked']!r}"

    def cmd_eval(self, rest: list[str]) -> str:
        tab, page = self._win(rest)
        if not rest:
            raise ValueError("eval needs a JS expression")
        result = page.evaluate(" ".join(rest))
        return f"[{tab}] {json.dumps(result, default=str)}"

    def cmd_wait(self, rest: list[str]) -> str:
        tab, page = self._win(rest)
        text = _flag(rest, "--text")
        seconds = float(_flag(rest, "--seconds") or 15)
        if text is not None:
            page.get_by_text(text).first.wait_for(
                state="visible", timeout=seconds * 1000
            )
            return f"[{tab}] {text!r} is on the page"
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        try:
            page.wait_for_load_state("networkidle", timeout=seconds * 1000)
            return f"[{tab}] network went quiet"
        except PlaywrightTimeout:
            return f"[{tab}] network never went quiet within {seconds:g}s — the page stands as it is"

    def cmd_screenshot(self, rest: list[str]) -> str:
        tab, page = self._win(rest)
        full = "--full" in rest
        if full:
            rest.remove("--full")
        out = _flag(rest, "--out")
        flags = [t for t in rest if t.startswith("-")]
        if flags:
            raise ValueError(
                f"screenshot has no flag {' '.join(flags)}: "
                "`screenshot wN [path | --out <path>] [--full]`"
            )
        if out is not None and rest:
            raise ValueError(
                f"screenshot writes one file, and two paths were named: {out} and {rest[0]}"
            )
        if out is not None or rest:
            path = Path(out if out is not None else rest[0])
        else:
            screenshots = self.home / "screenshots"
            screenshots.mkdir(parents=True, exist_ok=True)
            path = screenshots / f"{tab}-{int(time.time() * 1000)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Raw CDP capture, not page.screenshot: Playwright's screenshot blocks on
        # document.fonts, and a page with an unloadable webfont (a replay page opened
        # from file://, its recorded font origins refusing the opaque origin) would hang
        # the screenshot forever. The look captures the page as it stands.
        import base64

        # A CDP send has no deadline, and a capture asked of a document that is still
        # navigating — the page reloaded by the command before — never answers, which would
        # hold every agent's browser. The document is waited for, bounded, before the ask.
        page.wait_for_load_state("domcontentloaded", timeout=LANDING_MS)
        cdp = page.context.new_cdp_session(page)
        try:
            captured = cdp.send(
                "Page.captureScreenshot", {"captureBeyondViewport": full}
            )
        finally:
            cdp.detach()
        path.write_bytes(base64.b64decode(captured["data"]))
        return f"[{tab}] {path}"

    def cmd_dialog(self, rest: list[str]) -> str:
        tab, _page = self._win(rest)
        mode = rest[0] if rest else "dismiss"
        if mode not in ("accept", "dismiss"):
            raise ValueError("dialog takes accept or dismiss")
        self.dialog_mode[tab] = mode
        return f"[{tab}] next dialog: {mode}"

    def _scroll(self, page, rest: list[str], default: str) -> str:
        """The scroll grammar `scroll` and `show` share: `--to <ref>` centers a ref from `read`, `--css "sel"` centers the first match of a selector; otherwise down|up|top|bottom|<px>, `default` when nothing is said."""
        to = _flag(rest, "--to")
        css = _flag(rest, "--css")
        if to is not None and css is not None:
            raise ValueError("--to and --css name two targets; give one")
        if css is not None:
            hit = page.evaluate(FIND_BY_CSS_JS, css)
            if hit is None:
                raise ValueError(f"nothing matching {css!r}")
            if "err" in hit:
                raise ValueError(hit["err"])
            return f"scrolled to {hit['desc']}"
        if to is not None:
            if not (to.startswith("e") and to[1:].isdigit()):
                raise ValueError(
                    '--to takes a ref from `read`, like e3; a selector goes in --css "sel"'
                )
            hit = page.evaluate(REF_CENTER_JS, int(to[1:]) - 1)
            if hit is None:
                raise ValueError(f"no such ref {to} on this page — read again")
            return f"scrolled to {hit['desc']}"
        where = rest[0] if rest else default
        js = {
            "down": "scrollBy(0, innerHeight * .8)",
            "up": "scrollBy(0, -innerHeight * .8)",
            "top": "scrollTo(0, 0)",
            "bottom": "scrollTo(0, document.body.scrollHeight)",
        }.get(where)
        if js is None:
            if not where.lstrip("-").isdigit():
                raise ValueError(
                    'scroll takes down|up|top|bottom|<px>|--to <ref>|--css "sel"'
                )
            js = f"scrollBy(0, {int(where)})"
        page.evaluate(js)
        return f"scrolled {where}"

    def cmd_show(self, rest: list[str]) -> str:
        """Put the window on the operator's screen, at the place their eye should land. The window is scrolled to wherever the agent's last look left it, so showing it as it stands hands the operator the bottom of a page or the middle of a table; the show itself places the page — the top unless told otherwise, in `scroll`'s grammar — and then raises it."""
        wid, page = self._win(rest)
        placed = self._scroll(page, rest, "top")
        page.bring_to_front()
        return f"[{wid}] {placed}, brought to the front"

    def cmd_status(self, rest: list[str]) -> str:
        self._ensure()
        self._reconcile()
        lines = [
            f"browser: {'headless' if self.headless else 'headed'} · profile {self.home / 'profile'}"
        ]
        for wid, page in self.windows.items():
            lines.append(f"{wid}  {page.url}  — {page.title()}")
        if not self.windows:
            lines.append("(no windows — `open <url>` starts one)")
        return "\n".join(lines)

    def cmd_close(self, rest: list[str]) -> str:
        wid, page = self._win(rest)
        page.close()
        self._reconcile()
        if not self.windows:
            # The last window gone is the browser's whole reason gone — browser and daemon
            # close with it, so a fresh open boots a clean one rather than reviving this state.
            self.wants_exit = True
            return f"closed {wid} — no windows left, the browser is closed"
        return f"closed {wid}"

    def dispatch(self, argv: list[str]) -> str:
        """Run one command and answer it. An error the page threw while the command ran — a replay refusing the moment its fragment asked for, a site script failing under a click — fails the command, named, over whatever it had to say: a reply that reported the page as landed while the page had refused would have the agent reading a frame it was never shown. Errors thrown meanwhile on other windows ride along as notes. Nothing thrown is dropped: an error that lands between commands is reported by the next one."""
        cmd, rest = argv[0], list(argv[1:])
        fn = getattr(self, f"cmd_{cmd}", None)
        if fn is None:
            raise ValueError(
                f"unknown command {cmd!r} — `locus browse help` lists them"
            )
        out = fn(rest)
        acted = re.match(r"\[(w\d+)\]", out)
        acted = acted.group(1) if acted else None
        if acted in self.windows and not self.windows[acted].is_closed():
            # A handler the command's action queued — a hashchange after a fragment set
            # by eval — runs as its own task; one turn of the page's loop lets it throw
            # before the answer is read.
            self.windows[acted].evaluate("() => new Promise((r) => setTimeout(r, 0))")
        lines, failed = [], False
        for wid, page in self.windows.items():
            for error in self.page_errors.pop(id(page), []):
                lines.append(f"  page error ({wid}): {error}")
                failed = failed or wid == acted
        if not lines:
            return out
        report = "\n".join([out, *lines])
        if failed:
            raise RuntimeError(report)
        return report

    def idle(self) -> bool:
        """No window anywhere, judged on the browser's real windows after connecting to a running one. No browser running is idle too, and nothing is started to find that out."""
        if self.ctx is None and Browser(self._pw, self.home)._alive() is None:
            return True
        self._ensure()
        self._reconcile()
        return not self.windows

    def disconnect(self) -> None:
        """Let go of the browser and leave it running with every window — what a daemon reboot does."""
        if self.browser is not None:
            self.browser.close()
            self.browser = None
            self.ctx = None

    def shutdown(self) -> None:
        """End the browser, every window with it, and forget the window table — what `quit` and the last close do."""
        self.disconnect()
        Browser(self._pw, self.home).stop()
        self._table_path().unlink(missing_ok=True)


def main() -> None:
    from .browse import driver_digest, sock_for

    home = Path(sys.argv[1])
    home.mkdir(parents=True, exist_ok=True)
    booted = driver_digest()
    # One daemon per home. Concurrent first commands each spawn one of these, so boot is
    # gated on an exclusive flock held for the daemon's life: a loser exits right here,
    # before the socket and before the browser, and its client's connect poll finds the
    # winner. With the lock held, an existing socket file can only be stale — a live
    # daemon would be holding the lock — so unlinking it is safe.
    lock = (home / "daemon.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return
    sock_path = sock_for(home)
    sock_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    # Commands are served one at a time, so clients queue here while one runs; the
    # backlog holds them, and a connect refused means no daemon, never a busy one.
    server.listen(64)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        driver = Driver(
            pw, home, headless=bool(os.environ.get("LOCUS_BROWSE_HEADLESS"))
        )
        try:
            while True:
                conn, _ = server.accept()
                with conn:
                    buf = b""
                    while not buf.endswith(b"\n"):
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        buf += chunk
                    if not buf.strip():
                        continue
                    request = json.loads(buf)
                    argv = request["argv"]
                    stale = request.get("src") not in (None, booted)
                    if stale and argv and argv[0] != "quit":
                        # The daemon runs the code it booted with; the client sends the
                        # digest of the driver on disk, so a change lands as a mismatch
                        # here. Nothing is lost by rebooting — the browser and its windows
                        # outlive the daemon — so the reply says so and the client boots
                        # the code on disk for this same command.
                        server.close()
                        sock_path.unlink(missing_ok=True)
                        conn.sendall(
                            json.dumps({"code": 0, "out": "", "reboot": True}).encode()
                            + b"\n"
                        )
                        return
                    if argv and argv[0] == "quit":
                        # Stop answering the door, close the browser, and only then
                        # reply: "closed" means closed, so the caller's next command
                        # finds no browser rather than one still going down.
                        server.close()
                        sock_path.unlink(missing_ok=True)
                        driver.shutdown()
                        conn.sendall(
                            json.dumps({"code": 0, "out": "browser closed"}).encode()
                            + b"\n"
                        )
                        return
                    try:
                        out, code = driver.dispatch(argv), 0
                    # The daemon holds the live window every later command needs, so a
                    # command's failure is its answer — the message and a nonzero code —
                    # never the daemon's death; and dispatch drives playwright, so what
                    # a command can raise is open by nature.
                    except Exception as e:  # noqa: BLE001
                        out, code = str(e), 1
                    if driver.wants_exit or driver.idle():
                        # No window left — the last one closed, or none exists — and a
                        # browser with no windows has no reason to exist: browser and
                        # daemon go, before the reply, so what the caller reads is
                        # already true.
                        driver.wants_exit = True
                        server.close()
                        sock_path.unlink(missing_ok=True)
                        driver.shutdown()
                    try:
                        conn.sendall(
                            json.dumps({"code": code, "out": out}).encode() + b"\n"
                        )
                    except OSError:
                        # The client gave up waiting and closed its end; the command ran
                        # regardless, and the daemon holds the windows for the next one.
                        pass
                    if driver.wants_exit:
                        return
        finally:
            driver.disconnect()
            sock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
