import fcntl
import io
import os
import re
import socket
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
from locus.evidence.browse import run, sock_for
from locus.evidence.browse_driver import chord, target_url


def test_chords_translate_to_playwright_names():
    assert chord("ctrl+Enter") == "Control+Enter"
    assert chord("cmd+a") == "Meta+a"
    assert chord("Enter") == "Enter"
    assert chord("shift+alt+ArrowDown") == "Shift+Alt+ArrowDown"


def test_target_url_passes_urls_and_resolves_files(tmp_path):
    assert target_url("https://example.test/a") == "https://example.test/a"
    page = tmp_path / "p.html"
    page.write_text("<html></html>")
    assert target_url(str(page)) == page.resolve().as_uri()
    with pytest.raises(FileNotFoundError):
        target_url("/no/such/page.html")


def test_target_url_keeps_the_fragment_a_citation_rides_in(tmp_path):
    # A replay page opens at a citation as page.html#t=<ms>; the fragment is not part of the
    # file path, so the existence check must strip it and the URI must keep it.
    page = tmp_path / "v.html"
    page.write_text("<html></html>")
    assert target_url(f"{page}#t=123-456") == page.resolve().as_uri() + "#t=123-456"


def test_help_answers_without_booting_a_browser(tmp_path):
    code, out, _ = browse(tmp_path / "browse", ["help"])
    assert code == 0
    assert "open" in out
    assert not sock_for(tmp_path / "browse").exists()


def test_a_racing_daemon_defers_and_leaves_the_winners_socket_alone(tmp_path):
    # Two first commands can each find no daemon and each spawn one. The loser must
    # exit without ever touching the socket: unlinking or rebinding it would sever
    # the winner's socket and leave an owner-less browser running.
    home = tmp_path / "browse"
    home.mkdir(parents=True)
    lock = (home / "daemon.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sock_path = sock_for(home)
    winner = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    winner.bind(str(sock_path))
    winner.listen(1)
    try:
        loser = subprocess.run(
            [sys.executable, "-m", "locus.evidence.browse_driver", str(home)],
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert loser.returncode == 0, loser.stderr
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.connect(str(sock_path))
    finally:
        winner.close()
        sock_path.unlink(missing_ok=True)


def _daemons(home: Path) -> list[str]:
    out = subprocess.run(
        ["ps", "ax", "-o", "command"], capture_output=True, text=True, check=True
    ).stdout
    return [
        line
        for line in out.splitlines()
        if "locus.evidence.browse_driver" in line and str(home) in line
    ]


def _settles(cond, seconds: float = 15) -> bool:
    deadline = time.time() + seconds
    while not cond() and time.time() < deadline:
        time.sleep(0.2)
    return cond()


def test_racing_first_commands_converge_on_one_daemon(tmp_path_factory):
    # Four agents each opening their own page at once, none finding a daemon: every one
    # lands on the single daemon the flock admits and gets its own window from it.
    root = tmp_path_factory.mktemp("race-pages")
    targets = []
    for i in range(4):
        page = root / f"p{i}.html"
        page.write_text(f"<!doctype html><title>P{i}</title>")
        targets.append(str(page))
    home = tmp_path_factory.mktemp("race-deployment") / "browse"
    env = {**os.environ, "LOCUS_BROWSE_HEADLESS": "1"}
    client = (
        "import sys; from pathlib import Path; from locus.evidence.browse import run; "
        "sys.exit(run(Path(sys.argv[1]), ['open', sys.argv[2]]))"
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", client, str(home), target],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for target in targets
    ]
    try:
        results = []
        for p in procs:
            out, _ = p.communicate(timeout=120)
            results.append((p.returncode, out.decode()))
        assert all(code == 0 for code, _ in results), results
        assert len({win_of(out) for _, out in results}) == 4, results
        assert _settles(lambda: len(_daemons(home)) == 1), _daemons(home)
        _, status, _ = browse(home, ["status"])
        assert all(f"p{i}.html" in status for i in range(4)), status
    finally:
        code, out, _ = browse(home, ["quit"])
    assert code == 0, out
    assert _settles(lambda: not _daemons(home)), _daemons(home)
    assert _settles(lambda: not sock_for(home).exists())


def browse(home: Path, argv: list[str]):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = run(home, argv)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    root = tmp_path_factory.mktemp("pages")
    (root / "second.html").write_text(
        "<!doctype html><title>Second</title><h1>You navigated</h1>"
    )
    (root / "demo.html").write_text(
        "<!doctype html><title>Demo</title><body>"
        "<h1>Hello Locus</h1><p>A paragraph of visible text.</p>"
        "<a href='second.html'>go next</a>"
        "<input name='email' placeholder='your email'>"
        "<select name='pet'><option>cat</option><option>dog</option></select>"
        "<button onclick=\"document.querySelector('h1').textContent='Clicked'\">"
        "Submit</button>"
        "</body>"
    )
    (root / "foreign-locus.html").write_text(
        "<!doctype html><title>Foreign</title><script>window.locus = {}</script>"
        "<h1>a site with its own window.locus</h1>"
    )
    (root / "throws.html").write_text(
        "<!doctype html><title>Throws</title><body><h1>Quiet</h1><script>"
        "const boom = () => { if (location.hash.includes('boom')) "
        "throw new Error('boom at ' + location.hash); };"
        "addEventListener('hashchange', boom); boom();"
        "</script></body>"
    )
    return root


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    home = tmp_path_factory.mktemp("deployment") / "browse"
    os.environ["LOCUS_BROWSE_HEADLESS"] = "1"
    try:
        yield home
    finally:
        browse(home, ["quit"])
        os.environ.pop("LOCUS_BROWSE_HEADLESS", None)


def win_of(out: str) -> str:
    m = re.search(r"\[(w\d+)\]", out)
    assert m, out
    return m.group(1)


def open_page(home: Path, target: str) -> str:
    """The window holding `target`, the way an agent gets one: a bare open, and when that is refused because a window already holds the page, the reopen into the window the refusal named."""
    code, out, err = browse(home, ["open", target])
    if code == 1 and "already open as" in err:
        w = re.search(r"already open as (w\d+)", err).group(1)
        code, out, err = browse(home, ["open", w, target])
    assert code == 0, err
    return win_of(out)


def test_open_read_shows_refs_and_text(home, pages):
    code, out, err = browse(home, ["open", str(pages / "demo.html")])
    assert code == 0, err
    w = win_of(out)
    assert "Demo" in out and f"show {w}" in out

    code, out, _ = browse(home, ["read", w])
    assert code == 0
    assert "Hello Locus" in out
    assert '"go next"' in out
    assert "input[text]" in out
    assert "select" in out


def test_an_error_the_page_throws_during_a_command_fails_that_command(home, pages):
    # A page that throws while a command runs — a replay refusing the moment its fragment
    # names — has answered the command; reporting the open as landed would hand the agent
    # a page in a state it was never told about.
    code, out, err = browse(home, ["open", f"{pages / 'throws.html'}#boom-open"])
    assert code == 1, out
    w = win_of(err)
    assert w in err and "boom at #boom-open" in err
    assert "Throws" in err, "the command's own reply still rides above the error"

    code, out, err = browse(home, ["eval", w, "location.hash = '#boom-later'"])
    assert code == 1
    assert w in err and "boom at #boom-later" in err

    code, out, err = browse(home, ["eval", w, "location.hash = '#calm'"])
    assert code == 0, err
    assert "boom" not in out


def test_a_window_scoped_command_must_name_its_window(home, pages):
    # There is no shared active window, so a command without an id has no page to act on
    # and says so — the guard that makes parallel agents' commands never collide.
    code, _, err = browse(home, ["read"])
    assert code == 1
    assert "status" in err, "the reply names the command that lists windows"
    code, _, err = browse(home, ["read", "w999"])
    assert code == 1
    assert "w999" in err


def test_opening_the_address_a_window_already_shows_puts_it_back(home, pages):
    # Walking citations re-opens the same address whenever two of them name one moment, and
    # a page drifts from its address on its own — a replay scrubbed, a form filled. The
    # browser drops a navigation whose fragment did not change, so without a reload the open
    # reports landing while the page keeps whatever state it drifted into.
    target = f"{pages / 'demo.html'}#t=123"
    w = open_page(home, target)
    browse(
        home,
        [
            "eval",
            w,
            "() => { document.querySelector('h1').textContent = 'drifted'; }",
        ],
    )
    _, out, _ = browse(home, ["eval", w, "document.querySelector('h1').textContent"])
    assert "drifted" in out

    code, _, err = browse(home, ["open", w, target])
    assert code == 0, err
    _, out, _ = browse(home, ["eval", w, "document.querySelector('h1').textContent"])
    assert "Hello Locus" in out and "drifted" not in out


def test_click_by_ref_acts_on_the_page(home, pages):
    w = open_page(home, str(pages / "demo.html"))
    _, out, _ = browse(home, ["read", w])
    ref = next(line.split()[0] for line in out.splitlines() if '"Submit"' in line)
    code, out, err = browse(home, ["click", w, ref])
    assert code == 0, err
    _, out, _ = browse(home, ["eval", w, "document.querySelector('h1').textContent"])
    assert "Clicked" in out


def test_type_replaces_a_named_fields_content(home, pages):
    w = open_page(home, str(pages / "demo.html"))
    _, out, _ = browse(home, ["read", w])
    ref = next(line.split()[0] for line in out.splitlines() if "input[text]" in line)
    browse(home, ["type", w, ref, "a@b.test"])
    browse(home, ["type", w, ref, "c@d.test"])
    _, out, _ = browse(home, ["eval", w, "document.querySelector('input').value"])
    assert "c@d.test" in out and "a@b" not in out


def test_select_picks_an_option(home, pages):
    w = open_page(home, str(pages / "demo.html"))
    _, out, _ = browse(home, ["read", w])
    ref = next(
        line.split()[0]
        for line in out.splitlines()
        if len(line.split()) > 1 and line.split()[1] == "select"
    )
    code, out, err = browse(home, ["select", w, ref, "dog"])
    assert code == 0, err
    _, out, _ = browse(home, ["eval", w, "document.querySelector('select').value"])
    assert "dog" in out


def test_click_navigation_is_reported_and_refs_go_stale(home, pages):
    w = open_page(home, str(pages / "demo.html"))
    _, out, _ = browse(home, ["read", w])
    ref = next(line.split()[0] for line in out.splitlines() if '"go next"' in line)
    code, out, err = browse(home, ["click", w, ref])
    assert code == 0, err
    assert "second.html" in out

    # The old page's refs died with the navigation; acting on one must say so, not
    # silently click whatever occupies that index on the new page.
    code, _, err = browse(home, ["click", w, "e99"])
    assert code == 1
    assert "read" in err, "the reply names the command that refreshes refs"


def test_screenshot_lands_where_the_asker_said(home, pages, tmp_path):
    w = open_page(home, str(pages / "demo.html"))
    screenshot = tmp_path / "out" / "screenshot.png"
    code, out, err = browse(home, ["screenshot", w, str(screenshot)])
    assert code == 0, err
    assert str(screenshot) in out
    assert screenshot.exists() and screenshot.stat().st_size > 1000


def test_a_screenshot_asked_mid_reload_answers(home, pages, tmp_path):
    """An eval can set a navigation going; a capture asked of the document before it lands never answers over CDP, so the screenshot waits for the document first — bounded — and answers with the landed page."""
    w = open_page(home, str(pages / "demo.html"))
    browse(home, ["eval", w, "location.reload(); 'going'"])
    screenshot = tmp_path / "reloaded.png"
    code, _out, err = browse(home, ["screenshot", w, str(screenshot)])
    assert code == 0, err
    assert screenshot.exists() and screenshot.stat().st_size > 1000


def test_a_page_with_its_own_window_locus_opens_like_any_other(home, pages):
    """The landing wait belongs to the replay page's `window.locus.moved`; a site that happens to define `window.locus` for itself — a page carrying the recorder, say — has no seek to wait on and opens at once."""
    code, out, err = browse(home, ["open", str(pages / "foreign-locus.html")])
    assert code == 0, err
    assert out.startswith("[w")


def test_screenshot_takes_the_path_as_out_flag_or_bare(home, pages, tmp_path):
    """`--out <path>` and a bare path name the same file; a flag the command lacks is refused rather than written as a file name, and two paths for one file are refused."""
    w = open_page(home, str(pages / "demo.html"))
    code, out, err = browse(home, ["screenshot", w, "--out", str(tmp_path / "s.png")])
    assert code == 0, err
    assert (tmp_path / "s.png").stat().st_size > 1000
    assert str(tmp_path / "s.png") in out
    code, _out, err = browse(home, ["screenshot", w, "--png", str(tmp_path / "t.png")])
    assert code != 0 and "--png" in err
    assert not (tmp_path / "t.png").exists() and not (tmp_path / "--png").exists()
    code, _out, err = browse(
        home,
        ["screenshot", w, str(tmp_path / "u.png"), "--out", str(tmp_path / "v.png")],
    )
    assert code != 0 and "two paths" in err


def test_windows_are_separate_and_listed(home, pages):
    w1 = open_page(home, str(pages / "demo.html"))
    w2 = open_page(home, str(pages / "second.html"))
    assert w1 != w2

    # Each window keeps its own page — a command on one never touches the other.
    _, out, _ = browse(home, ["eval", w1, "document.title"])
    assert "Demo" in out
    _, out, _ = browse(home, ["eval", w2, "document.title"])
    assert "Second" in out

    _, out, _ = browse(home, ["status"])
    assert w1 in out and w2 in out and "second.html" in out

    code, out, _ = browse(home, ["close", w2])
    assert code == 0
    _, out, _ = browse(home, ["status"])
    assert not any(
        line.startswith(w2) for line in out.split("\n", 1)[1].splitlines()
    ), "the window is gone from the listing (its id may recur inside a tmp path)"


def test_a_new_window_says_it_is_behind_and_how_to_show_it(home, pages):
    # A window opens behind everything the operator has up; the reply names the one
    # command that puts it on their screen, so a page is never built and left unshown.
    fresh = pages / "fresh.html"
    fresh.write_text("<!doctype html><title>Fresh</title><p>never opened before</p>")
    code, out, err = browse(home, ["open", str(fresh)])
    assert code == 0, err
    w = win_of(out)
    assert f"show {w}" in out
    code, out, err = browse(home, ["open", w, str(pages / "second.html")])
    assert code == 0, err
    assert "show" not in out, (
        "re-navigating a window the operator may already see says nothing"
    )


def test_opening_a_page_some_window_already_holds_refuses_and_names_it(home, pages):
    # Which window the agent means — the one already on that page, a second one, or none —
    # is the agent's to say; the browser cannot know it, so it neither mints a duplicate
    # nor quietly hands back the existing window. It refuses, names the window, and names
    # the commands that would say either thing. A different moment is the same page.
    w = open_page(home, str(pages / "second.html"))
    code, out, err = browse(home, ["open", str(pages / "second.html") + "#t=1"])
    assert code == 1, out
    assert w in err
    assert f"show {w}" in err and f"open {w} " in err
    _, status, _ = browse(home, ["status"])
    assert status.count("second.html") == 1, "no duplicate window was made"


def test_scroll_to_takes_a_ref_or_a_css_selector(home, pages):
    tall = pages / "tall-css.html"
    tall.write_text(
        "<!doctype html><title>Tall</title><body style='height:5000px'>"
        "<h1>Top</h1><a id='deep' href='#' style='position:absolute;top:4000px'>Deep</a>"
    )
    code, out, err = browse(home, ["open", str(tall)])
    assert code == 0, err
    w = win_of(out)

    code, out, err = browse(home, ["scroll", w, "--css", "#deep"])
    assert code == 0, err
    assert "scrolled to a#deep" in out
    _, out, _ = browse(home, ["eval", w, "scrollY"])
    assert int(float(out.split("]")[1])) > 3000

    _, out, _ = browse(home, ["read", w])
    ref = next(tok for tok in out.split() if tok.startswith("e") and tok[1:].isdigit())
    browse(home, ["scroll", w, "top"])
    code, out, err = browse(home, ["scroll", w, "--to", ref])
    assert code == 0, err
    _, out, _ = browse(home, ["eval", w, "scrollY"])
    assert int(float(out.split("]")[1])) > 3000

    code, _, err = browse(home, ["scroll", w, "--css", "#nowhere"])
    assert code != 0 and "nothing matching" in err
    code, _, err = browse(home, ["scroll", w, "--to", "#deep"])
    assert code != 0 and "--css" in err
    code, _, err = browse(home, ["scroll", w, "--to", ref, "--css", "#deep"])
    assert code != 0 and "two targets" in err


def test_show_places_the_page_where_the_eye_should_land(home, pages):
    tall = pages / "tall.html"
    tall.write_text(
        "<!doctype html><title>Tall</title><body style='height:5000px'>"
        "<h1>Top</h1><p id='deep' style='position:absolute;top:4000px'>Deep</p>"
    )
    code, out, err = browse(home, ["open", str(tall)])
    assert code == 0, err
    w = win_of(out)
    browse(home, ["scroll", w, "bottom"])
    _, out, _ = browse(home, ["eval", w, "scrollY"])
    assert int(float(out.split("]")[1])) > 0

    code, out, err = browse(home, ["show", w])
    assert code == 0, err
    _, out, _ = browse(home, ["eval", w, "scrollY"])
    assert int(float(out.split("]")[1])) == 0

    code, out, err = browse(home, ["show", w, "bottom"])
    assert code == 0, err
    _, out, _ = browse(home, ["eval", w, "scrollY"])
    assert int(float(out.split("]")[1])) > 0


def test_closing_the_last_window_stops_the_daemon(tmp_path_factory, pages):
    home = tmp_path_factory.mktemp("solo") / "browse"
    os.environ["LOCUS_BROWSE_HEADLESS"] = "1"
    try:
        code, out, err = browse(home, ["open", str(pages / "demo.html")])
        assert code == 0, err
        w = win_of(out)
        code, out, _ = browse(home, ["close", w])
        assert code == 0 and "the browser is closed" in out
        # The daemon is gone, so a fresh open boots a clean one numbering from w1 again.
        code, out, err = browse(home, ["open", str(pages / "demo.html")])
        assert code == 0, err
        assert win_of(out) == "w1"
    finally:
        browse(home, ["quit"])
        os.environ.pop("LOCUS_BROWSE_HEADLESS", None)


def test_quit_closes_every_window_and_the_daemon(home):
    code, _, _ = browse(home, ["quit"])
    assert code == 0
    code, _, _ = browse(home, ["quit"])
    assert code == 0 and not (home / "browser.json").exists()


def _raw(home: Path, request: dict) -> dict:
    import json

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(str(sock_for(home)))
        s.settimeout(60)
        s.sendall(json.dumps(request).encode() + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf)


def test_a_browser_with_no_windows_does_not_exist(tmp_path_factory):
    # A command that finds no window — status on a fresh boot, here — gets its answer,
    # and then nothing is left running: browser and daemon end, no record beside the
    # profile, because a window is the only reason either exists.
    home = tmp_path_factory.mktemp("empty") / "browse"
    os.environ["LOCUS_BROWSE_HEADLESS"] = "1"
    try:
        code, out, err = browse(home, ["status"])
        assert code == 0, err
        assert "[w" not in out
        assert _settles(lambda: not _daemons(home)), _daemons(home)
        assert not (home / "browser.json").exists()
        code, _, _ = browse(home, ["quit"])
        assert code == 0 and not (home / "browser.json").exists()
    finally:
        os.environ.pop("LOCUS_BROWSE_HEADLESS", None)


def _browser_pid(home: Path) -> int:
    import json

    return json.loads((home / "browser.json").read_text())["pid"]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_a_daemon_on_stale_code_reboots_and_the_windows_survive_under_their_ids(
    tmp_path_factory, pages
):
    # The browser is its own process, so a daemon reboot loses nothing: the new daemon
    # finds the same browser, hands back the same ids for the same windows, and the refs
    # a page held are still in the page.
    home = tmp_path_factory.mktemp("stale-windows") / "browse"
    os.environ["LOCUS_BROWSE_HEADLESS"] = "1"
    try:
        code, out, err = browse(home, ["open", str(pages / "second.html")])
        assert code == 0, err
        w_first = win_of(out)
        code, out, err = browse(home, ["open", str(pages / "demo.html")])
        assert code == 0, err
        w = win_of(out)
        _, out, _ = browse(home, ["read", w])
        ref = next(line.split()[0] for line in out.splitlines() if '"Submit"' in line)
        pid = _browser_pid(home)

        reply = _raw(home, {"argv": ["status"], "src": "not-what-it-booted-with"})
        assert reply["reboot"] is True, reply
        assert _settles(lambda: not _daemons(home)), _daemons(home)
        assert _alive(pid), "the browser outlives the daemon"

        code, out, err = browse(home, ["status"])
        assert code == 0, err
        assert len(_daemons(home)) == 1
        assert _browser_pid(home) == pid, "the new daemon connected to the same browser"
        lines = out.splitlines()[1:]
        assert any(line.startswith(w_first) and "second.html" in line for line in lines)
        assert any(line.startswith(w) and "demo.html" in line for line in lines)
        code, out, err = browse(home, ["click", w, ref])
        assert code == 0, err
        _, out, _ = browse(
            home, ["eval", w, "document.querySelector('h1').textContent"]
        )
        assert "Clicked" in out
    finally:
        browse(home, ["quit"])
        os.environ.pop("LOCUS_BROWSE_HEADLESS", None)


def test_opening_a_replay_at_a_moment_lands_only_once_the_seek_has_moved(
    home, tmp_path
):
    # The component moves an opening seek once its players' pages are laid out, after
    # the document has parsed. An open that returned at parse time would hand the agent
    # the frame from before the seek; the reply waits for the movement.
    from test_replay import (
        BODY,
        compose,
        meta,
        page_snapshot,
        synthetic_page,
        text_mutation,
    )

    conn, ids = synthetic_page(
        tmp_path, [meta(1000), page_snapshot(1001), text_mutation(2000, 5, "goodbye")]
    )
    page_path = compose(tmp_path, conn, ids)

    code, out, err = browse(home, ["open", f"{page_path}#t=2000"])
    assert code == 0, err
    w = win_of(out)
    _, out, _ = browse(home, ["eval", w, BODY[6:]])
    assert "goodbye" in out
    code, out, err = browse(home, ["open", w, f"{page_path}#t=1500"])
    assert code == 0, err
    _, out, _ = browse(home, ["eval", w, BODY[6:]])
    assert "hello" in out and "goodbye" not in out


def test_an_unknown_verb_on_a_fresh_daemon_leaves_the_browsers_windows_alone(
    tmp_path_factory, pages
):
    # A daemon that has not yet connected holds an empty window table. A command that
    # fails before connecting — an unknown verb — must not have that emptiness read as
    # "no windows left": the browser still has its windows, and stays.
    home = tmp_path_factory.mktemp("unknown-verb") / "browse"
    os.environ["LOCUS_BROWSE_HEADLESS"] = "1"
    try:
        code, out, err = browse(home, ["open", str(pages / "demo.html")])
        assert code == 0, err
        w = win_of(out)
        pid = _browser_pid(home)
        reply = _raw(home, {"argv": ["status"], "src": "not-what-it-booted-with"})
        assert reply["reboot"] is True, reply
        assert _settles(lambda: not _daemons(home)), _daemons(home)

        code, out, err = browse(home, ["ls"])
        assert code == 1 and "unknown command 'ls'" in err
        assert _alive(pid), "an unknown verb ended the browser"
        code, out, err = browse(home, ["status"])
        assert code == 0, err
        assert any(
            line.startswith(w) and "demo.html" in line for line in out.splitlines()
        )
    finally:
        browse(home, ["quit"])
        os.environ.pop("LOCUS_BROWSE_HEADLESS", None)


def test_quit_with_no_daemon_still_closes_a_browser_left_running(
    tmp_path_factory, pages
):
    # After a reboot answer the daemon is gone and the browser is not; quit has to reach
    # the browser regardless, verified by the process dying.
    home = tmp_path_factory.mktemp("orphan") / "browse"
    os.environ["LOCUS_BROWSE_HEADLESS"] = "1"
    try:
        code, out, err = browse(home, ["open", str(pages / "demo.html")])
        assert code == 0, err
        pid = _browser_pid(home)
        _raw(home, {"argv": ["status"], "src": "not-what-it-booted-with"})
        assert _settles(lambda: not _daemons(home))
        assert _alive(pid)
        code, out, _ = browse(home, ["quit"])
        assert code == 0 and "browser closed" in out
        assert _settles(lambda: not _alive(pid)), "the browser process is dead"
        assert not (home / "browser.json").exists()
        code, _, _ = browse(home, ["quit"])
        assert code == 0 and not (home / "browser.json").exists()
    finally:
        os.environ.pop("LOCUS_BROWSE_HEADLESS", None)
