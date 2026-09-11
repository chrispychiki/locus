"""The deployment's browser — client half.

`locus browse <command>` drives one persistent, headed browser: the agent's eyes and hands on any page — a replay page at a citation, a page it composed, the live site — and the operator's window onto the same, since both look at the same screen. The daemon (`browse_driver`) drives the windows between commands; this module ships one command over the daemon's unix socket, prints the reply, and exits. The first command boots the daemon — concurrent first commands are safe: a lock under the home admits one daemon and every command lands on it — and `quit` shuts the windows, the browser, and the daemon down together. Every command carries the digest of the driver source on disk, so a daemon booted on older code learns it at the next command and reboots itself under that command; the browser is its own process and outlives the daemon, so the reboot costs no window. Browser state lives under `data/browse/` at the deployment root: the Chromium profile (logins persist), the browser's record and window table, the daemon log, the daemon's lock, and screenshots; the socket sits in the system tmpdir at a short path derived from the home.
"""

import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HELP = """locus browse drives this deployment's browser — headed, so the operator can watch the same
windows the agent drives. Each instance is its own window, addressed w1, w2, …; every command names
the window it acts on, so agents working in different windows never disturb each other's pages —
one the operator is looking at included.
Commands run one at a time across all windows, so a slow command in one window holds the rest. State
survives between commands: windows stay open, logins stick, refs from `read` stay clickable until
that window navigates. Every reply opens with the window it acted on.

  open [wN] <target>                   navigate — a URL; a page path; a slice set or an analysis
                                       directory, whose replay material and default page are
                                       materialized into pages/ beside the db, printed, then
                                       opened. wN re-navigates that window; no wN opens a new
                                       window and prints its id — and refuses, naming the
                                       window, when one already shows that page. The
                                       target may carry the moment
                                       to open at: #t=<epoch-ms> lands on a citation,
                                       #t=<start>-<end> marks the range and stops play at its end,
                                       &m=<mount> seeks one named mount instead of all. A
                                       moment the page holds no recording at fails the open,
                                       naming what the page can show; so does a fragment
                                       in any other shape
  read wN                              ref-tagged interactables (e1, e2, …) + visible text
  click wN <ref> | --text "label" | --css "sel" | <x> <y>
  type wN <ref> <text> | --css "sel" <text> | <text>   naming a field replaces its content;
                                       bare types into the focused field; no text = clear
  select wN <ref> <option> | --css "sel" <option>      pick an option in a native <select>
  press wN <chord...>                  Enter Tab Escape ArrowDown, or chords like ctrl+Enter
  scroll wN [down|up|top|bottom|<px>] | scroll wN --to <ref> | --css "sel"
  wait wN [--text <s>] [--seconds N]   until that text shows (default 15s); bare = network-quiet
  eval wN <js>                         run JS in the window's page, print the result
  screenshot wN [path | --out <path>] [--full]   PNG to that path (default data/browse/screenshots/),
                                       prints it
  show wN [top|bottom|<px>|--to <ref>|--css "sel"]  put the window on the operator's screen: scrolled to where
                                       their eye should land (default top), then raised. The only
                                       command that raises a window; a window never shown was
                                       never seen
  dialog wN [accept|dismiss]           how to answer the window's next confirm/prompt (default dismiss)
  status                               list the open windows
  close wN                             close one window; closing the last closes the browser
  quit                                 close every window and the browser — every agent's
                                       windows, any that were shown included; `close wN`
                                       ends one
"""


def is_help(argv: list[str]) -> bool:
    """Whether this invocation renders the browser's own documentation rather than driving the window — asked bare, or beside a verb (`open --help`), which is where an agent asks for one verb's usage."""
    asks = ("help", "--help", "-h")
    return not argv or argv[0] in asks or (len(argv) > 1 and argv[1] in asks)


def sock_for(home: Path) -> Path:
    """The daemon's socket, at a short stable path derived from the home — AF_UNIX paths cap at ~104 bytes on macOS, and the deployment root can sit arbitrarily deep."""
    digest = hashlib.sha1(str(home.resolve()).encode()).hexdigest()[:8]
    return Path(tempfile.gettempdir()) / f"locus-browse-{digest}.sock"


def _connect(sock_path: Path) -> socket.socket | None:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(str(sock_path))
        return s
    except OSError:
        s.close()
        return None


def driver_digest() -> str:
    """The driver's source as it is on disk right now. The daemon runs whatever it booted with; every command carries this digest, so an edit to the driver shows up at the daemon as a mismatch on the next command."""
    return hashlib.sha1(
        (Path(__file__).parent / "browse_driver.py").read_bytes()
    ).hexdigest()


def run(home: Path | None, argv: list[str], timeout_s: float = 180) -> int:
    """Ship one command to the deployment's browser daemon, print its reply, return its code. The browser's own documentation is not a deployment fact, so a help invocation renders it from no home at all — which is what lets it answer from outside a clone, like every other command's help. A daemon that finds itself running older code than is on disk answers by shutting down instead — so the command is shipped again to the fresh daemon this boots, which finds the same browser and windows."""
    if is_help(argv):
        print(HELP)
        return 0
    home.mkdir(parents=True, exist_ok=True)
    sock_path = sock_for(home)
    if argv[0] == "quit":
        # The browser outlives the daemon, so "is anything up" is either: a daemon
        # answering the socket, or a browser the daemon left running (its record beside
        # the profile). Either boots a daemon to do the closing; neither means nothing.
        probe = _connect(sock_path)
        if probe is None and not (home / "browser.json").exists():
            print("no browser is up")
            return 0
        if probe is not None:
            probe.close()
    for _ in range(2):
        s = _reach(home, sock_path, timeout_s)
        if s is None:
            return 1
        reply = _exchange(s, {"argv": argv, "src": driver_digest()}, timeout_s)
        if reply is None:
            return 1
        if reply.get("reboot"):
            continue
        print(reply["out"], file=sys.stderr if reply["code"] else sys.stdout)
        return reply["code"]
    print(
        "browser daemon kept rebooting — see " + str(home / "driver.log"),
        file=sys.stderr,
    )
    return 1


def _reach(home: Path, sock_path: Path, timeout_s: float) -> socket.socket | None:
    """A socket to the daemon, booting one when none answers; None when the boot failed, which is already printed."""
    s = _connect(sock_path)
    if s is None:
        # Racing clients may each spawn a daemon; the flock in browse_driver admits
        # exactly one, the rest exit untouched, and this poll converges on whichever
        # bound the socket. The socket file is the daemon's to clean up, never ours —
        # unlinking it here could sever a daemon that just bound it.
        # The wait is judged by liveness, never a wall-clock guess: a loaded
        # machine can hold a cold boot past any fixed patience, and a client that
        # gives up early strands its command while the daemon comes up anyway —
        # the next command then lands on a window in a state nobody asked for.
        # Liveness is read from the child's exit status, never by touching the
        # daemon.lock — acquiring it even for an instant can turn a booting
        # winner into a loser. A nonzero exit is a boot that died: fail now, the
        # log has it. Exit 0 is the flock-loser path and proves only that some
        # holder existed at that attempt — a booting winner, or a quitting
        # daemon still tearing down after its socket is gone, which no connect
        # will ever reach. Only a fresh contender converges both cases, so a
        # loser is respawned on a backoff until a socket answers or the
        # command's own deadline passes.
        log = (home / "driver.log").open("w")

        def spawn():
            return subprocess.Popen(
                [sys.executable, "-m", "locus.evidence.browse_driver", str(home)],
                stdout=log,
                stderr=log,
                start_new_session=True,
            )

        child = spawn()
        deadline = time.time() + timeout_s
        backoff = 0.5
        while s is None and time.time() < deadline:
            status = child.poll()
            if status is not None and status != 0:
                print(
                    f"browser daemon died during boot — see {home / 'driver.log'}",
                    file=sys.stderr,
                )
                return None
            if status == 0:
                time.sleep(backoff)
                backoff = min(backoff * 2, 2.0)
                s = _connect(sock_path)
                if s is None and time.time() < deadline:
                    child = spawn()
                continue
            time.sleep(0.15)
            s = _connect(sock_path)
        if s is None:
            print(
                f"no browser daemon came up within {timeout_s:g}s — see {home / 'driver.log'}",
                file=sys.stderr,
            )
            return None
    return s


def _exchange(s: socket.socket, request: dict, timeout_s: float) -> dict | None:
    """One request over the socket, the daemon's reply back; None when the daemon never answered (already printed)."""
    argv = request["argv"]
    with s:
        s.settimeout(timeout_s)
        s.sendall(json.dumps(request).encode() + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            try:
                chunk = s.recv(65536)
            except TimeoutError:
                print(
                    f"browse {argv[0]} did not finish within {timeout_s:g}s — the daemon "
                    "serves every window's commands one at a time, so another agent's "
                    "command may be holding it; this one still runs there to completion",
                    file=sys.stderr,
                )
                return None
            if not chunk:
                break
            buf += chunk
    if not buf.strip():
        print("browser daemon died mid-command — see its driver.log", file=sys.stderr)
        return None
    return json.loads(buf)
