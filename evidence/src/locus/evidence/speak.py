"""How a run speaks.

A run's whole conversation with its caller is one opening line — `started <stamp>  <address>`, the stamp its UTC anchor (clock.py owns the spelling) — printed the instant the address exists. Everything else it says lands at that address: narration, worker losses, heartbeats, and last its result, so a log that stops short of one is a run that died. Every line is one whole write, and the file is readable while the run still fills it. A span the caller waits minutes on beats on a clock, because a span that says nothing until it ends looks like one that will never end. A run that ends clean with a small say repeats the whole of it inline after the address, so the hop to the file is only ever paid for size. A verb's old logs are rotated out on a byte budget that never touches the log just announced or one still being spoken into.
"""

import fcntl
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .clock import utc_stamp


def started(path) -> None:
    """A run's opening line, printed the instant its log has an address and before the work that fills it. Flushed: a caller reading through a pipe is block-buffered, and the address is only useful before the work runs."""
    print(f"started {utc_stamp()}  {path}", flush=True)


def say(message: str) -> None:
    """One line of a run's narration, in one write. Concurrent writers share the run's stream — worker threads, a pulse, a subprocess on the same descriptor — and a line split across two writes is one another writer can tear."""
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


BEAT_S = 10.0


@contextmanager
def beating(what: str, detail=None, every_s: float | None = None):
    """A long span's pulse, off a clock and not off the work: a span wedged on a hung socket has nothing arriving to trigger a line. `what` names what is happening; `detail`, when given, is called per beat for how far it has got, so standing detail under climbing seconds reads as a wedge."""
    every_s = every_s if every_s is not None else BEAT_S
    opened = time.perf_counter()
    done = threading.Event()

    def beat() -> None:
        while not done.wait(every_s):
            said = f"{what}: {detail()}" if detail is not None else what
            say(f"{said} — {time.perf_counter() - opened:.0f}s")

    pulse = threading.Thread(target=beat, daemon=True)
    pulse.start()
    try:
        yield
    finally:
        done.set()
        pulse.join()


OUTPUT_HISTORY_BYTES = 128 * 1024 * 1024


def open_log(outputs: Path, verb: str) -> Path:
    """This run's own log, at the UTC moment it opened and the verb that opened it — reserved before the work so the run can name it up front. Created empty and exclusively, so no two runs land on one path: a deployment is read concurrently, and each run must find its own log where it was told."""
    outputs.mkdir(parents=True, exist_ok=True)
    moment = datetime.now(timezone.utc)
    while True:
        path = outputs / f"{moment.strftime('%Y-%m-%dT%H-%M-%S-%fZ')}_{verb}.log"
        try:
            path.open("x").close()
            return path
        except FileExistsError:
            moment += timedelta(microseconds=1)


@contextmanager
def saying_into(path: Path):
    """Everything a run says, into its own log — Python's stream objects and the process's file descriptors both. A library holding the sys.stderr it captured at import writes past a rebound stream but not past a redirected descriptor; a subprocess inherits the descriptors and knows nothing of either stream object. One file object serves as both streams, so a beat and a worker's line never tear each other, and it is line-buffered so the file can be watched while it fills.

    The streams are restored on the way out, so whatever ends the run reaches the caller rather than the log it just closed. It is said into the log first — a refusal in its own words, anything else as its traceback — or a run that stopped after minutes of work leaves a file that merely stops.

    The log is held under an exclusive lock for as long as the run speaks; `finished` is the lock's read side."""
    saying = path.open("w", buffering=1)
    fcntl.flock(saying.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    saved_streams = sys.stdout, sys.stderr
    saved_fds = os.dup(1), os.dup(2)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.dup2(saying.fileno(), 1)
        os.dup2(saying.fileno(), 2)
        sys.stdout = sys.stderr = saying
        yield
    except SystemExit as refusal:
        if isinstance(refusal.code, str):
            print(refusal.code)
        raise
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        sys.stdout, sys.stderr = saved_streams
        for saved, target in zip(saved_fds, (1, 2)):
            os.dup2(saved, target)
            os.close(saved)
        saying.close()


@contextmanager
def speaking_into(log: Path):
    """A run speaking into a log whose address its caller already holds: everything the run says captured into it, a small say echoed inline once it ends clean."""
    with saying_into(log):
        yield log
    echo(log)


@contextmanager
def speaking(outputs: Path, verb: str):
    """A run that speaks into its own log under `outputs`: the log opened and announced before the work, everything the run says captured into it, a small say echoed inline once it ends clean, and the verb's history rotated on the way out."""
    log = open_log(outputs, verb)
    started(log)
    try:
        with speaking_into(log):
            yield log
    finally:
        trim_outputs(log)


ECHO_BYTES = 5 * 1024


def echo(log: Path) -> None:
    """A small say repeated whole on the caller's stdout, byte-exact — all or nothing, never truncated. The log stays the artifact; this is a transport copy, so the hop to the file is only ever paid past the budget. A run that died echoes nothing: its refusal or traceback already reaches the caller through the raise.

    A caller that took the address and closed the pipe (`| head -1`) declined the copy: the write lands nowhere and the run still ends clean — a courtesy is never an error. Stdout is then parked on devnull so the interpreter's own exit flush stays quiet too."""
    if log.stat().st_size > ECHO_BYTES:
        return
    try:
        sys.stdout.write(log.read_text())
        sys.stdout.flush()
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        finally:
            os.close(devnull)


def trim_outputs(path: Path) -> None:
    """A verb's history dropped oldest-first past OUTPUT_HISTORY_BYTES, which bounds disk. The budget is in bytes because a log's size follows what its run answered rather than which verb answered: one listing is a handful of slices and the next is every slice a busy snippet opened this month. Concurrent runs trim the same history, so a listed log may already be gone by the time this one weighs it — disk it no longer occupies and nothing left to drop.

    Two logs are never dropped, whatever the budget says: this run's own, whose caller holds the address and has not read it yet, and any log another run is still speaking into."""
    _, _, name = path.stem.partition("_")
    kept = 0
    for older in sorted(path.parent.glob(f"*_{name}{path.suffix}"), reverse=True):
        if older == path:
            continue
        try:
            kept += older.stat().st_size
        except FileNotFoundError:
            continue
        if kept > OUTPUT_HISTORY_BYTES and finished(older):
            older.unlink(missing_ok=True)


def finished(log: Path) -> bool:
    """Whether no run is still speaking into this log: a run holds its log locked while it speaks, so a lock this can take is one nobody holds."""
    try:
        holding = os.open(log, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(holding, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False
    finally:
        os.close(holding)
