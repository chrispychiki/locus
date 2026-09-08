"""Shared test scaffolding: hand-built rrweb events, the store's chunk shape and an in-memory stand-in for it, real distilled dbs, a render stub, and the fixtures both suites run under. Real rendering is covered in test_render and the tests_live suite; here the browser is stubbed so the control flow itself is pinned.

Both packages' suites import from here: the analysis suite puts this directory on its import path (analysis/tests/conftest.py) so the fixtures and builders have exactly one home.
"""

import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from locus.evidence.db import connect
from locus.evidence.hydrate import hydrate
from locus.evidence.render import ScreenshotResult
from locus.evidence.rrweb_constants import (
    EventType,
    IncrementalSource,
    MouseInteractions,
    NodeType,
)
from locus.evidence.slices import materialize_slices, require_bun
from locus.evidence.store import S3Store

FIXTURE = Path(__file__).parent / "fixtures" / "demo_recording.json"
DISTILL = Path(__file__).parent.parent / "distill" / "distill.js"

URL = "https://x.test/page"
REPO = Path(__file__).resolve().parents[2]
RECORDER = REPO / "recorder"


def bun_install(package):
    """Install a bun package's modules if they aren't there — a clone carries the lockfiles, not the modules, so a suite installs what it is about to run rather than erroring on a fresh clone."""
    require_bun()
    if not (package / "node_modules").is_dir():
        subprocess.run(["bun", "install", "--frozen-lockfile"], cwd=package, check=True)


def build_recorder(*scripts):
    """Build the named recorder bundles for a browser suite to serve. The recorder's package.json owns how each one is built; a suite names only which ones it needs."""
    bun_install(RECORDER)
    for script in scripts:
        subprocess.run(
            ["bun", "run", script], cwd=RECORDER, check=True, capture_output=True
        )


@pytest.fixture(scope="session")
def recorder_dist():
    """The recorder's built bundles, ready to serve."""
    return RECORDER / "dist"


def meta(ts, href=URL):
    return {
        "type": EventType.Meta,
        "timestamp": ts,
        "data": {"href": href, "width": 1280, "height": 800},
    }


def full_snapshot(ts):
    return {
        "type": EventType.FullSnapshot,
        "timestamp": ts,
        "data": {
            "node": {
                "type": NodeType.Document,
                "id": 1,
                "childNodes": [
                    {
                        "type": NodeType.Element,
                        "tagName": "html",
                        "id": 2,
                        "attributes": {},
                        "childNodes": [
                            {
                                "type": NodeType.Element,
                                "tagName": "body",
                                "id": 3,
                                "attributes": {},
                                "childNodes": [
                                    {
                                        "type": NodeType.Text,
                                        "id": 4,
                                        "textContent": "hello",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        },
    }


def text_mutation(ts, node_id, value):
    return {
        "type": EventType.IncrementalSnapshot,
        "timestamp": ts,
        "data": {
            "source": IncrementalSource.Mutation,
            "texts": [{"id": node_id, "value": value}],
            "attributes": [],
            "removes": [],
            "adds": [],
        },
    }


def click(ts, node_id):
    return {
        "type": EventType.IncrementalSnapshot,
        "timestamp": ts,
        "data": {
            "source": IncrementalSource.MouseInteraction,
            "type": MouseInteractions.Click,
            "id": node_id,
            "x": 5,
            "y": 5,
        },
    }


def hidden(ts, url=URL):
    return {"type": EventType.PageHidden, "timestamp": ts, "data": {"url": url}}


def env(event, recorder_slice):
    return {**event, "_envelope": {"recorder_slice": recorder_slice}}


def read_recording(path: str | Path) -> tuple[str, list[dict]]:
    """Load a committed recording fixture into (visitor, events). Fixture loading lives in the suite — the only read path into a db is the store."""
    return Path(path).stem, json.loads(Path(path).read_text())["events"]


def stamped(events: list[dict], **envelope) -> list[dict]:
    """Hand-built events as the recorder actually emits them: the page context opens a head slice at start(), its record-start Meta fills that one, and each Meta after it opens a new slice. Every event carries a slice id, so a test that skips this is testing an input the product cannot receive."""
    current = f"{events[0]['timestamp'] - 1:014d}-head"
    filled = False
    out = []
    for event in events:
        if event["type"] == EventType.Meta:
            if filled:
                current = f"{event['timestamp']:014d}-{len(out):04d}"
            filled = True
        out.append({**event, "_envelope": {**envelope, "recorder_slice": current}})
    return out


UA_DESKTOP = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
VISITOR = "0f6e2a7c-9b1d-4e5f-8a3b-2c4d6e8f0a1b"
SNIPPET = "snip01"
T0 = 1_749_600_000_000


def counted(event: dict, seq: int) -> dict:
    """The recorder's ordering stamp: `{timestamp}{seq:06d}`, minted per event."""
    return {**event, "counter": f"{event['timestamp']}{seq:06d}"}


def chunk_blob(
    slice_id, events, user_agent=UA_DESKTOP, errors=(), visitor=VISITOR, envelope=None
) -> bytes:
    """One chunk exactly as the recorder posts it: gzipped JSON, self-describing."""
    return gzip.compress(
        json.dumps(
            {
                "visitorId": visitor,
                "sliceId": slice_id,
                "recorderVersion": "locus-recorder/0.1.0",
                "envelope": {"userAgent": user_agent, **(envelope or {})},
                "events": events,
                "errors": list(errors),
            }
        ).encode()
    )


def chunk_at(visitor, ms, snippet=SNIPPET, events=None):
    """A (key, blob) pair placed where the store's key layout puts it. The slice id carries the
    recorder's own grammar — 14 padded ms, then a 4-char base36 disambiguator — because the loader
    parses keys by that shape, and a fixture that minted something else would be testing an input
    the store's write gate cannot produce."""
    from locus.evidence.chunk import slice_date

    disambiguator = "".join(c for c in visitor.lower() if c.isalnum())[:4].ljust(4, "0")
    slice_id = f"{ms:014d}-{disambiguator}"
    events = events or [counted(meta(ms), 1)]
    return (
        (f"{snippet}/{slice_date(slice_id)}/{visitor}/{slice_id}/{ms}000001.json.gz"),
        chunk_blob(slice_id, events, visitor=visitor),
    )


class FakeStore:
    """The store read contract (S3Store's surface) over an in-memory {key: blob} dict. The ETag
    mirrors R2's — the body's MD5 — so a re-uploaded blob changes its ETag exactly as R2 would."""

    def __init__(self, blobs, uploaded=None):
        self.blobs = blobs
        self.uploaded = uploaded or {}

    def _etag(self, key):
        return hashlib.md5(self.blobs[key]).hexdigest()

    def shard_rows(self, prefix=""):
        # One shard, listed in key order — the real store's per-shard guarantee,
        # which is what a consumer folding keys into groups leans on.
        for row in self.objects(prefix):
            yield prefix, row

    def objects(self, prefix=""):
        for key in sorted(self.blobs):
            if key.startswith(prefix):
                yield (
                    key,
                    len(self.blobs[key]),
                    self.uploaded.get(key, 0),
                    self._etag(key),
                )

    def keys(self, prefix=""):
        for key, _size, _uploaded, _etag in self.objects(prefix):
            yield key

    def get(self, key):
        return self.blobs[key]

    # The real load loop, run against this fake's objects()/get() — the windowed
    # fetch, drain order, and stats ticking are S3Store's own, never a test copy.
    events_under = S3Store.events_under


class FakeVerify:
    """Stands in for the Cloudflare token-verify endpoint (urllib.request.urlopen), counting its calls."""

    def __init__(self, body):
        self.body = body
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append((request.full_url, request.headers.get("Authorization")))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.body).encode()


def build_distilled_db(base: Path) -> Path:
    db_path = base / "events.db"
    visitor_id, events = read_recording(FIXTURE)
    conn = connect(db_path)
    hydrate(conn, visitor_id, events)
    materialize_slices(conn, visitor_id)
    conn.close()
    subprocess.run(["bun", str(DISTILL), str(db_path)], check=True, capture_output=True)
    return db_path


SECOND_VISITOR = "second-visitor-0000-0000"
SECOND_VISITOR_SHIFT_MS = 3_600_000


def build_two_visitor_db(base: Path) -> Path:
    """The demo recording twice: once as recorded, once shifted an hour later
    under a second visitor — two visitors whose slices never overlap in time,
    the legal group-analysis shape."""
    db_path = base / "events.db"
    visitor_id, events = read_recording(FIXTURE)
    conn = connect(db_path)
    hydrate(conn, visitor_id, events)
    materialize_slices(conn, visitor_id)
    shifted = [
        {**e, "timestamp": e["timestamp"] + SECOND_VISITOR_SHIFT_MS} for e in events
    ]
    hydrate(conn, SECOND_VISITOR, shifted)
    materialize_slices(conn, SECOND_VISITOR)
    conn.close()
    subprocess.run(["bun", str(DISTILL), str(db_path)], check=True, capture_output=True)
    return db_path


def best_slice(conn) -> int:
    return conn.execute(
        "SELECT id FROM slices WHERE status='replayable' ORDER BY n_events DESC LIMIT 1"
    ).fetchone()["id"]


def slice_bounds(conn, slice_id: int) -> tuple[int, int]:
    row = conn.execute(
        "SELECT MIN(timestamp) lo, MAX(timestamp) hi FROM events WHERE slice_id = ?",
        (slice_id,),
    ).fetchone()
    return row["lo"], row["hi"]


class FakeRenderSession:
    """Stands in for RenderSession: writes a placeholder PNG per capture and
    records the (label, timestamp) addresses, no browser."""

    def __init__(self, conn, slices, out_dir):
        self.slices = dict(slices)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.captured = []
        self.open = False
        self.releases = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def release(self):
        self.releases.append(len(self.captured))
        self.open = False

    def capture(self, label, ts):
        assert label in self.slices, f"capture addressed an unknown label {label!r}"
        self.open = True
        path = self.out_dir / f"{label}_screenshot_{ts}.png"
        path.write_bytes(b"\x89PNG fake")
        self.captured.append((label, ts))
        return ScreenshotResult(timestamp=ts, path=str(path), faults={})


@pytest.fixture(autouse=True)
def contained(tmp_path):
    """Every test runs from its own directory, with its own environment, restored after.

    The deployment root and the credentials the CLI self-serves are both resolved by walking cwd
    upward, so a test left standing in the clone reads the machine's real `.env` files and writes
    its outputs and ledgers into the working tree. Running from `tmp_path` puts the real tree out of
    reach structurally: a test that wants a deployment points the code under test at its own fixture, and
    one that forgets fails loudly instead of quietly borrowing the machine's. os.environ is snapshotted and
    restored, so what a test exports — directly, or through the CLI's setdefault self-serve — dies
    with it rather than coloring every later test in the worker.
    """
    saved_cwd = Path.cwd()
    saved_env = dict(os.environ)
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(saved_cwd)
        os.environ.clear()
        os.environ.update(saved_env)


@pytest.fixture(scope="session")
def distilled_db(tmp_path_factory):
    return build_distilled_db(tmp_path_factory.mktemp("distilled"))


@pytest.fixture(scope="session")
def two_visitor_db(tmp_path_factory):
    return build_two_visitor_db(tmp_path_factory.mktemp("two_visitor"))
