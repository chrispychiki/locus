"""The live capture→store→replay chain: the recorder on a real page uploads through
the deployed store worker into R2, and the evidence package loads it back through the same
path `locus load` drives — the incremental load, then the derivation pass — into a
replayable db whose screenshots render at the driven moments. This crosses the deployed
worker + R2 that the hermetic suite stubs, which is why it lives in its own suite;
the conftest beside it owns the invocation and the configured-deployment check. The
worker URL and the store resolve from the deployment itself (deployment.py), exactly
as the CLI resolves them.

Every run mints its own snippet id under a shared test namespace. The store's key
layout leads with the snippet id, so a run's objects sit under a prefix no other run
lists, writes, or deletes — concurrent suites cannot destroy each other's chunks. A
run deletes its own prefix when it finishes; one that dies without deleting leaves
objects behind, so each run first reaps whatever in the namespace has sat in the
bucket past the stale horizon, which no live run's objects ever reach.
"""

import time
import uuid
from pathlib import Path

from _harness import chromium_page, serve_dir
from locus.evidence.db import connect
from locus.evidence.deployment import store as deployment_store
from locus.evidence.deployment import worker_url
from locus.evidence.derive import finish_derivations
from locus.evidence.load import load_chunks
from locus.evidence.render import RenderSession

SITE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "site"

# The namespace this test's runs live in, inside the snippet-id shape the worker admits
# (store/src/keys.js). Every run appends its own suffix.
NAMESPACE = "roundtrip"

# How long an object under the namespace must have sat in the bucket before a later run
# treats it as a dead run's litter. A run lasts under a minute, so the horizon is orders of
# magnitude beyond any live run — the reap cannot reach a concurrent run's objects.
STALE_MS = 60 * 60 * 1000


def _delete(store, keys) -> None:
    keys = list(keys)
    for batch in (keys[i : i + 1000] for i in range(0, len(keys), 1000)):
        store.client.delete_objects(
            Bucket=store.bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
        )


def _reap_litter(store) -> None:
    """Objects a crashed run left under the namespace. Staleness is the store's own
    arrival time from the LIST, not the recording clock in the key — the bucket's
    attestation of how long the object has been sitting there."""
    now = time.time() * 1000
    _delete(
        store,
        (
            key
            for key, _size, uploaded_ms, _etag in store.objects(prefix=NAMESPACE)
            if now - uploaded_ms > STALE_MS
        ),
    )


def _snippet_js(worker: str, snippet: str) -> bytes:
    """The page's install: the library bundle, started with storeSink aimed at the
    deployed worker, so the upload crosses the real ingest gate on the same URL
    contract the script-tag bundle uses. That bundle configures itself entirely
    off its own tag and accepts no options, and its local-environment gate
    suppresses capture on the loopback site this test serves — driving real ingest
    from a local page is the library build's job."""
    return (
        f"(function () {{\n"
        f'  import("/locus-recorder.lib.js").then(function (LocusRecorder) {{\n'
        f"    return LocusRecorder.start({{\n"
        f"      sink: LocusRecorder.storeSink({{\n"
        f'        origin: "{worker}",\n'
        f'        snippetId: "{snippet}",\n'
        f"      }}).sink,\n"
        f"      recordLocalEnvironment: true,\n"
        f"      botDetection: false,\n"
        f"      intervalMs: 1000,\n"
        f"    }});\n"
        f"  }}).then(function (recorder) {{ window.__rec = recorder; }});\n"
        f"}})();\n"
    ).encode()


def _drive(origin: str) -> dict:
    moments = {}
    with chromium_page() as page:
        page.goto(f"{origin}/index.html")
        page.wait_for_function("() => window.__rec")
        page.wait_for_timeout(2_000)
        moments["landing"] = page.evaluate("() => Date.now()")
        page.click("#view-1")
        page.wait_for_timeout(2_000)
        moments["product"] = page.evaluate("() => Date.now()")
        page.fill("#qty", "2")
        page.click("#add")
        page.wait_for_timeout(1_000)
        moments["added"] = page.evaluate("() => Date.now()")
        page.wait_for_timeout(3_000)
        for _ in range(5):
            page.evaluate("() => window.__rec.flush()")
            page.wait_for_timeout(300)
    return moments


def test_store_roundtrip(tmp_path, recorder_dist):
    worker = worker_url()
    store = deployment_store()
    snippet = f"{NAMESPACE}{uuid.uuid4().hex[:8]}"
    prefix = f"{snippet}/"

    _reap_litter(store)

    try:
        server, origin = serve_dir(
            SITE,
            {
                "/locus-recorder.lib.js": (
                    (recorder_dist / "locus-recorder.lib.js").read_bytes(),
                    "text/javascript",
                ),
                "/snippet.js": (_snippet_js(worker, snippet), "text/javascript"),
            },
        )
        try:
            moments = _drive(origin)
        finally:
            server.shutdown()

        keys, previous = [], -1
        for _ in range(20):
            keys = list(store.keys(prefix=prefix))
            if keys and len(keys) == previous:
                break
            previous = len(keys)
            time.sleep(3)
        assert keys, (
            "no chunks landed in R2 — the recorder→worker→R2 write path is broken"
        )

        visitors = sorted({key.split("/")[2] for key in keys})
        assert len(visitors) == 1, visitors
        db = tmp_path / "events.db"
        conn = connect(db)
        loaded = load_chunks(conn, store, prefix)
        assert loaded["inserted"], (
            "chunks were listed but the load landed no events — the store→db read path is broken"
        )
        finish_derivations(conn, str(db))

        slices = conn.execute(
            "SELECT id, start_ts, end_ts FROM slices WHERE status='replayable' ORDER BY start_ts"
        ).fetchall()
        assert slices, "the drive produced no replayable slice"
        homes = {}
        for name, ts in moments.items():
            home = next(
                (row["id"] for row in slices if row["start_ts"] <= ts <= row["end_ts"]),
                None,
            )
            assert home is not None, f"{name} @ {ts} falls in no slice"
            homes[name] = home
        labels = {sid: f"S{k}" for k, sid in enumerate(sorted(set(homes.values())), 1)}
        with RenderSession(
            conn,
            {label: sid for sid, label in labels.items()},
            tmp_path / "screenshots",
        ) as session:
            for name, ts in moments.items():
                screenshot = session.capture(labels[homes[name]], ts)
                assert "failedImages" not in screenshot.faults, name
    finally:
        _delete(store, store.keys(prefix=prefix))
