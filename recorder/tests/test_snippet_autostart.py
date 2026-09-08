"""The shipped script-tag artifact — dist/locus-recorder.min.js, the canonical one-tag install — executed in a real Chromium. The tag's designed execution signal is `window.LocusRecorder`: the bundle binds to its own tag via document.currentScript, so identification is certain and the global is planted whenever the bundle runs as a script tag — its presence discriminates "loaded and ran" from "not loaded as a tag at all". A malformed id refuses loudly on the console while the claim stands, and a tag with no ?id= refuses loudly too, naming the missing param. Capture is deliberately NOT asserted, and the recorder's gates are not defeated: this harness serves from 127.0.0.1, so the local-environment gate suppresses recording — production-correct behavior for an automated local page, leaving the global present but empty. Capture correctness is owned by the recorder↔evidence contract suite (test_recorder_integration.py beside this file) and the store roundtrip."""

import pytest
from _harness import chromium_page, serve_dir

# Registered in document order ahead of the snippet tag, so the traps are in
# place before the bundle runs. autostart() is fire-and-forget, so a throw past
# its first await surfaces as an unhandledrejection, not an error — catch both.
ERROR_TRAPS = (
    "<script>window.__errors=[];"
    "addEventListener('error',(e)=>window.__errors.push(String(e.message)));"
    "addEventListener('unhandledrejection',"
    "(e)=>window.__errors.push('unhandledrejection: '+String(e.reason)));"
    "</script>"
)


def host_page(snippet_tag):
    return (
        "<!DOCTYPE html><html><head><title>Snippet Host</title></head>"
        f"<body><p>hello</p>{ERROR_TRAPS}{snippet_tag}</body></html>"
    )


@pytest.fixture(scope="session")
def site(tmp_path_factory, recorder_dist):
    root = tmp_path_factory.mktemp("snippet_site")
    (root / "locus-recorder.min.js").write_bytes(
        (recorder_dist / "locus-recorder.min.js").read_bytes()
    )
    tag = '<script async src="/locus-recorder.min.js?id={}"></script>'
    (root / "valid.html").write_text(host_page(tag.format("oss123")))
    (root / "malformed.html").write_text(host_page(tag.format("BAD-ID")))
    (root / "unclaimed.html").write_text(
        host_page('<script async src="/locus-recorder.min.js"></script>')
    )
    server, origin = serve_dir(root)
    yield origin
    server.shutdown()


@pytest.fixture()
def page():
    with chromium_page() as page:
        yield page


def test_valid_id_claims_the_tag_and_gates_capture(page, site):
    uploads = []
    page.on(
        "request",
        lambda r: (
            uploads.append(r.url)
            if "/chunks/" in r.url or "/telemetry/" in r.url
            else None
        ),
    )
    page.goto(f"{site}/valid.html")
    page.wait_for_function("typeof window.LocusRecorder !== 'undefined'")
    page.wait_for_timeout(1_000)
    state = page.evaluate(
        "() => ({ keys: Object.keys(window.LocusRecorder),         errors: window.__errors })"
    )
    assert state["keys"] == [], (
        "gated off, the global carries no flush — present but empty"
    )
    assert state["errors"] == []
    assert uploads == []


def test_malformed_id_refuses_loudly_without_starting(page, site):
    uploads = []
    page.on(
        "request",
        lambda r: (
            uploads.append(r.url)
            if "/chunks/" in r.url or "/telemetry/" in r.url
            else None
        ),
    )
    with page.expect_console_message(
        lambda m: m.type == "error" and "locus-recorder" in m.text
    ) as info:
        page.goto(f"{site}/malformed.html")
    assert "does not match /^[a-z0-9]{3,64}$/" in info.value.text
    assert "capturing nothing" in info.value.text
    state = page.evaluate(
        "() => ({ claimed: typeof window.LocusRecorder !== 'undefined',"
        "         keys: Object.keys(window.LocusRecorder ?? {}),"
        "         errors: window.__errors })"
    )
    assert state["claimed"], "the tag is claimed before the id is judged"
    assert state["keys"] == []
    assert state["errors"] == [], "the refusal is a console error, not a throw"
    assert uploads == []


def test_tag_without_id_claims_the_tag_and_refuses_loudly(page, site):
    with page.expect_console_message(
        lambda m: m.type == "error" and "locus-recorder" in m.text
    ) as info:
        page.goto(f"{site}/unclaimed.html")
    assert "?id=" in info.value.text
    assert "Not starting" in info.value.text
    state = page.evaluate(
        "() => ({ claimed: typeof window.LocusRecorder !== 'undefined',"
        "         keys: Object.keys(window.LocusRecorder ?? {}),"
        "         errors: window.__errors })"
    )
    assert state["claimed"], (
        "currentScript identifies the tag with certainty — id or not, the bundle loaded and ran"
    )
    assert state["keys"] == []
    assert state["errors"] == [], "the refusal is a console error, not a throw"
