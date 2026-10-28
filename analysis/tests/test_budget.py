"""Analysis pricing.

The projection itself is covered by test_analyze.py. Pricing runs against an injected CostModel so every economic constant is the test's own; the composition underneath (compose_window, screenshot_moments, the walls) is real. The backends' own cost models run here on the offline count (_analysis_support.install_offline_count): the server's own count they call is proven against the live server in the mlx suites, so what is tested here is what each cost model hands it.
"""

import itertools
import json as _json

import pytest
from _analysis_support import GEMINI, install_offline_count, offline_image_tokens
from locus.analysis import budget
from locus.analysis.budget import CostModel, Count, price_payload, window_payload
from locus.analysis.select_screenshots import screenshot_moments
from locus.analysis.window import resolve_window, slice_pieces
from locus.evidence.db import connect
from locus.evidence.hydrate import pack_raw
from locus.evidence.kinds import Kind
from locus.evidence.rrweb_constants import EventType, IncrementalSource

LOCAL_CARD = "qwen3.6-35b-a3b-6bit"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    install_offline_count(monkeypatch)


SYSTEM = "system prompt for the pricing tests"
SITE = "a demo site"
TASK = "the entire question the caller sends last"
TOOL = {"name": "a_tool", "description": "four words of declaration"}


KIND_TYPES = {
    "Meta": EventType.Meta,
    "FullSnapshot": EventType.FullSnapshot,
    "PageLoad": EventType.PageLoad,
    "Click": EventType.IncrementalSnapshot,
}


def _insert_slice(conn, sid, visitor, start, end, events, status="replayable"):
    conn.execute(
        "INSERT INTO slices (id, visitor_id, recorder_slice, start_ts, end_ts, "
        "n_events, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (sid, visitor, f"s{sid}", start, end, len(events), status),
    )
    for ts, kind, url, script in events:
        typ = KIND_TYPES[kind]
        raw = {"type": typ, "timestamp": ts, "data": {}}
        if kind == "Meta":
            raw["data"] = {"href": url, "width": 390, "height": 663}
        if kind == "FullSnapshot":
            raw["data"] = {"node": {"id": 1}}
        if kind == "PageLoad":
            raw["data"] = {"url": url, "title": "t", "referrer": ""}
        conn.execute(
            "INSERT INTO events (visitor_id, timestamp, type, raw_json, "
            "content_hash, slice_id, type_str, url, script_version, md) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                visitor,
                ts,
                typ,
                pack_raw(_json.dumps(raw)),
                f"h{sid}_{ts}_{kind}",
                sid,
                kind,
                url,
                script,
                "# a projected page" if kind == "FullSnapshot" else None,
            ),
        )


def _plain_slice(conn, sid, visitor, start, url):
    """A 5s slice: the covering Meta+FullSnapshot pair, its own page-load PageLoad, two clicks."""
    rec = "locus-recorder/1.0"
    _insert_slice(
        conn,
        sid,
        visitor,
        start,
        start + 5_000,
        [
            (start, "Meta", url, rec),
            (start + 10, "FullSnapshot", None, rec),
            (start + 500, "PageLoad", url, rec),
            (start + 2_000, "Click", None, rec),
            (start + 4_500, "Click", None, rec),
        ],
    )


def _resize(conn, sid, ts, width, height):
    raw = {
        "type": EventType.IncrementalSnapshot,
        "timestamp": ts,
        "data": {
            "source": IncrementalSource.ViewportResize,
            "width": width,
            "height": height,
        },
    }
    conn.execute(
        "INSERT INTO events (visitor_id, timestamp, type, raw_json, content_hash, "
        "slice_id, type_str) VALUES ('v1', ?, ?, ?, ?, ?, ?)",
        (
            ts,
            raw["type"],
            pack_raw(_json.dumps(raw)),
            f"resize{sid}_{ts}",
            sid,
            Kind.VIEWPORT_RESIZE,
        ),
    )


@pytest.fixture
def conn(tmp_path):
    conn = connect(tmp_path / "events.db")
    _plain_slice(conn, 1, "v1", 1_000, "https://x.test/a")
    _plain_slice(conn, 2, "v1", 10_000, "https://x.test/b")
    _plain_slice(conn, 3, "v1", 20_000, "https://x.test/c")
    _plain_slice(conn, 4, "v2", 1_000, "https://x.test/other")
    conn.commit()
    return conn


def _words(texts):
    return sum(len(t.split()) for t in texts)


def _model(screenshot=10):
    def one(conn, window, contract, moments, site_contexts):
        system_prompt, task, _ = contract
        texts, n_events = window_payload(
            conn,
            window,
            system_prompt=system_prompt,
            task=task,
            site_contexts=site_contexts,
            moments=moments,
        )
        n_screenshots = sum(len(timestamps) for timestamps in moments.values())
        return Count(_words(texts), n_screenshots * screenshot, n_events)

    return CostModel(
        lambda conn, sets, site_contexts: [
            one(conn, *measured, site_contexts) for measured in sets
        ]
    )


def _prompts(labels):
    return SYSTEM, TASK, [TOOL]


def _cost(conn, model, sids, ws=None, we=None):
    window = resolve_window(conn, sids, ws, we)
    (counted,) = model.count(
        conn,
        [(window, _prompts(window.labels), screenshot_moments(conn, window))],
        {"site": SITE},
    )
    return counted.text_tokens + counted.screenshot_tokens


def _price(conn, sids, budget_tokens, model=None, **kwargs):
    return price_payload(
        conn,
        resolve_window(conn, sids),
        model="injected",
        prompts=_prompts,
        site_contexts={"site": SITE},
        cost_model=model or _model(),
        context_tokens=budget_tokens,
        headroom=1.0,
        **kwargs,
    )


def test_image_geometry_realizes_the_per_image_token_ceiling():
    model = "any"
    assert budget.image_geometry(model, 390, 663, 600) == (390, 663), (
        "a screenshot under the ceiling passes untouched — never upscaled"
    )
    width, height = budget.image_geometry(model, 1920, 921, 600)
    assert (width, height) != (1920, 921)
    assert offline_image_tokens(model, width, height) <= 600, (
        "the ceiling is a hard bound on what the sent image costs"
    )
    assert offline_image_tokens(model, width, height) > 540, (
        "the resize lands near the ceiling, not far under it"
    )
    assert abs(width / height - 1920 / 921) < 0.05, "aspect preserved"


def test_a_ceiling_below_the_smallest_image_fails_loud(monkeypatch):
    from locus.analysis.mlx import count

    monkeypatch.setattr(count, "image_tokens", lambda model, w, h: 66 + w * h // 1024)
    with pytest.raises(ValueError, match="ceiling"):
        budget.image_geometry("any", 1920, 921, 10)


def _local(conn, sids, moments, tools=(TOOL,)):
    window = resolve_window(conn, sids)
    model, _, _ = budget.cost_model_for(LOCAL_CARD)
    (counted,) = model.count(
        conn, [(window, (SYSTEM, TASK, list(tools)), moments)], {"site": SITE}
    )
    return counted


def test_a_local_screenshot_is_priced_at_its_moments_viewport_under_the_ceiling(conn):
    """The replayer sizes its frame at every Meta and ViewportResize it applies, so a visitor who resizes mid-slice changes the size of every screenshot after — and each is then sent shrunk under the card's activity ceiling, so that is the size the server counts."""
    from locus.analysis.model.cards import max_image_tokens

    _resize(conn, 1, 3_000, 1366, 917)
    ceiling = max_image_tokens(LOCAL_CARD)["activity_screenshots"]
    counted = _local(conn, [1], {1: [2_000, 4_500]})
    assert counted.screenshot_tokens == offline_image_tokens(
        None, 390, 663
    ) + offline_image_tokens(None, *budget.image_geometry("any", 1366, 917, ceiling))


def test_a_local_price_is_the_request_a_run_sends(conn):
    """Composed into the real client, the counted request carries what the tally composes — system prompt, window, question — and the tool declarations the run's request carries."""
    moments = {1: [2_000]}
    texts, _ = window_payload(
        conn,
        resolve_window(conn, [1]),
        system_prompt=SYSTEM,
        task=TASK,
        site_contexts={"site": SITE},
        moments=moments,
    )
    with_tool = _local(conn, [1], moments)
    assert with_tool.text_tokens == _words(texts) + len(
        _json.dumps([{"type": "function", "function": TOOL}]).split()
    )
    assert _local(conn, [1], moments, tools=()).text_tokens == _words(texts)


def test_a_moment_before_any_viewport_is_unpriceable(conn):
    with pytest.raises(ValueError, match="no viewport"):
        _local(conn, [1], {1: [999]})


def test_gemini_text_is_counted_by_counttokens_and_screenshots_at_their_tier(
    conn, monkeypatch
):
    from locus.analysis.model import gemini
    from locus.analysis.model.cards import media_resolution

    monkeypatch.setattr(gemini, "count_text_tokens", lambda model, texts: _words(texts))
    model, _, _ = budget.cost_model_for(GEMINI)
    window = resolve_window(conn, [1])
    moments = screenshot_moments(conn, window)
    (counted,) = model.count(
        conn, [(window, (SYSTEM, TASK, [TOOL]), moments)], {"site": SITE}
    )
    texts, n_events = window_payload(
        conn,
        window,
        system_prompt=SYSTEM,
        task=TASK,
        site_contexts={"site": SITE},
        moments=moments,
    )
    tiers = media_resolution(GEMINI)
    tier = tiers.get("activity_screenshots", tiers.get("pulled_screenshots"))
    assert counted == Count(
        _words(texts),
        len(moments[1]) * gemini.image_tier_tokens(tier),
        n_events,
    )


def test_gemini_counts_every_measured_set_at_once(conn, monkeypatch):
    """The whole window and each slice standalone go to countTokens together, one call per set — so a report costs about one round trip, whatever the sets — and each count lands on its own set."""
    import threading

    from locus.analysis.model import gemini

    sets = 4
    together = threading.Barrier(sets, timeout=10)

    def count_text_tokens(model, texts):
        together.wait()
        return _words(texts)

    monkeypatch.setattr(gemini, "count_text_tokens", count_text_tokens)
    model, context, _ = budget.cost_model_for(GEMINI)
    price = price_payload(
        conn,
        resolve_window(conn, [1, 2, 3]),
        model=GEMINI,
        prompts=_prompts,
        site_contexts={"site": SITE},
        cost_model=model,
        context_tokens=context,
        headroom=1.0,
    )
    from locus.analysis.model.cards import media_resolution

    tiers = media_resolution(GEMINI)
    tier = gemini.image_tier_tokens(
        tiers.get("activity_screenshots", tiers.get("pulled_screenshots"))
    )
    expected = _model(screenshot=tier)
    assert price["turn1"]["total_tokens"] == _cost(conn, expected, [1, 2, 3])
    for sid, entry in zip([1, 2, 3], price["per_slice"], strict=True):
        assert entry["standalone_tokens"] == _cost(conn, expected, [sid]), (
            "each slice's count is its own"
        )


def test_gemini_counts_fail_loud_when_any_call_fails(monkeypatch):
    from locus.analysis.model import gemini

    def count_text_tokens(model, texts):
        if texts == ["b"]:
            raise RuntimeError("quota")
        return len(texts)

    monkeypatch.setattr(gemini, "count_text_tokens", count_text_tokens)
    assert gemini.count_texts("m", [["a"], ["c", "d"]]) == [1, 2]
    with pytest.raises(RuntimeError, match="quota"):
        gemini.count_texts("m", [["a"], ["b"]])


def test_window_payload_is_the_real_composition(conn):
    window = resolve_window(conn, [1])
    moments = screenshot_moments(conn, window)
    texts, n_events = window_payload(
        conn,
        window,
        system_prompt=SYSTEM,
        task=TASK,
        site_contexts={"site": SITE},
        moments=moments,
    )
    joined = "\n".join(texts)
    assert SYSTEM in joined and SITE in joined
    assert texts[-1] == TASK, "the question rides last and is counted with the rest"
    assert "<SUMMARY>" in joined and "<SESSION_CONTEXT>" in joined
    assert "<RECORDING>" in joined and "</RECORDING>" in joined
    assert n_events == 5
    assert list(moments) == [1], (
        "the moments are per slice — a screenshot's identity is (label, moment)"
    )
    assert len(moments[1]) >= 2, "first/last always kept"


def test_price_totals_are_the_turn_one_payload(conn):
    model = _model()
    cost = _cost(conn, model, [1])
    price = _price(conn, [1], 10_000, model)
    turn1 = price["turn1"]
    assert turn1["total_tokens"] == cost
    assert turn1["total_tokens"] == turn1["text_tokens"] + turn1["screenshot_tokens"], (
        "the fit is judged on what is sent, never on a turn that has not happened"
    )
    assert turn1["context_pct"] == round(cost / 10_000 * 100, 1)
    assert turn1["fits"] is True
    assert "per_slice" not in price, "one slice has nothing to whittle"
    assert "pieces" not in price, "a fitting payload offers no pieces"


def test_price_carries_each_slice_standalone(conn):
    model = _model()
    price = _price(conn, [1, 2, 3], 100_000, model)
    per = price["per_slice"]
    assert [entry["slice"] for entry in per] == ["s1", "s2", "s3"]
    for sid, entry in zip([1, 2, 3], per):
        assert entry["standalone_tokens"] == _cost(conn, model, [sid]), (
            "the whittling surface: each slice priced as its own window"
        )


def test_each_measured_set_is_priced_as_its_own_window(conn):
    """A standalone slice would run as its own single-slice window, so it must
    be counted that way — the prompts callback is asked once per measured set,
    with that set's real labels."""
    seen = []

    def prompts(labels):
        seen.append(tuple(labels))
        return SYSTEM, TASK, [TOOL]

    price_payload(
        conn,
        resolve_window(conn, [1, 2]),
        model="injected",
        prompts=prompts,
        site_contexts={"site": SITE},
        cost_model=_model(),
        context_tokens=100_000,
        headroom=1.0,
    )
    assert ("S1", "S2") in seen, "the whole payload is a two-slice window"
    assert seen.count(("S1",)) == 2, (
        "each standalone slice is its own single-slice window"
    )


def test_an_over_budget_slice_offers_its_route_pieces(conn):
    rec = "locus-recorder/1.0"
    spa = "https://spa.test"
    _insert_slice(
        conn,
        10,
        "spa",
        30_000,
        40_000,
        [
            (30_000, "Meta", f"{spa}/#/home", rec),
            (30_010, "FullSnapshot", None, rec),
            (30_500, "PageLoad", f"{spa}/#/home", rec),
            (31_000, "Click", None, rec),
            (32_000, "Click", None, rec),
            (33_000, "PageLoad", f"{spa}/#/two", rec),
            (34_000, "Click", None, rec),
            (36_000, "PageLoad", f"{spa}/#/three", rec),
            (38_000, "Click", None, rec),
        ],
    )
    conn.commit()
    model = _model()
    whole = _cost(conn, model, [10])
    price = _price(conn, [10], whole - 1, model)
    assert price["turn1"]["fits"] is False
    pieces = price["pieces"]
    assert [s["address"] for s in pieces] == ["s10#1", "s10#2", "s10#3"], (
        "the refusal names structural units, never timestamps"
    )
    assert [s["page"] for s in pieces] == [
        f"{spa}/#/home",
        f"{spa}/#/two",
        f"{spa}/#/three",
    ]
    for piece, (url, ws, we) in zip(pieces, slice_pieces(conn, 10)):
        assert piece["turn1_tokens"] == _cost(conn, model, [10], ws, we), (
            "every offered piece is priced by the same instrument"
        )
    bounds = slice_pieces(conn, 10)
    assert bounds[0][1] is None and bounds[-1][2] is None
    for (_, _, end), (_, start, _) in itertools.pairwise(bounds):
        assert end == start, "the pieces tile the slice"


def _lead_slice(conn, sid, start, routes_after_snapshot):
    """The SPA lead-flip shape: a route PageLoad fires in the slice's lead,
    before the covering pair has captured any DOM."""
    rec = "locus-recorder/1.0"
    spa = "https://spa.test"
    events = [
        (start, "PageLoad", f"{spa}/#/home", rec),
        (start + 400, "PageLoad", f"{spa}/#/promo", rec),
        (start + 1_000, "Meta", f"{spa}/#/promo", rec),
        (start + 1_010, "FullSnapshot", None, rec),
        (start + 2_000, "Click", None, rec),
    ]
    for k, route in enumerate(routes_after_snapshot, 1):
        events.append((start + 2_000 + k * 1_000, "PageLoad", f"{spa}{route}", rec))
        events.append((start + 2_500 + k * 1_000, "Click", None, rec))
    _insert_slice(conn, sid, "spa", start, events[-1][0], events)
    conn.commit()


def test_a_route_flip_in_the_lead_is_not_a_cut(conn):
    """A cut at a lead arrival would offer a piece that is all lead — a window
    with no recorded DOM, which resolution refuses — so a lead arrival folds
    into the first piece, and every piece the price report offers is an analysis
    that will compose."""
    _lead_slice(conn, 11, 50_000, ["/#/two"])
    bounds = slice_pieces(conn, 11)
    assert bounds == [
        ("https://spa.test/#/home", None, 53_000),
        ("https://spa.test/#/two", 53_000, None),
    ], "the lead flip folds into the first piece; the post-snapshot route cuts"
    for _, ws, we in bounds:
        resolve_window(conn, [11], ws, we)

    model = _model()
    whole = _cost(conn, model, [11])
    price = _price(conn, [11], whole - 1, model)
    assert [p["address"] for p in price["pieces"]] == ["s11#1", "s11#2"]


def test_a_slice_whose_only_route_flip_is_in_the_lead_is_indivisible(conn):
    _lead_slice(conn, 12, 60_000, [])
    assert len(slice_pieces(conn, 12)) == 1
    model = _model()
    whole = _cost(conn, model, [12])
    price = _price(conn, [12], whole - 1, model)
    (entry,) = price["pieces"]
    assert "address" not in entry and "note" in entry


def test_a_slice_with_no_routes_offers_no_pieces(conn):
    model = _model()
    whole = _cost(conn, model, [1])
    price = _price(conn, [1], whole - 1, model)
    assert price["turn1"]["fits"] is False
    (entry,) = price["pieces"]
    assert "address" not in entry and "--screenshot-interval" in entry["note"], (
        "an indivisible slice says so instead of inventing cut points"
    )


def test_bad_sets_fail_at_resolution(conn):
    """The walls fire where the address is resolved — a bad set never reaches pricing or composition."""
    with pytest.raises(ValueError, match="duplicates"):
        resolve_window(conn, [1, 1])
    with pytest.raises(ValueError, match="unknown"):
        resolve_window(conn, [99])
    with pytest.raises(ValueError, match="at least one slice"):
        resolve_window(conn, [])
    conn.execute("UPDATE slices SET status='discarded' WHERE id=3")
    with pytest.raises(ValueError, match="not replayable"):
        resolve_window(conn, [3])


def test_argument_order_never_changes_the_window(conn):
    """Labels are data-determined — the same set passed in any order is the
    same window with the same table, because slice_table orders by time,
    not by the caller's argv."""
    from locus.analysis.window import slice_table

    assert slice_table(conn, [2, 1]) == slice_table(conn, [1, 2])
    assert [s.slice_id for s in slice_table(conn, [2, 1])] == [1, 2]
    model = _model()
    assert (
        _price(conn, [2, 1], 100_000, model)["turn1"]
        == _price(conn, [1, 2], 100_000, model)["turn1"]
    )


def test_overlapping_slices_price_as_one_window(conn):
    """Slices 1 and 4 cover the same instants under two visitors — the
    concurrent shape the shared window clock exists for; it prices like any
    other set."""
    model = _model()
    price = _price(conn, [1, 4], 100_000, model)
    assert price["turn1"]["fits"] is True
    assert {e["visitor"] for e in price["per_slice"]} == {"v1", "v2"}


def test_disjoint_cross_visitor_slices_price_as_one_analysis(conn):
    _plain_slice(conn, 5, "v2", 40_000, "https://x.test/late")
    conn.commit()
    model = _model()
    price = _price(conn, [1, 5], 100_000, model)
    assert price["turn1"]["fits"] is True
    assert {e["visitor"] for e in price["per_slice"]} == {"v1", "v2"}, (
        "a group analysis prices like any other, each slice attributed to its visitor"
    )


def test_a_gemini_count_rides_out_transient_pushback(monkeypatch):
    """Each countTokens call carries the API's retry policy, the one a conversation's calls carry, so one 429 or 5xx is waited out rather than failing the report."""
    from types import SimpleNamespace

    from locus.analysis.model import gemini

    seen = {}

    def client(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            models=SimpleNamespace(
                count_tokens=lambda model, contents: SimpleNamespace(total_tokens=7)
            )
        )

    monkeypatch.setattr(gemini, "_client", client)
    assert gemini.count_text_tokens("m", ["a b"]) == 7
    retry = seen["http_options"].retry_options
    assert (retry.initial_delay, retry.jitter, retry.attempts) == (
        gemini.RETRY["initial_delay"],
        gemini.RETRY["jitter"],
        gemini.RETRY["attempts"],
    )
