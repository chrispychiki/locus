"""Analysis pricing.

The tokenizer and preprocessor config are monkeypatched (no network, no HF cache); the projection itself is covered by test_analyze.py. The Qwen screenshot-token closed form is pinned against real processor output: the geometry cases below were computed by the actual pinned image processor on real screenshot dimensions — if the form or constants drift, these fail before a wrong count ships. Pricing runs against an injected CostModel so every economic constant is the test's own; the composition underneath (compose_window, screenshot_moments, the walls) is real.
"""

import itertools
import json as _json

import pytest
from _analysis_support import install_offline_tokenizer
from locus.analysis import budget
from locus.analysis.budget import CostModel, price_payload, window_payload
from locus.analysis.window import resolve_window, slice_pieces
from locus.evidence.db import connect
from locus.evidence.hydrate import pack_raw
from locus.evidence.rrweb_constants import EventType

PROCESSOR_VALIDATED = [
    ((390, 663), 252),
    ((501, 959), 480),
    ((1366, 917), 1247),
    ((1366, 981), 1333),
    ((1920, 921), 1740),
]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    install_offline_tokenizer(monkeypatch)


SYSTEM = "system prompt for the pricing tests"
SITE = "a demo site"
TASK = "the entire question the caller sends last"


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


@pytest.fixture
def conn(tmp_path):
    conn = connect(tmp_path / "events.db")
    _plain_slice(conn, 1, "v1", 1_000, "https://x.test/a")
    _plain_slice(conn, 2, "v1", 10_000, "https://x.test/b")
    _plain_slice(conn, 3, "v1", 20_000, "https://x.test/c")
    _plain_slice(conn, 4, "v2", 1_000, "https://x.test/other")
    conn.commit()
    return conn


def _model(count=None, verify=None, screenshot=10):
    return CostModel(
        count_text=count or (lambda texts: sum(len(t.split()) for t in texts)),
        verify_text=verify,
        screenshot_tokens=lambda c, moments: (
            sum(len(timestamps) for timestamps in moments.values()) * screenshot
        ),
    )


def _prompts(labels):
    return SYSTEM, TASK


def _cost(conn, model, sids, ws=None, we=None):
    texts, moments, _ = window_payload(
        conn,
        resolve_window(conn, sids, ws, we),
        system_prompt=SYSTEM,
        task=TASK,
        site_contexts={"site": SITE},
    )
    return model.count_text(texts) + model.screenshot_tokens(conn, moments)


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


@pytest.mark.parametrize("size,expected", PROCESSOR_VALIDATED)
def test_qwen_image_tokens_matches_processor(size, expected):
    width, height = size
    assert budget.qwen_image_tokens(width, height) == expected


def test_image_geometry_realizes_the_per_image_token_ceiling():
    assert budget.image_geometry(390, 663, 600) == (390, 663), (
        "a screenshot under the ceiling passes untouched — never upscaled"
    )
    width, height = budget.image_geometry(1920, 921, 600)
    assert (width, height) != (1920, 921)
    assert budget.qwen_image_tokens(width, height) <= 600, (
        "the ceiling is a hard bound on the sent geometry's price"
    )
    assert budget.qwen_image_tokens(width, height) > 540, (
        "the resize lands near the ceiling, not far under it"
    )
    assert abs(width / height - 1920 / 921) < 0.05, "aspect preserved"


def test_a_ceiling_below_the_model_floor_fails_loud():
    with pytest.raises(ValueError, match="ceiling"):
        budget.image_geometry(1920, 921, 10)


def test_slice_viewport_reads_meta(conn):
    assert budget.slice_viewport(conn, 1) == (390, 663)


def test_window_payload_is_the_real_composition(conn):
    texts, moments, n_events = window_payload(
        conn,
        resolve_window(conn, [1]),
        system_prompt=SYSTEM,
        task=TASK,
        site_contexts={"site": SITE},
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
    assert turn1["fits"] is True and turn1["verified"] is False
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
        return SYSTEM, TASK

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


def test_price_verifies_with_the_exact_instrument(conn):
    inflate = 50
    model = _model(verify=lambda texts: sum(len(t.split()) for t in texts) + inflate)
    proposed = _cost(conn, model, [1])
    price = _price(conn, [1], proposed + 10, model)
    assert price["turn1"]["verified"] is True
    assert price["turn1"]["text_tokens"] > 0
    assert price["turn1"]["total_tokens"] == proposed + inflate, (
        "the exact instrument's count is the one the fit is judged on"
    )
    assert price["turn1"]["fits"] is False, (
        "propose fit, verify overflowed — the verified number decides"
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


def test_tokenizer_repo_resolves_from_the_declared_local_cards(tmp_path, monkeypatch):
    """No card key is hardcoded: the instrument is whichever local card the deployment declares (they share it byte-identically), and a deployment declaring none has no local instrument to name — loud, with the countTokens alternative stated."""
    from _analysis_support import card_roster

    card_roster(
        monkeypatch,
        tmp_path,
        {
            "some-renamed-card": (
                'conversation = "openai-compatible"\n'
                'model = "someone/renamed-qwen"\n'
                'base_url = "http://127.0.0.1:9/v1"\n'
                'family = "qwen3"\n'
                "max_image_tokens = { activity_screenshots = 600, pulled_screenshots = 1200 }\n"
                "thinking_budget = 1\n"
                "max_tokens = 1\n"
                "context_tokens = 1000\n"
                "headroom = 0.8\n"
                "[sampling]\n"
                "temperature = 1.0\n"
            )
        },
    )
    assert budget.tokenizer_repo() == "someone/renamed-qwen"

    card_roster(
        monkeypatch,
        tmp_path / "gemini_only",
        {"gemini-x": "context_tokens = 1000\nheadroom = 0.5\nmax_output_tokens = 64\n"},
    )
    with pytest.raises(ValueError, match="openai-compatible"):
        budget.tokenizer_repo()


def test_gemini_pricing_without_local_cards_counts_with_counttokens(
    tmp_path, monkeypatch
):
    """A pure-Gemini deployment prices without any pinned tokenizer: the free countTokens API is the one text instrument, so propose is already exact and there is nothing to verify."""
    from _analysis_support import card_roster
    from locus.analysis.budget import cost_model_for
    from locus.analysis.model import gemini

    card_roster(
        monkeypatch,
        tmp_path,
        {
            "gemini-x": 'conversation = "gemini"\nthinking_levels = ["low"]\ncontext_tokens = 1000\nheadroom = 0.5\nmax_output_tokens = 64\n'
        },
    )
    monkeypatch.setattr(
        gemini,
        "count_text_tokens",
        lambda model, texts: sum(len(t.split()) for t in texts),
    )

    model, context, room = cost_model_for("gemini-x")
    assert context == 1000 and room == 0.5
    assert model.verify_text is None, "propose is already the exact instrument"
    assert model.count_text(["two words", "and three more"]) == 5


def test_every_local_card_shares_the_pinned_tokenizer():
    """The local count is claimed exact — it is the wire's own arithmetic, with no
    backend instrument behind it to verify against. That holds only while every local
    card tokenizes and preprocesses identically to the pinned repo. A card whose model
    breaks that identity must fail here, not silently mis-price every window it serves."""
    import hashlib

    from huggingface_hub import hf_hub_download
    from locus.analysis.model.cards import OPENAI_COMPATIBLE, load_card, models_speaking

    def digest(repo, name):
        with open(hf_hub_download(repo, name), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    pinned = {
        name: digest(budget.tokenizer_repo(), name)
        for name in (budget.TOKENIZER_FILE, budget.PREPROCESSOR_FILE)
    }

    local = models_speaking(OPENAI_COMPATIBLE)
    assert local, "the card roster declares no openai-compatible model"
    for model in local:
        repo = load_card(model)["model"]
        for name, expected in pinned.items():
            assert digest(repo, name) == expected, (
                f"{model} ({repo}) does not share the pinned {name} — the local token count is not exact for it"
            )

    # The suite's offline stub carries a hand copy of the pinned preprocessor's fields, and
    # every screenshot-token pin in this file validates against that copy — so the copy itself is
    # pinned to the real file here, where the repo is already in hand.
    from _analysis_support import PREPROCESSOR_CONFIG

    with open(hf_hub_download(budget.tokenizer_repo(), budget.PREPROCESSOR_FILE)) as f:
        real = _json.load(f)
    assert PREPROCESSOR_CONFIG == {
        "patch_size": real["patch_size"],
        "merge_size": real["merge_size"],
        "size": {
            "shortest_edge": real["size"]["shortest_edge"],
            "longest_edge": real["size"]["longest_edge"],
        },
    }, (
        "the offline preprocessor stub drifted from the pinned repo's real config — the "
        "screenshot-token tests are validating arithmetic the backend does not run"
    )
