"""Model cards — caller-side model facts, nothing else.

The roster is `config/cards/` under the deployment root: one TOML file per card, the file's stem the card's name — the deployment's name for the model, the name `locus analyze --model` and the spend ledger speak.
Every card declares `model`, the id its client speaks to the backend — the repo or local directory an openai-compatible server loads (a relative directory is the deployment root's), the model a Gemini call addresses — and no two cards declare it alike: a card is 1:1 with its model, held by refusal at load, which is what lets a card's name stand for its model everywhere.
Which card a bare invocation runs on is the one marked `default = true`. More than one marked card refuses at load; none marked refuses only at the moment something asks for the default.

Every card names the client that speaks to it — `conversation`, one of the names factory.py's registry holds (GEMINI for Gemini's own API, OPENAI_COMPATIBLE for an OpenAI-compatible server at the card's `base_url`).
That declaration is what routes; the card's name is a label.

Every entry, either client, declares two composition facts: context_tokens (the model's native context — max_position_embeddings, no RoPE scaling) and headroom, the fraction of it one call's model-payload may claim.
Headroom is the whole reserve.
What follows turn 1 — the model's reasoning and answers, the screenshots they pull, a citation repair — cannot be counted before the model has spoken, so it is not counted; a share of the context is left free for it (and on a local card, whose context holds generation too, for the output cap every request needs free when it is sent), and every turn after the first is measured exactly, at the wire, against what is actually left.
The share is a round number wide enough for what ordinary runs do after turn 1, repair included.

An openai-compatible card additionally is the caller's complete statement of generation intent: the repo to request and its sampling regime (the model vendor's published recommendations for thinking, the one mode the bundled server runs).
Config is a client-side notion only: every value is sent explicitly on every request, so the persisted payload is the full record of what was asked for.
The server is upstream mlx-vlm behind an operational layer; it holds no opinion of ours. max_tokens caps total generation, thinking included, and is sized from measurement: thinking_budget plus 8192 of visible output — every visible output measured in operation (answers, verdict batches up to 29 claims, claim-extraction JSON) sits under ~2.5k tokens, so 8192 is ~3x the observed maximum.
Resize from new measurements, never by feel.

An openai-compatible card also declares its `family`, `image_tokens`, and `base_url`.
The bundled server bounds its memory with patches bound to each supported family's modules (Qwen's in mlx/fused_attention.py and mlx/vision_chunking.py, Gemma 4's in mlx/gemma4.py), so a model outside those families would be served without those bounds, and the count's per-image shortcut has been judged on those families' processors alone (mlx/count.py JUDGED_PROCESSORS); `family` states the card's membership, and a card outside SUPPORTED_FAMILIES is refused at load.
`image_tokens` is a table keyed by an analysis's two kinds of screenshot, `activity_screenshots` (the pushed set) and `pulled_screenshots` (the served pulls). Each value is the token count of one screenshot of that kind at any viewport: the renderer draws the screenshot at the largest size on the model's grid whose cost in the server's count fits the value (budget.image_geometry). Pricing counts the window's screenshots at the activity value, so how much a window holds follows the card.
An openai-compatible card whose model thinks inside other markers than upstream's `<think>` and `</think>` declares them as `thinking_markers = [start, end]`: the bundled server's thinking budget and thinking-aware grammar match those tokens (mlx/serve.py serving_env).
An openai-compatible card may also declare `chat_template_kwargs`, a table of options to the model's chat template, sent verbatim on every request in the field of that name, the one vLLM and SGLang take; the bundled server hands it to the template (mlx/chat_turns.py).
`base_url` is the OpenAI-compatible endpoint the backend dials — and the one home for that address: the bundled mlx server derives its bind host/port from the same field at boot, so analysis and the server cannot drift.
Gemini cards carry none of these: their image dial is `media_resolution`, the same table keyed by the same two kinds, mapping each to the tier it rides — flat-priced per image within a tier, with the choosing rationale beside the dial in the card file.

A `pricing` block — per-token prices in USD per million tokens (fresh input, cached input, output; thinking bills at the output rate) — is what declares a card's calls billed, the same way `conversation` declares its client: a card with prices has every generation call spend-governed (walled and ledgered, spend.py) at exactly those rates, and a card without them is free.
Every Gemini card declares the block, because its wire always bills — a Gemini conversation refuses at construction without one.
The shipped openai-compatible cards declare none: a call to the bundled server spends nothing real.

The supported-fields set is the declared boundary: a card may only carry sampling fields the bundled server verifiably honors end to end.
Upstream's batched sampler implements temperature/top_p/top_k (min_p is accepted at the wire and dropped before the draw) — verified at mlx-vlm 0.7.10 (https://github.com/Blaizzy/mlx-vlm/blob/v0.7.10/mlx_vlm/server/generation.py: _make_sampler builds _PositionedTargetSampler from temperature/top_p/top_k/seed alone); presence_penalty runs as a logit processor over the tokens the request has generated, the bundled server's own (mlx/presence_penalty.py).
Extend SUPPORTED_SAMPLING only after verifying the engine honors the new field at the draw.
"""

import tomllib
from pathlib import Path
from urllib.parse import urlparse

CARDS_DIR = "cards"

SUPPORTED_SAMPLING = {
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
}

SUPPORTED_FAMILIES = {"qwen3", "gemma4"}

GEMINI = "gemini"
OPENAI_COMPATIBLE = "openai-compatible"


def cards_dir() -> Path:
    """The deployment's card roster: config/cards/ under the deployment root, one TOML file per card, the file's stem the card's name. Resolved fresh so an edit takes effect on the next lookup."""
    from locus.evidence.deployment import CONFIG_DIR, deployment_root

    return deployment_root() / CONFIG_DIR / CARDS_DIR


def local_model(model: str) -> str:
    """The model an openai-compatible card names, as every reader takes it — the server loads it, the count reads it, the wire and the record carry it: a Hub repo id, or a local directory. A relative path is the deployment's, so it resolves against the deployment root the roster lives in (the roster's own, so a card and the directory it names always come from one deployment), whatever the working directory; the test is upstream's own (utils.get_model_path) — a path that exists is local, anything else is a repo id."""
    root = cards_dir().parent.parent
    path = Path(model)
    if not path.is_absolute() and (root / path).exists():
        return str(root / path)
    return model


def _card_file(name: str) -> Path:
    return cards_dir() / f"{name}.toml"


def _cards() -> dict:
    """Every card the roster declares, read fresh: the files are the truth, so an edit takes effect on the next lookup. The cross-card invariants hold here, so an incoherent roster refuses every read: every card declares its `model`, no two declare the same one, and at most one is marked `default`."""
    directory = cards_dir()
    if not directory.is_dir():
        raise ValueError(
            f"no {directory}: the card roster ships with the repo, one "
            f"<card>.toml per model — restore it"
        )
    cards = {
        path.stem: tomllib.loads(path.read_text())
        for path in sorted(directory.glob("*.toml"))
    }
    by_model: dict[str, list[str]] = {}
    for name, card in cards.items():
        declared = card.get("model")
        if not isinstance(declared, str) or not declared:
            raise ValueError(
                f"card {name!r} declares no `model` in {_card_file(name)} — "
                f"every card names the model its client speaks to the "
                f"backend; the filename is a label and never stands in for it"
            )
        if card.get("conversation") == OPENAI_COMPATIBLE:
            card["model"] = local_model(declared)
        by_model.setdefault(card["model"], []).append(name)
        marked = card.get("default")
        if marked is not None and not isinstance(marked, bool):
            raise ValueError(
                f"card {name!r} declares default = {marked!r} in "
                f"{_card_file(name)} — `default` is a bool: `default = true` "
                f"marks the card a bare invocation runs on"
            )
    duplicated = {m: names for m, names in by_model.items() if len(names) > 1}
    if duplicated:
        raise ValueError(
            f"two cards declare the same model in {directory} — a card is 1:1 "
            f"with its model; keep one card per model: {duplicated}"
        )
    marked = sorted(name for name, card in cards.items() if card.get("default"))
    if len(marked) > 1:
        raise ValueError(
            f"{len(marked)} cards declare `default = true` in {directory} "
            f"({marked}) — exactly one card may be the default; unmark the rest"
        )
    return cards


def declared_models() -> list[str]:
    """Every model the deployment declares — the full roster, every client."""
    return sorted(_cards())


def conversation(name: str) -> str:
    """The client a card names — its `conversation` declaration. A card naming none fails loud: which client speaks to a model is configuration, never inferred from the card's other fields. Whether the name is one the registry holds is factory.py's check, at construction."""
    card = declared_card(name)
    client = card.get("conversation")
    if not isinstance(client, str) or not client:
        raise ValueError(
            f"card {name!r} declares no `conversation` in {_card_file(name)} — every card names the client that speaks to it ({GEMINI!r} or {OPENAI_COMPATIBLE!r})"
        )
    return client


def models_speaking(client: str) -> list[str]:
    """The declared cards naming one client, in declaration order."""
    return [
        name for name, card in _cards().items() if card.get("conversation") == client
    ]


def default_card() -> str:
    """The card an invocation runs on when it names none: the roster card marked `default = true`.
    Which card that is lives on the card itself and nowhere else — presence in the roster never picks one, more than one marked card already refused at load, and a roster with none marked refuses here, at the moment something actually asks for the default."""
    cards = _cards()
    marked = [name for name, card in cards.items() if card.get("default")]
    if not marked:
        raise ValueError(
            f"no card declares `default = true` in {cards_dir()} — mark "
            f"exactly one of {sorted(cards)} as the default, or name a model "
            f"per invocation with --model"
        )
    return marked[0]


def model_id(name: str) -> str:
    """The `model` a card declares — the id its client speaks to the backend: the repo an openai-compatible server loads, the model a Gemini call addresses. The card's name is the deployment's label for it and never rides the wire."""
    return _declared(name, "model")


def declared_card(name: str) -> dict:
    """The declared entry for a model, either backend — the one lookup an undeclared model fails through, loud and naming the roster."""
    cards = _cards()
    if name not in cards:
        raise ValueError(
            f"no card declares model {name!r} — a model is named by its card's name, one <name>.toml per model in {cards_dir()}; declared models: {sorted(cards)}"
        )
    return cards[name]


def _declared(name: str, field: str):
    """A composition fact every model declares, either backend. A model missing one fails loud naming the file — these are declared facts, never guessed."""
    cards = _cards()
    if name not in cards or field not in cards[name]:
        raise ValueError(
            f"no {field} declared for model {name!r} in {_card_file(name)} — add an entry (models with one: {sorted(cards)})"
        )
    return cards[name][field]


def context_tokens(name: str) -> int:
    """The model's context size in tokens."""
    return _declared(name, "context_tokens")


def headroom(name: str) -> float:
    """The fraction of the model's context one call's model-payload may claim; the rest is what the conversation grows into."""
    return _declared(name, "headroom")


def thinking_level(name: str) -> str | None:
    """The reasoning level a Gemini card declares, or None if it declares none."""
    return _cards().get(name, {}).get("thinking_level")


def thinking_levels(name: str) -> list[str]:
    """The reasoning levels a Gemini card declares its model accepts, lowest first — the model page's list, which differs per model. The card's thinking_level must be one of them; a turn whose thinking exhausts the output ceiling is retried at the first."""
    levels = _declared(name, "thinking_levels")
    level = thinking_level(name)
    if level is not None and level not in levels:
        raise ValueError(
            f"card {name!r} declares thinking_level {level!r}, which is not among its thinking_levels {levels} in {_card_file(name)}"
        )
    return levels


IMAGE_KINDS = ("activity_screenshots", "pulled_screenshots")
MEDIA_RESOLUTION_KINDS = IMAGE_KINDS


def media_resolution(name: str) -> dict:
    """The image-tier table a Gemini card declares — {kind: tier} for an analysis's two kinds of screenshot: `activity_screenshots` (the pushed set sampling the session) and `pulled_screenshots` (the served pulls). There is no single resolution; a kind the card leaves out rides the API's own default tier. Unknown kinds fail loud."""
    tiers = _cards().get(name, {}).get("media_resolution") or {}
    unknown = sorted(set(tiers) - set(MEDIA_RESOLUTION_KINDS))
    if unknown:
        raise ValueError(
            f"card {name!r} media_resolution declares unknown keys "
            f"{unknown} in {_card_file(name)} — the kinds are "
            f"{list(MEDIA_RESOLUTION_KINDS)}"
        )
    return tiers


def max_output_tokens(name: str) -> int:
    """The combined thinking+output ceiling a Gemini card declares — required (a Gemini card without its safety ceiling can hang thinking indefinitely), so a card missing it fails loud. Only resolved for Gemini models; local cards bound generation with their own max_tokens."""
    return _declared(name, "max_output_tokens")


def image_tokens(name: str) -> dict:
    """The per-image token values a local card declares, {kind: tokens}, one per kind of screenshot; every screenshot of a kind is drawn and priced at its value (budget.image_geometry). Local-only: Gemini's image cost is flat per image, so its cards carry no image dial."""
    return _validate_image_tokens(name, _declared(name, "image_tokens"))


def _validate_image_tokens(name: str, table) -> dict:
    ok = isinstance(table, dict) and set(table) == set(IMAGE_KINDS)
    if ok:
        ok = all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0
            for v in table.values()
        )
    if not ok:
        raise ValueError(
            f"card {name!r} image_tokens in {_card_file(name)} is malformed — "
            f"it declares a table with exactly the keys {list(IMAGE_KINDS)}, each "
            f"a positive per-image token count; got {table!r}"
        )
    return table


PRICING_FIELDS = ("input_per_mtok", "cached_input_per_mtok", "output_per_mtok")


def declared_pricing(name: str) -> dict | None:
    """The per-token price block a card declares — USD per million tokens for fresh input, cached input, and output (thinking bills at the output rate) — or None when it declares none, which is the declaration that its calls are free. This is the lenient read for surfaces that price when they can (the price report, the engine's early wall check); the Gemini backend's strict requirement is pricing()."""
    card = _cards().get(name, {})
    return card.get("pricing")


def _validate_pricing(name: str, prices: dict) -> dict:
    missing = [field for field in PRICING_FIELDS if field not in prices]
    unknown = sorted(set(prices) - set(PRICING_FIELDS))
    bad = [
        field
        for field in PRICING_FIELDS
        if field in prices
        and (
            isinstance(prices[field], bool)
            or not isinstance(prices[field], (int, float))
            or prices[field] < 0
        )
    ]
    if missing or unknown or bad:
        raise ValueError(
            f"card {name!r} pricing block in {_card_file(name)} is malformed — "
            f"missing {missing}, unknown {unknown}, non-price values {bad}; "
            f"it declares exactly {list(PRICING_FIELDS)}, each USD per "
            f"million tokens"
        )
    return prices


def pricing(name: str) -> dict:
    """The price block a Gemini conversation runs on, validated whole: all three per-mtok fields, finite and non-negative. A Gemini card missing the block fails loud here, at conversation construction — before anything is spent — because that wire always bills, and without declared prices neither the ledger nor a spend wall can compute a dollar."""
    prices = declared_pricing(name)
    if prices is None:
        raise ValueError(
            f"card {name!r} declares no pricing block in {_card_file(name)} — its "
            f"calls bill money, and spend cannot be accounted without one; "
            f"declare {list(PRICING_FIELDS)} (USD per million tokens, from "
            f"the model's pricing page)"
        )
    return _validate_pricing(name, prices)


def load_card(name: str) -> dict:
    """The validated openai-compatible card, with its file name carried along as `name` — the deployment's name for the model, which is what a ledger entry and a --model flag speak, where `model` is the repo the server loads."""
    card = declared_card(name)
    if "pricing" in card:
        _validate_pricing(name, card["pricing"])
    if conversation(name) != OPENAI_COMPATIBLE:
        raise ValueError(
            f"{name!r} is not an openai-compatible card — it names "
            f"conversation {conversation(name)!r}; openai-compatible cards: "
            f"{models_speaking(OPENAI_COMPATIBLE)}"
        )
    if "sampling" not in card:
        raise ValueError(
            f"card '{name}' declares no sampling block — an openai-compatible "
            f"card states its whole generation config explicitly, sent on "
            f"every request; declare [sampling] in {_card_file(name)}."
        )
    if card.get("family") not in SUPPORTED_FAMILIES:
        raise ValueError(
            f"card '{name}' declares family {card.get('family')!r}; the bundled "
            f"server's memory bounds (mlx/fused_attention.py, mlx/vision_chunking.py, "
            f"mlx/gemma4.py) patch {sorted(SUPPORTED_FAMILIES)}'s modules, and the count's "
            f"per-image shortcut is judged on those families' processors (mlx/count.py) — a model "
            f"outside those families would be served without the bounds. Declare "
            f"family membership only for a model those patches reach."
        )
    if "image_tokens" not in card:
        raise ValueError(
            f"card '{name}' declares no image_tokens, the per-image token value "
            f"for each kind of screenshot ({list(IMAGE_KINDS)}) that screenshots "
            f"are drawn and priced at; declare the table in {_card_file(name)}."
        )
    _validate_image_tokens(name, card["image_tokens"])
    url = card.get("base_url")
    if not isinstance(url, str) or not url:
        raise ValueError(
            f"card '{name}' declares no base_url — the OpenAI-compatible "
            f"endpoint the local backend dials (and the bundled server binds) "
            f"is the card's one home for that address; declare it in "
            f"{_card_file(name)}."
        )
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname or parsed.port is None:
        raise ValueError(
            f"card '{name}' base_url {url!r} must be an absolute URL with an "
            f"explicit host and port (scheme://host:port/...), so analysis "
            f"and the server agree on one endpoint without a second copy of "
            f"the number anywhere else"
        )
    markers = card.get("thinking_markers")
    if markers is not None and not (
        isinstance(markers, list)
        and len(markers) == 2
        and all(isinstance(marker, str) and marker for marker in markers)
    ):
        raise ValueError(
            f"card '{name}' thinking_markers in {_card_file(name)} is {markers!r} — "
            f"it declares the model's [start, end] thinking tokens, two non-empty strings"
        )
    template_options = card.get("chat_template_kwargs", {})
    if not isinstance(template_options, dict):
        raise ValueError(  # noqa: TRY004 — every malformed card is refused alike
            f"card '{name}' chat_template_kwargs in {_card_file(name)} is "
            f"{template_options!r} — it declares a table of options to the "
            f"model's chat template, sent as the request's chat_template_kwargs"
        )
    unsupported = sorted(set(card["sampling"]) - SUPPORTED_SAMPLING)
    if unsupported:
        raise ValueError(
            f"card '{name}' sampling claims {unsupported}, which the upstream "
            f"batched engine does not honor — a claimed-but-ignored config "
            f"field is a lie in the surface. The supported set, and the "
            f"upstream verification behind it, is SUPPORTED_SAMPLING in "
            f"{Path(__file__).name}; extend it only after verifying the "
            f"engine honors the field end to end."
        )
    return {**card, "name": name}
