"""Model cards — caller-side model facts, nothing else.

The roster is `config/cards/` under the deployment root: one TOML file per card, the file's stem the card's name — the deployment's name for the model, the name `locus analyze --model` and the spend ledger speak.
Every card declares `model`, the id its client speaks to the backend — the repo an openai-compatible server loads, the model a Gemini call addresses — and no two cards declare it alike: a card is 1:1 with its model, held by refusal at load, which is what lets a card's name stand for its model everywhere.
Which card a bare invocation runs on is the one marked `default = true`. More than one marked card refuses at load; none marked refuses only at the moment something asks for the default.

Every card names the client that speaks to it — `conversation`, one of the names factory.py's registry holds (GEMINI for Gemini's own API, OPENAI_COMPATIBLE for an OpenAI-compatible server at the card's `base_url`).
That declaration is what routes; the card's name is a label.

Every entry, either client, declares two composition facts: context_tokens (the model's native context — max_position_embeddings, no RoPE scaling) and headroom, the fraction of it one call's model-payload may claim.
Headroom is the whole reserve.
What follows turn 1 — the model's thinking and answer, the screenshots it pulls, the history re-sent on the next turn — cannot be counted before the model has spoken, so it is not counted; a share of the context is left free for it, and every turn after the first is measured exactly, at the wire, against what is actually left.
The right share differs per model because what fills it differs: a local pull arrives at native pixels and a Gemini one at a flat price.

An openai-compatible card additionally is the caller's complete statement of generation intent: the repo to request and its sampling regime (the model vendor's published recommendations for thinking, the one mode the bundled server runs).
Config is a client-side notion only: every value is sent explicitly on every request, so the persisted payload is the full record of what was asked for.
The server is upstream mlx-vlm behind an operational layer; it holds no opinion of ours. max_tokens caps total generation, thinking included, and is sized from measurement: thinking_budget plus 8192 of visible output — every visible output measured in operation (answers, verdict batches up to 29 claims, claim-extraction JSON) sits under ~2.5k tokens, so 8192 is ~3x the observed maximum.
Resize from new measurements, never by feel.

An openai-compatible card also declares its `family`, `max_image_tokens`, and `base_url`.
The composition plane's counting and resize arithmetic (budget.py: the pinned tokenizer, the smart_resize closed form) replicates exactly one model family's wire behavior; `family` states the card's membership, and a card outside SUPPORTED_FAMILIES is refused at load.
`max_image_tokens` is a table keyed by an analysis's two kinds of screenshot — `activity_screenshots` (the pushed set) and `pulled_screenshots` (the served pulls) — each the per-image token ceiling the backend realizes at the wire for that kind; pricing counts the window's screenshots at the activity ceiling, so window capacity follows the card.
`base_url` is the OpenAI-compatible endpoint the backend dials — and the one home for that address: the bundled mlx server derives its bind host/port from the same field at boot, so analysis and the server cannot drift.
Gemini cards carry none of these: their image dial is `media_resolution`, the same table keyed by the same two kinds, mapping each to the tier it rides — flat-priced per image within a tier, with the choosing rationale beside the dial in the card file.

A `pricing` block — per-token prices in USD per million tokens (fresh input, cached input, output; thinking bills at the output rate) — is what declares a card's calls billed, the same way `conversation` declares its client: a card with prices has every generation call spend-governed (walled and ledgered, spend.py) at exactly those rates, and a card without them is free.
Every Gemini card declares the block, because its wire always bills — a Gemini conversation refuses at construction without one.
The shipped openai-compatible card declares none: a call to the bundled server spends nothing real.

The supported-fields set is the declared boundary: a card may only carry sampling fields the upstream engine verifiably honors end to end.
Upstream's batched sampler implements temperature/top_p (top_k and min_p are accepted at the wire and dropped before the draw); penalties run as windowed logit processors — verified at mlx-vlm 0.6.2 (https://github.com/Blaizzy/mlx-vlm/blob/v0.6.2/mlx_vlm/server/generation.py: _PositionedTargetSampler.__init__ takes only temperature/top_p/seed, and _make_logits_processors passes presence_context_size/repetition_context_size to make_logits_processors).
Extend SUPPORTED_SAMPLING only after verifying the engine honors the new field at the draw.
"""

from pathlib import Path
from urllib.parse import urlparse

import tomllib

CARDS_DIR = "cards"

SUPPORTED_SAMPLING = {
    "temperature",
    "top_p",
    "presence_penalty",
    "presence_context_size",
}

SUPPORTED_FAMILIES = {"qwen3"}

GEMINI = "gemini"
OPENAI_COMPATIBLE = "openai-compatible"


def cards_dir() -> Path:
    """The deployment's card roster: config/cards/ under the deployment root, one TOML file per card, the file's stem the card's name. Resolved fresh so an edit takes effect on the next lookup."""
    from locus.evidence.deployment import CONFIG_DIR, deployment_root

    return deployment_root() / CONFIG_DIR / CARDS_DIR


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
        by_model.setdefault(declared, []).append(name)
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


def max_image_tokens(name: str) -> dict:
    """The per-image token ceilings a local card declares — {kind: tokens} for both of an analysis's kinds of screenshot, each realized by its backend as a downscale at the wire and priced at the sent geometry. Local-only: Gemini's image cost is flat per image, so its cards carry no image dial."""
    return _validate_image_ceilings(name, _declared(name, "max_image_tokens"))


def _validate_image_ceilings(name: str, table) -> dict:
    ok = isinstance(table, dict) and set(table) == set(IMAGE_KINDS)
    if ok:
        ok = all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0
            for v in table.values()
        )
    if not ok:
        raise ValueError(
            f"card {name!r} max_image_tokens in {_card_file(name)} is malformed — "
            f"it declares a table with exactly the keys {list(IMAGE_KINDS)}, each "
            f"a positive per-image token ceiling; got {table!r}"
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
            f"card '{name}' declares family {card.get('family')!r}; the "
            f"composition plane's counting and resize arithmetic (the pinned "
            f"tokenizer, the smart_resize closed form) implements "
            f"{sorted(SUPPORTED_FAMILIES)} — a model outside those families "
            f"would silently be priced and resized with another model's math. "
            f"Declare family membership only for a model whose tokenizer and "
            f"vision preprocessor match the family's."
        )
    if "max_image_tokens" not in card:
        raise ValueError(
            f"card '{name}' declares no max_image_tokens — the per-image token "
            f"ceilings the backend realizes at the wire, one per kind of "
            f"screenshot ({list(IMAGE_KINDS)}), and pricing counts windows with. "
            f"A local card without them has no stated image economics; declare "
            f"the table in {_card_file(name)}."
        )
    _validate_image_ceilings(name, card["max_image_tokens"])
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
