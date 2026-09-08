"""Model-payload pricing — the composition plane's only token surface.

Token costs are computed fresh at decision time, never stored: a stored count is a staleness bug waiting for a projection or distillation change to make it a lie.

Counting is exact by construction: the model-payload is composed through the real composer (compose_window) into a tally conversation, so the counted text is byte-for-byte what the wire would carry. The local denomination runs that text through the pinned tokenizer and prices screenshots by the ViT closed form over the geometry the card's per-image token ceiling actually sends; the Gemini denomination verifies the same text against the live countTokens API (free, the wire's own arithmetic) — or, on a deployment declaring no local card and so pinning no tokenizer, counts with countTokens alone — and prices each activity-screenshot flat at its tier; within a tier image cost is independent of dimensions, content, and transport (gemini.py owns the per-tier numbers). One request term stays uncounted, deliberately: the tool declaration the run's request carries (~100 billed tokens — measured +98 on the single-slice request_screenshots shape) — countTokens refuses a tools config on this API tier, and a hand-formula for its serialization would trade exact-by-construction for a guess — so a billed turn-1 runs that constant over the price, a property of the request shape, never drift.

The bound is turn 1, and only turn 1: an analysis fits when the model-payload actually sent on the opening call comes in under the card's headroom x context. Nothing later is counted here, because nothing later exists here — what the model will think, write, or ask to see cannot be measured before it has been asked. The conversation grows into the context left below that ceiling, and every turn past the first is measured exactly, at the wire, against the context that is genuinely left (the engine serves pulls while they fit and refuses the rest). Caching changes none of this — cached tokens still occupy the context; the discount is dollars, never counted toward fit.

Fitting is the caller's judgment, informed: the price report carries the whole payload's turn-1 total as a percentage of the card's context, each slice's standalone price when the set holds several, and — when one slice alone out-prices the budget — that slice's route-boundary pieces, individually priced and addressable as `<slice>#<k>`. Nothing here ever cuts or splits on its own.

The prompts are a callback, not strings: every set measured here — the whole window, each slice standalone, each route-boundary piece — is its own window with its own labels, so the caller supplies prompts(labels) → (system_prompt, task) and each measurement composes the exact bytes that set's run would send.
"""

import json
import math
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from locus.evidence.db import CANONICAL_ORDER
from locus.evidence.hydrate import raw_event
from locus.evidence.rrweb_constants import EventType

from .model.protocol import Conversation
from .select_screenshots import SCREENSHOT_INTERVAL_MS, screenshot_moments

TOKENIZER_FILE = "tokenizer.json"
PREPROCESSOR_FILE = "preprocessor_config.json"

_tokenizer = None
_preprocessor_config = None


def tokenizer_repo() -> str:
    """The pinned local instrument's repo, resolved from whatever local cards the deployment declares — any of them, because every local card carries byte-identical tokenizer and preprocessor files, so the count is each of their own (a card that breaks that identity fails the suite's identity check before anything is priced with it). A deployment declaring no local card pins no local instrument — and needs none: local pricing is unreachable without a local card, and the Gemini denomination then counts with the API's own free countTokens alone (cost_model_for)."""
    from .model.cards import OPENAI_COMPATIBLE, load_card, models_speaking

    models = models_speaking(OPENAI_COMPATIBLE)
    if not models:
        raise ValueError(
            "no openai-compatible card is declared in config/cards/ under "
            "the deployment root, so "
            "there is no pinned local tokenizer to count with — on such a "
            "deployment pricing runs on Gemini's own countTokens"
        )
    return load_card(models[0])["model"]


def _load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        _tokenizer = Tokenizer.from_file(
            hf_hub_download(tokenizer_repo(), TOKENIZER_FILE)
        )
    return _tokenizer


def _load_preprocessor_config() -> dict:
    global _preprocessor_config
    if _preprocessor_config is None:
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(tokenizer_repo(), PREPROCESSOR_FILE)) as f:
            _preprocessor_config = json.load(f)
    return _preprocessor_config


def count_text_tokens(text: str) -> int:
    """Pinned-tokenizer count of any text bound for the local model — the one text-counting instrument, shared by pricing and the local wire-boundary check."""
    if not text:
        return 0
    return len(_load_tokenizer().encode(text).ids)


def qwen_image_tokens(width: int, height: int) -> int:
    """Vision tokens for one screenshot of the given pixel dimensions.

    Replicates the pinned processor's smart_resize (round each dimension to factor = patch_size x merge_size, rescale into the pixel bounds preserving aspect), then patches / merge^2.
    """
    cfg = _load_preprocessor_config()
    patch, merge = cfg["patch_size"], cfg["merge_size"]
    factor = patch * merge
    min_pixels = cfg["size"]["shortest_edge"]
    max_pixels = cfg["size"]["longest_edge"]

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return (h_bar // patch) * (w_bar // patch) // (merge * merge)


def image_geometry(width: int, height: int, max_image_tokens: int) -> tuple[int, int]:
    """The geometry a screenshot is actually sent at under the card's per-image token ceiling. Pricing must count the sent geometry, so this is the one place the ceiling turns into pixels — the local backend resizes by the same rule. A screenshot already under the ceiling passes untouched (never upscaled); over it, the screenshot scales down preserving aspect to the largest geometry whose price fits. The model's own per-image floor bounds what a ceiling may ask: the preprocessor never tokenizes below its minimum pixels, so a ceiling under that floor is unrealizable and fails loud."""
    cfg = _load_preprocessor_config()
    factor = cfg["patch_size"] * cfg["merge_size"]
    floor_tokens = qwen_image_tokens(factor, factor)
    if max_image_tokens < floor_tokens:
        raise ValueError(
            f"per-image token ceiling {max_image_tokens} is below the model's "
            f"own floor of {floor_tokens} tokens (the preprocessor's minimum "
            f"pixels) — no geometry realizes it; raise max_image_tokens in the "
            f"card"
        )
    if qwen_image_tokens(width, height) <= max_image_tokens:
        return width, height
    scale = math.sqrt(max_image_tokens * factor * factor / (width * height))
    sent_w = max(1, math.floor(width * scale))
    sent_h = max(1, math.floor(height * scale))
    # smart_resize rounds each dimension to the factor grid, which can round up
    # past the ceiling; step down until the priced geometry fits.
    while qwen_image_tokens(sent_w, sent_h) > max_image_tokens:
        sent_w = max(1, math.floor(sent_w * 0.98))
        sent_h = max(1, math.floor(sent_h * 0.98))
    return sent_w, sent_h


def slice_viewport(conn: sqlite3.Connection, slice_id: int) -> tuple[int, int] | None:
    row = conn.execute(
        f"SELECT raw_json FROM events WHERE slice_id = ? AND type = ? {CANONICAL_ORDER} LIMIT 1",
        (slice_id, EventType.Meta),
    ).fetchone()
    if row is None:
        return None
    data = raw_event(row["raw_json"]).get("data") or {}
    if not data.get("width") or not data.get("height"):
        return None
    return data["width"], data["height"]


class TallyConversation(Conversation):
    """Composition-faithful counter: compose_window writes into it exactly as into a real backend, so the tallied text and image count are the wire payload by construction — no second copy of composition arithmetic to drift. Never calls a model."""

    def __init__(self):
        super().__init__()
        self.texts: list[str] = []
        self.n_images = 0

    def _add_user_text(self, text: str) -> None:
        self.texts.append(text)

    def _add_user_image(self, image_path: str, pushed: bool) -> str:
        self.n_images += 1
        return str(image_path)

    def image_tokens(self, image_path: str) -> int:
        raise RuntimeError(
            "a tally conversation prices a window, not an image — the backend's own cost model does that"
        )

    def context_remaining(self) -> int:
        raise RuntimeError(
            "a tally conversation holds no context — pricing compares its total against the budget itself"
        )

    def _add_tool_result(self, call_id: str, content: str) -> None:
        raise RuntimeError("a tally conversation only composes, never converses")

    def _get_response(self, **kwargs):
        raise RuntimeError("a tally conversation never calls a model")

    @property
    def execution_mode(self) -> str:
        return "tally"


def window_payload(
    conn: sqlite3.Connection,
    window,
    *,
    system_prompt: str,
    task: str,
    site_contexts: dict[str, str],
    screenshot_interval_ms: int = SCREENSHOT_INTERVAL_MS,
) -> tuple[list[str], dict[int, list[int]], int]:
    """The full turn-1 model-payload as (text parts, per-slice screenshot timestamps, n_events), composed by the real composer against the real moment selection over a resolved window (window.resolve_window) — nothing rendered, nothing sent. task is the entire question as the caller sends it last on the turn that writes the answer; it rides the wire, so it is counted."""
    from .analyze import compose_window

    pushed = screenshot_moments(
        conn, window, screenshot_interval_ms=screenshot_interval_ms
    )
    labels = {s.slice_id: s.label for s in window.slices}
    tally = TallyConversation()
    manifest = compose_window(
        conn,
        tally,
        window,
        site_contexts=site_contexts,
        screenshots={
            (labels[sid], ts): Path(f"screenshot_{labels[sid]}_{ts}.png")
            for sid, timestamps in pushed.items()
            for ts in timestamps
        },
    )
    return [system_prompt, *tally.texts, task], pushed, manifest.n_events


@dataclass
class CostModel:
    """How one backend turns a composed payload into token totals. count_text is the propose instrument — the pinned local tokenizer where the deployment declares a local card (offline, fast enough to probe candidates), else the backend's own exact instrument; verify_text is the backend's exact instrument for the close (None where propose already is that instrument); screenshot_tokens prices a window's screenshot moments ({slice_id: timestamps}, the shape screenshot_moments emits)."""

    count_text: Callable
    verify_text: Callable | None
    screenshot_tokens: Callable


def _local_screenshot_tokens(
    conn, moments: dict[int, list[int]], max_image_tokens
) -> int:
    """A screenshot's local cost is the viewport of the slice it renders from, under the card's per-image token ceiling it will be sent at. The moments are already per slice — a screenshot belongs to exactly the slice whose replay produces it — so what a screenshot costs and what it shows come from the same slice by construction."""
    total = 0
    for slice_id, timestamps in moments.items():
        viewport = slice_viewport(conn, slice_id)
        if viewport is None:
            raise ValueError(
                f"slice {slice_id} carries no viewport (no Meta width/height) "
                f"— local screenshot cost is unpriceable and the render would have "
                f"nothing to size the browser with"
            )
        total += len(timestamps) * qwen_image_tokens(
            *image_geometry(*viewport, max_image_tokens)
        )
    return total


def cost_model_for(model: str) -> tuple[CostModel, int, float]:
    """(CostModel, context_tokens, headroom) for a model name, either client. What a screenshot costs is the model card's: an openai-compatible card's activity-screenshot ceiling prices exactly the geometry its server sends; Gemini prices every image flat."""
    from .model.cards import OPENAI_COMPATIBLE, context_tokens, conversation, headroom

    context, room = context_tokens(model), headroom(model)
    if conversation(model) == OPENAI_COMPATIBLE:
        from .model.cards import max_image_tokens

        ceiling = max_image_tokens(model)["activity_screenshots"]
        return (
            CostModel(
                count_text=lambda texts: count_text_tokens("\n".join(texts)),
                verify_text=None,
                screenshot_tokens=lambda conn, moments: _local_screenshot_tokens(
                    conn, moments, ceiling
                ),
            ),
            context,
            room,
        )

    from .model.cards import media_resolution, models_speaking
    from .model.gemini import count_text_tokens as gemini_count
    from .model.gemini import image_tier_tokens

    # The activity-screenshots are the priced payload's whole image spend, and
    # they ride the card's activity_screenshots tier (falling back to the
    # pulled_screenshots tier on a card that declares none) — flat per image
    # within the tier, whatever the pixels.
    tiers = media_resolution(model)
    per_screenshot = image_tier_tokens(
        tiers.get("activity_screenshots", tiers.get("pulled_screenshots"))
    )
    screenshot_tokens = lambda conn, moments: (
        sum(len(timestamps) for timestamps in moments.values()) * per_screenshot
    )
    if models_speaking(OPENAI_COMPATIBLE):
        return (
            CostModel(
                count_text=lambda texts: count_text_tokens("\n".join(texts)),
                verify_text=lambda texts: gemini_count(model, texts),
                screenshot_tokens=screenshot_tokens,
            ),
            context,
            room,
        )
    # No local card, no pinned tokenizer: the free countTokens API is the one
    # text instrument, so propose and close are the same exact number and
    # there is nothing left to verify.
    return (
        CostModel(
            count_text=lambda texts: gemini_count(model, texts),
            verify_text=None,
            screenshot_tokens=screenshot_tokens,
        ),
        context,
        room,
    )


def _pricing_block(model: str, turn1_total: int) -> dict:
    """The price report's dollar side, present exactly when the model's card declares per-token prices (a free model has no dollar side). turn1_input_usd is the composed turn-1 payload at the declared fresh-input rate — exact for what it prices, undiscounted (implicit caching pays back in dollars only when it hits), and stated so the caller's own arithmetic against the deployment's spend periods (`locus status`) needs no price lookup. What generation will add cannot be counted before the model has spoken, so it is not counted. The declared prices ride along with their best-effort registry verification — the card is the price of record; the verification only says whether the registry agrees."""
    from .model.cards import declared_pricing

    prices = declared_pricing(model)
    if not prices:
        return {}
    from .spend import verify_prices

    return {
        "pricing": {
            "declared_usd_per_mtok": prices,
            "turn1_input_usd": round(turn1_total * prices["input_per_mtok"] / 1e6, 6),
            "verification": verify_prices(model, prices),
        }
    }


def price_payload(
    conn: sqlite3.Connection,
    window,
    *,
    model: str,
    prompts: Callable[[list[str]], tuple[str, str]],
    site_contexts: dict[str, str],
    screenshot_interval_ms: int = SCREENSHOT_INTERVAL_MS,
    cost_model: CostModel | None = None,
    context_tokens: int | None = None,
    headroom: float | None = None,
) -> dict:
    """Price one model-payload, exactly, spending nothing — the default `locus analyze` invocation, over a resolved window (window.resolve_window).

    The whole composed turn-1 payload is counted (exact-verified where the backend has an instrument) and stated against the card's context: total tokens, context percentage, and whether it fits under headroom x context. When the window holds several slices, each is also priced standalone — the whittling surface. When one slice alone out-prices the budget, its route-boundary pieces (window.slice_pieces) are enumerated and priced, each addressable as `<slice>#<k>`.

    prompts(labels) → (system_prompt, task) supplies the contract for a window with those slice labels, so every measured set is counted under the exact bytes its own run would carry. cost_model/context_tokens/headroom exist for tests to inject; production callers pass the model name and let cost_model_for resolve them.
    """
    from .window import resolve_window, slice_pieces

    if cost_model is None or context_tokens is None or headroom is None:
        cost_model, context_tokens, headroom = cost_model_for(model)
    budget = int(context_tokens * headroom)

    def measure(win) -> dict:
        system_prompt, task = prompts(win.labels)
        texts, moments, n_events = window_payload(
            conn,
            win,
            system_prompt=system_prompt,
            task=task,
            site_contexts=site_contexts,
            screenshot_interval_ms=screenshot_interval_ms,
        )
        n_screenshots = sum(len(timestamps) for timestamps in moments.values())
        text = cost_model.count_text(texts)
        screenshots = cost_model.screenshot_tokens(conn, moments)
        return {
            "texts": texts,
            "n_events": n_events,
            "n_screenshots": n_screenshots,
            "text_tokens": text,
            "screenshot_tokens": screenshots,
            "total": text + screenshots,
        }

    def pct(total: int) -> float:
        return round(total / context_tokens * 100, 1)

    whole = measure(window)
    text, verified = whole["text_tokens"], False
    if cost_model.verify_text is not None:
        text = cost_model.verify_text(whole["texts"])
        verified = True
    total = text + whole["screenshot_tokens"]

    price: dict = {
        "model": model,
        "context_tokens": context_tokens,
        "headroom": headroom,
        "budget_tokens": budget,
        "turn1": {
            "text_tokens": text,
            "screenshot_tokens": whole["screenshot_tokens"],
            "total_tokens": total,
            "context_pct": pct(total),
            "verified": verified,
            "fits": total <= budget,
        },
        **_pricing_block(model, total),
        "n_events": whole["n_events"],
        "n_activity_screenshots": whole["n_screenshots"],
        "cache_note": (
            "input tokens count toward fit regardless of caching — the discount is dollars, never context"
        ),
    }

    oversized = []
    if len(window.slices) > 1:
        per_slice = []
        for s in window.slices:
            m = measure(
                resolve_window(
                    conn, [s.slice_id], window.window_start, window.window_end
                )
            )
            if m["total"] > budget:
                oversized.append(s)
            per_slice.append(
                {
                    "slice": s.slice,
                    "visitor": s.visitor,
                    "standalone_tokens": m["total"],
                    "context_pct": pct(m["total"]),
                }
            )
        price["per_slice"] = per_slice
    elif total > budget:
        oversized = list(window.slices)

    if oversized:
        pieces = []
        for s in oversized:
            bounds = slice_pieces(conn, s.slice_id)
            if len(bounds) < 2:
                pieces.append(
                    {
                        "slice": s.slice,
                        "note": (
                            "no internal route boundaries to divide at — "
                            "widen --screenshot-interval to shrink this "
                            "slice's screenshot spend"
                        ),
                    }
                )
                continue
            for k, (url, ws, we) in enumerate(bounds, 1):
                m = measure(resolve_window(conn, [s.slice_id], ws, we))
                pieces.append(
                    {
                        "address": f"{s.slice}#{k}",
                        "page": url,
                        "turn1_tokens": m["total"],
                        "context_pct": pct(m["total"]),
                        "fits": m["total"] <= budget,
                    }
                )
        price["pieces"] = pieces

    return price
