"""Model-payload pricing — the composition plane's only token surface.

Token costs are computed fresh at decision time, never stored: a stored count is a staleness bug waiting for a projection or distillation change to make it a lie.

Counting is exact by construction: the model-payload is composed through the real composer (compose_window), and each backend counts it with its own arithmetic. A local card's request is composed into the real client and counted by its server's own code, run in this process (mlx/count.py), with each activity-screenshot stood in for at the size it will render at; a Gemini card's text is counted by the free countTokens API, with each activity-screenshot flat at its tier (the cost models below say what each leaves out, and why).

The bound is turn 1, and only turn 1: an analysis fits when the model-payload actually sent on the opening call comes in under the card's headroom x context. Nothing later is counted here, because nothing later exists here — what the model will think, write, or ask to see cannot be measured before they have been asked. The conversation grows into the context left below that ceiling, and every turn past the first is measured exactly, at the wire, against the context that is genuinely left (the engine serves pulls while they fit and refuses the rest). Caching changes none of this — cached tokens still occupy the context; the discount is dollars, never counted toward fit.

Fitting is the caller's judgment, informed: the price report carries the whole payload's turn-1 total as a percentage of the card's context, each slice's standalone price when the set holds several, and — when one slice alone out-prices the budget — that slice's route-boundary pieces, individually priced and addressable as `<slice>#<k>`. Nothing here ever cuts or splits on its own.

The prompts are a callback, not strings: every set measured here — the whole window, each slice standalone, each route-boundary piece — is its own window with its own labels, so the caller supplies prompts(labels) → (system_prompt, task, tools) and each measurement composes the exact request that set's run would send.
"""

import math
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .model.protocol import Conversation
from .select_screenshots import SCREENSHOT_INTERVAL_MS, screenshot_moments


def image_geometry(
    model: str, width: int, height: int, max_image_tokens: int
) -> tuple[int, int]:
    """The geometry a screenshot is actually sent at under the card's per-image token ceiling — the one place the ceiling turns into pixels: the local backend resizes by it, and pricing counts what it returns. An image's cost is the server's own count of it (mlx/count.py image_tokens), so the ceiling bounds everything the image adds to the prompt. A screenshot already under the ceiling passes untouched (never upscaled); over it, the screenshot scales down preserving aspect to the largest geometry whose cost fits. The model's processor never encodes an image below its own minimum size, so a ceiling under what the smallest image costs is unrealizable and fails loud."""
    from .mlx.count import image_tokens

    floor_tokens = image_tokens(model, 1, 1)
    if max_image_tokens < floor_tokens:
        raise ValueError(
            f"per-image token ceiling {max_image_tokens} is below the "
            f"{floor_tokens} tokens the model's smallest image costs — no "
            f"geometry realizes it; raise max_image_tokens in the card"
        )
    cost = image_tokens(model, width, height)
    if cost <= max_image_tokens:
        return width, height
    scale = math.sqrt(max_image_tokens / cost)
    sent_w = max(1, math.floor(width * scale))
    sent_h = max(1, math.floor(height * scale))
    # The processor rounds each side to its patch grid, which can round the cost back up
    # past the ceiling; step down until it fits.
    while image_tokens(model, sent_w, sent_h) > max_image_tokens:
        sent_w = max(1, math.floor(sent_w * 0.98))
        sent_h = max(1, math.floor(sent_h * 0.98))
    return sent_w, sent_h


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


def _placeholders(window, moments: dict[int, list[int]]) -> dict:
    labels = {s.slice_id: s.label for s in window.slices}
    return {
        (labels[sid], ts): Path(f"screenshot_{labels[sid]}_{ts}.png")
        for sid, timestamps in moments.items()
        for ts in timestamps
    }


def window_payload(
    conn: sqlite3.Connection,
    window,
    *,
    system_prompt: str,
    task: str,
    site_contexts: dict[str, str],
    moments: dict[int, list[int]],
) -> tuple[list[str], int]:
    """The turn-1 model-payload's text as (text parts, n_events), composed by the real composer over a resolved window (window.resolve_window) and its activity-screenshot moments (screenshot_moments' shape) — nothing rendered, nothing sent. task is the entire question as the caller sends it last on the turn that writes the answer; it is sent, so it is counted."""
    from .analyze import compose_window

    tally = TallyConversation()
    manifest = compose_window(
        conn,
        tally,
        window,
        site_contexts=site_contexts,
        screenshots=_placeholders(window, moments),
    )
    return [system_prompt, *tally.texts, task], manifest.n_events


@dataclass
class Count:
    text_tokens: int
    screenshot_tokens: int
    n_events: int


@dataclass
class CostModel:
    """How one backend counts the turn-1 model-payloads of the sets a price report measures: count(conn, sets, site_contexts) returns one Count per set, in order. Each set is (window, contract, moments): contract is (system_prompt, task, tools) — the system prompt, the question sent last, and the tool declarations the request carries; moments are the set's activity-screenshot moments ({slice_id: timestamps}, the shape screenshot_moments emits). A backend is handed every set at once, so it can count them however its instrument runs fastest."""

    count: Callable[..., list[Count]]


def _local_count(model: str) -> Callable[..., list[Count]]:
    """A local card's payload counted by its server's own code (mlx/count.py): the window composed into the real client, so the request is the one a run sends, with each activity-screenshot stood in for by a blank image of the size the renderer will give it — the slice's viewport at that moment (slices.viewport_at) — which the client then shrinks under the card's ceiling exactly as it shrinks the real one. The count reads an image's size, never its pixels."""

    def one(conn, window, contract, moments, site_contexts) -> Count:
        import tempfile

        from locus.evidence.slices import viewport_at, viewport_events
        from PIL import Image

        from .analyze import compose_window
        from .mlx.count import image_tokens, prompt_tokens
        from .model.factory import make_conversation

        system_prompt, task, tools = contract
        labels = {s.slice_id: s.label for s in window.slices}
        with tempfile.TemporaryDirectory(prefix="locus_price_") as tmp:
            shots = {}
            for slice_id, timestamps in moments.items():
                events = viewport_events(conn, slice_id)
                for ts in timestamps:
                    width, height = viewport_at(events, ts)
                    blank = Path(tmp) / f"{width}x{height}.png"
                    if not blank.exists():
                        Image.new("RGB", (width, height)).save(blank)
                    shots[(labels[slice_id], ts)] = blank
            conv = make_conversation(model, system_prompt, record_dir=None)
            manifest = compose_window(
                conn, conv, window, site_contexts=site_contexts, screenshots=shots
            )
            conv.add_user_text(task)
            request = conv.wire_request(tools)
            total = prompt_tokens(request)
            screenshots = 0
            for url in conv.sent_images():
                with Image.open(url) as image:
                    screenshots += image_tokens(conv.model, *image.size)
        return Count(total - screenshots, screenshots, manifest.n_events)

    def count(conn, sets, site_contexts) -> list[Count]:
        # One after another: the count is CPU work over one shared tokenizer, which
        # refuses concurrent use (mlx/serve.py's preprocessing lock says why).
        return [one(conn, *measured, site_contexts) for measured in sets]

    return count


def _gemini_count(model: str) -> Callable[..., list[Count]]:
    """A Gemini card's payloads counted by the free countTokens API — Gemini's own arithmetic, so exact — with each activity-screenshot flat at the tier it rides (the card's activity_screenshots tier, falling back to its pulled_screenshots tier): within a tier an image's cost is independent of its pixels (gemini.py owns the per-tier numbers). Every set is composed first, then all are counted at once (gemini.count_texts). The tool declaration the run's request carries (~100 tokens — measured +98 on the single-slice request_screenshots shape) stays uncounted: the SDK's countTokens refuses a tools config on the Developer API, and a hand formula for its serialization would trade exact for a guess, so a billed turn-1 runs that constant over the price."""
    from .model.cards import media_resolution, model_id
    from .model.gemini import count_texts, image_tier_tokens

    tiers = media_resolution(model)
    per_screenshot = image_tier_tokens(
        tiers.get("activity_screenshots", tiers.get("pulled_screenshots"))
    )

    def count(conn, sets, site_contexts) -> list[Count]:
        payloads = []
        for window, (system_prompt, task, _), moments in sets:
            texts, n_events = window_payload(
                conn,
                window,
                system_prompt=system_prompt,
                task=task,
                site_contexts=site_contexts,
                moments=moments,
            )
            n_screenshots = sum(len(timestamps) for timestamps in moments.values())
            payloads.append((texts, n_screenshots, n_events))
        text_tokens = count_texts(model_id(model), [texts for texts, _, _ in payloads])
        return [
            Count(tokens, n_screenshots * per_screenshot, n_events)
            for tokens, (_, n_screenshots, n_events) in zip(text_tokens, payloads)
        ]

    return count


def cost_model_for(model: str) -> tuple[CostModel, int, float]:
    """(CostModel, context_tokens, headroom) for a model name, either client."""
    from .model.cards import OPENAI_COMPATIBLE, context_tokens, conversation, headroom

    context, room = context_tokens(model), headroom(model)
    if conversation(model) == OPENAI_COMPATIBLE:
        return CostModel(_local_count(model)), context, room
    return CostModel(_gemini_count(model)), context, room


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
    prompts: Callable[[list[str]], tuple[str, str, list[dict]]],
    site_contexts: dict[str, str],
    screenshot_interval_ms: int = SCREENSHOT_INTERVAL_MS,
    cost_model: CostModel | None = None,
    context_tokens: int | None = None,
    headroom: float | None = None,
) -> dict:
    """Price one model-payload, exactly, spending nothing — the default `locus analyze` invocation, over a resolved window (window.resolve_window).

    The whole composed turn-1 payload is counted and stated against the card's context: total tokens, context percentage, and whether it fits under headroom x context. When the window holds several slices, each is also priced standalone — the whittling surface. When one slice alone out-prices the budget, its route-boundary pieces (window.slice_pieces) are enumerated and priced, each addressable as `<slice>#<k>`.

    prompts(labels) → (system_prompt, task, tools) supplies the contract for a window with those slice labels, so every measured set is counted under the exact request its own run would send. cost_model/context_tokens/headroom exist for tests to inject; production callers pass the model name and let cost_model_for resolve them.
    """
    from .window import resolve_window, slice_pieces

    if cost_model is None or context_tokens is None or headroom is None:
        cost_model, context_tokens, headroom = cost_model_for(model)
    budget = int(context_tokens * headroom)

    def measure(windows: list) -> list[dict]:
        sets = []
        for win in windows:
            moments = screenshot_moments(
                conn, win, screenshot_interval_ms=screenshot_interval_ms
            )
            sets.append((win, prompts(win.labels), moments))
        counted = cost_model.count(conn, sets, site_contexts)
        return [
            {
                "n_events": c.n_events,
                "n_screenshots": sum(
                    len(timestamps) for timestamps in moments.values()
                ),
                "text_tokens": c.text_tokens,
                "screenshot_tokens": c.screenshot_tokens,
                "total": c.text_tokens + c.screenshot_tokens,
            }
            for c, (_, _, moments) in zip(counted, sets, strict=True)
        ]

    def pct(total: int) -> float:
        return round(total / context_tokens * 100, 1)

    # The whole window and each slice standalone are one round; the pieces of whichever
    # slice that round finds over budget are the next.
    standalone = (
        [
            resolve_window(conn, [s.slice_id], window.window_start, window.window_end)
            for s in window.slices
        ]
        if len(window.slices) > 1
        else []
    )
    whole, *per_slice_measured = measure([window, *standalone])
    total = whole["total"]

    price: dict = {
        "model": model,
        "context_tokens": context_tokens,
        "headroom": headroom,
        "budget_tokens": budget,
        "turn1": {
            "text_tokens": whole["text_tokens"],
            "screenshot_tokens": whole["screenshot_tokens"],
            "total_tokens": total,
            "context_pct": pct(total),
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
    if standalone:
        per_slice = []
        for s, m in zip(window.slices, per_slice_measured):
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
        pieces, windows = [], []
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
                pieces.append({"address": f"{s.slice}#{k}", "page": url})
                windows.append(resolve_window(conn, [s.slice_id], ws, we))
        addressed = [piece for piece in pieces if "address" in piece]
        for piece, m in zip(addressed, measure(windows), strict=True):
            piece.update(
                turn1_tokens=m["total"],
                context_pct=pct(m["total"]),
                fits=m["total"] <= budget,
            )
        price["pieces"] = pieces

    return price
