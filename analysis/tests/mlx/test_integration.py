"""Live integration: the running server, real inference, images that must be apprehended — not merely received.

Three levels of proof: reception (prompt_tokens reflect vision tokens), tokenization (accounting stays stable across cache hits), and semantics (the model states visual facts — counts, colors, positions — that exist only in pixels).

Requires the server (uv run locus-mlx-serve <models.toml key>) on the card's endpoint; every test skips when it isn't there. Fixtures are drawn with PIL at test time — nothing binary is committed.

Contract notes for the upstream server: thinking arrives as `reasoning` (not `reasoning_content`); rich per-request stats live in `timings`, not usage extensions; images travel as plain absolute paths (upstream does not parse file:// for images); structured output is llguidance via response_format json_schema.
"""

import json
import platform
from pathlib import Path

import pytest
import requests
from locus.analysis.mlx.chat import loaded_model
from locus.analysis.mlx.serve import sole_bind
from PIL import Image, ImageDraw

_host, _port = sole_bind()
SERVER = f"http://{_host}:{_port}"
CHAT = f"{SERVER}/v1/chat/completions"
TIMEOUT = 900

COLORS = {
    "red": (220, 30, 30),
    "green": (30, 160, 60),
    "blue": (40, 70, 220),
    "yellow": (230, 210, 40),
    "purple": (140, 60, 180),
    "orange": (240, 140, 30),
}


def server_up() -> bool:
    try:
        return requests.get(f"{SERVER}/docs", timeout=2).status_code == 200
    except requests.RequestException:
        return False


pytestmark = [
    pytest.mark.skipif(platform.system() != "Darwin", reason="MLX is macOS-only"),
    pytest.mark.skipif(
        not server_up(), reason=f"MLX server not running on {_host}:{_port}"
    ),
]


def chat(
    messages,
    *,
    schema=None,
    enable_thinking=True,
    max_tokens=4096,
    stream=False,
):
    body = {
        "model": loaded_model(),
        "messages": messages,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "schema": schema},
        }
    response = requests.post(CHAT, json=body, timeout=TIMEOUT, stream=stream)
    response.raise_for_status()
    return response


def image_part(path):
    return {"type": "image_url", "image_url": {"url": str(path)}}


def smiley_image(tmp_path, count=4, color="red"):
    img = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(img)
    for i in range(count):
        x = 30 + i * 95
        draw.ellipse([x, 60, x + 70, 130], fill=COLORS[color])
        draw.ellipse([x + 18, 80, x + 28, 90], fill="black")
        draw.ellipse([x + 42, 80, x + 52, 90], fill="black")
        draw.arc([x + 15, 90, x + 55, 120], 20, 160, fill="black", width=4)
    path = tmp_path / "smileys.png"
    img.save(path)
    return path


def color_card(tmp_path, index, color):
    img = Image.new("RGB", (320, 240), COLORS[color])
    path = tmp_path / f"card_{index}.png"
    img.save(path)
    return path


def test_text_only_thinking_split():
    response = chat(
        [
            {"role": "user", "content": "What is 17 * 23? Answer with the number."},
        ]
    ).json()
    message = response["choices"][0]["message"]
    assert "391" in message["content"]
    assert message.get("reasoning"), "thinking enabled but no reasoning returned"
    usage = response["usage"]
    assert usage["completion_tokens"] > 0
    assert usage["prompt_tokens"] > 0


def test_image_counting_and_color_via_schema(tmp_path):
    path = smiley_image(tmp_path, count=4, color="red")
    schema = {
        "type": "object",
        "properties": {
            "object_count": {"type": "integer"},
            "object_color": {"type": "string", "enum": list(COLORS)},
        },
        "required": ["object_count", "object_color"],
    }
    text_only_tokens = chat(
        [
            {"role": "user", "content": "Reply with exactly: ok"},
        ],
        enable_thinking=False,
        max_tokens=8,
    ).json()["usage"]["prompt_tokens"]
    response = chat(
        [
            {
                "role": "user",
                "content": [
                    image_part(path),
                    {
                        "type": "text",
                        "text": "How many smiley faces are in this image and what color are they? Respond with JSON.",
                    },
                ],
            },
        ],
        schema=schema,
    ).json()

    message = response["choices"][0]["message"]
    parsed = json.loads(message["content"])
    assert parsed["object_count"] == 4, f"model saw {parsed['object_count']} smileys"
    assert parsed["object_color"] == "red"
    assert response["usage"]["prompt_tokens"] > text_only_tokens + 50, (
        "prompt_tokens do not reflect vision tokens — image not received"
    )


def test_interleaved_order_survives_many_images(tmp_path):
    """12 solid-color cards interleaved with their index labels, and the model must report the color at one specific position. Passes only if interleaving and order preserve image identity through the vision tower."""
    sequence = [
        "red",
        "green",
        "blue",
        "yellow",
        "purple",
        "orange",
        "blue",
        "red",
        "yellow",
        "green",
        "purple",
        "red",
    ]
    parts = []
    for i, color in enumerate(sequence, start=1):
        parts.append({"type": "text", "text": f"Image {i}:"})
        parts.append(image_part(color_card(tmp_path, i, color)))
    parts.append(
        {
            "type": "text",
            "text": "Each image above is a solid color card and is "
            "preceded by its number. What is the color of "
            "Image 5? Respond with JSON.",
        }
    )
    schema = {
        "type": "object",
        "properties": {"color": {"type": "string", "enum": list(COLORS)}},
        "required": ["color"],
    }
    response = chat([{"role": "user", "content": parts}], schema=schema).json()

    parsed = json.loads(response["choices"][0]["message"]["content"])
    assert parsed["color"] == sequence[4], (
        f"asked for image 5 ({sequence[4]}), model said {parsed['color']} — interleaving scrambled image identity"
    )

    # The wire half, mechanical: the server records each request's rendered prompt
    # (serve.py install_prompt_recording), and this request's labels must sit exactly where the
    # message put them — each "Image N:" immediately before its own vision block, the question
    # after the last. The semantic assert above cannot prove placement (ordinal matching answers
    # it even from a collapsed rendering); this reads the rendering itself.
    rendered = (
        Path(__file__).resolve().parents[3] / "data" / "mlx" / "rendered_prompt.txt"
    ).read_text()
    vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
    segments = rendered.split(vision_block)
    assert len(segments) == len(sequence) + 1, (
        f"expected {len(sequence)} vision blocks in the rendered prompt, found {len(segments) - 1}"
    )
    assert segments[0].endswith("Image 1:"), segments[0][-40:]
    for i in range(2, len(sequence) + 1):
        assert segments[i - 1] == f"Image {i}:", (
            f"between vision blocks {i - 1} and {i} the rendering carries "
            f"{segments[i - 1]!r}, not the label the message put there"
        )
    assert segments[-1].startswith("Each image above"), segments[-1][:40]


def test_followup_turn_uses_prior_answer(tmp_path):
    """KNOWN UPSTREAM GAP: assistant `reasoning` is parsed but not replayed into the chat template, so only content carries across turns. This asserts the conversational floor that must hold regardless: a follow-up referencing the prior answer works."""
    path = smiley_image(tmp_path, count=3, color="blue")
    first = chat(
        [
            {
                "role": "user",
                "content": [
                    image_part(path),
                    {"type": "text", "text": "How many smiley faces? Just the number."},
                ],
            },
        ]
    ).json()["choices"][0]["message"]
    assert "3" in first["content"]

    second = chat(
        [
            {
                "role": "user",
                "content": [
                    image_part(path),
                    {"type": "text", "text": "How many smiley faces? Just the number."},
                ],
            },
            {
                "role": "assistant",
                "content": first["content"],
                "reasoning": first.get("reasoning"),
            },
            {
                "role": "user",
                "content": "Now double that count and answer with just the number.",
            },
        ]
    ).json()["choices"][0]["message"]
    assert "6" in second["content"]


def test_streaming_emits_reasoning_then_content_then_usage():
    deltas = {"reasoning": "", "content": ""}
    usage = None
    with chat(
        [{"role": "user", "content": "Name three primary colors, comma-separated."}],
        stream=True,
    ) as response:
        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[len(b"data: ") :]
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("reasoning"):
                    deltas["reasoning"] += delta["reasoning"]
                if delta.get("content"):
                    deltas["content"] += delta["content"]
    assert deltas["reasoning"], "no reasoning deltas streamed"
    assert "red" in deltas["content"].lower()
    assert usage is not None and usage["prompt_tokens"] > 0


def test_prompt_token_accounting_is_stable_across_cache_hits(tmp_path):
    """A repeat of an image request is served from the vision feature cache (MLX_VLM_VISION_CACHE_SIZE, keyed on the image path) — reported prompt_tokens must describe the request, not what the server skipped computing for it."""
    path = smiley_image(tmp_path, count=2, color="green")
    messages = [
        {
            "role": "user",
            "content": [
                image_part(path),
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
    first = chat(messages, max_tokens=256).json()["usage"]
    second = chat(messages, max_tokens=256).json()["usage"]
    assert first["prompt_tokens"] > 0
    assert second["prompt_tokens"] == first["prompt_tokens"], (
        f"prompt_tokens drifted across identical requests: "
        f"{first['prompt_tokens']} → {second['prompt_tokens']} — cache trim "
        f"leaked into accounting"
    )


def test_reads_ui_scale_text_from_screenshot(tmp_path):
    """Images pass the preprocessor at native resolution; this asserts the property that actually matters downstream — UI-scale text in a screenshot-sized image is legible to the model, not just received by it."""
    from PIL import ImageFont

    img = Image.new("RGB", (1280, 800), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=16)
    draw.text(
        (48, 40), "Meridian Coffee", fill="black", font=ImageFont.load_default(size=28)
    )
    draw.text((48, 120), "Huila, Colombia — washed process", fill="black", font=font)
    draw.text((48, 150), "Price: $23 / 250g", fill="black", font=font)
    draw.text((48, 180), "Order code: KX-740", fill="black", font=font)
    path = tmp_path / "screenshot.png"
    img.save(path)

    schema = {
        "type": "object",
        "properties": {
            "price_dollars": {"type": "integer"},
            "order_code": {"type": "string"},
        },
        "required": ["price_dollars", "order_code"],
    }
    response = chat(
        [
            {
                "role": "user",
                "content": [
                    image_part(path),
                    {
                        "type": "text",
                        "text": "Read the price in dollars and the order code from this page. Respond with JSON.",
                    },
                ],
            },
        ],
        schema=schema,
    ).json()
    parsed = json.loads(response["choices"][0]["message"]["content"])
    assert parsed["price_dollars"] == 23, f"misread price: {parsed}"
    assert parsed["order_code"].strip().upper() == "KX-740", f"misread code: {parsed}"


def test_generation_speed_floor():
    """The floor is on the server's own decode rate for this request (timings.predicted_per_second), never on client wall clock: the server is shared, so wall time includes unbounded admission-queue and prefill-serialization wait, and a healthy server queued behind a large window would read as crawling. Decode sharing is the one contention that survives into the measurement, and the admission cap (models.toml) bounds it to one concurrent stream — measured 4.5 tok/s beside a 210k-context decode, still above this floor — so a reading below it is pathology (lockstep decode, compression thrash), not load."""
    response = chat(
        [{"role": "user", "content": "Write one sentence about coffee."}],
        enable_thinking=False,
        max_tokens=128,
    ).json()
    timings = response["timings"]
    assert timings, "the server stopped reporting timings — the stats contract moved"
    assert timings["predicted_per_second"] > 3, (
        f"{timings['predicted_n']} tokens decoded at "
        f"{timings['predicted_per_second']:.1f} tok/s — generation is crawling"
    )
