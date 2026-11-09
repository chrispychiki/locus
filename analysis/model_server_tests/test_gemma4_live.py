"""Gemma 4 on the running server: what gemma4.py patches, held to the model's own behavior. Runs only while the server holds a Gemma 4 card."""

import pytest
import requests
from locus.analysis.mlx.chat import loaded_model
from PIL import Image
from test_integration import CHAT, COLORS, TIMEOUT, chat, image_part

pytestmark = pytest.mark.family("gemma4")


def test_a_thought_comes_back_without_the_channel_header():
    message = chat(
        [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]
    ).json()["choices"][0]["message"]
    assert "391" in message["content"]
    assert message["reasoning"], "thinking enabled but no reasoning returned"
    assert not message["reasoning"].startswith("thought"), message["reasoning"][:40]


def test_the_thinking_budget_ends_thinking_at_gemmas_own_marker():
    """The budget counts and closes thinking by the card's thinking_markers; under the wrong markers it never fires and the thinking runs on."""
    from locus.analysis.mlx import count

    budget = 32
    response = requests.post(
        CHAT,
        json={
            "model": loaded_model(),
            "messages": [
                {
                    "role": "user",
                    "content": "What is 17 * 23? Check it three different ways, then answer with the number.",
                }
            ],
            "max_tokens": 2048,
            "enable_thinking": True,
            "thinking_budget": budget,
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    with count._serving(loaded_model()):
        tokenizer = count._resources(loaded_model())[0].tokenizer
    thought = tokenizer.encode(message["reasoning"], add_special_tokens=False)
    assert len(thought) <= budget + 2, f"{len(thought)} tokens of thinking"
    assert "391" in message["content"]


def test_an_image_prompt_longer_than_a_prefill_chunk_is_read_in_order(tmp_path):
    """Sixteen 960x720 cards, 20x15 cells each, make a prompt of several prefill chunks, each ending at an image boundary; the count is the server's, and the model still finds the card at one position."""
    from locus.analysis.mlx.count import prompt_tokens

    sequence = list(COLORS) * 3
    sequence = sequence[:16]
    parts = []
    for i, color in enumerate(sequence, start=1):
        path = tmp_path / f"card_{i}.png"
        Image.new("RGB", (960, 720), COLORS[color]).save(path)
        parts.append({"type": "text", "text": f"Image {i}:"})
        parts.append(image_part(path))
    parts.append(
        {
            "type": "text",
            "text": "Each image above is a solid color card preceded by its number. What is the color of Image 11? Answer with the color alone.",
        }
    )
    body = {
        "model": loaded_model(),
        "messages": [{"role": "user", "content": parts}],
        "max_tokens": 2048,
        "enable_thinking": True,
    }
    response = requests.post(CHAT, json=body, timeout=TIMEOUT)
    response.raise_for_status()
    served = response.json()
    assert served["usage"]["prompt_tokens"] == prompt_tokens(body)
    assert served["usage"]["prompt_tokens"] > 4096, "the prompt spans several chunks"
    assert sequence[10] in served["choices"][0]["message"]["content"].lower()
