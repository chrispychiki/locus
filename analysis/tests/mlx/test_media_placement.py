"""Unit tests for the media-placement patch — no model checkpoint, no server.

The patch rests on upstream properties: that the template layer places a request's media by the markers left in each message (falling back to the last user message only when there are none), and that a message's parts, handed through in order, reach the built message in that order.
Both are asserted against upstream's own message building (`return_messages=True`), not read off a comment. What the built messages then render as is the model's own chat template's doing, asserted against the live server's recorded rendering in test_integration.
"""

import locus.analysis.mlx.media_placement as mp
import pytest
from locus.analysis.mlx.media_placement import carries_media, install_media_placement
from mlx_vlm import prompt_utils
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.server import openai as server_openai

CONFIG = {"model_type": "qwen3_5_moe"}

EVIDENCE = {
    "role": "user",
    "content": [
        {"type": "text", "text": "the event stream"},
        {"type": "image_url", "image_url": {"url": "/tmp/shot.png"}},
    ],
}
DRAFT = {"role": "assistant", "content": "a draft answer"}
CHECK = {"role": "user", "content": [{"type": "text", "text": "check the draft"}]}


@pytest.fixture(autouse=True)
def restore_symbols():
    originals = (
        server_openai.extract_text_from_content,
        prompt_utils.extract_text_from_content,
        prompt_utils.get_message_json,
    )
    yield
    (
        server_openai.extract_text_from_content,
        prompt_utils.extract_text_from_content,
        prompt_utils.get_message_json,
    ) = originals


def image_bearing_message(messages):
    """Index of the message upstream renders the image into."""
    built = apply_chat_template(
        None, CONFIG, messages, num_images=1, return_messages=True
    )

    def holds_image(content):
        if isinstance(content, list):
            return any(part.get("type") in ("image", "image_url") for part in content)
        return "<|vision_start|>" in (content or "")

    carrying = [i for i, message in enumerate(built) if holds_image(message["content"])]
    assert len(carrying) == 1, built
    return carrying[0]


class TestCarriesMedia:
    def test_a_string_carries_none(self):
        assert carries_media("just text") is False

    def test_a_text_only_part_list_carries_none(self):
        assert carries_media([{"type": "text", "text": "hi"}]) is False

    def test_an_image_part_counts(self):
        assert carries_media(EVIDENCE["content"]) is True


class TestUpstreamPlacement:
    """What the patch buys, stated against upstream's own allocation."""

    def test_stripped_content_hangs_the_image_on_the_newest_user_turn(self):
        stripped = [
            {"role": "user", "content": "the event stream"},
            DRAFT,
            {"role": "user", "content": "check the draft"},
        ]
        assert image_bearing_message(stripped) == 2

    def test_marker_content_keeps_the_image_in_the_turn_that_carried_it(self):
        assert image_bearing_message([EVIDENCE, DRAFT, CHECK]) == 0

    def test_a_single_turn_renders_the_same_either_way(self):
        assert image_bearing_message([EVIDENCE]) == 0
        assert (
            image_bearing_message([{"role": "user", "content": "the event stream"}])
            == 0
        )


class TestInstall:
    def test_refuses_boot_on_upstream_source_drift(self, monkeypatch):
        monkeypatch.setattr(mp, "EXPECTED_ENDPOINT_SHA256", "0" * 64)
        before = server_openai.extract_text_from_content
        with pytest.raises(SystemExit, match="upstream drift"):
            install_media_placement()
        assert server_openai.extract_text_from_content is before

    def test_installed_it_keeps_media_and_flattens_everything_else(self):
        install_media_placement()
        keep = server_openai.extract_text_from_content
        assert keep(EVIDENCE["content"]) == EVIDENCE["content"]
        assert keep(CHECK["content"]) == "check the draft"
        assert keep("already text") == "already text"


INTERLEAVED = {
    "role": "user",
    "content": [
        {"type": "text", "text": "before the first shot"},
        {"type": "image_url", "image_url": {"url": "/tmp/a.png"}},
        {"type": "text", "text": "between the shots"},
        {"type": "image_url", "image_url": {"url": "/tmp/b.png"}},
        {"type": "text", "text": "after the last shot"},
    ],
}


def rendered_kinds(content):
    return [
        "image" if part.get("type") in ("image", "image_url") else part["text"]
        for part in content
    ]


class TestWithinMessageOrder:
    """The other half of placement: a message's own interleaving survives to the built message."""

    def test_upstream_collapses_the_message_to_images_first(self):
        built = apply_chat_template(
            None, CONFIG, [INTERLEAVED], num_images=2, return_messages=True
        )
        assert rendered_kinds(built[0]["content"]) == [
            "image",
            "image",
            "before the first shot between the shots after the last shot",
        ]

    def test_installed_parts_build_in_the_order_the_message_carried(self):
        install_media_placement()
        built = apply_chat_template(
            None, CONFIG, [INTERLEAVED], num_images=2, return_messages=True
        )
        assert rendered_kinds(built[0]["content"]) == [
            "before the first shot",
            "image",
            "between the shots",
            "image",
            "after the last shot",
        ]

    def test_a_part_type_outside_the_patch_takes_upstreams_own_path(self):
        install_media_placement()
        exotic = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "/tmp/a.png"}},
                {"type": "audio", "audio": "x"},
                {"type": "text", "text": "hi"},
            ],
        }
        built = apply_chat_template(
            None, CONFIG, [exotic], num_images=1, return_messages=True
        )
        assert rendered_kinds(built[0]["content"]) == ["image", "hi"]

    def test_image_markers_honor_the_allocated_count(self):
        install_media_placement()
        built = apply_chat_template(
            None, CONFIG, [INTERLEAVED], num_images=1, return_messages=True
        )
        assert rendered_kinds(built[0]["content"]) == [
            "before the first shot",
            "image",
            "between the shots",
            "after the last shot",
        ]
