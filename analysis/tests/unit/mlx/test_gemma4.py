import re

import mlx.core as mx
import numpy as np
import pytest
from locus.analysis.mlx import gemma4
from mlx_vlm.models.gemma4 import language, processing_gemma4

PATCH, POOL = 16, 3
CELL = PATCH * POOL


class TestEncodedSize:
    @pytest.mark.parametrize(
        "height,width", [(2160, 1152), (3000, 3000), (900 * 4, 1440 * 4), (14080, 750)]
    )
    def test_an_image_over_the_largest_budget_is_sized_as_upstream_sizes_it_at_that_budget(
        self, height, width
    ):
        upstream = processing_gemma4.get_aspect_ratio_preserving_size(
            height, width, PATCH, gemma4.MAX_SOFT_TOKENS * POOL**2, POOL
        )
        assert gemma4.encoded_size(height, width, PATCH, POOL) == upstream

    @pytest.mark.parametrize("height,width", [(1536, 816), (704, 375), (96, 1968)])
    def test_an_image_within_the_budget_is_never_scaled_up(self, height, width):
        h, w = gemma4.encoded_size(height, width, PATCH, POOL)
        assert (h, w) == (height // CELL * CELL, width // CELL * CELL)

    def test_an_image_on_the_grid_within_the_budget_is_returned_as_sent(self):
        from transformers.image_utils import ChannelDimension

        image = np.zeros((1536, 816, 3), dtype=np.uint8)
        out = gemma4.aspect_ratio_preserving_resize(
            None, image, PATCH, 280 * POOL**2, POOL, ChannelDimension.LAST
        )
        assert out is image

    def test_its_cost_never_exceeds_the_largest_budget(self):
        for height in range(48, 20000, 997):
            for width in range(48, 20000, 1013):
                h, w = gemma4.encoded_size(height, width, PATCH, POOL)
                assert (h // CELL) * (w // CELL) <= gemma4.MAX_SOFT_TOKENS


class TestChunkAlignment:
    # Text, a 4-token image, text, a 3-token image, text.
    VISION = np.array([[0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0]], dtype=bool)

    def test_a_boundary_inside_an_image_cuts_it(self):
        assert gemma4.cuts_an_image(self.VISION, 3)
        assert not gemma4.cuts_an_image(self.VISION, 1)
        assert not gemma4.cuts_an_image(self.VISION, 5)

    def test_a_chunk_ending_inside_an_image_moves_back_to_its_start(self):
        assert gemma4.aligned_chunk_end(self.VISION, 3, 10) == 1
        assert gemma4.aligned_chunk_end(self.VISION, 8, 10) == 7

    def test_an_image_longer_than_the_chunk_at_its_start_is_taken_whole(self):
        vision = self.VISION[:, 1:]
        assert gemma4.aligned_chunk_end(vision, 2, 9) == 4

    def test_an_image_running_past_the_limit_raises(self):
        vision = np.array([[1, 1, 1, 1]], dtype=bool)
        with pytest.raises(RuntimeError, match="no chunk boundary outside it"):
            gemma4.aligned_chunk_end(vision, 2, 3)


class _Masks:
    _block_sequence_ids_for_mask = language.Gemma4TextModel._block_sequence_ids_for_mask


def _reference(queries, keys, types, window):
    """Google's sliding mask for a chunk of `queries` new tokens after keys - queries cached ones: the window AND (causal OR same image)."""
    offset = keys - queries
    image = [None] * offset + [None] * queries
    group, previous = -1, False
    for i, t in enumerate(types):
        vision = t in (1, 2)
        if vision and not previous:
            group += 1
        image[offset + i] = group if vision else None
        previous = vision
    out = np.zeros((queries, keys), dtype=bool)
    for qi in range(queries):
        q = offset + qi
        for k in range(keys):
            same = image[q] is not None and image[q] == image[k]
            out[qi, k] = abs(q - k) < window and (k <= q or same)
    return out


class TestSlidingOverlay:
    @pytest.mark.parametrize("cached", [0, 5])
    def test_the_overlay_on_a_windowed_causal_mask_is_googles_composition(self, cached):
        types = [0, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0]
        queries, window = len(types), 4
        keys = cached + queries
        q_pos = np.arange(cached, keys)[:, None]
        k_pos = np.arange(keys)[None, :]
        base = mx.array((k_pos <= q_pos) & (q_pos - k_pos < window))[None, None]
        mask = gemma4._apply_blockwise_bidirectional_overlay(
            _Masks(), base, mx.array([types]), window_size=window
        )
        assert np.array_equal(
            np.array(mask[0, 0]), _reference(queries, keys, types, window)
        )

    def test_token_types_that_do_not_match_the_chunk_raise(self):
        base = mx.ones((1, 1, 4, 4), dtype=mx.bool_)
        with pytest.raises(RuntimeError, match="attend causally"):
            gemma4._apply_blockwise_bidirectional_overlay(
                _Masks(), base, mx.array([[0, 1, 1, 1, 0, 0]]), window_size=4
            )


class TestReasoningMarkers:
    def test_a_pair_another_pair_refines_is_dropped(self):
        markers = (
            ("<|channel>", "<channel|>"),
            ("<|channel>thought", "<channel|>"),
            ("<think>", "</think>"),
        )
        assert gemma4.refined(markers) == (
            ("<|channel>thought", "<channel|>"),
            ("<think>", "</think>"),
        )

    def test_the_cleanup_keeps_a_thought_that_begins_with_the_word_thought(self):
        assert (
            gemma4.clean_reasoning("thought about it", "<|channel>thought")
            == "thought about it"
        )


START, END = "<|channel>", "<channel|>"
CALL = "<|tool_call>call:request_screenshots{timestamps:[12000]}<tool_call|>"
# (generated text, whether the prompt opened the channel, reasoning, content), in the shapes
# Gemma 4 generates under the canonical template: the channel opened by the model with its
# header, or by the prompt after a tool reply; closed, or cut by max_tokens.
SPLITS = [
    (
        "<|channel>thought\nI reason.<channel|>The answer.",
        False,
        "I reason.",
        "The answer.",
    ),
    (
        "<|channel>thought\nthought experiments.<channel|>A.",
        False,
        "thought experiments.",
        "A.",
    ),
    ("<|channel>thought\n<channel|>The answer.", False, None, "The answer."),
    ("I reason.<channel|>The answer.", True, "I reason.", "The answer."),
    ("thought experiments.<channel|>A.", True, "thought experiments.", "A."),
    ("<|channel>thought\nI reason and", False, "I reason and", ""),
    (
        "<|channel>thought\nthought experiments and",
        False,
        "thought experiments and",
        "",
    ),
    ("I reason and", True, "I reason and", ""),
    ("thought experiments and", True, "thought experiments and", ""),
    ("<|channel>thought\nI need one.<channel|>" + CALL, False, "I need one.", CALL),
    (
        '<|channel>thought\nPlan.<channel|>{"product": 391}',
        False,
        "Plan.",
        '{"product": 391}',
    ),
    ("Just an answer.", False, None, "Just an answer."),
]


def _pieces(text):
    """The text cut where Gemma 4's tokens fall around the markers: each marker, `thought`, and a newline are tokens of their own."""
    return [p for p in re.split(r"(<\|channel>|<channel\|>|thought|\n)", text) if p]


@pytest.fixture
def installed(monkeypatch):
    from mlx_vlm.server import responses_state
    from mlx_vlm.server.responses_state import ThinkingStreamState

    monkeypatch.setattr(
        ThinkingStreamState,
        "_build_open_close_markers",
        ThinkingStreamState.__dict__["_build_open_close_markers"],
    )
    monkeypatch.setattr(
        responses_state, "_clean_reasoning", responses_state._clean_reasoning
    )
    gemma4.install_reasoning_markers()
    return responses_state


class TestInstalledSplit:
    @pytest.mark.parametrize("text,opened,reasoning,content", SPLITS)
    def test_the_whole_text_split(self, installed, text, opened, reasoning, content):
        assert installed._split_thinking(text, START, END, opened) == (
            reasoning,
            content,
        )

    @pytest.mark.parametrize("text,opened,reasoning,content", SPLITS)
    @pytest.mark.parametrize("chunking", ["tokens", "characters"])
    def test_the_streamed_split_reassembled_as_the_client_does(
        self, installed, text, opened, reasoning, content, chunking
    ):
        state = installed.ThinkingStreamState(opened, START, END)
        chunks = _pieces(text) if chunking == "tokens" else list(text)
        streamed_reasoning, streamed_content = [], []
        for i, chunk in enumerate(chunks):
            delta = state.feed(chunk, last=i == len(chunks) - 1)
            streamed_reasoning.append(delta.reasoning or "")
            streamed_content.append(delta.content or "")
        assert ("".join(streamed_reasoning).strip() or None) == reasoning
        assert "".join(streamed_content).strip() == content

    def test_a_prompt_that_opens_the_channel_is_seen_open(self, installed):
        prompt = "<tool_response|><|channel>thought\n"
        assert installed.prompt_has_open_thinking(prompt, True, START, END)
        assert not installed.prompt_has_open_thinking(
            "<|turn>model\n", True, START, END
        )


def test_the_largest_budget_is_the_largest_gemma4_was_trained_at():
    assert gemma4.MAX_SOFT_TOKENS == max(processing_gemma4._SUPPORTED_SOFT_TOKENS)


def _tiny_language_model():
    """A Gemma 4 language model small enough to run here: sliding and full-attention layers alternating, a sliding window of 8, and the bidirectional image attention of the real one."""
    from mlx_vlm.models.gemma4.config import TextConfig

    config = TextConfig(
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=128,
        num_attention_heads=2,
        head_dim=16,
        global_head_dim=32,
        num_key_value_heads=1,
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        vocab_size=300,
        vocab_size_per_layer_input=300,
        sliding_window=8,
        sliding_window_pattern=2,
        use_bidirectional_attention="vision",
        use_double_wide_mlp=False,
    )
    mx.random.seed(0)
    model = language.LanguageModel(config)
    mx.eval(model.parameters())
    return model


# Text, a 6-token image, text, an 11-token image longer than the chunk, text, a 4-token image, text.
PROMPT_TYPES = [0] * 3 + [1] * 6 + [0] * 4 + [1] * 11 + [0] * 5 + [1] * 4 + [0] * 3
STEP = 5


class _NeverStop:
    def add_eos_token_ids(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def __call__(self, token):
        return False


def _kv(cache):
    """The keys and values a layer's cache holds, oldest first."""
    from mlx_vlm.models import cache as cache_module

    if isinstance(cache, cache_module.RotatingKVCache):
        held = min(cache.offset, cache.keys.shape[2])
        keys = cache._temporal_order(cache.keys)[..., -held:, :]
        values = cache._temporal_order(cache.values)[..., -held:, :]
        return keys, values
    return cache.keys[..., : cache.offset, :], cache.values[..., : cache.offset, :]


@pytest.fixture
def aligned_prefill(monkeypatch):
    from mlx_vlm.generate import ar

    for owner, name in [
        (language.LanguageModel, "chunked_prefill_policy"),
        (language.Gemma4TextModel, "_apply_blockwise_bidirectional_overlay"),
        (language.Gemma4TextModel, "_make_masks"),
        (ar.PromptProcessingBatch, "__init__"),
        (ar.PromptProcessingBatch, "prompt_step"),
    ]:
        monkeypatch.setattr(owner, name, owner.__dict__[name])
    gemma4.install_image_aligned_prefill()
    return ar


def _prefill(ar, model, embeds, prompt_kwargs, chunks=None):
    """Prefill one row through the batch the server builds, at STEP, and the log-probabilities of its first token. Each chunk's length is appended to `chunks`."""
    rows = embeds.shape[0]
    batch = ar.PromptProcessingBatch(
        model,
        list(range(rows)),
        [list(range(1, embeds.shape[1] + 1))] * rows,
        [1] * rows,
        embeds,
        prompt_kwargs,
        prefill_step_size=STEP,
        greedy_sampling=True,
    )
    while batch.needs_processing():
        taken = batch.prompt_step()
        assert batch.prefill_step_size == STEP, "the chunk length is restored"
        if chunks is not None:
            chunks.append(taken)
    seen = {}

    def sampler(logprobs):
        seen["logprobs"] = logprobs
        return mx.argmax(logprobs, axis=-1)

    generation = batch.generate(sampler, _NeverStop(), compute_logprobs=False)
    return seen["logprobs"], generation.prompt_cache


class TestImageAlignedPrefill:
    def test_chunked_prefill_is_the_one_pass_under_googles_masks(self, aligned_prefill):
        model = _tiny_language_model()
        types = mx.array([PROMPT_TYPES])
        length = len(PROMPT_TYPES)
        embeds = mx.random.normal((1, length, 64))
        cache = model.make_cache()
        logits = model(
            mx.array([list(range(1, length + 1))]),
            inputs_embeds=embeds,
            cache=cache,
            mm_token_type_ids=types,
        ).logits[:, -1]
        one_pass = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        chunks = []
        logprobs, chunked_cache = _prefill(
            aligned_prefill, model, embeds, {"mm_token_type_ids": types}, chunks
        )

        boundaries = np.cumsum(chunks)[:-1]
        vision = np.array(PROMPT_TYPES) == 1
        assert len(chunks) > 3, "the prompt prefilled in chunks"
        assert not any(vision[b - 1] and vision[b] for b in boundaries), (
            f"a chunk ended inside an image: {chunks}"
        )
        assert mx.allclose(logprobs, one_pass, atol=1e-4)
        for chunked, whole in zip(chunked_cache, cache):
            for got, ref in zip(_kv(chunked), _kv(whole)):
                ref = ref[..., -got.shape[2] :, :]
                assert mx.allclose(got, ref, atol=1e-4)

    def test_a_chunk_cut_inside_an_image_is_what_the_comparison_catches(
        self, aligned_prefill, monkeypatch
    ):
        model = _tiny_language_model()
        types = mx.array([PROMPT_TYPES])
        embeds = mx.random.normal((1, len(PROMPT_TYPES), 64))
        aligned, _ = _prefill(
            aligned_prefill, model, embeds, {"mm_token_type_ids": types}
        )
        monkeypatch.setattr(gemma4, "cuts_an_image", lambda vision, k: False)
        cut, _ = _prefill(aligned_prefill, model, embeds, {"mm_token_type_ids": types})
        assert not mx.allclose(cut, aligned, atol=1e-4)

    def test_chunking_stays_on_for_an_image_prompt_only_while_the_batch_is_built(
        self, aligned_prefill
    ):
        model = _tiny_language_model()
        kwargs = {"mm_token_type_ids": mx.array([PROMPT_TYPES])}
        assert not gemma4.chunked_prefill_policy(model, prefill_kwargs=kwargs)
        gemma4._building.batch = True
        try:
            assert gemma4.chunked_prefill_policy(model, prefill_kwargs=kwargs)
        finally:
            gemma4._building.batch = False
        _prefill(
            aligned_prefill,
            model,
            mx.random.normal((1, len(PROMPT_TYPES), 64)),
            kwargs,
        )
        assert not getattr(gemma4._building, "batch", False), (
            "the flag ends with the build"
        )

    def test_two_rows_with_images_raise(self, aligned_prefill):
        types = mx.array([PROMPT_TYPES, PROMPT_TYPES])
        with pytest.raises(RuntimeError, match="2 rows with images"):
            _prefill(
                aligned_prefill,
                _tiny_language_model(),
                mx.random.normal((2, len(PROMPT_TYPES), 64)),
                {"mm_token_type_ids": types},
            )

    def test_token_types_the_batch_cannot_slice_raise(self, aligned_prefill):
        types = mx.array([PROMPT_TYPES[:-1]])
        with pytest.raises(RuntimeError, match="cannot slice"):
            _prefill(
                aligned_prefill,
                _tiny_language_model(),
                mx.random.normal((1, len(PROMPT_TYPES), 64)),
                {"mm_token_type_ids": types},
            )

    def test_an_apc_checkpoint_inside_an_image_raises(
        self, aligned_prefill, monkeypatch
    ):
        monkeypatch.setattr(
            aligned_prefill.PromptProcessingBatch,
            "_next_apc_checkpoint_column",
            lambda self: 6,
        )
        with pytest.raises(RuntimeError, match="APC checkpoint"):
            _prefill(
                aligned_prefill,
                _tiny_language_model(),
                mx.random.normal((1, len(PROMPT_TYPES), 64)),
                {"mm_token_type_ids": mx.array([PROMPT_TYPES])},
            )


def _tiny_vision_model():
    from mlx_vlm.models.gemma4.config import VisionConfig
    from mlx_vlm.models.gemma4.vision import VisionModel

    config = VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        default_output_length=12,
    )
    mx.random.seed(0)
    model = VisionModel(config)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def chunked_encode(monkeypatch):
    from mlx_vlm.models.gemma4.vision import VisionModel

    original = VisionModel.__dict__["__call__"]
    monkeypatch.setattr(VisionModel, "__call__", original)

    def install(budget):
        gemma4.install_vision_encode_chunking(budget)
        return original

    return install


class TestEncodeChunking:
    def test_images_of_different_sizes_encode_as_one_call_would(self, chunked_encode):
        model = _tiny_vision_model()
        images = [
            mx.random.uniform(shape=(3, h, w))
            for h, w in [(96, 144), (144, 96), (48, 96)]
        ]
        original = chunked_encode(96 * 144)
        whole = original(model, images)
        assert mx.array_equal(model(images), whole)

    def test_a_stacked_batch_encodes_as_one_call_would(self, chunked_encode):
        model = _tiny_vision_model()
        stack = mx.random.uniform(shape=(3, 3, 96, 144))
        original = chunked_encode(96 * 144)
        whole = original(model, stack)
        chunked = model(stack)
        assert chunked.shape == whole.shape
        assert mx.allclose(chunked, whole, atol=1e-5)

    def test_a_budget_holding_every_image_takes_the_one_call(
        self, chunked_encode, monkeypatch
    ):
        model = _tiny_vision_model()
        images = [mx.random.uniform(shape=(3, 96, 144)) for _ in range(2)]
        chunked_encode(10**9)
        calls = []
        monkeypatch.setattr(
            gemma4, "image_groups", lambda *a: calls.append(a) or [(0, 2)]
        )
        model(images)
        assert len(calls) == 1


TEMPLATE_MESSAGES = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You read "},
            {"type": "text", "text": "screenshots."},
        ],
    },
    {"role": "user", "content": "What changed?"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should look.",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "request_screenshots", "arguments": {}},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "c1",
        "content": [
            {"type": "text", "text": "at [00:01.000]:\n"},
            {"type": "image_url", "image_url": {"url": "one.png"}},
            {"type": "text", "text": "\nat [00:02.000]:\n"},
            {"type": "image_url", "image_url": {"url": "two.png"}},
            {"type": "text", "text": "\nNow answer."},
        ],
    },
]


class TestTemplateRepairs:
    """Each change to Google's template, in the template the server renders: the two repairs against the format Google's own formatter writes, and a tool reply's parts in the order sent."""

    @staticmethod
    def _render(template):
        from transformers.utils.chat_template_utils import render_jinja_template

        rendered, _ = render_jinja_template(
            conversations=[TEMPLATE_MESSAGES],
            chat_template=template,
            add_generation_prompt=True,
            enable_thinking=True,
            preserve_thinking=True,
        )
        return rendered[0]

    def test_a_system_messages_parts_render_as_the_string_they_concatenate_to(self):
        rendered = self._render(gemma4.canonical_template())
        assert "<|think|>\nYou read screenshots.<turn|>" in rendered

    def test_a_replayed_thought_closes_right_after_its_last_character(self):
        rendered = self._render(gemma4.canonical_template())
        assert "<|channel>thought\nI should look.<channel|>" in rendered

    def test_a_tool_replys_parts_render_in_the_order_sent_inside_the_replys_string(
        self,
    ):
        rendered = self._render(gemma4.canonical_template())
        assert (
            '{value:<|"|>at [00:01.000]:\n<|image|>\nat [00:02.000]:\n<|image|>\nNow answer.<|"|>}<tool_response|><|channel>thought'
            in rendered
        )
        assert rendered.count("<|image|>") == 2

    def test_googles_template_unchanged_renders_each_differently(self):
        from pathlib import Path

        from huggingface_hub import hf_hub_download

        raw = Path(
            hf_hub_download(
                gemma4.CANONICAL_TEMPLATE_REPO,
                "chat_template.jinja",
                revision=gemma4.CANONICAL_TEMPLATE_REVISION,
            )
        ).read_text()
        rendered = self._render(raw)
        assert "You read screenshots. <turn|>" in rendered
        assert "I should look.\n<channel|>" in rendered
        assert (
            '\nNow answer.<|"|>}<tool_response|><|image|><|image|><|channel>thought'
            in rendered
        )
