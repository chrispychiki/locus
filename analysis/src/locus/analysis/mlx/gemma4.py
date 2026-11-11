"""Gemma 4 on the upstream server: an image-dense conversation prefilled in bounded chunks under the masks Google's reference uses, each image encoded at the size the client sent, the canonical chat template, and thoughts returned without the channel's header.

- **Image prompts prefill in chunks.** Gemma 4 attends bidirectionally within each image on its sliding layers, so a chunk boundary inside an image would hide the image's later tokens from its earlier ones. Upstream's `chunked_prefill_policy` therefore prefills any prompt carrying an image in one forward pass, with its masks materialized at prompt x prompt and its attention scores with them: measured here, image prompts of 5k, 11k and 16k tokens peak at 25, 30 and 37 GB, about 20 of it the weights, where chunked, the same screenshots peak 24 GB at 14k tokens and 28 GB at 109k. `install_image_aligned_prefill` keeps chunking on for an image prompt while a `PromptProcessingBatch` is built (the one prefill loop the continuous-batching backend runs) and ends each of its chunks at an image boundary, so no image straddles a chunk and every key of an image is among the chunk's own keys; every other loop that consults the policy, and a draft-model request, keep upstream's unchunked prefill. Upstream applies the image overlay only to a square mask, so the patch applies it to a chunk prefilled against cached keys too. It raises instead of prefilling wrongly when prompt_step would shorten a chunk to an APC checkpoint column inside an image, when the token types cannot be sliced to the chunk, or when the batch holds more than one row (upstream's batch sliding cache prefills a left-padded row wrongly when a chunk of several tokens follows a single-token one, which aligned chunks produce).
- **The image overlay belongs to the sliding layers, within their window.** Google's reference keeps the full-attention layers causal through an image and composes the sliding mask as the window AND (causal OR same image), the window bounding distance in both directions. Upstream's `_make_masks` puts the overlay on the full-attention layers too and drops its window bound. The patched builder follows the reference, which also leaves the full-attention layers on mlx's causal mask, the one its fused kernel takes.
- **An image is encoded at the size it was sent.** Upstream resizes every image to fill one budget, the processor config's `max_soft_tokens` (280), so a screenshot drawn at a smaller or larger kind's size is scaled to 280 tokens. `install_processor_contract` raises the budget to the largest Gemma 4 was trained at (1120) and scales an image up only where a side is shorter than one cell of its grid, so an image on the processor's grid within that budget is encoded as sent and the token count is the geometry's (budget.image_geometry).
- **A request's images encode in bounded chunks.** Upstream encodes all of a request's images as one graph, so the encode's transient grows with their total pixels. `install_vision_encode_chunking` encodes them in contiguous groups of at most the processor's pixel limit, each materialized before the next, with the same output.
- **The checkpoint carries a stale chat template.** mlx-community's conversion ships an older template; Google's canonical template fixes tool-calling loops, turn closures, and thinking order, and the stale one has the model continue bare after a tool reply, where it echoes the prompt and emits stray tool-call markers. Every gemma4 processor the server loads renders with the canonical template at a pinned revision, refused on any other content, with two repairs, each to the format Google's own formatter (the `dialog` package the gemma library samples through) writes: a system message's content list renders as the string its parts concatenate to (the template follows every part with a space), and a replayed thought is closed directly after its last character (the template adds a newline, which the model then imitates at its next close).
- **The canonical template moves a tool reply's images after its text.** It renders the reply's text in the reply block's quoted string and every image after the block's close, so screenshots sent each under a line naming its moment reach the model as all the lines, then all the screenshots: measured here on a reply of 24, the model gave 6 to 14 of them the wrong moment in seven of nine runs. The served template renders the parts in the order sent, each image's marker inside the quoted string where the image was sent, and in nine runs the model misplaced none. Google's formatter writes tool replies as text alone, so there is no reference layout to follow.
- **Returned thinking begins with the channel's name.** The thinking budget and the thinking-aware grammar match single tokens, so the card's `thinking_markers` set the channel's own delimiters (`<|channel>` / `<channel|>`). Upstream's reasoning split tries that configured pair before its built-in `("<|channel>thought", "<channel|>")`, so every returned thought begins with `thought`, and replayed it renders the header twice. `install_reasoning_markers` drops a configured pair that a built-in pair refines, so the split removes the whole header; its whole-text cleanup, which also strips a leading `thought` after the built-in marker, is replaced by one that removes only the marker, since the canonical template opens the channel with its whole header and the text left is the thought itself.

Each rewrite is keyed to the upstream source it replaces, and patching.py refuses the boot on drift.
"""

import logging
import threading

import numpy as np

from .patching import expect_source, refuse
from .vision_chunking import image_groups

logger = logging.getLogger("locus.analysis.mlx.gemma4")

# The largest vision budget Gemma 4 was trained at, in soft tokens (70/140/280/560/1120).
MAX_SOFT_TOKENS = 1120

EXPECTED_POLICY_SHA256 = (
    "aa22e47f20ad1751a96950fd25c5018ac9a6fb9e733fbef54c37f99c269f4672"
)
EXPECTED_OVERLAY_SHA256 = (
    "e542a4ff72c1846ae1610da3e8895df93fdf7ff8c3d1bba40eadea73aac06480"
)
EXPECTED_MAKE_MASKS_SHA256 = (
    "193a8b64e571163717f771b4ff8d36e40d50817aee4527956892dcd6ac845e6c"
)
EXPECTED_PROMPT_STEP_SHA256 = (
    "d7c445866a7b6d38a4aeb934f2652feff73abd2f74e43158834c72d94954d0c7"
)
EXPECTED_BATCH_INIT_SHA256 = (
    "463151a660348eb1b33871c7e8ec88c14327655067f6ef34fb0bd1b99ab278f1"
)
EXPECTED_RESIZE_SHA256 = (
    "fcef0bc8c4f2abfdb358e05a71a91b00820219a9fd0550232eb442be3b3c9adc"
)
EXPECTED_ASPECT_SIZE_SHA256 = (
    "e07f968c28d9490ce462ae49c4490079038e473944f6b7fb521d3dfbe2cf9109"
)
EXPECTED_PREPROCESS_SHA256 = (
    "aae0b75da3c8aa9778b2febc64423ae97dd3518d13a0cd0311a8b04af8333a1c"
)
EXPECTED_FROM_PRETRAINED_SHA256 = (
    "e0bc1cafac1c6a862d58376618ec7ca87f3d9731e4c04a45208f7dc4406edde3"
)
EXPECTED_MARKERS_SHA256 = (
    "30fce43f3938d8bb0c5ed063382544486d7c9d39089dc449e5904790e68686cb"
)
EXPECTED_CLEAN_REASONING_SHA256 = (
    "623e53f4a345cdaf3440b35d5cb372f50432a30a8c8afb56bb54a58a4a977483"
)

CANONICAL_TEMPLATE_REPO = "google/gemma-4-26B-A4B-it"
CANONICAL_TEMPLATE_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
EXPECTED_CANONICAL_TEMPLATE_SHA256 = (
    "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
)

# The loop that renders a system message's content list, every part followed by a space.
TRAILING_SYSTEM_PARTS = """\
            {%- for item in messages[0]['content'] -%}
                {{- item['text'] | trim + ' '-}}
            {%- endfor -%}"""

# The text parts concatenated as they are, then rendered as the template renders a string
# system message. A part without text renders as nothing, as it does in the loop above.
JOINED_SYSTEM_PARTS = """\
            {%- set system_text = namespace(value='') -%}
            {%- for item in messages[0]['content'] -%}
                {%- set system_text.value = system_text.value + (item['text'] | default('')) -%}
            {%- endfor -%}
            {{- system_text.value | trim -}}"""

PADDED_THOUGHT_CLOSE = (
    """        {{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}"""
)
FAITHFUL_THOUGHT_CLOSE = (
    """        {{- '<|channel>thought\\n' + thinking_text + '<channel|>' -}}"""
)

# The branch that renders a tool message's content list: its text parts inside the reply block,
# then a marker for each media part after the block's close.
TOOL_TEXT_THEN_MEDIA = """\
                            {%- for part in tool_body -%}
                                {%- if part.get('type') == 'text' -%}
                                    {%- set ns_txt.s = ns_txt.s + (part.get('text') | default('')) -%}
                                {%- endif -%}
                            {%- endfor -%}
                            {{- format_tool_response_block(ns_tname.name, ns_txt.s) -}}
                            {%- for part in tool_body -%}
                                {%- if part.get('type') in ['image', 'image_url'] -%}
                                    {{- '<|image|>' -}}
                                {%- elif part.get('type') in ['audio', 'input_audio'] -%}
                                    {{- '<|audio|>' -}}
                                {%- elif part.get('type') == 'video' -%}
                                    {{- '<|video|>' -}}
                                {%- endif -%}
                            {%- endfor -%}"""

# Every part in the order sent, inside the reply block's quoted string: a media part's marker
# where the part sits.
TOOL_PARTS_IN_ORDER = """\
                            {%- for part in tool_body -%}
                                {%- if part.get('type') == 'text' -%}
                                    {%- set ns_txt.s = ns_txt.s + (part.get('text') | default('')) -%}
                                {%- elif part.get('type') in ['image', 'image_url'] -%}
                                    {%- set ns_txt.s = ns_txt.s + '<|image|>' -%}
                                {%- elif part.get('type') in ['audio', 'input_audio'] -%}
                                    {%- set ns_txt.s = ns_txt.s + '<|audio|>' -%}
                                {%- elif part.get('type') == 'video' -%}
                                    {%- set ns_txt.s = ns_txt.s + '<|video|>' -%}
                                {%- endif -%}
                            {%- endfor -%}
                            {{- format_tool_response_block(ns_tname.name, ns_txt.s) -}}"""

PREFILL_OWES = (
    "Re-read the policy, the overlay, the mask builder, the batch constructor, and prompt_step "
    "together — the overlay belongs to the sliding layers alone and within their window, "
    "chunking under it is exact only while no chunk boundary falls inside an image and the "
    "overlay covers a chunk's own keys, and only prompt_step aligns its chunks — then "
    "re-validate and re-pin (analysis/src/locus/analysis/mlx/gemma4.py)."
)

# Set on the thread building a PromptProcessingBatch, for as long as its constructor runs.
_building = threading.local()


def bidirectional_vision(model) -> bool:
    """Whether the served model attends bidirectionally within each image — the property that makes a chunk boundary inside an image change the math."""
    language = getattr(model, "language_model", model)
    config = getattr(language, "config", None)
    return getattr(config, "use_bidirectional_attention", None) == "vision"


def cuts_an_image(vision: np.ndarray, k: int) -> bool:
    """Whether a chunk boundary after k tokens falls inside an image: tokens k-1 and k are both image tokens in some row. vision is (rows, remaining) bool over the prompt still to prefill, True on image tokens; a run of consecutive image tokens is one image, since every image is bracketed by text markers."""
    return 0 < k < vision.shape[1] and bool(np.any(vision[:, k - 1] & vision[:, k]))


def aligned_chunk_end(vision: np.ndarray, n: int, limit: int) -> int:
    """The chunk length to prefill: n, moved back to the start of any image it would cut, or — when an image longer than n starts the chunk — forward to that image's end. limit is the most a chunk may take."""
    k = n
    while cuts_an_image(vision, k):
        k -= 1
    if k > 0:
        return k
    k = n
    while cuts_an_image(vision, k):
        k += 1
    if k > limit:
        raise RuntimeError(
            f"an image runs to the end of the prompt ({vision.shape[1]} tokens left) — "
            f"no chunk boundary outside it exists"
        )
    return k


def chunked_prefill_policy(
    self,
    *,
    input_ids=None,
    inputs_embeds=None,
    prompt_cache=None,
    draft_model=None,
    draft_kind=None,
    prefill_kwargs=None,
) -> bool:
    """Upstream's policy, with chunking kept on for a prompt carrying images only while a PromptProcessingBatch is being built without a draft model: that batch's prompt_step is the one loop that aligns its chunks to images. Every other caller gets upstream's answer, an unchunked prefill."""
    del input_ids, inputs_embeds, prompt_cache
    prefill_kwargs = prefill_kwargs or {}
    if getattr(self, "no_chunked_prefill", False):
        return False

    token_types = prefill_kwargs.get("mm_token_type_ids", None)
    if token_types is None:
        token_types = prefill_kwargs.get("token_type_ids", None)
    if (
        getattr(self.config, "use_bidirectional_attention", None) == "vision"
        and token_types is not None
        and (draft_model is not None or not getattr(_building, "batch", False))
    ):
        import mlx.core as mx

        has_visual = int(mx.sum((token_types == 1) | (token_types == 2)).item()) > 0
        has_audio = int(mx.sum(token_types == 3).item()) > 0
        if has_visual and not has_audio:
            return False

    if draft_model is not None:
        return (
            draft_kind == "mtp"
            and bool(prefill_kwargs.get("return_hidden", False))
            and bool(prefill_kwargs.get("return_shared_kv", False))
        )

    return True


def _apply_blockwise_bidirectional_overlay(
    self, base_mask, mm_token_type_ids, window_size=None
):
    """Upstream's overlay, extended to a chunk prefilled against cached keys and bounded by the layer's window: the mask is (queries, cached + queries), the chunk's own keys its last columns, and every image the chunk holds lies wholly among them. A same-image pair is kept only while the two positions sit within `window_size` of each other, the bound the sliding layers' attention has to every other key."""
    import mlx.core as mx

    if mm_token_type_ids is None:
        return base_mask
    queries, keys = base_mask.shape[-2], base_mask.shape[-1]
    if mm_token_type_ids.shape[1] != queries or keys < queries:
        raise RuntimeError(
            f"gemma4 image overlay: {mm_token_type_ids.shape[1]} token types for a mask of "
            f"{queries} queries over {keys} keys, so the chunk's images would attend causally"
        )

    block_sequence_ids = self._block_sequence_ids_for_mask(mm_token_type_ids)
    q_blocks = mx.expand_dims(block_sequence_ids, -1)
    k_blocks = mx.expand_dims(block_sequence_ids, -2)
    same_block = (q_blocks != -1) & (q_blocks == k_blocks)
    if keys > queries:
        cached = mx.zeros((*same_block.shape[:-1], keys - queries), dtype=mx.bool_)
        same_block = mx.concatenate([cached, same_block], axis=-1)
    if window_size is not None:
        q_pos = mx.arange(keys - queries, keys)[:, None]
        k_pos = mx.arange(keys)[None, :]
        same_block = same_block & (mx.abs(q_pos - k_pos) < window_size)
    return base_mask | mx.expand_dims(same_block, 1)


def _make_masks(self, h, cache, mm_token_type_ids=None):
    """Upstream's mask builder with the image overlay on the sliding layers alone, bounded by their window: the full-attention mask is the causal mask (the string for a single sequence, an array only under batch left padding), and the sliding mask is the window AND (causal OR same image) — Google's reference composition, its window bounding distance in both directions as Google's code does (gemma4/_modules.py _create_sliding_mask), where transformers bounds one side."""
    import mlx.core as mx
    from mlx_vlm.models.gemma4 import language

    mask = {}
    masks = []
    has_visual_tokens = (
        mm_token_type_ids is not None
        and int(mx.sum((mm_token_type_ids == 1) | (mm_token_type_ids == 2)).item()) > 0
    )
    use_bidirectional_vision = (
        getattr(self.config, "use_bidirectional_attention", None) == "vision"
        and mm_token_type_ids is not None
        and has_visual_tokens
        and h.shape[1] > 1
    )
    for layer, c in zip(self.layers, cache):
        if layer.layer_type not in mask:
            if layer.layer_type == "full_attention":
                mask["full_attention"] = language.create_attention_mask(
                    h, c, return_array=getattr(c, "left_padding", None) is not None
                )
            elif layer.layer_type == "sliding_attention":
                return_array = (
                    h.shape[1] > 1
                    and c is not None
                    and int(mx.max(mx.array(c.offset)).item()) > 0
                ) or use_bidirectional_vision
                sliding = language.create_attention_mask(
                    h, c, window_size=self.window_size, return_array=return_array
                )
                if use_bidirectional_vision:
                    if isinstance(sliding, str):
                        sliding = language.create_causal_mask(
                            h.shape[1], window_size=self.window_size
                        )
                    sliding = self._apply_blockwise_bidirectional_overlay(
                        sliding, mm_token_type_ids, window_size=self.window_size
                    )
                mask["sliding_attention"] = sliding
        masks.append(mask[layer.layer_type])
    return masks


def install_image_aligned_prefill() -> None:
    """Keep gemma4's prefill chunked with images in the prompt, every chunk ending at an image boundary, under Google's mask composition; on drift, refuse to boot (patching.py)."""
    from mlx_vlm.generate import ar
    from mlx_vlm.models.gemma4 import language

    for what, fn, sha in (
        (
            "LanguageModel.chunked_prefill_policy",
            language.LanguageModel.chunked_prefill_policy,
            EXPECTED_POLICY_SHA256,
        ),
        (
            "Gemma4TextModel._apply_blockwise_bidirectional_overlay",
            language.Gemma4TextModel._apply_blockwise_bidirectional_overlay,
            EXPECTED_OVERLAY_SHA256,
        ),
        (
            "Gemma4TextModel._make_masks",
            language.Gemma4TextModel._make_masks,
            EXPECTED_MAKE_MASKS_SHA256,
        ),
        (
            "PromptProcessingBatch.prompt_step",
            ar.PromptProcessingBatch.prompt_step,
            EXPECTED_PROMPT_STEP_SHA256,
        ),
        (
            "PromptProcessingBatch.__init__",
            ar.PromptProcessingBatch.__init__,
            EXPECTED_BATCH_INIT_SHA256,
        ),
    ):
        expect_source("image-aligned prefill", what, fn, sha, PREFILL_OWES)

    language.LanguageModel.chunked_prefill_policy = chunked_prefill_policy
    language.Gemma4TextModel._apply_blockwise_bidirectional_overlay = (
        _apply_blockwise_bidirectional_overlay
    )
    language.Gemma4TextModel._make_masks = _make_masks

    init = ar.PromptProcessingBatch.__init__

    def building_init(self, *args, **kwargs):
        _building.batch = True
        try:
            init(self, *args, **kwargs)
        finally:
            _building.batch = False

    prompt_step = ar.PromptProcessingBatch.prompt_step

    def aligned_prompt_step(self) -> int:
        if not bidirectional_vision(self.model) or not self.needs_processing():
            return prompt_step(self)
        # The token types the mask builder reads: mm_token_type_ids, else token_type_ids.
        key = (
            "mm_token_type_ids"
            if self._prompt_kwargs.get("mm_token_type_ids") is not None
            else "token_type_ids"
        )
        types = self._prompt_kwargs.get(key)
        if types is None:
            return prompt_step(self)
        types = np.array(types)
        vision = (types == 1) | (types == 2)
        if not vision.any():
            return prompt_step(self)
        if key not in self._prompt_length_aware_keys:
            raise RuntimeError(
                f"gemma4 prefill cannot slice the prompt's {key} {types.shape} to its "
                f"chunks, so a chunk's images would attend causally"
            )
        if self._inputs_embeds.shape[0] > 1:
            raise RuntimeError(
                f"gemma4 prefill of {self._inputs_embeds.shape[0]} rows with images: "
                f"upstream's batch sliding cache prefills a left-padded row wrongly when a "
                f"chunk of several tokens follows a single-token one, which aligned chunks "
                f"produce. Serve image prompts one row at a time."
            )
        step = self.prefill_step_size
        limit = self._inputs_embeds.shape[1] - 1
        n = min(step or self._inputs_embeds.shape[1], limit)
        aligned = n if step is None else aligned_chunk_end(vision, n, limit)
        # prompt_step shortens the chunk to the next APC checkpoint column. Its other
        # shortening, to a right-padded row's end, never shortens a single row.
        taken = aligned
        checkpoint = self._next_apc_checkpoint_column()
        if checkpoint is not None:
            taken = min(aligned, checkpoint - self._processed_prompt_columns)
        if cuts_an_image(vision, taken):
            raise RuntimeError(
                f"gemma4 prefill would end a chunk inside an image ({taken} of "
                f"{vision.shape[1]} tokens left), where the image's later tokens are "
                f"hidden from its earlier ones: the chunk was shortened to an APC "
                f"checkpoint column. Serve image prompts with APC checkpoints off."
            )
        if step is None or aligned == n:
            return prompt_step(self)
        self.prefill_step_size = aligned
        try:
            return prompt_step(self)
        finally:
            self.prefill_step_size = step

    ar.PromptProcessingBatch.__init__ = building_init
    ar.PromptProcessingBatch.prompt_step = aligned_prompt_step


EXPECTED_VISION_CALL_SHA256 = (
    "0eee171d85792566c55b1b687d2752bab7bf35fde9c7c5e6fa6f7bba6aa15b19"
)


def install_vision_encode_chunking(budget: int) -> None:
    """Encode gemma4's images in pixel-bounded chunks, each materialized before the next. Upstream hands a request's whole image set to VisionModel.__call__ as one graph, so the encode's transient grows with the request's total pixels: measured here, 60 screenshots at 280 soft tokens peak 19.5 GB above the prefill that follows them. The encoder treats every image independently — a list of differently sized images runs image by image, a stacked batch attends within each row under its own padding mask and pools each row alone — so encoding contiguous groups and concatenating their tokens is the same output. Keyed to the pinned source; on drift, refuse to boot (patching.py).

    A chunk holds at most `budget` pixels, the processor's pixel limit (count.pixel_limit), which no image it encodes exceeds: an image is never split, so no smaller budget lowers the peak, and a larger one only batches images that each already fill the GPU, raising the transient for no speed."""
    import mlx.core as mx
    from mlx_vlm.models.gemma4.vision import VisionModel

    expect_source(
        "gemma4 vision-encode chunking",
        "VisionModel.__call__",
        VisionModel.__call__,
        EXPECTED_VISION_CALL_SHA256,
        "Chunking is output-identical only because the pinned source encodes every image "
        "independently — re-verify that against the installed version, then re-pin "
        "(analysis/src/locus/analysis/mlx/gemma4.py).",
    )
    original = VisionModel.__call__

    def chunked(self, pixel_values, pixel_position_ids=None):
        images = (
            list(pixel_values)
            if isinstance(pixel_values, list)
            else [pixel_values[i] for i in range(pixel_values.shape[0])]
            if pixel_position_ids is None and getattr(pixel_values, "ndim", 0) == 4
            else None
        )
        if images is None or len(images) <= 1:
            return original(self, pixel_values, pixel_position_ids)
        positions = (
            pixel_position_ids
            if isinstance(pixel_position_ids, list)
            else [None] * len(images)
        )
        groups = image_groups([img.shape[-2] * img.shape[-1] for img in images], budget)
        if len(groups) == 1:
            return original(self, pixel_values, pixel_position_ids)
        parts = []
        for start, end in groups:
            if isinstance(pixel_values, list):
                pos = positions[start:end]
                out = original(
                    self,
                    images[start:end],
                    pos if any(p is not None for p in pos) else None,
                )
            else:
                out = original(self, pixel_values[start:end], None)
            # Materialize this chunk and release the allocator's cache, so each chunk's encode
            # transient exists alone.
            mx.eval(out)
            mx.clear_cache()
            parts.append(out)
        return mx.concatenate(parts, axis=1)

    VisionModel.__call__ = chunked


def encoded_size(
    height: int, width: int, patch_size: int, pooling_kernel_size: int
) -> tuple[int, int]:
    """The (height, width) the patched processor encodes an image of this size at: upstream's own resize (get_aspect_ratio_preserving_size) at a budget of the image's own patches, capped at MAX_SOFT_TOKENS. An image on the grid within that budget comes back at its own size and a larger one is shrunk to the largest trained budget. The only image scaled up has a side shorter than one cell, which upstream's resize widens to one cell (40x2000 comes back 48x1632); one shorter than a cell on both sides raises."""
    from mlx_vlm.models.gemma4.processing_gemma4 import (
        get_aspect_ratio_preserving_size,
    )

    own = height * width // patch_size**2
    return get_aspect_ratio_preserving_size(
        height,
        width,
        patch_size,
        min(MAX_SOFT_TOKENS * pooling_kernel_size**2, own),
        pooling_kernel_size,
    )


def aspect_ratio_preserving_resize(
    self, image, patch_size, max_patches, pooling_kernel_size, input_data_format
):
    """Upstream's resize at encoded_size: within the largest trained budget, scaling an image up only where a side is shorter than one cell."""
    from transformers.image_utils import ChannelDimension

    del max_patches
    if input_data_format == ChannelDimension.FIRST:
        height, width = image.shape[1], image.shape[2]
    else:
        height, width = image.shape[0], image.shape[1]
    target_height, target_width = encoded_size(
        height, width, patch_size, pooling_kernel_size
    )
    if target_height == height and target_width == width:
        return image

    from PIL import Image

    array = (
        np.transpose(image, (1, 2, 0))
        if input_data_format == ChannelDimension.FIRST
        else image
    )
    if array.dtype in (np.float32, np.float64):
        array = (array * 255).clip(0, 255).astype(np.uint8)
    resized = np.array(
        Image.fromarray(array).resize(
            (target_width, target_height), resample=Image.BICUBIC
        )
    )
    return (
        np.transpose(resized, (2, 0, 1))
        if input_data_format == ChannelDimension.FIRST
        else resized
    )


def _replace_once(
    template: str, repair: str, fragment: str, replacement: str, owes: str
) -> str:
    count = template.count(fragment)
    if count != 1:
        refuse(
            repair,
            "the gemma4 chat template",
            f"holds the branch it replaces {count} times, not once",
            owes,
        )
    return template.replace(fragment, replacement)


def canonical_template() -> str:
    """Google's canonical Gemma 4 chat template at the pinned revision, fetched once into the Hugging Face cache, with the two repairs, and a tool reply's parts rendered in the order sent; content that does not hash to the pin refuses. A fetch that fails exits with its cause, in the server and in pricing alike: the server loads the processor inside upstream's model load, whose `except Exception` would otherwise answer the failure as a failed request."""
    import hashlib
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    source = (
        f"{CANONICAL_TEMPLATE_REPO}@{CANONICAL_TEMPLATE_REVISION} chat_template.jinja"
    )
    try:
        template = Path(
            hf_hub_download(
                CANONICAL_TEMPLATE_REPO,
                "chat_template.jinja",
                revision=CANONICAL_TEMPLATE_REVISION,
            )
        ).read_text()
    except Exception as error:
        raise SystemExit(
            f"canonical template: fetching {source} failed ({type(error).__name__}: {error}). "
            "Every gemma4 processor renders with it, so neither the server nor pricing runs a "
            "Gemma 4 card without it; "
            "fetch it once with the Hub reachable, after which the Hugging Face cache serves it."
        ) from error
    actual = hashlib.sha256(template.encode()).hexdigest()
    if actual != EXPECTED_CANONICAL_TEMPLATE_SHA256:
        refuse(
            "canonical template",
            source,
            f"hashes to {actual[:12]}, not the pinned {EXPECTED_CANONICAL_TEMPLATE_SHA256[:12]}",
            "Re-read the template at that revision, re-validate the repairs and the tool reply's part order against it, then re-pin.",
        )
    owes = (
        "Read the canonical chat template, re-derive the repair against what Google's own "
        "formatter writes, then update the fragment (analysis/src/locus/analysis/mlx/gemma4.py)."
    )
    template = _replace_once(
        template, "system-part join", TRAILING_SYSTEM_PARTS, JOINED_SYSTEM_PARTS, owes
    )
    template = _replace_once(
        template, "thought close", PADDED_THOUGHT_CLOSE, FAITHFUL_THOUGHT_CLOSE, owes
    )
    return _replace_once(
        template,
        "tool reply order",
        TOOL_TEXT_THEN_MEDIA,
        TOOL_PARTS_IN_ORDER,
        "Read the canonical chat template's tool-message branch and have it render a content "
        "list's parts in the order sent, inside the reply block's quoted string, then update the fragment "
        "(analysis/src/locus/analysis/mlx/gemma4.py).",
    )


def install_processor_contract() -> None:
    """Have every gemma4 processor encode an image at encoded_size and render with canonical_template's template; on drift, refuse to boot (patching.py)."""
    from mlx_vlm.models.gemma4 import processing_gemma4

    owes = (
        "Re-read the processor's resize, preprocess, and from_pretrained — an image on the "
        "grid within the largest trained budget must be encoded as sent, and the template "
        "set on load must be the one rendered — then re-validate and re-pin "
        "(analysis/src/locus/analysis/mlx/gemma4.py)."
    )
    image_processor = processing_gemma4.Gemma4ImageProcessor
    processor = processing_gemma4.Gemma4Processor
    if image_processor.aspect_ratio_preserving_resize is aspect_ratio_preserving_resize:
        return
    for what, fn, sha in (
        (
            "Gemma4ImageProcessor.aspect_ratio_preserving_resize",
            image_processor.aspect_ratio_preserving_resize,
            EXPECTED_RESIZE_SHA256,
        ),
        (
            "get_aspect_ratio_preserving_size",
            processing_gemma4.get_aspect_ratio_preserving_size,
            EXPECTED_ASPECT_SIZE_SHA256,
        ),
        (
            "Gemma4ImageProcessor.preprocess",
            image_processor.preprocess,
            EXPECTED_PREPROCESS_SHA256,
        ),
        (
            "Gemma4Processor.from_pretrained",
            processor.from_pretrained,
            EXPECTED_FROM_PRETRAINED_SHA256,
        ),
    ):
        expect_source("gemma4 processor", what, fn, sha, owes)

    image_processor.aspect_ratio_preserving_resize = aspect_ratio_preserving_resize
    load = processor.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        loaded = load(cls, pretrained_model_name_or_path, **kwargs)
        template = canonical_template()
        loaded.tokenizer.chat_template = template
        loaded.chat_template = template
        return loaded

    processor.from_pretrained = from_pretrained


def refined(markers) -> tuple:
    """The marker pairs in upstream's order, without any pair that another pair refines (the same close marker, and an open marker that extends its open marker). A streamed split buffers a partial open marker until it is complete, so once the coarser pair is removed the split matches the whole refined marker, however the tokens arrive."""

    def refines(other, pair):
        return (
            other[1] == pair[1]
            and len(other[0]) > len(pair[0])
            and other[0].startswith(pair[0])
        )

    return tuple(
        pair for pair in markers if not any(refines(other, pair) for other in markers)
    )


def clean_reasoning(reasoning: str, start_marker: str) -> str:
    """Upstream's cleanup of the thinking a whole-text split finds without its open marker, minus its removal of a leading `thought` after the channel's header: the canonical template opens the channel with its whole header, so the text here begins with the thought itself."""
    return reasoning.replace(start_marker, "").strip()


def install_reasoning_markers() -> None:
    """Have upstream's reasoning split delimit by the most specific marker pair, and clean a thought without cutting its first word; on drift, refuse to boot (patching.py)."""
    from mlx_vlm.server import responses_state
    from mlx_vlm.server.responses_state import ThinkingStreamState

    owes = (
        "Re-read how the reasoning split orders the configured marker pair against its "
        "built-in ones and cleans what it finds, then re-validate and re-pin "
        "(analysis/src/locus/analysis/mlx/gemma4.py)."
    )
    expect_source(
        "reasoning markers",
        "ThinkingStreamState._build_open_close_markers",
        ThinkingStreamState._build_open_close_markers,
        EXPECTED_MARKERS_SHA256,
        owes,
    )
    expect_source(
        "reasoning markers",
        "responses_state._clean_reasoning",
        responses_state._clean_reasoning,
        EXPECTED_CLEAN_REASONING_SHA256,
        owes,
    )
    build = ThinkingStreamState._build_open_close_markers.__func__

    @classmethod
    def _build_open_close_markers(cls, thinking_start_token, thinking_end_token):
        return refined(build(cls, thinking_start_token, thinking_end_token))

    ThinkingStreamState._build_open_close_markers = _build_open_close_markers
    responses_state._clean_reasoning = clean_reasoning
