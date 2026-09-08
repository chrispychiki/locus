"""Bound the vision-encode memory transient by chunking the encode at a pixel budget.

Upstream encodes a request's entire vision batch as one graph: qwen3_vl's get_input_embeddings
hands every image's patches to VisionModel.__call__ in a single call, so the encode's command
buffer grows with the request's total pixels, and past the machine's wired headroom it fails as
an uncatchable Metal command-buffer OOM that kills the server process.

The bound is at the encode itself: the batch is partitioned into contiguous whole-image chunks
whose pixel sum stays within LOCUS_MLX_ENCODE_CHUNK_MPX, each chunk runs upstream's own
__call__ with the outputs materialized and the allocator cache released between chunks, and the
per-chunk outputs concatenate. An image is never split; a single image over the budget encodes
alone (the preprocessor caps one image at 16.7 Mpx).

Chunking is output-identical to the unchunked encode on the pinned upstream source: patch
embedding, position-embedding interpolation, and the mergers are per-image (or per-token)
operations concatenated across the batch; attention splits q/k/v at cu_seqlens and runs SDPA per
image segment, so no information crosses image boundaries; and the rotary frequency table's rows
are position-indexed values independent of the batch-max table height.

That independence is a property of the exact upstream source, so the patch is keyed to it: when
the installed mlx-vlm's VisionModel.__call__ no longer hashes to the pinned version, the server
refuses to boot (patching.py owns that policy). qwen3_5 and qwen3_5_moe inherit __call__ from
qwen3_vl's VisionModel, so the one patch point covers the dense and MoE models alike, on the
server path and the vision-feature-cache path both (the cache stores this function's output,
which chunking does not change).
"""

import logging
import os

logger = logging.getLogger("locus.analysis.mlx.vision_chunking")

DEFAULT_CHUNK_MPX = 16.0

EXPECTED_CALL_SHA256 = (
    "012ef942326f600189fa96b338a22d90c18cdad4ce73313669e1a84ddbdee183"
)


def _budget_pixels(budget_mpx: float | None) -> float:
    if budget_mpx is None:
        budget_mpx = float(
            os.environ.get("LOCUS_MLX_ENCODE_CHUNK_MPX", DEFAULT_CHUNK_MPX)
        )
    return budget_mpx * 1e6


def chunk_rows(rows, patch_pixels, budget_pixels):
    """Partition grid rows [(t, h, w), ...] into contiguous groups whose source-pixel sum
    (t*h*w patches × patch_pixels each) stays within the budget. A row is never split; a single
    row over the budget forms its own group. Returns [(start, end), ...] half-open index pairs
    covering every row in order."""
    groups = []
    start, used = 0, 0
    for i, (t, h, w) in enumerate(rows):
        pixels = t * h * w * patch_pixels
        if i > start and used + pixels > budget_pixels:
            groups.append((start, i))
            start, used = i, 0
        used += pixels
    groups.append((start, len(rows)))
    return groups


def install_vision_encode_chunking(budget_mpx: float | None = None) -> None:
    """Replace qwen3_vl VisionModel.__call__ with the pixel-chunked equivalent. On upstream
    source drift it refuses to boot (patching.py owns that policy)."""
    from mlx_vlm.models.qwen3_vl.vision import VisionModel

    from .patching import expect_source

    expect_source(
        "vision-encode chunking",
        "VisionModel.__call__",
        VisionModel.__call__,
        EXPECTED_CALL_SHA256,
        "Chunking is output-identical only because the pinned source processes images "
        "independently — re-verify that property against the installed version, then re-pin.",
    )

    import mlx.core as mx

    budget = _budget_pixels(budget_mpx)
    original = VisionModel.__call__

    def chunked(self, hidden_states, grid_thw, **kwargs):
        if grid_thw is None or grid_thw.shape[0] <= 1:
            return original(self, hidden_states, grid_thw, **kwargs)
        rows = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
        patch_pixels = self.patch_embed.patch_size**2
        groups = chunk_rows(rows, patch_pixels, budget)
        if len(groups) == 1:
            return original(self, hidden_states, grid_thw, **kwargs)

        patches = [t * h * w for t, h, w in rows]
        if hidden_states.shape[0] != sum(patches):
            logger.warning(
                f"vision-encode chunking skipped for this call: {hidden_states.shape[0]} patch rows != {sum(patches)} expected from grid_thw — the patch-row layout differs from the pinned upstream's; encoding unchunked"
            )
            return original(self, hidden_states, grid_thw, **kwargs)
        offsets = [0]
        for count in patches:
            offsets.append(offsets[-1] + count)

        merged_parts, deep_parts = [], []
        for start, end in groups:
            merged, deepstacks = original(
                self,
                hidden_states[offsets[start] : offsets[end]],
                grid_thw[start:end],
                **kwargs,
            )
            # Materialize this chunk and release the allocator's cache so each chunk's encode
            # transient exists alone instead of one graph holding every chunk's at once.
            mx.eval(merged, *deepstacks)
            mx.clear_cache()
            merged_parts.append(merged)
            deep_parts.append(deepstacks)

        return (
            mx.concatenate(merged_parts, axis=0),
            [mx.concatenate(layer, axis=0) for layer in zip(*deep_parts)],
        )

    VisionModel.__call__ = chunked
