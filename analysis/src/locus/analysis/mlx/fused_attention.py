"""Read every Qwen attention call through mlx's fused attention kernel.

For Qwen's head size (256), mlx's default dispatch on GPUs without neural accelerators reads a multi-token chunk by materializing its whole score table: new tokens × every key so far, held for the call. Prefill reads the prompt in chunks against everything before them, so the table grows with the prompt — 16 GiB for one 2,048-token chunk at 256k keys — and it sits wired on top of the weights, the cache, and the vision encode. The fused kernel walks the keys in tiles with a running softmax and never holds the table; the output is the same to rounding. mlx exposes the choice as `force_fused` because the trade — bounded memory for some prompt throughput — depends on the runtime's memory budget, which mlx cannot see. Here the budget is the dial, so every call asks for fused. Where mlx already picks the fused kernel for a shape (full chunks on GPUs with neural accelerators, single-token decode everywhere), nothing changes.

The patch point is qwen3_5.language's binding of base.scaled_dot_product_attention, the name Qwen3_5Attention calls (qwen3_5_moe's attention is the same class). The wrapper stands in for the helper's plain-cache branch only: quantized and TurboQuant caches go to the helper unchanged. It is keyed to the helper's exact source and to the attention call that reaches it by that name, so on drift in either the server refuses to boot (patching.py owns that policy).
"""

EXPECTED_HELPER_SHA256 = (
    "9b12ea1ee54be83003091a88dd8657b1448e1a68681dd791e69fbe971cd27cb5"
)
EXPECTED_ATTENTION_CALL_SHA256 = (
    "b051c66057acc90f4ed78691f90254685cfdb20d5814b0d58beec4d2bb34c184"
)


def install_fused_attention() -> None:
    """Rebind Qwen's attention helper to one that asks mlx for the fused kernel on plain caches."""
    import mlx.core as mx
    from mlx_vlm.models import base
    from mlx_vlm.models.qwen3_5 import language

    from .patching import expect_source

    expect_source(
        "fused attention",
        "base.scaled_dot_product_attention",
        base.scaled_dot_product_attention,
        EXPECTED_HELPER_SHA256,
        "The wrapper reproduces the helper's plain-cache branch with force_fused added — "
        "re-verify that branch and the quantized/TurboQuant routing against the installed "
        "version, then re-pin.",
    )
    expect_source(
        "fused attention",
        "Qwen3_5Attention.__call__",
        language.Qwen3_5Attention.__call__,
        EXPECTED_ATTENTION_CALL_SHA256,
        "The patch reaches Qwen's attention only through the module-level "
        "scaled_dot_product_attention name this call uses — re-verify the call still reads "
        "every attention through that name, then re-pin.",
    )
    original = base.scaled_dot_product_attention
    quantized = (base.TurboQuantKVCache, base.BatchTurboQuantKVCache)

    def fused(queries, keys, values, cache, scale, mask, sinks=None):
        if isinstance(cache, quantized) or hasattr(cache, "bits"):
            return original(queries, keys, values, cache, scale, mask, sinks)
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
            force_fused=True,
        )

    language.scaled_dot_product_attention = fused
