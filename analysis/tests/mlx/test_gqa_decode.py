"""The shared-tile decode-attention kernel — no model, no server; runs the GPU.

Exactness is the whole contract: the kernel replaces mx.fast.scaled_dot_product_attention on eligible decode calls, so every test here compares against that kernel on the same inputs and tolerates only bf16 output rounding. Eligibility is the other half — the router must decline everything the kernel was not built for and hand it back to upstream untouched.
"""

import mlx.core as mx
import pytest
from locus.analysis.mlx.gqa_decode import _MIN_SEQ_LEN, _eligible, gqa_decode
from mlx_vlm.models.cache import BatchKVCache, KVCache

HQ, HKV, DIM = 16, 2, 256
SCALE = DIM**-0.5


def rand(shape):
    return mx.random.normal(shape).astype(mx.bfloat16)


def max_diff(a, b):
    return mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()


def within_bf16_rounding(ref, out) -> bool:
    """The docstring's contract made measurable: both sides run the same math in float32 and round once to bf16, so they may differ by the format's rounding at the tensor's own scale — one bf16 mantissa step (2^-7) of the largest output — and no further. The bound is scale-relative, not flat and not per-element, because both alternatives misstate the contract: a flat epsilon under one step at magnitude ~1 fails legitimate single-step rounding (0.0039 beside an output of 0.9), and a per-element step count fails legitimate near-zero elements, which are differences of large softmax terms and so carry the arithmetic's scale, not their own."""
    return max_diff(ref, out) <= 2**-7 * mx.abs(ref).max().item()


@pytest.mark.parametrize("length", [1, 17, 255, 4097, 32768, 65531])
def test_matches_sdpa(length):
    q, k, v = (
        rand((1, HQ, 1, DIM)),
        rand((1, HKV, length, DIM)),
        rand((1, HKV, length, DIM)),
    )
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
    assert within_bf16_rounding(ref, gqa_decode(q, k, v, SCALE))


def test_matches_sdpa_on_cache_slices():
    """KVCache hands attention non-contiguous views of its padded buffer; the kernel must read them in place."""
    length = 33000
    q = rand((1, HQ, 1, DIM))
    kbuf, vbuf = rand((1, HKV, length + 280, DIM)), rand((1, HKV, length + 280, DIM))
    k, v = kbuf[..., :length, :], vbuf[..., :length, :]
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
    assert within_bf16_rounding(ref, gqa_decode(q, k, v, SCALE))


def test_matches_sdpa_batched():
    q, k, v = (
        rand((2, HQ, 1, DIM)),
        rand((2, HKV, 40000, DIM)),
        rand((2, HKV, 40000, DIM)),
    )
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
    assert within_bf16_rounding(ref, gqa_decode(q, k, v, SCALE))


def test_eligibility_gate():
    long = _MIN_SEQ_LEN
    q, k, v = (
        rand((1, HQ, 1, DIM)),
        rand((1, HKV, long, DIM)),
        rand((1, HKV, long, DIM)),
    )
    for cache in (KVCache(), BatchKVCache([0])):
        assert _eligible(q, k, v, cache, None, None)

    cache = KVCache()
    assert not _eligible(q, k, v, cache, mx.zeros((1, 1, 1, long)), None)  # masked
    assert not _eligible(q, k, v, cache, None, mx.zeros((HQ,)))  # sinks
    assert not _eligible(q, k, v, object(), None, None)  # exotic cache
    assert not _eligible(rand((1, HQ, 2, DIM)), k, v, cache, None, None)  # prefill
    assert not _eligible(
        q, rand((1, HKV, long - 1, DIM)), v, cache, None, None
    )  # short
    assert not _eligible(rand((1, HKV * 2, 1, DIM)), k, v, cache, None, None)  # low gqa
    assert not _eligible(q.astype(mx.float16), k, v, cache, None, None)  # dtype
    assert not _eligible(
        rand((1, HQ, 1, 128)),
        rand((1, HKV, long, 128)),
        rand((1, HKV, long, 128)),
        cache,
        None,
        None,
    )  # head_dim


def test_install_routes_and_falls_through():
    from locus.analysis.mlx.gqa_decode import install_gqa_decode
    from mlx_vlm.models.qwen3_5 import language

    original = language.scaled_dot_product_attention
    try:
        install_gqa_decode()
        routed = language.scaled_dot_product_attention
        assert routed is not original

        length = _MIN_SEQ_LEN
        q, k, v = (
            rand((1, HQ, 1, DIM)),
            rand((1, HKV, length, DIM)),
            rand((1, HKV, length, DIM)),
        )
        via_router = routed(q, k, v, cache=KVCache(), scale=SCALE, mask=None)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
        assert within_bf16_rounding(ref, via_router)

        short_k, short_v = rand((1, HKV, 64, DIM)), rand((1, HKV, 64, DIM))
        via_fallthrough = routed(
            q, short_k, short_v, cache=KVCache(), scale=SCALE, mask=None
        )
        ref_short = mx.fast.scaled_dot_product_attention(
            q, short_k, short_v, scale=SCALE
        )
        assert max_diff(ref_short, via_fallthrough) == 0.0
    finally:
        language.scaled_dot_product_attention = original


def test_refuses_boot_on_foreign_upstream_signature():
    """The router stands in for upstream's exact call signature; on drift the boot must refuse and leave the binding untouched, because a mismatched stand-in would instead raise TypeError mid-inference, in the middle of a window."""
    from locus.analysis.mlx.gqa_decode import install_gqa_decode
    from mlx_vlm.models.qwen3_5 import language

    original = language.scaled_dot_product_attention

    def foreign(queries, keys, values, cache, scale, mask, sinks, window):
        return None

    try:
        language.scaled_dot_product_attention = foreign
        with pytest.raises(SystemExit, match="upstream drift"):
            install_gqa_decode()
        assert language.scaled_dot_product_attention is foreign
    finally:
        language.scaled_dot_product_attention = original
