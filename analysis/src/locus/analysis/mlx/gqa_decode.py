"""Decode attention for high-GQA shapes, where the stock kernel leaves half the memory bandwidth unused.

At single-token decode, mlx's `mx.fast.scaled_dot_product_attention` (`sdpa_vector_2pass`) assigns one simdgroup per *query* head, so the simdgroups of a GQA group each stream the same K/V bytes from device memory: at the Qwen3.6 shape (16 query heads over 2 KV heads, head_dim 256) the kernel sustains ~118 GiB/s of unique KV traffic on an M4 Pro whose streaming rate is ~220 GiB/s, and long-context decode is bound by exactly that traffic. The kernel here stages K/V tiles through threadgroup memory once per KV head — one simdgroup per query head consumes the shared tile — and sustains ~170-180 GiB/s at 224k tokens, a ~1.4x on the attention step.

The math is the one the stock kernel does: fp32 accumulation, online softmax within each sequence block, block partials combined by a second reduction pass. Outputs agree with `mx.fast.scaled_dot_product_attention` to bf16 output rounding (max |diff| ~6e-5 at 224k tokens).

`install_gqa_decode()` rebinds `scaled_dot_product_attention` inside `mlx_vlm.models.qwen3_5.language` — the module both Qwen3.5/3.6 dense and MoE attention resolve it from — with a wrapper that routes to this kernel only when every condition holds, and to upstream otherwise:

- query length 1 (decode), no mask, no sinks — the single-sequence decode path; batched left-padded decode carries a mask and stays upstream
- plain `KVCache`/`BatchKVCache` (quantized and TurboQuant caches keep their own paths)
- bfloat16 q/k/v, head_dim 256, GQA factor >= 4, and a sequence long enough that the stock kernel's redundancy costs more than this kernel's fixed overhead (two dispatches and the fp32 block-partial intermediate the second pass folds — 2 MiB per sequence at this model's 16 query heads, head_dim 256, and the 128-block ceiling)

The kernel reads K/V through their strides, so the zero-copy cache slices `KVCache.update_and_fetch` returns (contiguous within a token, strided across tokens) are consumed in place.

Keyed to the pinned mlx (last re-measured on 0.32.1: stock 122 GiB/s vs 180 GiB/s here at 224k, 1.48x): re-measure against the stock kernel on any mlx upgrade — if `sdpa_vector_2pass` learns to share K/V reads across a GQA group, this kernel is deletable.
"""

import mlx.core as mx
from mlx_vlm.models.cache import BatchKVCache, KVCache

_HEAD_DIM = 256
_MIN_GQA_FACTOR = 4
_MIN_SEQ_LEN = 32768
_MAX_BLOCKS = 128
_TILE = 16

_P1_SRC = """
    // One threadgroup per (batch*kv_head, block): G simdgroups, one per query
    // head of the group. K/V tiles are staged through threadgroup memory once,
    // so device memory sees each K/V byte once per KV head. Tokens are consumed
    // four at a time to overlap the per-token simd_sum reduction chains.
    constexpr int PER_LANE = D / 32;
    constexpr int VEC = 4;
    constexpr int NVEC = PER_LANE / VEC;

    uint lane = thread_position_in_threadgroup.x;
    uint h    = thread_position_in_threadgroup.y;
    uint bh   = threadgroup_position_in_grid.y;
    uint blk  = threadgroup_position_in_grid.z;
    uint nb   = threadgroups_per_grid.z;

    const int HKV = k_shape[1];
    const int L   = k_shape[2];
    const int b   = bh / HKV;
    const int kv  = bh % HKV;
    const int HQ  = HKV * G;

    int chunk = (L + nb - 1) / nb;
    int start = blk * chunk;
    int end   = min(start + chunk, L);

    typedef vec<T, VEC> Tv;
    threadgroup T k_tile[TILE * D];
    threadgroup T v_tile[TILE * D];

    float4 q_reg[NVEC];
    {
        const device Tv* qp = (const device Tv*)(q + ((size_t)b * HQ + kv * G + h) * D + lane * PER_LANE);
        for (int j = 0; j < NVEC; j++) {
            q_reg[j] = float4(qp[j]) * scale[0];
        }
    }

    float o_reg[PER_LANE] = {0.0f};
    float max_s = -1e30f;
    float sum_s = 0.0f;

    const size_t k_base = (size_t)b * k_strides[0] + (size_t)kv * k_strides[1];
    const size_t v_base = (size_t)b * v_strides[0] + (size_t)kv * v_strides[1];

    constexpr int THREADS = 32 * G;
    constexpr int TILE_VECS = TILE * D / VEC;
    constexpr int LOADS = (TILE_VECS + THREADS - 1) / THREADS;
    uint tix = h * 32 + lane;

    for (int t0 = start; t0 < end; t0 += TILE) {
        int ntok = min(TILE, end - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int m = 0; m < LOADS; m++) {
            uint vi = tix + m * THREADS;
            if (vi < (uint)(ntok * D / VEC)) {
                uint tok = vi / (D / VEC);
                uint off = vi % (D / VEC);
                const device Tv* kp = (const device Tv*)(k + k_base + (size_t)(t0 + tok) * k_strides[2]);
                const device Tv* vp = (const device Tv*)(v + v_base + (size_t)(t0 + tok) * v_strides[2]);
                ((threadgroup Tv*)k_tile)[tok * (D / VEC) + off] = kp[off];
                ((threadgroup Tv*)v_tile)[tok * (D / VEC) + off] = vp[off];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int i = 0;
        for (; i + 4 <= ntok; i += 4) {
            const threadgroup Tv* kt = ((const threadgroup Tv*)k_tile) + i * (D / VEC) + lane * NVEC;
            float4 pv0 = 0.0f, pv1 = 0.0f, pv2 = 0.0f, pv3 = 0.0f;
            for (int j = 0; j < NVEC; j++) {
                pv0 += q_reg[j] * float4(kt[j]);
                pv1 += q_reg[j] * float4(kt[j + (D/VEC)]);
                pv2 += q_reg[j] * float4(kt[j + 2*(D/VEC)]);
                pv3 += q_reg[j] * float4(kt[j + 3*(D/VEC)]);
            }
            float s0 = simd_sum(pv0.x + pv0.y + pv0.z + pv0.w);
            float s1 = simd_sum(pv1.x + pv1.y + pv1.z + pv1.w);
            float s2 = simd_sum(pv2.x + pv2.y + pv2.z + pv2.w);
            float s3 = simd_sum(pv3.x + pv3.y + pv3.z + pv3.w);
            float new_max = max(max(max_s, max(s0, s1)), max(s2, s3));
            float factor = metal::fast::exp(max_s - new_max);
            float p0 = metal::fast::exp(s0 - new_max);
            float p1 = metal::fast::exp(s1 - new_max);
            float p2 = metal::fast::exp(s2 - new_max);
            float p3 = metal::fast::exp(s3 - new_max);
            max_s = new_max;
            sum_s = sum_s * factor + p0 + p1 + p2 + p3;
            const threadgroup Tv* vt = ((const threadgroup Tv*)v_tile) + i * (D / VEC) + lane * NVEC;
            for (int j = 0; j < NVEC; j++) {
                float4 vv0 = float4(vt[j]);
                float4 vv1 = float4(vt[j + (D/VEC)]);
                float4 vv2 = float4(vt[j + 2*(D/VEC)]);
                float4 vv3 = float4(vt[j + 3*(D/VEC)]);
                float4 acc = float4(o_reg[VEC*j], o_reg[VEC*j+1], o_reg[VEC*j+2], o_reg[VEC*j+3]);
                acc = acc * factor + p0 * vv0 + p1 * vv1 + p2 * vv2 + p3 * vv3;
                o_reg[VEC*j] = acc.x; o_reg[VEC*j+1] = acc.y; o_reg[VEC*j+2] = acc.z; o_reg[VEC*j+3] = acc.w;
            }
        }
        for (; i < ntok; i++) {
            const threadgroup Tv* kt = ((const threadgroup Tv*)k_tile) + i * (D / VEC) + lane * NVEC;
            const threadgroup Tv* vt = ((const threadgroup Tv*)v_tile) + i * (D / VEC) + lane * NVEC;
            float4 pv = 0.0f;
            for (int j = 0; j < NVEC; j++) {
                pv += q_reg[j] * float4(kt[j]);
            }
            float score = simd_sum(pv.x + pv.y + pv.z + pv.w);
            float new_max = max(max_s, score);
            float factor = metal::fast::exp(max_s - new_max);
            float p = metal::fast::exp(score - new_max);
            max_s = new_max;
            sum_s = sum_s * factor + p;
            for (int j = 0; j < NVEC; j++) {
                float4 vv = float4(vt[j]);
                o_reg[VEC*j+0] = o_reg[VEC*j+0] * factor + p * vv.x;
                o_reg[VEC*j+1] = o_reg[VEC*j+1] * factor + p * vv.y;
                o_reg[VEC*j+2] = o_reg[VEC*j+2] * factor + p * vv.z;
                o_reg[VEC*j+3] = o_reg[VEC*j+3] * factor + p * vv.w;
            }
        }
    }

    size_t oh = ((size_t)b * HQ + kv * G + h) * nb + blk;
    device float* pop = po + oh * D + lane * PER_LANE;
    for (int j = 0; j < PER_LANE; j++) {
        pop[j] = o_reg[j];
    }
    if (lane == 0) {
        pm[oh] = max_s;
        ps[oh] = sum_s;
    }
"""

_P2_SRC = """
    // One simdgroup per (batch, query head): fold the block partials with the
    // global max, exactly the stock kernel's second pass.
    constexpr int PER_LANE = D / 32;
    uint lane = thread_position_in_threadgroup.x;
    uint bq   = thread_position_in_grid.y;
    const int NB = pm_shape[1];

    const device float* pmh = pm + (size_t)bq * NB;
    const device float* psh = ps + (size_t)bq * NB;

    float m = -1e30f;
    for (int i = lane; i < NB; i += 32) {
        m = max(m, pmh[i]);
    }
    m = simd_max(m);

    float total = 0.0f;
    for (int i = lane; i < NB; i += 32) {
        total += metal::fast::exp(pmh[i] - m) * psh[i];
    }
    total = simd_sum(total);

    float acc[PER_LANE] = {0.0f};
    for (int i = 0; i < NB; i++) {
        float w = metal::fast::exp(pmh[i] - m);
        if (w == 0.0f) continue;
        const device float* pop = po + ((size_t)bq * NB + i) * D + lane * PER_LANE;
        for (int j = 0; j < PER_LANE; j++) {
            acc[j] += w * pop[j];
        }
    }
    device T* op = out + (size_t)bq * D + lane * PER_LANE;
    for (int j = 0; j < PER_LANE; j++) {
        op[j] = (T)(acc[j] / total);
    }
"""

_p1 = mx.fast.metal_kernel(
    name="locus_gqa_decode_p1",
    input_names=["q", "k", "v", "scale"],
    output_names=["po", "ps", "pm"],
    source=_P1_SRC,
    ensure_row_contiguous=False,
)
_p2 = mx.fast.metal_kernel(
    name="locus_gqa_decode_p2",
    input_names=["po", "ps", "pm"],
    output_names=["out"],
    source=_P2_SRC,
    ensure_row_contiguous=False,
)


def gqa_decode(q, k, v, scale):
    """Decode attention (q_len 1, maskless) for bf16 GQA shapes; exact SDPA math."""
    B, HQ, _, D = q.shape
    HKV, L = k.shape[1], k.shape[2]
    G = HQ // HKV
    NB = max(1, min(_MAX_BLOCKS, L // (_TILE * 4)))
    q = mx.contiguous(q)
    scale_arr = mx.array([scale], dtype=mx.float32)
    po, ps, pm = _p1(
        inputs=[q, k, v, scale_arr],
        template=[("T", q.dtype), ("D", D), ("G", G), ("TILE", _TILE)],
        grid=(32, B * HKV * G, NB),
        threadgroup=(32, G, 1),
        output_shapes=[(B * HQ, NB, D), (B * HQ, NB), (B * HQ, NB)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = _p2(
        inputs=[po, ps, pm],
        template=[("T", q.dtype), ("D", D)],
        grid=(32, B * HQ, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, HQ, 1, D)],
        output_dtypes=[q.dtype],
    )
    return out


def _eligible(queries, keys, values, cache, mask, sinks):
    return (
        mask is None
        and sinks is None
        and isinstance(cache, (KVCache, BatchKVCache))
        and queries.dtype == mx.bfloat16
        and keys.dtype == mx.bfloat16
        and values.dtype == mx.bfloat16
        and queries.ndim == 4
        and queries.shape[2] == 1
        and queries.shape[3] == _HEAD_DIM
        and keys.shape[3] == _HEAD_DIM
        and values.shape[3] == _HEAD_DIM
        and keys.shape[1] > 0
        and queries.shape[1] % keys.shape[1] == 0
        and queries.shape[1] // keys.shape[1] >= _MIN_GQA_FACTOR
        and keys.shape[2] >= _MIN_SEQ_LEN
    )


# The upstream call this patch stands in front of. A signature it no longer matches would raise
# TypeError mid-inference, in the middle of a window, so it is checked at boot and a mismatch
# refuses the boot (patching.py owns that policy).
UPSTREAM_SDPA_PARAMS = ("queries", "keys", "values", "cache", "scale", "mask", "sinks")


def install_gqa_decode() -> None:
    """Route eligible decode-attention calls in qwen3_5 to the shared-tile kernel.

    Rebinds the `scaled_dot_product_attention` name inside
    `mlx_vlm.models.qwen3_5.language` (the resolution point for both the dense
    and MoE attention classes); every ineligible call falls through to the
    binding it replaces. On upstream signature drift it refuses to boot
    (patching.py owns that policy).
    """
    import inspect

    from mlx_vlm.models.qwen3_5 import language

    from .patching import refuse

    upstream = language.scaled_dot_product_attention
    params = tuple(inspect.signature(upstream).parameters)
    if params != UPSTREAM_SDPA_PARAMS:
        refuse(
            "gqa decode kernel",
            "scaled_dot_product_attention",
            f"takes {params}, this patch stands in for {UPSTREAM_SDPA_PARAMS}",
            "Re-validate the wrapper's routing and eligibility checks against the new "
            "signature, then update UPSTREAM_SDPA_PARAMS.",
        )

    def routed(queries, keys, values, cache, scale, mask, sinks=None):
        if _eligible(queries, keys, values, cache, mask, sinks):
            return gqa_decode(queries, keys, values, scale)
        return upstream(
            queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks
        )

    language.scaled_dot_product_attention = routed
