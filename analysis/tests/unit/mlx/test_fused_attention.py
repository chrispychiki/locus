"""Unit tests for the fused-attention patch — no model checkpoint, no server.

The patch's claims are that Qwen's attention reads through the fused kernel without holding the score table, that its output matches mlx's default dispatch, and that quantized caches keep upstream's own path; each is asserted against real mlx calls.
"""

import locus.analysis.mlx.fused_attention as fa
import mlx.core as mx
import pytest
from locus.analysis.mlx.fused_attention import install_fused_attention
from mlx_vlm.models import base
from mlx_vlm.models.qwen3_5 import language

HQ, HKV, D = 16, 2, 256


@pytest.fixture(autouse=True)
def restore_binding():
    original = language.scaled_dot_product_attention
    yield
    language.scaled_dot_product_attention = original


def qkv(q_len, k_len):
    mx.random.seed(0)
    q = mx.random.normal((1, HQ, q_len, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, HKV, k_len, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, k_len, D)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v


def peak_transient(fn):
    mx.synchronize()
    mx.clear_cache()
    base_memory = mx.get_active_memory()
    mx.reset_peak_memory()
    out = fn()
    mx.eval(out)
    mx.synchronize()
    return out, mx.get_peak_memory() - base_memory


class TestInstall:
    @pytest.mark.parametrize(
        "pin", ["EXPECTED_HELPER_SHA256", "EXPECTED_ATTENTION_CALL_SHA256"]
    )
    def test_refuses_boot_on_upstream_source_drift(self, monkeypatch, pin):
        monkeypatch.setattr(fa, pin, "0" * 64)
        before = language.scaled_dot_product_attention
        with pytest.raises(SystemExit, match="upstream drift"):
            install_fused_attention()
        assert language.scaled_dot_product_attention is before

    def test_rebinds_qwens_helper_and_nothing_else(self):
        install_fused_attention()
        assert (
            language.scaled_dot_product_attention
            is not base.scaled_dot_product_attention
        )


class TestFused:
    @pytest.mark.parametrize(
        "q_len,k_len", [(2048, 2048), (2048, 16384), (300, 8492), (1, 8192)]
    )
    def test_matches_the_default_dispatch(self, q_len, k_len):
        q, k, v = qkv(q_len, k_len)
        scale = D**-0.5
        mask = "causal" if q_len > 1 else None
        expected = base.scaled_dot_product_attention(
            q, k, v, cache=None, scale=scale, mask=mask
        )
        install_fused_attention()
        got = language.scaled_dot_product_attention(
            q, k, v, cache=None, scale=scale, mask=mask
        )
        diff = mx.max(
            mx.abs(got.astype(mx.float32) - expected.astype(mx.float32))
        ).item()
        # bf16 output: one rounding of values of order 1 is ~4e-3; the two kernels differ only
        # in accumulation order inside fp32.
        assert diff < 4e-3

    def test_a_chunk_against_long_context_never_holds_the_table(self):
        q, k, v = qkv(2048, 32768)
        table = HQ * 2048 * 32768 * 2
        install_fused_attention()
        _, transient = peak_transient(
            lambda: language.scaled_dot_product_attention(
                q, k, v, cache=None, scale=D**-0.5, mask="causal"
            )
        )
        assert transient < table / 16

    def test_quantized_caches_keep_upstreams_path(self, monkeypatch):
        calls = []

        def spy(*args):
            calls.append(args)
            return "upstream"

        monkeypatch.setattr(base, "scaled_dot_product_attention", spy)

        class QuantizedCache:
            bits = 8

        # The spy stands in for the helper, so its source no longer matches the pin.
        monkeypatch.setattr(
            "locus.analysis.mlx.patching.expect_source", lambda *a, **kw: None
        )
        install_fused_attention()
        out = language.scaled_dot_product_attention(
            "q", "k", "v", cache=QuantizedCache(), scale=1.0, mask=None
        )
        assert out == "upstream" and len(calls) == 1
