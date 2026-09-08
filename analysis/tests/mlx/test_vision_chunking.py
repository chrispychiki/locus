"""Unit tests for the vision-encode chunking patch — no model checkpoint, no server.

The output-identity test runs a tiny randomly-initialized VisionModel through the real upstream
tower code, chunked and unchunked, and compares features: the whole patch rests on the claim that
the tower processes images independently, so that claim is asserted against the actual math, not
read off a comment.
"""

import itertools

import locus.analysis.mlx.vision_chunking as vc
import mlx.core as mx
import pytest
from locus.analysis.mlx.vision_chunking import (
    chunk_rows,
    install_vision_encode_chunking,
)
from mlx_vlm.models.qwen3_vl.config import VisionConfig
from mlx_vlm.models.qwen3_vl.vision import VisionModel

PATCH_PIXELS = 16 * 16


@pytest.fixture(autouse=True)
def restore_call():
    original = VisionModel.__call__
    yield
    VisionModel.__call__ = original


class TestChunkRows:
    def test_everything_under_budget_is_one_group(self):
        rows = [(1, 4, 4), (1, 4, 4)]
        assert chunk_rows(rows, PATCH_PIXELS, 1e9) == [(0, 2)]

    def test_splits_at_row_boundaries_when_budget_fills(self):
        # Each row is 1*8*8*256 = 16384 px; budget fits exactly two.
        rows = [(1, 8, 8)] * 5
        assert chunk_rows(rows, PATCH_PIXELS, 2 * 16384) == [(0, 2), (2, 4), (4, 5)]

    def test_a_single_row_over_budget_encodes_alone(self):
        rows = [(1, 4, 4), (1, 100, 100), (1, 4, 4)]
        groups = chunk_rows(rows, PATCH_PIXELS, 5000)
        assert groups == [(0, 1), (1, 2), (2, 3)]

    def test_rows_are_never_split_and_order_is_preserved(self):
        rows = [(1, 6, 6), (2, 8, 4), (1, 2, 2), (1, 10, 10)]
        groups = chunk_rows(rows, PATCH_PIXELS, 20000)
        assert groups[0][0] == 0
        assert groups[-1][1] == len(rows)
        for (a, b), (c, d) in itertools.pairwise(groups):
            assert b == c
            assert a < b and c < d

    def test_exact_fit_stays_in_one_group(self):
        rows = [(1, 8, 8), (1, 8, 8)]
        assert chunk_rows(rows, PATCH_PIXELS, 2 * 16384) == [(0, 2)]


class TestInstall:
    def test_refuses_boot_on_upstream_source_drift(self, monkeypatch):
        monkeypatch.setattr(vc, "EXPECTED_CALL_SHA256", "0" * 64)
        before = VisionModel.__call__
        with pytest.raises(SystemExit, match="upstream drift"):
            install_vision_encode_chunking()
        assert VisionModel.__call__ is before

    def test_installs_against_the_pinned_upstream(self):
        before = VisionModel.__call__
        install_vision_encode_chunking()
        assert VisionModel.__call__ is not before

    def test_explicit_budget_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("LOCUS_MLX_ENCODE_CHUNK_MPX", "7.5")
        assert vc._budget_pixels(None) == 7.5e6
        assert vc._budget_pixels(2.0) == 2.0e6

    def test_default_budget_reflects_the_measured_slope(self):
        # ~0.25 GiB of transient per Mpx: the default bounds the encode spike to ~4 GiB.
        assert vc._budget_pixels(None) == vc.DEFAULT_CHUNK_MPX * 1e6
        assert 2.0 <= vc.DEFAULT_CHUNK_MPX * 0.25 <= 8.0


def tiny_vision_model():
    config = VisionConfig(
        model_type="qwen3_vl",
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        out_hidden_size=32,
        num_heads=4,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0],
    )
    return VisionModel(config)


def patch_input(rows, seed=0):
    total_patches = sum(t * h * w for t, h, w in rows)
    patch_dim = 3 * 2 * 16 * 16
    pixels = mx.random.normal((total_patches, patch_dim), key=mx.random.key(seed))
    grid = mx.array([list(r) for r in rows], dtype=mx.int32)
    return pixels, grid


class TestOutputIdentity:
    def test_chunked_encode_matches_unchunked(self):
        model = tiny_vision_model()
        rows = [(1, 4, 4), (1, 8, 4), (1, 6, 6), (1, 4, 8)]
        pixels, grid = patch_input(rows)

        merged_ref, deep_ref = VisionModel.__call__(model, pixels, grid)
        mx.eval(merged_ref, *deep_ref)

        # A budget of one row's pixels forces every image into its own chunk.
        install_vision_encode_chunking(budget_mpx=4 * 4 * PATCH_PIXELS / 1e6)
        merged, deep = model(pixels, grid)
        mx.eval(merged, *deep)

        assert merged.shape == merged_ref.shape
        assert mx.allclose(merged, merged_ref, atol=1e-5, rtol=1e-5)
        assert len(deep) == len(deep_ref)
        for got, ref in zip(deep, deep_ref):
            assert got.shape == ref.shape
            assert mx.allclose(got, ref, atol=1e-5, rtol=1e-5)

    def test_partial_grouping_matches_too(self):
        model = tiny_vision_model()
        rows = [(1, 4, 4), (1, 4, 4), (1, 8, 8), (1, 4, 4)]
        pixels, grid = patch_input(rows, seed=7)

        merged_ref, deep_ref = VisionModel.__call__(model, pixels, grid)

        # Budget of two small rows: groups land as [2 rows][1 row][1 row].
        install_vision_encode_chunking(budget_mpx=2 * 4 * 4 * PATCH_PIXELS / 1e6)
        merged, deep = model(pixels, grid)

        assert mx.allclose(merged, merged_ref, atol=1e-5, rtol=1e-5)
        for got, ref in zip(deep, deep_ref):
            assert mx.allclose(got, ref, atol=1e-5, rtol=1e-5)

    def test_single_image_takes_the_unchunked_path(self, monkeypatch):
        """A one-image call must go straight to upstream — no grouping, no per-chunk eval/clear overhead — so the chunker being consulted at all is the failure."""
        model = tiny_vision_model()
        rows = [(1, 4, 4)]
        pixels, grid = patch_input(rows, seed=3)
        merged_ref, deep_ref = VisionModel.__call__(model, pixels, grid)

        install_vision_encode_chunking(budget_mpx=0.000001)

        def never(*args, **kwargs):
            raise AssertionError("single-image call consulted the chunker")

        monkeypatch.setattr(vc, "chunk_rows", never)
        merged, deep = model(pixels, grid)

        assert mx.allclose(merged, merged_ref, atol=1e-6, rtol=1e-6)
        assert len(deep) == len(deep_ref)

    def test_layout_mismatch_warns_and_delegates_instead_of_slicing(self, caplog):
        """The chunker slices hidden_states at offsets computed from grid_thw, an arithmetic the hash pin does not protect (it pins __call__, not the callers' patch-row layout). On a mismatch the patched call must warn and hand the untouched inputs to upstream — silently slicing by wrong offsets would scramble features. The bar is behaving exactly as the unchunked encode does on the same inputs, whatever that behavior is."""
        model = tiny_vision_model()
        rows = [(1, 4, 4), (1, 4, 4)]
        pixels, grid = patch_input(rows, seed=11)
        extra = mx.concatenate([pixels, mx.zeros((16, pixels.shape[1]))], axis=0)

        def outcome(call):
            try:
                merged, deep = call()
                mx.eval(merged, *deep)
                return ("ok", merged.shape, [d.shape for d in deep])
            except Exception as exc:  # noqa: BLE001 — the outcome, not a swallow
                return ("raise", type(exc))

        reference = outcome(lambda: VisionModel.__call__(model, extra, grid))
        install_vision_encode_chunking(budget_mpx=4 * 4 * PATCH_PIXELS / 1e6)
        assert outcome(lambda: model(extra, grid)) == reference
        assert "chunking skipped" in caplog.text
