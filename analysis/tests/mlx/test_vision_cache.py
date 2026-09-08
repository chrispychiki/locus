"""Unit tests for upstream's vision feature cache — no model checkpoint, no server, no generation.

The property the cache owes the analysis path: a hit returns bit-exactly what a miss stored, and only for the same image. Upstream's read path uses a hit verbatim as the vision tower's output (models/qwen3_5: `hidden_states = cached`, no transformation), so cache correctness IS feature-reuse correctness — asserted here at the tensor, where the claim lives, rather than inferred through sampled text. Whether the model can read images at all is the live integration suite's job.
"""

import mlx.core as mx
from mlx_vlm.vision_cache import VisionFeatureCache
from PIL import Image


def features(seed: int) -> mx.array:
    return mx.random.uniform(shape=(4, 8), key=mx.random.key(seed))


class TestHitIsExactlyWhatWasStored:
    def test_a_hit_returns_the_stored_array_bit_exact(self):
        cache = VisionFeatureCache()
        stored = features(0)
        cache.put("/tmp/a.png", stored)
        hit = cache.get("/tmp/a.png")
        assert hit is stored
        assert mx.array_equal(hit, stored).item()

    def test_distinct_paths_never_alias(self):
        cache = VisionFeatureCache()
        cache.put("/tmp/a.png", features(0))
        cache.put("/tmp/b.png", features(1))
        assert not mx.array_equal(
            cache.get("/tmp/a.png"), cache.get("/tmp/b.png")
        ).item()

    def test_a_replaced_entry_serves_the_new_features(self):
        cache = VisionFeatureCache()
        cache.put("/tmp/a.png", features(0))
        replacement = features(1)
        cache.put("/tmp/a.png", replacement)
        assert cache.get("/tmp/a.png") is replacement
        assert len(cache) == 1


class TestKeying:
    def test_pil_images_key_on_content(self):
        cache = VisionFeatureCache()
        same_a = Image.new("RGB", (8, 8), (10, 20, 30))
        same_b = Image.new("RGB", (8, 8), (10, 20, 30))
        differs = Image.new("RGB", (8, 8), (10, 20, 31))
        cache.put(same_a, features(0))
        assert cache.get(same_b) is not None, "same pixels must hit"
        assert cache.get(differs) is None, "a one-value pixel change must miss"

    def test_a_composite_key_is_order_sensitive(self):
        cache = VisionFeatureCache()
        cache.put(["/tmp/a.png", "/tmp/b.png"], features(0))
        assert cache.get(["/tmp/b.png", "/tmp/a.png"]) is None
        assert cache.get(["/tmp/a.png"]) is None
        assert cache.get(["/tmp/a.png", "/tmp/b.png"]) is not None

    def test_a_miss_is_none_not_an_empty_array(self):
        assert VisionFeatureCache().get("/tmp/never.png") is None


class TestLru:
    def test_the_oldest_entry_is_evicted_at_capacity(self):
        cache = VisionFeatureCache(max_size=2)
        cache.put("/tmp/a.png", features(0))
        cache.put("/tmp/b.png", features(1))
        cache.put("/tmp/c.png", features(2))
        assert cache.get("/tmp/a.png") is None
        assert cache.get("/tmp/b.png") is not None
        assert cache.get("/tmp/c.png") is not None

    def test_a_hit_refreshes_recency(self):
        cache = VisionFeatureCache(max_size=2)
        cache.put("/tmp/a.png", features(0))
        cache.put("/tmp/b.png", features(1))
        cache.get("/tmp/a.png")
        cache.put("/tmp/c.png", features(2))
        assert cache.get("/tmp/a.png") is not None
        assert cache.get("/tmp/b.png") is None

    def test_clear_empties(self):
        cache = VisionFeatureCache()
        cache.put("/tmp/a.png", features(0))
        cache.clear()
        assert len(cache) == 0
        assert cache.get("/tmp/a.png") is None
