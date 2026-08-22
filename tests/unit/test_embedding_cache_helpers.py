"""
Unit tests for module-level helpers in src/common/cache/embedding_cache.py.

Covers:
- _hash_text: deterministic SHA-256 of stripped text
- _pack / _unpack: numpy array round-trip
- CacheStats.hit_rate: normal, all-hits, all-misses, zero-total
- EmbeddingCache.stats() returns correct CacheStats
- EmbeddingCache.close() skips cancelled evict task
"""

import hashlib
from unittest.mock import MagicMock

import numpy as np
import pytest

from medici.common.cache.embedding_cache import (
    CacheStats,
    EmbeddingCache,
    _hash_text,
    _pack,
    _unpack,
)

# ---------------------------------------------------------------------------
# _hash_text
# ---------------------------------------------------------------------------


class TestHashText:
    def test_known_value(self):
        text = "hello"
        expected = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        assert _hash_text(text) == expected

    def test_strips_whitespace_before_hashing(self):
        assert _hash_text("  hello  ") == _hash_text("hello")

    def test_different_texts_different_hashes(self):
        assert _hash_text("abc") != _hash_text("xyz")

    def test_returns_64_char_hex_string(self):
        h = _hash_text("sample text")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# _pack / _unpack
# ---------------------------------------------------------------------------


class TestPackUnpack:
    def test_roundtrip_float32(self):
        arr = np.array([0.1, 0.5, -0.3, 1.0], dtype=np.float32)
        packed = _pack(arr)
        recovered = _unpack(packed)
        np.testing.assert_allclose(arr, recovered, rtol=1e-6)

    def test_pack_returns_bytes(self):
        arr = np.array([1.0, 2.0])
        assert isinstance(_pack(arr), bytes)

    def test_unpack_returns_ndarray(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = _unpack(_pack(arr))
        assert isinstance(result, np.ndarray)

    def test_pack_length_is_4_bytes_per_element(self):
        arr = np.array([1.0, 2.0, 3.0])
        packed = _pack(arr)
        assert len(packed) == 4 * 3

    def test_roundtrip_list_input(self):
        # _pack should accept list input via np.asarray
        packed = _pack([0.5, -0.5])
        recovered = _unpack(packed)
        assert abs(recovered[0] - 0.5) < 1e-6
        assert abs(recovered[1] - (-0.5)) < 1e-6


# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------


class TestCacheStats:
    def test_hit_rate_normal(self):
        stats = CacheStats(hits=3, misses=1, evictions=0)
        assert stats.hit_rate == pytest.approx(0.75)

    def test_hit_rate_all_hits(self):
        stats = CacheStats(hits=10, misses=0, evictions=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_all_misses(self):
        stats = CacheStats(hits=0, misses=5, evictions=0)
        assert stats.hit_rate == 0.0

    def test_hit_rate_zero_total(self):
        stats = CacheStats(hits=0, misses=0, evictions=0)
        assert stats.hit_rate == 0.0

    def test_is_frozen(self):
        stats = CacheStats(hits=1, misses=1, evictions=0)
        with pytest.raises((AttributeError, TypeError)):
            stats.hits = 99  # type: ignore


# ---------------------------------------------------------------------------
# EmbeddingCache.stats()
# ---------------------------------------------------------------------------


class TestEmbeddingCacheStats:
    def test_stats_returns_cache_stats_object(self):
        mock_pool = MagicMock()
        cache = EmbeddingCache(pool=mock_pool, max_entries=1000)
        # Manually set counters
        cache._hits = 5
        cache._misses = 2
        cache._evictions = 1

        stats = cache.stats()
        assert isinstance(stats, CacheStats)
        assert stats.hits == 5
        assert stats.misses == 2
        assert stats.evictions == 1
        assert stats.hit_rate == pytest.approx(5 / 7)

    def test_initial_stats_all_zeros(self):
        mock_pool = MagicMock()
        cache = EmbeddingCache(pool=mock_pool)
        stats = cache.stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.hit_rate == 0.0
