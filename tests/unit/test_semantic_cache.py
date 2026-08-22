"""
Unit tests for SemanticQueryCache static/pure methods
(src/common/cache/semantic_cache.py).

All Redis I/O is mocked — no real Redis needed.
Covers:
- _normalize: lowercase, whitespace collapse, punctuation strip
- _hash: consistent SHA-256 output
- _cosine: correct values, zero-vector handling, dimension mismatch
- _pack_vector / _unpack_vector: round-trip
- CacheEntry dataclass fields
- lookup() — exact hit, cosine hit, miss, cache miss paths
- store() — pipeline calls
- stats() — entry count
- aclose() — delegates to redis.aclose()
"""

import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from medici.common.cache.semantic_cache import CacheEntry, SemanticQueryCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cache(
    threshold: float = 0.9, ttl: int = 300, max_entries: int = 100
) -> SemanticQueryCache:
    cache = SemanticQueryCache.__new__(SemanticQueryCache)
    cache._redis = MagicMock()
    cache._embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    cache.threshold = threshold
    cache.ttl = ttl
    cache.max_entries = max_entries
    return cache


def _pack(floats):
    return struct.pack(f"<{len(floats)}f", *floats)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_lowercases_input(self):
        assert SemanticQueryCache._normalize("Hello World") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        assert SemanticQueryCache._normalize("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        result = SemanticQueryCache._normalize("hello    world")
        assert result == "hello world"

    def test_removes_punctuation(self):
        result = SemanticQueryCache._normalize("What is AI?")
        assert "?" not in result
        assert "what is ai" == result

    def test_removes_multiple_punctuation_marks(self):
        result = SemanticQueryCache._normalize("Hello, World! How are you?")
        assert "," not in result
        assert "!" not in result

    def test_empty_string_stays_empty(self):
        assert SemanticQueryCache._normalize("") == ""


# ---------------------------------------------------------------------------
# _hash
# ---------------------------------------------------------------------------


class TestHash:
    def test_same_input_produces_same_hash(self):
        h1 = SemanticQueryCache._hash("hello world")
        h2 = SemanticQueryCache._hash("hello world")
        assert h1 == h2

    def test_different_input_different_hash(self):
        h1 = SemanticQueryCache._hash("hello")
        h2 = SemanticQueryCache._hash("world")
        assert h1 != h2

    def test_hash_is_64_char_hex_string(self):
        h = SemanticQueryCache._hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical_vectors_give_1(self):
        v = [1.0, 2.0, 3.0]
        score = SemanticQueryCache._cosine(v, v)
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_vectors_give_0(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert SemanticQueryCache._cosine(a, b) == 0.0

    def test_opposite_vectors_give_negative_1(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        score = SemanticQueryCache._cosine(a, b)
        assert abs(score - (-1.0)) < 1e-6

    def test_zero_vector_returns_0(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert SemanticQueryCache._cosine(a, b) == 0.0

    def test_dimension_mismatch_returns_0(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert SemanticQueryCache._cosine(a, b) == 0.0

    def test_realistic_similar_vectors(self):
        a = [0.9, 0.1, 0.0]
        b = [0.8, 0.2, 0.0]
        score = SemanticQueryCache._cosine(a, b)
        assert score > 0.99  # should be very similar


# ---------------------------------------------------------------------------
# _pack_vector / _unpack_vector
# ---------------------------------------------------------------------------


class TestVectorSerialization:
    def test_pack_unpack_roundtrip(self):
        original = [0.1, 0.5, -0.3, 1.0]
        packed = SemanticQueryCache._pack_vector(original)
        unpacked = SemanticQueryCache._unpack_vector(packed)
        for a, b in zip(original, unpacked, strict=False):
            assert abs(a - b) < 1e-6

    def test_pack_returns_bytes(self):
        packed = SemanticQueryCache._pack_vector([1.0, 2.0])
        assert isinstance(packed, bytes)

    def test_pack_length_is_4_bytes_per_float(self):
        v = [1.0, 2.0, 3.0]
        packed = SemanticQueryCache._pack_vector(v)
        assert len(packed) == 4 * len(v)

    def test_empty_vector_roundtrip(self):
        packed = SemanticQueryCache._pack_vector([])
        unpacked = SemanticQueryCache._unpack_vector(packed)
        assert unpacked == []


# ---------------------------------------------------------------------------
# CacheEntry dataclass
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_cache_entry_fields(self):
        entry = CacheEntry(
            answer="test answer",
            sources=["doc1"],
            token_usage={"total": 100},
            similarity=0.95,
            cache_key="abc123",
        )
        assert entry.answer == "test answer"
        assert entry.similarity == 0.95
        assert entry.cache_key == "abc123"

    def test_cache_entry_is_frozen(self):
        entry = CacheEntry(answer="a", sources=[], token_usage={}, similarity=1.0, cache_key="k")
        with pytest.raises((AttributeError, TypeError)):
            entry.answer = "modified"  # type: ignore


# ---------------------------------------------------------------------------
# lookup() and store() — mocked Redis
# ---------------------------------------------------------------------------


class TestLookup:
    async def test_exact_hit_returns_entry(self):
        cache = _make_cache()
        # _fetch_entry will be called for exact key
        answer_bytes = b"Cached answer"
        fake_data = {
            b"answer": answer_bytes,
            b"sources": b'["doc1"]',
            b"token_usage": b"{}",
        }
        cache._redis.hgetall = AsyncMock(return_value=fake_data)
        cache._redis.zrange = AsyncMock(return_value=[])

        result = await cache.lookup("Hello world?")
        assert result is not None
        assert result.answer == "Cached answer"
        assert result.similarity == 1.0

    async def test_miss_returns_none_when_no_candidates(self):
        cache = _make_cache()
        # No exact match and no candidates in ZSET
        cache._redis.hgetall = AsyncMock(return_value={})  # empty → no entry
        cache._redis.zrange = AsyncMock(return_value=[])

        result = await cache.lookup("No match query")
        assert result is None

    async def test_cosine_hit_returns_entry_above_threshold(self):
        cache = _make_cache(threshold=0.5)

        v = [1.0, 0.0, 0.0]
        cache._embed = AsyncMock(return_value=v)

        # No exact hit
        cache._redis.hgetall = AsyncMock(
            side_effect=[
                {},  # first call: exact hit check → miss
                {  # second call: cosine candidate fetch
                    b"answer": b"Similar answer",
                    b"sources": b"[]",
                    b"token_usage": b"{}",
                },
            ]
        )
        # Return one candidate key
        candidate_key = SemanticQueryCache._hash("hello world")
        cache._redis.zrange = AsyncMock(return_value=[candidate_key.encode()])
        packed_same_vec = SemanticQueryCache._pack_vector(v)
        cache._redis.hget = AsyncMock(return_value=packed_same_vec)

        result = await cache.lookup("hello world?")
        assert result is not None
        assert result.answer == "Similar answer"


class TestStore:
    async def test_store_calls_pipeline_execute(self):
        cache = _make_cache()
        mock_pipe = MagicMock()
        mock_pipe.hset = AsyncMock()
        mock_pipe.expire = AsyncMock()
        mock_pipe.zadd = AsyncMock()
        mock_pipe.zremrangebyrank = AsyncMock()
        mock_pipe.execute = AsyncMock()
        cache._redis.pipeline = MagicMock(return_value=mock_pipe)

        await cache.store("My question", "My answer", ["doc1"], {"calls": 1})
        mock_pipe.execute.assert_called_once()

    async def test_store_uses_normalized_query_for_key(self):
        cache = _make_cache()
        mock_pipe = MagicMock()
        mock_pipe.hset = AsyncMock()
        mock_pipe.expire = AsyncMock()
        mock_pipe.zadd = AsyncMock()
        mock_pipe.zremrangebyrank = AsyncMock()
        mock_pipe.execute = AsyncMock()
        cache._redis.pipeline = MagicMock(return_value=mock_pipe)

        await cache.store("HELLO WORLD?", "Answer", [])
        # embed should have been called with normalized form
        call_args = cache._embed.call_args[0][0]
        assert call_args == "hello world"


class TestStats:
    async def test_stats_returns_dict_with_entries(self):
        cache = _make_cache(max_entries=50)
        cache._redis.zcard = AsyncMock(return_value=10)

        result = await cache.stats()
        assert result["entries"] == 10
        assert result["max_entries"] == 50

    async def test_aclose_calls_redis_aclose(self):
        cache = _make_cache()
        cache._redis.aclose = AsyncMock()
        await cache.aclose()
        cache._redis.aclose.assert_called_once()
