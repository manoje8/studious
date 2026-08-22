"""
Redis-backed semantic query cache

Design
------
Two-phase lookup:

1. **Exact match** (O(1)) — SHA-256 of the normalized query string is tried first.
   A repeated identical query returns instantly without a cosine scan.

2. **Cosine similarity scan** (O(N)) — iterates over the most-recent
   ``max_entries`` keys from a Redis ZSET (``sqc:index``, scored by epoch) and
   computes cosine similarity against the stored embedding.  For N ≤ 500 this
   is pure-Python arithmetic that completes in < 5 ms on any server.

Cache entries
-------------
Each cached query is stored as a Redis hash:
    sqc:<sha256> → {embedding_bytes, answer, sources_json, stored_at, token_usage_json}

The ZSET ``sqc:index`` tracks all hashed keys sorted by insertion timestamp so
the scan always operates on the most-recent N entries, and so LRU eviction is
a single ZREMRANGEBYRANK call.

TTL
---
Both the hash and the ZSET score use the configurable TTL.  Expired entries are
lazily evicted by Redis on next access.

Thread safety
-------------
All Redis operations are async (aioredis-compatible via redis-py ≥ 4.2).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import time
from dataclasses import dataclass
from typing import Any

import logfire
import redis.asyncio as aioredis

from medici.common.utils.config import config

_INDEX_KEY = "sqc:index"
_ENTRY_PREFIX = "sqc:"


@dataclass(frozen=True)
class CacheEntry:
    answer: str
    sources: list[str]
    token_usage: dict[str, Any]
    similarity: float  # 1.0 for exact match
    cache_key: str


class SemanticQueryCache:
    """Redis-backed semantic cache for RAG query answers.

    Parameters
    ----------
    redis_url:
        Redis connection string (same as short-term memory).
    embedding_fn:
        Async callable ``(text: str) -> list[float]``.
    similarity_threshold:
        Cosine similarity score above which a cache hit is declared.
    ttl_seconds:
        Redis key TTL.  Expired entries are evicted by Redis automatically.
    max_entries:
        Maximum number of entries kept in the ZSET index.  Older entries are
        pruned when this limit is exceeded.
    """

    def __init__(
        self,
        redis_url: str,
        embedding_fn,
        similarity_threshold: float = config.SEMANTIC_CACHE_THRESHOLD,
        ttl_seconds: int = config.SEMANTIC_CACHE_TTL_SECONDS,
        max_entries: int = config.SEMANTIC_CACHE_MAX_ENTRIES,
    ) -> None:
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, encoding="utf-8", decode_responses=False
        )
        self._embed = embedding_fn
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries

    async def lookup(self, query: str) -> CacheEntry | None:
        """Return a cached result for ``query`` or ``None`` on miss."""

        normalised = self._normalize(query)
        exact_key = self._hash(normalised)

        result = await self._fetch_entry(exact_key, similarity=1.0)
        if result is not None:
            logfire.info(
                "SemanticCache EXACT HIT",
                key_prefix=exact_key[:8],
                query_preview=query[:60],
            )
            return result

        #  Cosine scan over recent entries
        query_embedding = await self._embed(normalised)
        candidate_keys = await self._recent_keys()

        best_score = 0.0
        best_key: str | None = None

        for raw_key in candidate_keys:
            if isinstance(raw_key, bytes):
                raw_key = raw_key.decode()
            stored_emb = await self._load_embedding(raw_key)
            if stored_emb is None:
                continue
            score = self._cosine(query_embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_key = raw_key

        if best_key is not None and best_score >= self.threshold:
            result = await self._fetch_entry(best_key, similarity=best_score)
            if result is not None:
                logfire.info(
                    "SemanticCache SIMILARITY HIT",
                    similarity=round(best_score, 4),
                    key_prefix=best_key[:8],
                    query_preview=query[:60],
                )
                return result

        logfire.debug("SemanticCache MISS", query_preview=query[:60])
        return None

    async def store(
        self,
        query: str,
        answer: str,
        sources: list[str],
        token_usage: dict | None = None,
    ) -> None:
        """Store a query/answer pair in the cache."""

        normalised = self._normalize(query)
        key = self._hash(normalised)
        entry_redis_key = f"{_ENTRY_PREFIX}{key}"

        embedding = await self._embed(normalised)
        emb_bytes = self._pack_vector(embedding)
        now = time.time()

        payload = {
            b"embedding": emb_bytes,
            b"answer": answer.encode(),
            b"sources": json.dumps(sources).encode(),
            b"stored_at": str(now).encode(),
            b"token_usage": json.dumps(token_usage or {}).encode(),
        }

        pipe = self._redis.pipeline()
        await pipe.hset(entry_redis_key, mapping=payload)
        await pipe.expire(entry_redis_key, self.ttl)
        # Add to the time-sorted index so the cosine scan finds it
        await pipe.zadd(_INDEX_KEY, {key.encode(): now})
        # Evict oldest entries beyond max_entries
        await pipe.zremrangebyrank(_INDEX_KEY, 0, -(self.max_entries + 1))
        await pipe.execute()

        logfire.info(
            "SemanticCache STORE",
            key_prefix=key[:8],
            answer_length=len(answer),
            query_preview=query[:60],
        )

    async def stats(self) -> dict:
        """Return cache statistics."""
        total = await self._redis.zcard(_INDEX_KEY)
        return {
            "entries": total,
            "max_entries": self.max_entries,
            "threshold": self.threshold,
            "ttl_seconds": self.ttl,
        }

    async def aclose(self) -> None:
        await self._redis.aclose()

    @staticmethod
    def _normalize(query: str) -> str:
        """Lowercase + collapse whitespace + strip punctuation for stable hashing."""
        text = query.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    @staticmethod
    def _pack_vector(v: list[float]) -> bytes:
        """Serialize a float list as little-endian 32-bit floats for compact storage."""
        return struct.pack(f"<{len(v)}f", *v)

    @staticmethod
    def _unpack_vector(data: bytes) -> list[float]:
        n = len(data) // 4
        return list(struct.unpack(f"<{n}f", data))

    async def _recent_keys(self) -> list[str]:
        """Return the most-recent ``max_entries`` hash-keys from the ZSET index."""
        raw = await self._redis.zrange(
            _INDEX_KEY,
            -self.max_entries,  # start (from oldest within window)
            -1,  # stop  (newest)
        )
        return [r.decode() if isinstance(r, bytes) else r for r in raw]

    async def _load_embedding(self, key: str) -> list[float] | None:
        raw = await self._redis.hget(f"{_ENTRY_PREFIX}{key}", "embedding")
        if raw is None:
            return None
        return self._unpack_vector(raw)

    async def _fetch_entry(self, key: str, similarity: float) -> CacheEntry | None:
        data = await self._redis.hgetall(f"{_ENTRY_PREFIX}{key}")
        if not data:
            return None
        answer = data.get(b"answer", b"").decode()
        sources = json.loads(data.get(b"sources", b"[]").decode())
        token_usage = json.loads(data.get(b"token_usage", b"{}").decode())
        return CacheEntry(
            answer=answer,
            sources=sources,
            token_usage=token_usage,
            similarity=similarity,
            cache_key=key,
        )
