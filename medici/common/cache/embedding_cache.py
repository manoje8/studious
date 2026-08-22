"""
Embedding vector cache

Stores (text_hash -> vector) in a PostgreSQL table. Uses an asyncpg
connection pool for concurrent reads and writes; a single asyncio.Lock
serializes to write + LRU-eviction step so eviction stays consistent
under concurrent writers within this process. (Across multiple processes,
eviction races are possible; see note on advisory locks below if that
matters for your deployment.)

Cache key: SHA-256 of the stripped chunk text.
Eviction:  LRU -- oldest ``accessed_at`` rows are pruned when ``max_entries``
           is exceeded.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta

import asyncpg
import logfire
import numpy as np

_ADVISORY_LOCK_KEY = 0x656D625F65766963
SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash    CHAR(64) PRIMARY KEY,
    embedding    BYTEA NOT NULL,
    dim          INTEGER NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    accessed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_accessed_at
    ON embedding_cache (accessed_at);
"""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _unpack(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32)


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class EmbeddingCache:
    """Postgres-backed embedding vector cache with LRU eviction.

    Parameters
    ----------
    dsn:
        PostgreSQL connection string, e.g.
        ``postgresql://user:pass@host:5432/dbname``.
    max_entries:
        Maximum number of cached vectors before LRU eviction kicks in.
    pool_min_size / pool_max_size:
        Bounds for the underlying asyncpg connection pool.

    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        max_entries: int = 50_000,
        access_update_interval: timedelta = timedelta(minutes=5),
    ) -> None:
        self._pool = pool
        self._max_entries = max_entries
        self._access_update_interval = access_update_interval
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._evict_task: asyncio.Task | None = None
        self._touch_task: set[asyncio.Task] = set()
        self.dsn = None

    @classmethod
    async def create(
        cls, dsn: str, *, min_size: int = 1, max_size: int = 10, **kwargs
    ) -> EmbeddingCache:
        pool = await asyncpg.create_pool(dsn=dsn, min_size=min_size, max_size=max_size)
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA)
        return cls(pool, **kwargs)

    async def close(self) -> None:
        if self._evict_task and not self._evict_task.done():
            await self._evict_task

        for t in list(self._touch_task):
            if not t.done():
                await t

        await self._pool.close()

    async def get(self, text: str) -> np.ndarray | None:
        result = await self.get_many([text])
        return result.get(_hash_text(text))

    async def get_many(self, texts: Iterable[str]) -> dict[str, np.ndarray]:
        hashes = {_hash_text(t): t for t in texts}

        if not hashes:
            return {}

        rows = await self._pool.fetch(
            "SELECT text_hash, embedding FROM embedding_cache WHERE text_hash = ANY($1::char(64)[])",
            list(hashes.keys()),
        )

        found = {row["text_hash"]: _unpack(row["embedding"]) for row in rows}

        self._hits += len(found)
        self._misses += len(hashes) - len(found)

        if found:
            self._touch_async(list(found.keys()))

        return found

    def _touch_async(self, text_hashes: list[str]) -> None:
        task = asyncio.create_task(self._touch(text_hashes))
        self._touch_task.add(task)
        task.add_done_callback(self._touch_task.discard)

    async def _touch(self, text_hashes: list[str]):
        try:
            await self._pool.execute(
                """
                UPDATE embedding_cache
                SET accessed_at = now()
                WHERE text_hash = ANY($1::char(64)[])
                  AND accessed_at < now() - $2::interval
                """,
                text_hashes,
                self._access_update_interval,
            )
        except asyncpg.PostgresError:
            logfire.warning("embedding_cache: accessed_at touch failed")

    async def set(self, text: str, embedding: np.ndarray) -> None:
        await self.set_many([(text, embedding)])

    async def set_many(self, items: Iterable[tuple[str, np.ndarray]]) -> None:
        rows = [
            (_hash_text(text), _pack(vec), int(np.asarray(vec).shape[0])) for text, vec in items
        ]

        if not rows:
            return

        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO embedding_cache (text_hash, embedding, dim, created_at, accessed_at)
                VALUES ($1, $2, $3, now(), now())
                ON CONFLICT (text_hash)
                DO UPDATE SET embedding = EXCLUDED.embedding,
                              dim = EXCLUDED.dim,
                              accessed_at = now()
                """,
                rows,
            )

        self._maybe_evict()

    def _maybe_evict(self) -> None:
        """Kick off an eviction pass if one isn't already running in this process."""

        if self._evict_task is None or self._evict_task.done():
            self._evict_task = asyncio.create_task(self._evict_if_needed())

    async def _evict_if_needed(self) -> None:
        try:
            async with self._pool.acquire() as conn:
                got_lock = await conn.fetchval(
                    "SELECT pg_try_advisory_lock($1)", _ADVISORY_LOCK_KEY
                )

                if not got_lock:
                    return

                try:
                    result = await conn.execute(
                        """
                        WITH victims AS (
                            SELECT text_hash FROM embedding_cache
                            ORDER BY accessed_at ASC
                            LIMIT GREATEST(0, (SELECT COUNT(*) FROM embedding_cache) - $1)
                        )
                        DELETE FROM embedding_cache
                        USING victims
                        WHERE embedding_cache.text_hash = victims.text_hash
                        """,
                        self._max_entries,
                    )
                    deleted = int(result.split()[-1]) if result else 0
                    if deleted:
                        self._evictions += deleted
                        logfire.info(f"embedding_cache: evicted {deleted} entries")
                finally:
                    await conn.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)
        except asyncpg.PostgresError:
            logfire.warning("embedding_cache: eviction failed")

    def stats(self) -> CacheStats:
        return CacheStats(hits=self._hits, misses=self._misses, evictions=self._evictions)
