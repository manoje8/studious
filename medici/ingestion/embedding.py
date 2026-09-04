import asyncio

import logfire
from google import genai

from medici.common.cache.embedding_cache import EmbeddingCache
from medici.common.utils.config import config
from medici.common.utils.tokenizer import TikTokenTokenizer
from medici.ingestion.chunking.chunk import Chunk

VERTEX_MAX_TEXTS_PER_REQUEST = 250
VERTEX_MAX_TOKENS_PER_REQUEST = 20_000

_tokenizer = TikTokenTokenizer()


def _is_token_limit_error(error: Exception) -> bool:
    message = str(error)
    return (
        "INVALID_ARGUMENT" in message and "token count" in message and "supports up to" in message
    )


def _estimate_tokens(text: str) -> int:
    """Return the exact tiktoken token count for *text* (minimum 1)."""
    return max(1, _tokenizer.count(text))


from medici.common.models import EmbeddedChunk  # noqa: F401 — re-export for backward compatibility


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "text-embedding-004",
        dimensions: int = 768,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache: EmbeddingCache | None = None,
        max_tokens_per_batch: int = 15_000,
    ):
        self.client = genai.Client(
            vertexai=True, project=config.PROJECT_ID, location=config.LOCATION
        )
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = min(batch_size, VERTEX_MAX_TEXTS_PER_REQUEST)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._cache = cache
        self.max_tokens_per_batch = min(max_tokens_per_batch, VERTEX_MAX_TOKENS_PER_REQUEST)

    @property
    def vector_size(self) -> int:
        return self.dimensions or config.VECTOR_SIZE

    async def embed_single(self, text: str) -> list[float]:
        """Embed one piece of text, returning a cached vector when available."""

        text = text.strip()

        if not text:
            raise ValueError("Cannot embed empty text")

        if self._cache is not None:
            cached = await self._cache.get(text)
            if cached is not None:
                return cached

        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.models.embed_content,
                    model=self.model_name,
                    contents=text,
                    config=genai.types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=self.dimensions,
                    ),
                )

                vector = response.embeddings[0].values
                if self._cache is not None:
                    await self._cache.set(text, vector)
                return vector
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logfire.error(
                        "Embedding failed after {max_retries} attempts",
                        max_retries=self.max_retries,
                        error=str(e),
                        text_length=len(text),
                    )
                    raise
                logfire.warning(
                    "Embedding attempt {attempt} failed, retrying...",
                    attempt=attempt + 1,
                    error=str(e),
                    text_length=len(text),
                )
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    def _make_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """
        Group chunks into request-sized batches, respecting both Vertex's
        item-count cap (self.batch_size) and an estimated token budget
        (self.max_tokens_per_batch). Order is preserved so callers can zip
        results back against the input list.
        """
        batches: list[list[Chunk]] = []
        current: list[Chunk] = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = _estimate_tokens(chunk.text)

            if current and (
                len(current) >= self.batch_size
                or current_tokens + chunk_tokens > self.max_tokens_per_batch
            ):
                batches.append(current)
                current = []
                current_tokens = 0

            current.append(chunk)
            current_tokens += chunk_tokens

        if current:
            batches.append(current)

        return batches

    async def _embed_batch(self, batch: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Embed one batch, splitting it in half and recursing whenever
        Vertex rejects it for exceeding the per-request token budget.

        Ordinary transient errors (timeouts, 5xxs, rate limits) still go
        through the existing retry-with-backoff loop. Only the
        deterministic token-limit error triggers a split — retrying an
        over-budget batch unchanged would just fail the same way again,
        so there's no point burning retries/backoff on it first.
        """
        for attempt in range(self.max_retries):
            try:
                batch_texts = [chunk.text for chunk in batch]
                response = await asyncio.to_thread(
                    self.client.models.embed_content,
                    model=self.model_name,
                    contents=batch_texts,
                    config=genai.types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=self.dimensions,
                    ),
                )

                results: list[EmbeddedChunk] = []
                for chunk, emb in zip(batch, response.embeddings, strict=False):
                    ec = EmbeddedChunk(chunk=chunk, vector=emb.values, model_name=self.model_name)
                    results.append(ec)
                    if self._cache is not None:
                        await self._cache.set(chunk.text, emb.values)
                return results

            except Exception as e:
                if _is_token_limit_error(e):
                    if len(batch) == 1:
                        preview = batch[0].text[:120].replace("\n", " ")
                        raise ValueError(
                            "A single chunk exceeds Vertex's per-request token "
                            f"budget on its own (chunk_index={batch[0].chunk_index}, "
                            f"{len(batch[0].text)} chars): {preview!r}... "
                            "This chunk needs to be split further upstream in "
                            "chunking — likely a large table/multimodal item "
                            "that isn't going through the size-limited text "
                            "splitter (chunk_multimodal_items doesn't take a "
                            "size/overlap config the way the text chunker does)."
                        ) from e

                    mid = len(batch) // 2
                    logfire.warning(
                        "Batch of {size} exceeded Vertex's token budget, "
                        "splitting into {left}/{right} and retrying",
                        size=len(batch),
                        left=mid,
                        right=len(batch) - mid,
                    )
                    left = await self._embed_batch(batch[:mid])
                    right = await self._embed_batch(batch[mid:])
                    return left + right

                if attempt == self.max_retries - 1:
                    logfire.error(
                        "Batch of {size} failed after {max_retries} attempts",
                        size=len(batch),
                        max_retries=self.max_retries,
                        error=str(e),
                    )
                    raise

                logfire.warning(
                    "Batch attempt {attempt} failed, retrying...",
                    attempt=attempt + 1,
                    error=str(e),
                )
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> tuple[list[EmbeddedChunk], list[dict]]:
        """
        Embed a list of chunks in batches, skipping API calls for cached chunks.

        Returns
        -------
        embedded : list[EmbeddedChunk]
            Successfully embedded chunks (order matches *chunks*).
        dead_letter : list[dict]
            Chunks that could not be embedded after all retries.  Each entry
            has the shape::

                {
                    "chunk": Chunk,
                    "chunk_index": int,      # original position in *chunks*
                    "error": str,
                }

            An empty list means every chunk was embedded successfully.
        """

        if not chunks:
            logfire.debug("No chunks to embed")
            return [], []

        result_map: dict[int, EmbeddedChunk] = {}
        uncached_indices: list[int] = []

        if self._cache is not None:
            for idx, chunk in enumerate(chunks):
                cached_vec = await self._cache.get(chunk.text)
                if cached_vec is not None:
                    result_map[idx] = EmbeddedChunk(
                        chunk=chunk, vector=cached_vec, model_name=self.model_name
                    )
                else:
                    uncached_indices.append(idx)
        else:
            uncached_indices = list(range(len(chunks)))

        cache_hits = len(result_map)
        if cache_hits:
            logfire.info(
                "Cache hit ratio: {hits}/{total} ({ratio:.1%})",
                hits=cache_hits,
                total=len(chunks),
                ratio=cache_hits / len(chunks),
            )

        uncached_chunks = [chunks[i] for i in uncached_indices]
        dead_letter: list[dict] = []

        if uncached_chunks:
            batches = self._make_batches(uncached_chunks)

            logfire.debug(
                "Embedding {total_chunks} chunks in {total_batches} batches "
                "(<= {batch_size} items / ~{max_tokens} est. tokens each)",
                total_chunks=len(uncached_chunks),
                total_batches=len(batches),
                batch_size=self.batch_size,
                max_tokens=self.max_tokens_per_batch,
            )

            uncached_cursor = 0
            for batch_num, batch in enumerate(batches):
                logfire.debug(
                    "Embedding batch {batch_num}/{total_batches} ({batch_size} chunks)",
                    batch_num=batch_num + 1,
                    total_batches=len(batches),
                    batch_size=len(batch),
                )
                try:
                    batch_results = await self._embed_batch(batch)
                    for ec in batch_results:
                        orig_idx = uncached_indices[uncached_cursor]
                        result_map[orig_idx] = ec
                        uncached_cursor += 1
                except Exception as e:
                    logfire.error(
                        "Dead-letter: batch {batch_num}/{total_batches} failed permanently "
                        "({batch_size} chunks skipped). error={error}",
                        batch_num=batch_num + 1,
                        total_batches=len(batches),
                        batch_size=len(batch),
                        error=str(e),
                    )
                    for chunk in batch:
                        orig_idx = uncached_indices[uncached_cursor]
                        dead_letter.append(
                            {
                                "chunk": chunk,
                                "chunk_index": orig_idx,
                                "error": str(e),
                            }
                        )
                        uncached_cursor += 1

        if dead_letter:
            logfire.warn(
                "Dead-letter queue: {count}/{total} chunks could not be embedded and were skipped.",
                count=len(dead_letter),
                total=len(chunks),
            )

        embedded = [result_map[i] for i in range(len(chunks)) if i in result_map]
        logfire.info(
            "Embedding complete: {total} vectors ({dead} dead-lettered), {unique} unique from cache",
            total=len(embedded),
            dead=len(dead_letter),
            unique=len(embedded) - cache_hits,
        )
        return embedded, dead_letter
