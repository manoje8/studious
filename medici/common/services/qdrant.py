import logging
import uuid
from functools import partial

import logfire
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    Document,
    FieldCondition,
    Filter,
    FilterSelector,
    Fusion,
    FusionQuery,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVectorParams,
    VectorParams,
)
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from medici.common.models import EmbeddedChunk
from medici.common.utils.config import config

_logger = logging.getLogger(__name__)


def _is_qdrant_transient(exc: BaseException) -> bool:
    """
    Return True only for errors that are worth retrying.

    - ResponseHandlingException: transport-level failures (connection refused,
      read timeout, DNS resolution error).
    - UnexpectedResponse with status_code >= 500: server-side fault (5xx)
      that may resolve on the next attempt.
    - OSError / subclasses: low-level network resets (ConnectionResetError,
      BrokenPipeError, etc.) that are inherently transient.

    4xx responses (bad auth, malformed filter, model not found) are NOT retried
    because they represent a client-side bug that will never self-heal.
    """
    if isinstance(exc, ResponseHandlingException):
        return True
    if isinstance(exc, UnexpectedResponse):
        return exc.status_code is not None and exc.status_code >= 500
    if isinstance(exc, OSError):
        return True
    return False


_QDRANT_RETRY_POLICY = dict(
    retry=retry_if_exception(_is_qdrant_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
    before_sleep=before_sleep_log(_logger, logging.WARNING),
    reraise=True,
)


async def _with_retry(coro):
    """Execute an awaitable under the standard Qdrant retry policy."""
    async for attempt in AsyncRetrying(**_QDRANT_RETRY_POLICY):
        with attempt:
            return await coro()


class QdrantStorageService:
    def __init__(
        self,
        url: str,
        collection_name: str = config.QDRANT_COLLECTION_NAME,
        vector_size: int = 1536,
        upsert_batch_size: int = 100,
    ):
        self.client = AsyncQdrantClient(
            url=url or config.QDRANT_CLUSTER_ENDPOINT,
            api_key=config.QDRANT_API_KEY,
            timeout=30,
            check_compatibility=False,
        )
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.upsert_batch_size = upsert_batch_size

    async def ping(self) -> bool:
        """Return True if Qdrant is reachable, False otherwise.

        Uses the same retry policy as all other operations so a single
        transient blip does not immediately report unhealthy.
        """
        try:
            await _with_retry(partial(self.client.get_collections))
            return True
        except Exception as exc:
            logfire.warning("Qdrant health-check failed", error=str(exc))
            return False

    async def validate_vector_dimension(self) -> None:
        info = await _with_retry(partial(self.client.get_collection, self.collection_name))
        actual = info.config.params.vectors["dense"].size
        if actual != self.vector_size:
            raise ValueError(
                f"Vector dimension mismatch for collection '{self.collection_name}': "
                f"expected {self.vector_size}, got {actual}. "
                f"Either update vector_size to {actual} or recreate the collection."
            )

    async def _ensure_doc_id_index(self) -> None:
        """
        Create a keyword payload index on 'doc_id' if it does not already exist.

        Qdrant requires an index on any field used in a filter for delete
        operations. This call is idempotent — creating an index that already
        exists is a no-op on the server side.
        """
        await _with_retry(
            partial(
                self.client.create_payload_index,
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        )
        logfire.info("Ensured keyword payload index on 'doc_id'")

    async def ensure_collection_exists(self) -> None:
        exists = await _with_retry(partial(self.client.collection_exists, self.collection_name))

        if not exists:
            await _with_retry(
                partial(
                    self.client.create_collection,
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=self.vector_size, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
                )
            )
            logfire.info(f"Created Qdrant collection: {self.collection_name}")
        else:
            await self.validate_vector_dimension()
            logfire.info(f"Collection already exists: {self.collection_name}")

        await self._ensure_doc_id_index()

    async def upsert_embedded_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Store embedded chunks in Qdrant in batches.

        Each batch is independently retried so a transient error mid-way
        through a large upload does not force a full re-ingestion.
        """

        await self.ensure_collection_exists()

        resolved_doc_id = embedded_chunks[0].chunk.doc_id if embedded_chunks else None

        if resolved_doc_id is not None:
            await self._delete_doc_chunks(resolved_doc_id)
        else:
            logfire.warning(
                "upsert_embedded_chunks called with no doc_id and no chunks; skipping stale-chunk cleanup"
            )

        total = len(embedded_chunks)
        total_batches = (total + self.upsert_batch_size - 1) // self.upsert_batch_size

        for batch_num in range(total_batches):
            start = batch_num * self.upsert_batch_size
            end = start + self.upsert_batch_size

            batch = embedded_chunks[start:end]

            points = [
                PointStruct(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            f"{ec.chunk.doc_id}_{ec.chunk.chunk_index}",
                        )
                    ),
                    vector={
                        "dense": ec.vector,
                        "sparse": Document(text=ec.chunk.text, model=config.QDRANT_SPARSE_MODEL),
                    },
                    payload={
                        **ec.chunk.to_quant_payload(),
                        "embedding_model": ec.model_name,
                    },
                )
                for ec in batch
            ]

            try:
                await _with_retry(
                    partial(self.client.upsert, collection_name=self.collection_name, points=points)
                )
            except Exception as e:
                logfire.error(
                    f"Batch {batch_num + 1}/{total_batches} failed after all retries",
                    error=str(e),
                    start_idx=start,
                    end_idx=end,
                )
                raise

            logfire.info(f"Upserted batch {batch_num + 1}/{total_batches}({len(points)} points)")

        logfire.info(f"Storage complete: {total} vectors in Qdrant")

    async def search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 5,
        doc_id_filter: str | None = None,
    ) -> list[dict]:
        search_filter = None

        if doc_id_filter:
            search_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))]
            )

        result = await _with_retry(
            partial(
                self.client.query_points,
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=Document(text=query, model=config.QDRANT_SPARSE_MODEL),
                        using="sparse",
                        limit=top_k,
                    ),
                    Prefetch(
                        query=query_vector,
                        using="dense",
                        limit=top_k,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                query_filter=search_filter,
                limit=top_k,
                with_payload=True,
            )
        )

        return [
            {
                "text": r.payload.get("text", ""),
                "score": float(r.score) if r.score is not None else None,
                "section": r.payload.get("section_title", ""),
                "source": r.payload.get("source_file", ""),
                "doc_id": r.payload.get("doc_id", ""),
                "chunk_index": r.payload.get("chunk_index"),
                "content_type": r.payload.get("content_type", "text"),
                "image_path": r.payload.get("image_path", ""),
                "page_numbers": r.payload.get("page_numbers", []),
            }
            for r in result.points
        ]

    async def _delete_doc_chunks(self, doc_id: str) -> None:
        """
        Delete all existing points for doc_id ahead of a re-upsert.
        """
        try:
            await _with_retry(
                partial(
                    self.client.delete,
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="doc_id",
                                    match=MatchValue(value=doc_id),
                                )
                            ]
                        )
                    ),
                )
            )
            logfire.info(f"Deleted existing chunks for document {doc_id}")
        except Exception as e:
            logfire.error(
                f"Failed to delete chunks for document {doc_id}",
                error=str(e),
            )
            raise

        logfire.info(f"Deleted existing chunks for doc_id={doc_id} prior to upsert")

    async def chunk_count(self) -> int:
        current_count = await _with_retry(
            partial(self.client.count, collection_name=self.collection_name)
        )
        return current_count.count

    async def scroll_all_chunks(self) -> list[dict]:
        """Scroll through all chunks in Qdrant and return them as a list of dicts."""

        all_chunks = []
        next_page_offset = None

        while True:
            result, next_page_offset = await _with_retry(
                partial(
                    self.client.scroll,
                    collection_name=self.collection_name,
                    scroll_filter=None,
                    limit=500,
                    with_payload=True,
                    with_vectors=False,
                    offset=next_page_offset,
                )
            )

            for point in result:
                all_chunks.append(
                    {
                        "text": point.payload.get("text", ""),
                        "doc_id": point.payload.get("doc_id", ""),
                        "chunk_index": point.payload.get("chunk_index"),
                        "section_title": point.payload.get("section_title", ""),
                        "source_file": point.payload.get("source_file", ""),
                        "content_type": point.payload.get("content_type", "text"),
                        "image_path": point.payload.get("image_path", ""),
                        "page_numbers": point.payload.get("page_numbers", []),
                    }
                )

            if next_page_offset is None:
                break

        logfire.info(f"Scrolled {len(all_chunks)} chunks from Qdrant")
        return all_chunks

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch chunk payloads from Qdrant by ``doc_id:chunk_index`` identifiers.

        Used by the KG retriever to resolve graph-traversed entity source
        chunks back to their full text content.

        Parameters
        ----------
        chunk_ids : list[str]
            Identifiers in ``"<doc_id>:<chunk_index>"`` format.

        Returns
        -------
        list[dict]
            Chunk payloads with text, metadata, and a ``score`` of 1.0
            (KG-sourced chunks have no vector similarity score).
        """
        if not chunk_ids:
            return []

        # Parse chunk_ids into (doc_id, chunk_index) pairs
        filters: list[tuple[str, int]] = []
        for cid in chunk_ids:
            parts = cid.rsplit(":", 1)
            if len(parts) == 2:
                try:
                    filters.append((parts[0], int(parts[1])))
                except ValueError:
                    logfire.warning("Invalid chunk_id format: {cid}", cid=cid)
            else:
                logfire.warning("Invalid chunk_id format: {cid}", cid=cid)

        if not filters:
            return []

        results: list[dict] = []
        for doc_id, chunk_index in filters:
            try:
                points, _ = await _with_retry(
                    partial(
                        self.client.scroll,
                        collection_name=self.collection_name,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                                FieldCondition(
                                    key="chunk_index",
                                    match=MatchValue(value=chunk_index),
                                ),
                            ]
                        ),
                        limit=1,
                        with_payload=True,
                        with_vectors=False,
                    )
                )
                for point in points:
                    results.append(
                        {
                            "text": point.payload.get("text", ""),
                            "doc_id": point.payload.get("doc_id", ""),
                            "chunk_index": point.payload.get("chunk_index"),
                            "section_title": point.payload.get("section_title", ""),
                            "source_file": point.payload.get("source_file", ""),
                            "source": point.payload.get("source_file", ""),
                            "section": point.payload.get("section_title", ""),
                            "content_type": point.payload.get("content_type", "text"),
                            "image_path": point.payload.get("image_path", ""),
                            "page_numbers": point.payload.get("page_numbers", []),
                            "parent_text": point.payload.get("parent_text", ""),
                            "score": 1.0,  # KG-sourced chunks have no similarity score
                        }
                    )
            except Exception as exc:
                logfire.warning(
                    "Failed to fetch chunk {doc_id}:{chunk_index}",
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    error=str(exc),
                )

        logfire.info(
            "get_chunks_by_ids",
            requested=len(chunk_ids),
            resolved=len(results),
        )
        return results
