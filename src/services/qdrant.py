import uuid

import logfire
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.ingestion.embedding import EmbeddedChunk
from src.utils.config import config


class QdrantStorageService:
    def __init__(
        self,
        url: str,
        collection_name: str = config.QDRANT_COLLECTION_NAME,
        vector_size: int = 1536,
        upsert_batch_size: int = 100,
    ):
        self.client = AsyncQdrantClient(
            url=url or config.QDRANT_CLUSTER_ENDPOINT, api_key=config.QDRANT_API_KEY
        )
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.upsert_batch_size = upsert_batch_size

    async def validate_vector_dimension(self) -> None:
        info = await self.client.get_collection(self.collection_name)
        actual = info.config.params.vectors.size
        if actual != self.vector_size:
            raise ValueError(
                f"Vector dimension mismatch for collection '{self.collection_name}': "
                f"expected {self.vector_size}, got {actual}. "
                f"Either update vector_size to {actual} or recreate the collection."
            )

    async def ensure_collection_exists(self) -> None:
        exists = await self.client.collection_exists(self.collection_name)

        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )
            logfire.info(f"Created Qdrant collection: {self.collection_name}")
        else:
            await self.validate_vector_dimension()
            logfire.info(f"Collection already exists: {self.collection_name}")

    async def upsert_embedded_chunks(
        self, embedded_chunks: list[EmbeddedChunk]
    ) -> None:
        """Store embedded chunks in Qdrant in batches."""

        await self.ensure_collection_exists()

        total = len(embedded_chunks)
        total_batches = (total + self.upsert_batch_size - 1) // self.upsert_batch_size

        for batch_num in range(total_batches):
            start = batch_num * self.upsert_batch_size
            end = start + self.upsert_batch_size

            batch = embedded_chunks[start:end]

            points = [
                ec.to_qdrant_point(
                    point_id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS,
                            f"{ec.chunk.doc_id}_{ec.chunk.chunk_index}",
                        )
                    )
                )
                for ec in batch
            ]

            try:
                await self.client.upsert(
                    collection_name=self.collection_name, points=points
                )
            except Exception as e:
                logfire.error(
                    f"Batch {batch_num + 1}/{total_batches} failed",
                    error=str(e),
                    start_idx=start,
                    end_idx=end,
                )
                raise

            logfire.info(
                f"Upserted batch {batch_num + 1}/{total_batches}"
                f"({len(points)} points)"
            )

        logfire.info(f"Storage complete: {total} vectors in Qdrant")

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        doc_id_filter: str | None = None,
    ) -> list[dict]:
        search_filter = None

        if doc_id_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))
                ]
            )

        result = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "text": r.payload.get("text", ""),
                "score": float(r.score) if r.score is not None else None,
                "section": r.payload.get("section_title", ""),
                "source": r.payload.get("source_file", ""),
                "doc_id": r.payload.get("doc_id", ""),
                "chunk_index": r.payload.get("chunk_index"),
            }
            for r in result.points
        ]

    async def chunk_count(self) -> int:
        current_count = await self.client.count(collection_name=self.collection_name)
        return current_count.count

    async def scroll_all_chunks(self) -> list[dict]:
        """Scroll through all chunks in Qdrant and return them as a list of dicts."""

        all_chunks = []
        next_page_offset = None

        while True:
            result, next_page_offset = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=500,
                with_payload=True,
                with_vectors=False,
                offset=next_page_offset,
            )

            for point in result:
                all_chunks.append(
                    {
                        "text": point.payload.get("text", ""),
                        "doc_id": point.payload.get("doc_id", ""),
                        "chunk_index": point.payload.get("chunk_index"),
                        "section_title": point.payload.get("section_title", ""),
                        "source_file": point.payload.get("source_file", ""),
                    }
                )

            if next_page_offset is None:
                break

        logfire.info(f"Scrolled {len(all_chunks)} chunks from Qdrant")
        return all_chunks
