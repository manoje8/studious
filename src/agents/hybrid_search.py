import asyncio
import logfire

from src.ingestion.embedding import EmbeddingService
from src.services.qdrant import QdrantStorageService
from src.services.sparse_index import SparseSearchIndex


class HybridSearch:
    def __init__(
        self,
        storage_service: QdrantStorageService,
        embedding_service: EmbeddingService,
        sparse_index: SparseSearchIndex,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
    ):
        self.storage_service = storage_service
        self.embedding_service = embedding_service
        self.sparse_index = sparse_index
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k

    def _reciprocal_rank_fusion(self, result_lists: list, k: int = 10):
        """Merge multiple ranked result lists using a compound doc_id:chunk_index key."""

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for result_list in result_lists:
            for rank, chunk in enumerate(result_list):
                chunk_id = f"{chunk.get('doc_id', '')}:{chunk['chunk_index']}"

                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (
                    1 / (k + rank + 1)
                )

                chunk_map[chunk_id] = chunk

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        merged = []

        for chunk_id in sorted_ids:
            chunk = dict(chunk_map[chunk_id])
            chunk["rrf_score"] = rrf_scores[chunk_id]
            merged.append(chunk)

        return merged

    async def _search_one(self, query: str, doc_id_filter: str | None = None):
        query_vector = await self.embedding_service.embed_single(query)
        dense_result = await self.storage_service.search(
            query_vector=query_vector,
            top_k=self.dense_top_k,
            doc_id_filter=doc_id_filter,
        )
        sparse_result = self.sparse_index.search(query, top_k=self.sparse_top_k)

        return dense_result, sparse_result

    async def search(
        self, queries: list[str], doc_id_filter: str | None = None, timeout: float = 8.0
    ) -> list[dict]:
        if not isinstance(queries, list) or not all(
            isinstance(q, str) for q in queries
        ):
            msg = f"queries must be list[str], got {type(queries)!r}"
            logfire.error(msg)
            raise ValueError(msg)

        if len(self.sparse_index.chunks) <= self.dense_top_k:
            queries = queries[:1]

        async def _run():
            search_one = [self._search_one(query, doc_id_filter) for query in queries]
            results = await asyncio.gather(*search_one, return_exceptions=True)
            all_dense, all_sparse = [], []

            for query, result in zip(queries, results):
                if isinstance(result, Exception):
                    logfire.error(
                        f"Search failed for the query variant {query!r}: {result}"
                    )
                    continue
                dense_result, sparse_result = result
                all_dense.append(dense_result)
                all_sparse.append(sparse_result)

            return all_dense, all_sparse

        try:
            all_dense, all_sparse = await asyncio.wait_for(_run(), timeout=timeout)
        except asyncio.TimeoutError:
            logfire.error(
                f"Hybrid search timeout after {timeout}s ({len(queries)} variants)"
            )
            return []

        if not all_dense or not all_sparse:
            return []

        dense_merged = self._reciprocal_rank_fusion(all_dense) if all_sparse else []
        sparse_merged = self._reciprocal_rank_fusion(all_sparse) if all_dense else []
        final_merged = self._reciprocal_rank_fusion([dense_merged, sparse_merged])

        logfire.info(
            f"Hybrid search: {len(final_merged)} unique candidates "
            f"from {len(queries)} query variants"
        )

        return final_merged
