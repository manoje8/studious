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

        for list_idx, result_list in enumerate(result_lists):
            for rank, chunk in enumerate(result_list):
                chunk_id = f"{chunk.get('doc_id', '')}:{chunk['chunk_index']}"

                score = 1 / (k + rank + 1)

                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score

                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = chunk
                else:
                    logfire.debug(
                        "duplicate_chunk_detected_in_rrf",
                        chunk_id=chunk_id,
                        list_idx=list_idx,
                        rank=rank,
                        previous_score=rrf_scores[chunk_id] - score,
                        new_score=rrf_scores[chunk_id],
                    )

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        merged = []

        for chunk_id in sorted_ids:
            chunk = dict(chunk_map[chunk_id])
            chunk["rrf_score"] = rrf_scores[chunk_id]
            merged.append(chunk)

        logfire.debug(
            "reciprocal_rank_fusion_complete",
            unique_chunks=len(merged),
            max_rrf_score=merged[0]["rrf_score"] if merged else 0,
            min_rrf_score=merged[-1]["rrf_score"] if merged else 0,
        )

        return merged

    async def _search_one(self, query: str, doc_id_filter: str | None = None):
        """Perform a single search with both dense and sparse retrieval."""

        with logfire.span("single_search", query=query[:50] + "..." if len(query) > 50 else query):
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
        """Perform hybrid search with multiple query variants."""

        if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
            error_msg = f"queries must be list[str], got {type(queries)!r}"
            logfire.error("invalid_queries_type", error=error_msg)
            raise ValueError(error_msg)

        logfire.info(
            "hybrid_search_start",
            num_queries=len(queries),
            queries=queries[:3] + ["..."] if len(queries) > 3 else queries,
            doc_id_filter=doc_id_filter,
            timeout=timeout,
            dense_top_k=self.dense_top_k,
            sparse_top_k=self.sparse_top_k,
        )

        if len(self.sparse_index.chunks) <= self.dense_top_k:
            queries = queries[:1]

        async def _run():
            search_one = [self._search_one(query, doc_id_filter) for query in queries]
            results = await asyncio.gather(*search_one, return_exceptions=True)
            all_dense, all_sparse = [], []
            failed_queries = []

            for query, result in zip(queries, results, strict=False):
                if isinstance(result, Exception):
                    failed_queries.append(query)
                    logfire.warning(
                        "search_failed_for_query",
                        query=query[:50] + "..." if len(query) > 50 else query,
                        error=str(result),
                        error_type=type(result).__name__,
                    )
                    continue
                dense_result, sparse_result = result
                all_dense.append(dense_result)
                all_sparse.append(sparse_result)

            if failed_queries:
                logfire.warning(
                    "some_queries_failed",
                    failed_queries=(
                        failed_queries[:3] + ["..."] if len(failed_queries) > 3 else failed_queries
                    ),
                    failure_count=len(failed_queries),
                    total_queries=len(queries),
                )

            return all_dense, all_sparse

        try:
            all_dense, all_sparse = await asyncio.wait_for(_run(), timeout=timeout)
        except asyncio.TimeoutError:
            logfire.error(
                "hybrid_search_timeout",
                timeout_seconds=timeout,
                num_queries=len(queries),
                dense_top_k=self.dense_top_k,
                sparse_top_k=self.sparse_top_k,
                doc_id_filter=doc_id_filter,
            )
            return []

        if not all_dense or not all_sparse:
            logfire.warning(
                "hybrid_search_no_results",
                has_dense_results=bool(all_dense),
                has_sparse_results=bool(all_sparse),
                num_queries=len(queries),
            )
            return []

        dense_merged = self._reciprocal_rank_fusion(all_dense) if all_sparse else []
        sparse_merged = self._reciprocal_rank_fusion(all_sparse) if all_dense else []
        final_merged = self._reciprocal_rank_fusion([dense_merged, sparse_merged])

        logfire.info(
            "hybrid_search_complete",
            num_unique_candidates=len(final_merged),
            num_queries=len(queries),
            doc_id_filter=doc_id_filter,
            top_score=final_merged[0].get("rrf_score", 0.0) if final_merged else None,
            top_doc_id=final_merged[0].get("doc_id") if final_merged else None,
        )

        return final_merged
