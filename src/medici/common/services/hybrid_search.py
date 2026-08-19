from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import logfire

from medici.common.services.qdrant import QdrantStorageService

if TYPE_CHECKING:
    from medici.ingestion.embedding import EmbeddingService


class HybridSearch:
    def __init__(
        self,
        storage_service: QdrantStorageService,
        embedding_service: EmbeddingService,
        top_k: int = 20,
    ):
        self.storage_service = storage_service
        self.embedding_service = embedding_service
        self.top_k = top_k

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
            hybrid_result = await self.storage_service.search(
                query=query,
                query_vector=query_vector,
                top_k=self.top_k,
                doc_id_filter=doc_id_filter,
            )

            return hybrid_result

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
            top_k=self.top_k,
        )

        async def _run():
            search_one = [self._search_one(query, doc_id_filter) for query in queries]
            return await asyncio.gather(*search_one, return_exceptions=True)

        try:
            per_query_results = await asyncio.wait_for(_run(), timeout=timeout)
            logfire.info("hybrid_search_result", results=len(per_query_results))
        except TimeoutError:
            logfire.error(
                "hybrid_search_timeout",
                timeout_seconds=timeout,
                num_queries=len(queries),
                top_k=self.top_k,
                doc_id_filter=doc_id_filter,
            )
            return []

        flat: list[dict] = []
        for query, result in zip(queries, per_query_results, strict=False):
            if isinstance(result, BaseException):
                logfire.warning("single_search_failed", query=query[:50], error=str(result))
                continue
            flat.extend(result)

        if not flat:
            logfire.warning(
                "hybrid_search_no_results",
                has_results=bool(flat),
                num_queries=len(queries),
            )
            return []

        best: dict[tuple, dict] = {}
        for item in flat:
            key = (item.get("doc_id"), item.get("chunk_index"))
            if key not in best or (item.get("score") or 0) > (best[key].get("score") or 0):
                best[key] = item
        final = sorted(best.values(), key=lambda d: d.get("score") or 0, reverse=True)[: self.top_k]
        logfire.info(
            "hybrid_search_complete",
            num_unique_candidates=len(final),
            num_queries=len(queries),
            doc_id_filter=doc_id_filter,
            top_score=final[0].get("score", 0.0) if final else None,
            top_doc_id=final[0].get("doc_id") if final else None,
        )

        return final
