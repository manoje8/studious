import logfire
from flashrank import Ranker, RerankRequest


class Reranker:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._ranker = None

    def _get_ranker(self) -> Ranker:
        if self._ranker is None:
            logfire.info(
                "initializing_reranker_model",
                cache_dir="/tmp/flashrank",
                top_k=self.top_k,
            )
            try:
                with logfire.span("ranker_model_loading"):
                    self._ranker = Ranker(cache_dir="/tmp/flashrank")
                    logfire.info(
                        "reranker_model_loaded_successfully",
                        model_type=type(self._ranker).__name__,
                        cache_dir="/tmp/flashrank",
                    )
            except Exception as e:
                logfire.warning(
                    "reranker_model_loading_fallback",
                    error=str(e),
                    error_type=type(e).__name__,
                    cache_dir="/tmp/flashrank",
                    using_fallback=True,
                )
                self._ranker = Ranker()
                logfire.info(
                    "reranker_model_loaded_with_fallback",
                    model_type=type(self._ranker).__name__,
                    using_fallback=True,
                )

        return self._ranker

    async def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            logfire.debug(
                "rerank_skipped_empty_candidates",
                query=query[:50] + "..." if len(query) > 50 else query,
                reason="No candidates provided",
            )
            return []

        try:
            with logfire.span("reranking_operation", query=query, num_candiates=len(candidates)):
                ranker = self._get_ranker()

                text_to_chunk: dict[str, dict] = {c["text"]: c for c in candidates}

                pairs = [{"id": i, "text": chunk["text"]} for i, chunk in enumerate(candidates)]
                request = RerankRequest(query=query, passages=pairs)
                logfire.info("rerank_preparation", paires=len(pairs))

                reranked = ranker.rerank(request)

                logfire.debug(
                    "rerank_execution_complete",
                    num_reranked=len(reranked),
                    top_score=reranked[0].get("score") if reranked else None,
                )

                with logfire.span("rerank_result_merging"):
                    merged = []
                    for result in reranked[: self.top_k]:
                        text = result.get("text", "")
                        original = text_to_chunk.get(text)
                        if original:
                            chunk = dict(original)
                            score = result.get("score", original.get("score", 0.0))
                            chunk["score"] = float(score) if score is not None else None
                            merged.append(chunk)
                        else:
                            logfire.warning(
                                "rerank_result_mismatch",
                                text_preview=(text[:50] + "..." if len(text) > 50 else text),
                                reason="Text from reranker not found in original candidates",
                            )

                if merged:
                    logfire.info(
                        "rerank_complete",
                        query=query[:50] + "..." if len(query) > 50 else query,
                        num_candidates=len(candidates),
                        num_results=len(merged),
                        top_score=merged[0].get("score"),
                        top_doc_id=merged[0].get("doc_id"),
                        top_chunk_index=merged[0].get("chunk_index"),
                        top_k_used=min(self.top_k, len(merged)),
                    )
                else:
                    logfire.warning(
                        "rerank_no_valid_results",
                        query=query[:50] + "..." if len(query) > 50 else query,
                        num_candidates=len(candidates),
                        reason="No matching texts found after reranking",
                    )

                return merged

        except Exception as e:
            logfire.error(
                "rerank_failed",
                query=query[:50] + "..." if len(query) > 50 else query,
                num_candidates=len(candidates),
                error=str(e),
                error_type=type(e).__name__,
                top_k=self.top_k,
                fallback_to_original=True,
            )
            import traceback

            logfire.debug(
                "rerank_error_traceback",
                traceback=traceback.format_exc(),
            )

            fallback_results = candidates[: self.top_k]
            logfire.info(
                "rerank_fallback_original_results",
                num_results=len(fallback_results),
                top_k=self.top_k,
            )
            return fallback_results
