import os

import logfire
from flashrank import Ranker, RerankRequest

from medici.common.utils.config import config


class Reranker:
    def __init__(
        self,
        top_k: int = 5,
        cache_dir: str = config.FLASHRANK_CACHE_DIR,
        min_score: float = config.MIN_RERANK_SCORE if hasattr(config, "MIN_RERANK_SCORE") else 0.0,
    ):
        self.top_k = top_k
        self.cache_dir = cache_dir
        self.min_score = min_score
        self._ranker = None
        self._check_cache_path()

    def _check_cache_path(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_dir = self.cache_dir
        except PermissionError:
            fallback_dir = os.path.expanduser("~/.flashrank_cache")
            os.makedirs(fallback_dir, exist_ok=True)
            self.cache_dir = fallback_dir
            logfire.warning(
                "using_fallback_cache_directory", original=self.cache_dir, fallback=fallback_dir
            )

    def _get_ranker(self) -> Ranker:
        if self._ranker is None:
            logfire.info(
                "initializing_reranker_model",
                cache_dir=self.cache_dir,
                top_k=self.top_k,
            )
            try:
                with logfire.span("ranker_model_loading"):
                    self._ranker = Ranker(cache_dir=self.cache_dir)
                    logfire.info(
                        "reranker_model_loaded_successfully",
                        model_type=type(self._ranker).__name__,
                        cache_dir=self.cache_dir,
                    )
            except Exception as e:
                logfire.warning(
                    "reranker_model_loading_fallback",
                    error=str(e),
                    error_type=type(e).__name__,
                    cache_dir=self.cache_dir,
                    using_fallback=True,
                )
                self._ranker = Ranker()
                logfire.info(
                    "reranker_model_loaded_with_fallback",
                    model_type=type(self._ranker).__name__,
                    using_fallback=True,
                )

        return self._ranker

    def _safe_get_id(self, result: dict) -> int | None:
        """Safely extract and validate the ID from a reranker result."""
        try:
            result_id = result.get("id")
            if result_id is None:
                return None
            return int(result_id)
        except (ValueError, TypeError):
            logfire.warning(
                "invalid_reranker_result_id",
                raw_id=result.get("id"),
                raw_id_type=type(result.get("id")).__name__,
                reason="ID could not be converted to int",
            )
            return None

    def _safe_get_score(self, result: dict) -> float | None:
        """Safely extract and validate the score from a reranker result."""
        try:
            score = result.get("score")
            if score is None:
                return None
            return float(score)
        except (ValueError, TypeError):
            logfire.warning(
                "invalid_reranker_score",
                raw_score=result.get("score"),
                raw_score_type=type(result.get("score")).__name__,
                reason="Score could not be converted to float",
            )
            return None

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

                id_to_chunk: dict[int, dict] = {i: c for i, c in enumerate(candidates)}

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
                    filtered_count = 0

                    for result in reranked[: self.top_k]:
                        result_id = self._safe_get_id(result)

                        if result_id is None:
                            logfire.warning(
                                "rerank_result_skipped_invalid_id",
                                raw_result=result,
                                reason="Invalid or missing ID in reranker result",
                            )
                            continue

                        original = id_to_chunk.get(result_id)
                        if original is None:
                            logfire.warning(
                                "rerank_result_mismatch",
                                result_id=result_id,
                                reason="ID from reranker not found in original candidates",
                            )
                            continue

                        score = self._safe_get_score(result)
                        if score is None:
                            score = original.get("score", 0.0)

                        if self.min_score > 0 and score < self.min_score:
                            filtered_count += 1
                            logfire.debug(
                                "rerank_result_filtered_low_score",
                                result_id=result_id,
                                score=score,
                                min_score=self.min_score,
                                doc_id=original.get("doc_id"),
                                chunk_index=original.get("chunk_index"),
                            )
                            continue

                        chunk = dict(original)
                        chunk["score"] = score
                        merged.append(chunk)

                    if filtered_count > 0:
                        logfire.info(
                            "rerank_score_filtering_stats",
                            total_considered=min(len(reranked), self.top_k),
                            passed=len(merged),
                            filtered=filtered_count,
                            min_score_threshold=self.min_score,
                        )

                if merged:
                    logfire.info(
                        "rerank_complete",
                        query=query[:50] + "..." if len(query) > 50 else query,
                        num_candidates=len(candidates),
                        num_results=len(merged),
                        top_score=merged[0].get("score"),
                        lowest_score=merged[-1].get("score"),
                        top_doc_id=merged[0].get("doc_id"),
                        top_chunk_index=merged[0].get("chunk_index"),
                        top_k_used=min(self.top_k, len(merged)),
                        min_score_threshold=self.min_score,
                    )
                else:
                    logfire.warning(
                        "rerank_no_valid_results",
                        query=query[:50] + "..." if len(query) > 50 else query,
                        num_candidates=len(candidates),
                        reason=(
                            f"All results filtered out (min_score={self.min_score})"
                            if filtered_count > 0
                            else "No matching texts found after reranking"
                        ),
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
                min_score=self.min_score,
                fallback_to_original=True,
            )
            import traceback

            logfire.debug(
                "rerank_error_traceback",
                traceback=traceback.format_exc(),
            )

            fallback_results = candidates[: self.top_k]
            if self.min_score > 0:
                filtered_fallback = [
                    c for c in fallback_results if float(c.get("score", 0) or 0) >= self.min_score
                ]
                if filtered_fallback:
                    fallback_results = filtered_fallback
                    logfire.debug(
                        "fallback_score_filtering_applied",
                        original_count=min(len(candidates), self.top_k),
                        filtered_count=len(fallback_results),
                        min_score=self.min_score,
                    )

            logfire.info(
                "rerank_fallback_original_results",
                num_results=len(fallback_results),
                top_k=self.top_k,
            )
            return fallback_results
