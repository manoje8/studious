import logfire
from flashrank import Ranker, RerankRequest


class Reranker:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def _get_ranker(self) -> Ranker:
        global _ranker

        if _ranker is None:
            logfire.info("Initializing re-ranking")
            try:
                _ranker = Ranker(cache_dir="/tmp/flashrank")
            except Exception:
                _ranker = Ranker()

        return _ranker

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        try:
            ranker = self._get_ranker()
            pairs = [{"id": query, "text": chunk["text"]} for chunk in candidates]
            request = RerankRequest(query=query, passages=pairs)

            results = ranker.rerank(request)

            return results

        except Exception as e:
            logfire.error(f"Sematic reranking failed: {str(e)}")
            return candidates[: self.top_k]
