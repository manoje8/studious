import logfire
from flashrank import Ranker, RerankRequest

_ranker = None


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

    async def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        try:
            ranker = self._get_ranker()

            text_to_chunk: dict[str, dict] = {c["text"]: c for c in candidates}

            pairs = [
                {"id": i, "text": chunk["text"]} for i, chunk in enumerate(candidates)
            ]
            request = RerankRequest(query=query, passages=pairs)

            reranked = ranker.rerank(request)

            merged = []
            for result in reranked[: self.top_k]:
                text = result.get("text", "")
                original = text_to_chunk.get(text)
                if original:
                    chunk = dict(original)
                    chunk["score"] = result.get("score", original.get("score", 0.0))
                    merged.append(chunk)

            return merged

        except Exception as e:
            logfire.error(f"Semantic reranking failed: {str(e)}")
            return candidates[: self.top_k]
