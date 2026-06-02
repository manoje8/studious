import logfire
from rank_bm25 import BM25Okapi
import numpy as np


class SparseSearchIndex:
    """In-memory BM25 index built from your chunks at ingestion time."""

    def __init__(self):
        self.index: BM25Okapi | None = None
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]):
        """Build BM25 index from chunk texts."""

        tokenized = [chunk["text"].lower().split() for chunk in chunks]

        self.index = BM25Okapi(tokenized)
        self.chunks = chunks

        logfire.info(f"BM25 index build with {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.index:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokenized_query = query.lower().split()
        score = self.index.get_batch_scores(tokenized_query)

        top_indices = np.argsort(score)[::-1][:top_k]

        results = []

        for idx in top_indices:
            if score[idx] > 0:
                result = dict(self.chunks[idx])
                result["bm25_score"] = float(score[idx])
                results.append(result)

        return results
