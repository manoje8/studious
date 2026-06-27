import pickle

import logfire
from rank_bm25 import BM25Okapi
import numpy as np
from pathlib import Path

BM25_CACHE_PATH = ".cache/sparse/bm25_index.pkl"


class SparseSearchIndex:
    """In-memory BM25 index built from your chunks at ingestion time."""

    def __init__(self):
        self.index: BM25Okapi | None = None
        self.chunks: list[dict] = []

    def save(self, path: Path = BM25_CACHE_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logfire.info(f"Saving BM25 index to {path}")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "index": self.index,
                    "chunks": self.chunks,
                },
                f,
            )

    def load(self, path: Path = BM25_CACHE_PATH) -> bool:
        path = Path(path)

        if not path.exists():
            return False
        logfire.info(f"Loading BM25 index from {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.index = data["index"]
        self.chunks = data["chunks"]
        return True

    def build(self, chunks: list[dict]):
        """Build BM25 index from chunk texts."""

        if not chunks:
            logfire.warning("No chunks provided to build BM25 index")
            self.index = None
            self.chunks = []
            return

        tokenized = [chunk["text"].lower().split() for chunk in chunks]

        self.index = BM25Okapi(tokenized)
        self.chunks = chunks

        logfire.info(f"BM25 index build with {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self.index is None:
            logfire.warning("BM25 index not built yet; returning empty sparse results")
            return []

        tokenized_query = query.lower().split()
        score = self.index.get_scores(tokenized_query)

        top_indices = np.argsort(score)[::-1][:top_k]

        results = []

        for idx in top_indices:
            if score[idx] > 0:
                result = dict(self.chunks[idx])
                result["bm25_score"] = float(score[idx])
                results.append(result)

        return results
