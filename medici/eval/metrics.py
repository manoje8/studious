"""
Retrieval Quality Metrics
=========================

Pure functions for computing standard information retrieval metrics.
All functions work on lists of chunk identifiers or text fragments,
making them testable without any infrastructure dependencies.

Metrics
-------
- **Recall@K**: fraction of expected items found in top-K retrieved
- **Precision@K**: fraction of top-K retrieved items that are relevant
- **MRR (Mean Reciprocal Rank)**: 1/rank of the first relevant item
- **NDCG@K**: normalised discounted cumulative gain at K
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _build_retrieved_ids(
    accepted_chunks: list[dict],
) -> list[str]:
    """Build ``doc_id:chunk_index`` identifiers from accepted chunks."""
    return [f"{c.get('doc_id', '')}:{c.get('chunk_index', '')}" for c in accepted_chunks]


def _build_retrieved_texts(accepted_chunks: list[dict]) -> list[str]:
    """Extract lowered text from accepted chunks for substring matching."""
    return [(c.get("text") or "").lower() for c in accepted_chunks]


def _is_relevant(
    chunk_id: str,
    chunk_text_lower: str,
    expected_ids: set[str],
    expected_texts_lower: list[str],
) -> bool:
    """Check whether a single retrieved chunk matches any expected item."""
    if chunk_id in expected_ids:
        return True
    for substr in expected_texts_lower:
        if substr in chunk_text_lower:
            return True
    return False


def recall_at_k(
    accepted_chunks: list[dict],
    expected_chunk_ids: Sequence[str] = (),
    expected_chunks_text: Sequence[str] = (),
    k: int | None = None,
) -> dict:
    """Compute Recall@K.

    Parameters
    ----------
    accepted_chunks:
        Chunks returned by the retrieval pipeline.
    expected_chunk_ids:
        ``doc_id:chunk_index`` strings that *should* appear.
    expected_chunks_text:
        Text substrings that should appear in at least one chunk.
    k:
        Evaluate only the top-K chunks.  ``None`` means all.

    Returns
    -------
    dict with ``recall``, ``matched``, ``missed``, ``total_expected``.
    """
    if not expected_chunk_ids and not expected_chunks_text:
        return {"recall": 1.0, "matched": [], "missed": [], "total_expected": 0}

    top_chunks = accepted_chunks[:k] if k else accepted_chunks
    retrieved_ids = set(_build_retrieved_ids(top_chunks))
    retrieved_texts = _build_retrieved_texts(top_chunks)

    matched: list[dict[str, str]] = []
    missed: list[dict[str, str]] = []

    for eid in expected_chunk_ids:
        if eid in retrieved_ids:
            matched.append({"type": "id", "value": eid})
        else:
            missed.append({"type": "id", "value": eid})

    for substr in expected_chunks_text:
        substr_lower = substr.lower()
        if any(substr_lower in rt for rt in retrieved_texts):
            matched.append({"type": "text", "value": substr})
        else:
            missed.append({"type": "text", "value": substr})

    total = len(matched) + len(missed)
    recall = len(matched) / total if total > 0 else 0.0

    return {
        "recall": round(recall, 4),
        "matched": matched,
        "missed": missed,
        "total_expected": total,
    }


def precision_at_k(
    accepted_chunks: list[dict],
    expected_chunk_ids: Sequence[str] = (),
    expected_chunks_text: Sequence[str] = (),
    k: int = 5,
) -> float:
    """Compute Precision@K — fraction of retrieved items that are relevant."""
    top_chunks = accepted_chunks[:k]
    if not top_chunks:
        return 0.0

    expected_ids = set(expected_chunk_ids)
    expected_texts_lower = [t.lower() for t in expected_chunks_text]

    retrieved_ids = _build_retrieved_ids(top_chunks)
    retrieved_texts = _build_retrieved_texts(top_chunks)

    relevant_count = sum(
        1
        for cid, ctext in zip(retrieved_ids, retrieved_texts, strict=False)
        if _is_relevant(cid, ctext, expected_ids, expected_texts_lower)
    )

    return round(relevant_count / len(top_chunks), 4)


def mean_reciprocal_rank(
    accepted_chunks: list[dict],
    expected_chunk_ids: Sequence[str] = (),
    expected_chunks_text: Sequence[str] = (),
) -> float:
    """Compute MRR — reciprocal of the rank of the first relevant result.

    Returns 0.0 if no relevant chunk is found.
    """
    if not expected_chunk_ids and not expected_chunks_text:
        return 1.0

    expected_ids = set(expected_chunk_ids)
    expected_texts_lower = [t.lower() for t in expected_chunks_text]

    retrieved_ids = _build_retrieved_ids(accepted_chunks)
    retrieved_texts = _build_retrieved_texts(accepted_chunks)

    for rank, (cid, ctext) in enumerate(zip(retrieved_ids, retrieved_texts, strict=False), start=1):
        if _is_relevant(cid, ctext, expected_ids, expected_texts_lower):
            return round(1.0 / rank, 4)

    return 0.0


def ndcg_at_k(
    accepted_chunks: list[dict],
    expected_chunk_ids: Sequence[str] = (),
    expected_chunks_text: Sequence[str] = (),
    k: int = 5,
) -> float:
    """Compute NDCG@K.

    Uses binary relevance: 1 if the chunk matches an expected item, 0 otherwise.
    The ideal ranking places all relevant items at the top.
    """
    top_chunks = accepted_chunks[:k]
    if not top_chunks:
        return 0.0

    expected_ids = set(expected_chunk_ids)
    expected_texts_lower = [t.lower() for t in expected_chunks_text]
    total_relevant = len(expected_chunk_ids) + len(expected_chunks_text)

    if total_relevant == 0:
        return 1.0

    retrieved_ids = _build_retrieved_ids(top_chunks)
    retrieved_texts = _build_retrieved_texts(top_chunks)

    # Actual DCG
    dcg = 0.0
    for i, (cid, ctext) in enumerate(zip(retrieved_ids, retrieved_texts, strict=False)):
        if _is_relevant(cid, ctext, expected_ids, expected_texts_lower):
            dcg += 1.0 / math.log2(i + 2)  # i+2 because positions are 1-indexed

    # Ideal DCG: all relevant items at the top
    ideal_at_k = min(total_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_at_k))

    if idcg == 0:
        return 0.0

    return round(dcg / idcg, 4)
