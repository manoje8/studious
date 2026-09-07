"""
Unit Tests for Retrieval Metrics
=================================

Tests the pure-function metrics in ``medici.eval.metrics`` without
requiring any infrastructure (no Qdrant, no LLM, no network).
"""

import pytest

from medici.eval.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# Fixtures


@pytest.fixture
def chunks_perfect():
    """5 chunks where all expected items are present."""
    return [
        {"doc_id": "doc-1", "chunk_index": 0, "text": "The Transformer achieved 28.4 BLEU score"},
        {"doc_id": "doc-1", "chunk_index": 1, "text": "English-to-German translation task"},
        {"doc_id": "doc-1", "chunk_index": 2, "text": "h = 8 parallel attention heads"},
        {"doc_id": "doc-1", "chunk_index": 3, "text": "Some other chunk about dropout"},
        {"doc_id": "doc-1", "chunk_index": 4, "text": "More about residual connections"},
    ]


@pytest.fixture
def chunks_partial():
    """5 chunks where only some expected items are found."""
    return [
        {"doc_id": "doc-1", "chunk_index": 0, "text": "The Transformer achieved 28.4 BLEU score"},
        {"doc_id": "doc-1", "chunk_index": 1, "text": "Unrelated chunk about CNNs"},
        {"doc_id": "doc-1", "chunk_index": 2, "text": "Another unrelated chunk"},
        {"doc_id": "doc-1", "chunk_index": 3, "text": "Nothing useful here"},
        {"doc_id": "doc-1", "chunk_index": 4, "text": "Still nothing relevant"},
    ]


@pytest.fixture
def chunks_empty():
    """No chunks retrieved."""
    return []


# Recall@K


class TestRecallAtK:
    def test_perfect_recall(self, chunks_perfect):
        result = recall_at_k(
            chunks_perfect,
            expected_chunks_text=["28.4 BLEU", "English-to-German"],
        )
        assert result["recall"] == 1.0
        assert len(result["matched"]) == 2
        assert len(result["missed"]) == 0

    def test_partial_recall(self, chunks_partial):
        result = recall_at_k(
            chunks_partial,
            expected_chunks_text=["28.4 BLEU", "English-to-German", "8 parallel attention"],
        )
        assert result["recall"] == pytest.approx(1 / 3, abs=0.01)
        assert len(result["matched"]) == 1
        assert len(result["missed"]) == 2

    def test_no_recall(self, chunks_partial):
        result = recall_at_k(
            chunks_partial,
            expected_chunks_text=["completely missing content"],
        )
        assert result["recall"] == 0.0

    def test_empty_chunks(self, chunks_empty):
        result = recall_at_k(
            chunks_empty,
            expected_chunks_text=["28.4 BLEU"],
        )
        assert result["recall"] == 0.0

    def test_no_expectations(self, chunks_perfect):
        result = recall_at_k(chunks_perfect)
        assert result["recall"] == 1.0

    def test_id_matching(self, chunks_perfect):
        result = recall_at_k(
            chunks_perfect,
            expected_chunk_ids=["doc-1:0", "doc-1:2"],
        )
        assert result["recall"] == 1.0

    def test_id_and_text_matching(self, chunks_perfect):
        result = recall_at_k(
            chunks_perfect,
            expected_chunk_ids=["doc-1:0"],
            expected_chunks_text=["English-to-German"],
        )
        assert result["recall"] == 1.0

    def test_k_limits_search(self, chunks_perfect):
        result = recall_at_k(
            chunks_perfect,
            expected_chunks_text=["h = 8 parallel attention"],
            k=2,  # Only look at first 2 chunks
        )
        assert result["recall"] == 0.0  # It's in chunk index 2, but k=2


# Precision@K


class TestPrecisionAtK:
    def test_all_relevant(self, chunks_perfect):
        # First 2 chunks match our expectations
        p = precision_at_k(
            chunks_perfect,
            expected_chunks_text=["28.4 BLEU", "English-to-German"],
            k=2,
        )
        assert p == 1.0

    def test_half_relevant(self, chunks_perfect):
        p = precision_at_k(
            chunks_perfect,
            expected_chunks_text=["28.4 BLEU"],
            k=2,
        )
        assert p == 0.5

    def test_none_relevant(self, chunks_partial):
        p = precision_at_k(
            chunks_partial,
            expected_chunks_text=["totally absent"],
            k=5,
        )
        assert p == 0.0

    def test_empty_chunks(self, chunks_empty):
        p = precision_at_k(chunks_empty, expected_chunks_text=["something"], k=5)
        assert p == 0.0


# MRR


class TestMRR:
    def test_first_result_relevant(self, chunks_perfect):
        mrr = mean_reciprocal_rank(
            chunks_perfect,
            expected_chunks_text=["28.4 BLEU"],
        )
        assert mrr == 1.0

    def test_second_result_relevant(self, chunks_perfect):
        mrr = mean_reciprocal_rank(
            chunks_perfect,
            expected_chunks_text=["English-to-German"],
        )
        assert mrr == 0.5

    def test_third_result_relevant(self, chunks_perfect):
        mrr = mean_reciprocal_rank(
            chunks_perfect,
            expected_chunks_text=["h = 8 parallel attention"],
        )
        assert mrr == pytest.approx(1 / 3, abs=0.01)

    def test_no_relevant_result(self, chunks_partial):
        mrr = mean_reciprocal_rank(
            chunks_partial,
            expected_chunks_text=["something not in any chunk"],
        )
        assert mrr == 0.0

    def test_no_expectations(self, chunks_perfect):
        mrr = mean_reciprocal_rank(chunks_perfect)
        assert mrr == 1.0


# NDCG@K


class TestNDCG:
    def test_perfect_ranking(self, chunks_perfect):
        ndcg = ndcg_at_k(
            chunks_perfect,
            expected_chunks_text=["28.4 BLEU", "English-to-German"],
            k=5,
        )
        assert ndcg == 1.0

    def test_no_relevant(self, chunks_partial):
        ndcg = ndcg_at_k(
            chunks_partial,
            expected_chunks_text=["missing content"],
            k=5,
        )
        assert ndcg == 0.0

    def test_relevant_at_bottom(self):
        """Relevant item at position 5 should get lower NDCG than at position 1."""
        chunks = [
            {"doc_id": "d", "chunk_index": i, "text": f"irrelevant {i}"} for i in range(4)
        ] + [
            {"doc_id": "d", "chunk_index": 4, "text": "target content here"},
        ]

        ndcg = ndcg_at_k(
            chunks,
            expected_chunks_text=["target content"],
            k=5,
        )
        assert 0 < ndcg < 1.0

    def test_empty(self, chunks_empty):
        ndcg = ndcg_at_k(chunks_empty, expected_chunks_text=["x"], k=5)
        assert ndcg == 0.0

    def test_no_expectations(self, chunks_perfect):
        ndcg = ndcg_at_k(chunks_perfect, k=5)
        assert ndcg == 1.0


# Case-insensitivity


class TestCaseInsensitivity:
    def test_recall_case_insensitive(self):
        chunks = [{"doc_id": "d", "chunk_index": 0, "text": "BLEU Score Was 28.4"}]
        result = recall_at_k(chunks, expected_chunks_text=["bleu score was 28.4"])
        assert result["recall"] == 1.0

    def test_mrr_case_insensitive(self):
        chunks = [{"doc_id": "d", "chunk_index": 0, "text": "ATTENTION Mechanism"}]
        mrr = mean_reciprocal_rank(chunks, expected_chunks_text=["attention mechanism"])
        assert mrr == 1.0
