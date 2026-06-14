"""
Unit tests for the services layer.

Covers:
- SparseSearchIndex: build(), search(), error when not built
- Reranker: rerank() happy path, empty candidates, fallback on failure
- QdrantStorageService: ensure_collection_exists(), upsert_embedded_chunks(),
  search(), scroll_all_chunks()
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.sparse_index import SparseSearchIndex
from src.services.reranker import Reranker
from src.services.qdrant import QdrantStorageService

# SparseSearchIndex


class TestSparseSearchIndex:
    """Tests for the BM25-based SparseSearchIndex."""

    @pytest.fixture
    def chunks(self):
        return [
            {
                "text": "retrieval augmented generation RAG",
                "doc_id": "d1",
                "chunk_index": 0,
            },
            {
                "text": "vector database qdrant embeddings",
                "doc_id": "d1",
                "chunk_index": 1,
            },
            {
                "text": "language model transformer attention",
                "doc_id": "d2",
                "chunk_index": 0,
            },
            {
                "text": "python programming language syntax",
                "doc_id": "d2",
                "chunk_index": 1,
            },
        ]

    @pytest.fixture
    def built_index(self, chunks):
        index = SparseSearchIndex()
        index.build(chunks)
        return index, chunks

    def test_search_before_build_raises(self):
        index = SparseSearchIndex()
        with pytest.raises(RuntimeError, match="BM25 index not built"):
            index.search("some query")

    def test_build_stores_chunks(self, chunks):
        index = SparseSearchIndex()
        index.build(chunks)
        assert index.chunks == chunks

    def test_build_creates_index(self, chunks):
        index = SparseSearchIndex()
        index.build(chunks)
        assert index.index is not None

    def test_search_returns_list(self, built_index):
        index, _ = built_index
        results = index.search("retrieval", top_k=2)
        assert isinstance(results, list)

    def test_search_finds_relevant_chunk(self, built_index):
        index, _ = built_index
        results = index.search("retrieval augmented generation")
        assert len(results) > 0
        texts = [r["text"] for r in results]
        assert any("retrieval" in t or "RAG" in t for t in texts)

    def test_search_adds_bm25_score(self, built_index):
        index, _ = built_index
        results = index.search("qdrant")
        assert all("bm25_score" in r for r in results)

    def test_search_scores_are_positive(self, built_index):
        index, _ = built_index
        results = index.search("embeddings")
        assert all(r["bm25_score"] > 0 for r in results)

    def test_search_respects_top_k(self, built_index):
        index, _ = built_index
        results = index.search("language", top_k=1)
        assert len(results) <= 1

    def test_search_returns_empty_for_nonsense_query(self, built_index):
        index, _ = built_index
        results = index.search("xyzabc123nonsensetoken")
        # All scores should be 0, so nothing returned
        assert results == []

    def test_search_is_case_insensitive(self, built_index):
        index, _ = built_index
        lower = index.search("retrieval")
        upper = index.search("RETRIEVAL")
        # Both should return results for the same concept
        assert len(lower) == len(upper)

    def test_rebuild_replaces_old_index(self, chunks):
        index = SparseSearchIndex()
        index.build(chunks)
        new_chunks = [
            {"text": "brand new content here", "doc_id": "d3", "chunk_index": 0}
        ]
        index.build(new_chunks)
        assert index.chunks == new_chunks


# Reranker


class TestReranker:
    """Tests for the flashrank-based Reranker."""

    @pytest.fixture
    def candidates(self):
        return [
            {
                "text": "RAG stands for Retrieval Augmented Generation",
                "score": 0.9,
                "source": "a.pdf",
            },
            {
                "text": "Vector databases store embeddings",
                "score": 0.7,
                "source": "b.pdf",
            },
            {"text": "LLMs are large language models", "score": 0.5, "source": "c.pdf"},
        ]

    @pytest.fixture
    def mock_ranker(self):
        """Mock flashrank Ranker."""
        with patch("src.services.reranker.Ranker") as mock_ranker_cls:
            ranker_instance = MagicMock()
            mock_ranker_cls.return_value = ranker_instance
            yield ranker_instance

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self):
        reranker = Reranker(top_k=5)
        result = await reranker.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_returns_list(self, candidates, mock_ranker):
        mock_ranker.rerank.return_value = [
            {"id": 0, "text": candidates[0]["text"], "score": 0.95},
            {"id": 1, "text": candidates[1]["text"], "score": 0.75},
        ]

        reranker = Reranker(top_k=2)
        result = await reranker.rerank("What is RAG?", candidates)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_rerank_respects_top_k(self, candidates, mock_ranker):
        mock_ranker.rerank.return_value = [
            {"id": 0, "text": candidates[0]["text"], "score": 0.95},
            {"id": 1, "text": candidates[1]["text"], "score": 0.75},
            {"id": 2, "text": candidates[2]["text"], "score": 0.55},
        ]

        reranker = Reranker(top_k=2)
        result = await reranker.rerank("query", candidates)

        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_rerank_updates_score(self, candidates, mock_ranker):
        mock_ranker.rerank.return_value = [
            {"id": 0, "text": candidates[0]["text"], "score": 0.99},
        ]

        reranker = Reranker(top_k=5)
        result = await reranker.rerank("What is RAG?", candidates)

        if result:
            assert result[0]["score"] == pytest.approx(0.99)

    @pytest.mark.asyncio
    async def test_rerank_preserves_original_fields(self, candidates, mock_ranker):
        mock_ranker.rerank.return_value = [
            {"id": 0, "text": candidates[0]["text"], "score": 0.9},
        ]

        reranker = Reranker(top_k=5)
        result = await reranker.rerank("query", candidates)

        if result:
            assert "source" in result[0]

    @pytest.mark.asyncio
    async def test_rerank_fallback_on_exception(self, candidates, mock_ranker):
        """When ranker raises, should return top_k candidates unchanged."""
        mock_ranker.rerank.side_effect = RuntimeError("flashrank exploded")

        reranker = Reranker(top_k=2)
        result = await reranker.rerank("query", candidates)

        # Fallback returns up to top_k from original
        assert len(result) <= 2
        assert all(isinstance(r, dict) for r in result)

    @pytest.mark.asyncio
    async def test_ranker_is_lazily_initialised(self, mock_ranker):
        """Ranker should not be created until rerank() is first called."""
        reranker = Reranker(top_k=3)
        assert reranker._ranker is None

        mock_ranker.rerank.return_value = []
        await reranker.rerank("q", [{"text": "x", "score": 0.1}])

        # Now it has been initialised
        assert reranker._ranker is not None


# QdrantStorageService


class TestQdrantStorageService:
    """Tests for QdrantStorageService using a mocked AsyncQdrantClient."""

    @pytest.fixture
    def mock_client(self):
        with (
            patch("src.services.qdrant.AsyncQdrantClient") as mock_cls,
            patch("src.services.qdrant.config") as mock_config,
            patch("src.services.qdrant.logfire"),
        ):
            mock_config.QDRANT_COLLECTION_NAME = "test_collection"
            mock_config.QDRANT_CLUSTER_ENDPOINT = "http://localhost:6333"
            mock_config.QDRANT_API_KEY = "test-key"

            mock_qdrant = AsyncMock()
            mock_cls.return_value = mock_qdrant

            yield mock_qdrant

    @pytest.fixture
    def service(self, mock_client):
        return QdrantStorageService(
            url="http://localhost:6333",
            collection_name="test_collection",
            vector_size=4,
            upsert_batch_size=2,
        )

    # --- ensure_collection_exists ---

    @pytest.mark.asyncio
    async def test_creates_collection_when_not_exists(self, service, mock_client):
        mock_client.collection_exists.return_value = False

        await service.ensure_collection_exists()

        mock_client.create_collection.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_creation_when_collection_exists(self, service, mock_client):
        mock_client.collection_exists.return_value = True
        # Simulate dimension match
        info = MagicMock()
        info.config.params.vectors.size = 4
        mock_client.get_collection.return_value = info

        await service.ensure_collection_exists()

        mock_client.create_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_validate_vector_dimension_mismatch_raises(
        self, service, mock_client
    ):
        info = MagicMock()
        info.config.params.vectors.size = 99  # Wrong dimension
        mock_client.get_collection.return_value = info

        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            await service.validate_vector_dimension()

    @pytest.mark.asyncio
    async def test_validate_vector_dimension_match_ok(self, service, mock_client):
        info = MagicMock()
        info.config.params.vectors.size = 4  # Matches service.vector_size
        mock_client.get_collection.return_value = info

        # Should not raise
        await service.validate_vector_dimension()

    # --- upsert_embedded_chunks ---

    @pytest.fixture
    def make_embedded_chunk(self):
        """Factory for EmbeddedChunk-like mocks."""

        def _make(doc_id="d1", chunk_index=0, text="hello"):
            ec = MagicMock()
            ec.chunk.doc_id = doc_id
            ec.chunk.chunk_index = chunk_index
            ec.to_qdrant_point.return_value = MagicMock()
            return ec

        return _make

    @pytest.mark.asyncio
    async def test_upsert_calls_ensure_collection(
        self, service, mock_client, make_embedded_chunk
    ):
        mock_client.collection_exists.return_value = True
        info = MagicMock()
        info.config.params.vectors.size = 4
        mock_client.get_collection.return_value = info

        chunks = [make_embedded_chunk(chunk_index=i) for i in range(3)]
        await service.upsert_embedded_chunks(chunks)

        mock_client.collection_exists.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upsert_batches_correctly(
        self, service, mock_client, make_embedded_chunk
    ):
        """With batch_size=2 and 5 chunks → 3 upsert calls."""
        mock_client.collection_exists.return_value = False

        chunks = [make_embedded_chunk(chunk_index=i) for i in range(5)]
        await service.upsert_embedded_chunks(chunks)

        assert mock_client.upsert.await_count == 3  # ceil(5/2)

    @pytest.mark.asyncio
    async def test_upsert_propagates_exception(
        self, service, mock_client, make_embedded_chunk
    ):
        mock_client.collection_exists.return_value = False
        mock_client.upsert.side_effect = RuntimeError("Connection failed")

        chunks = [make_embedded_chunk()]
        with pytest.raises(RuntimeError, match="Connection failed"):
            await service.upsert_embedded_chunks(chunks)

    # --- search ---

    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self, service, mock_client):
        point = MagicMock()
        point.score = 0.95
        point.payload = {
            "text": "Some chunk text",
            "section_title": "Intro",
            "source_file": "doc.pdf",
            "doc_id": "d1",
            "chunk_index": 0,
        }
        result_mock = MagicMock()
        result_mock.points = [point]
        mock_client.query_points.return_value = result_mock

        results = await service.search(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=5)

        assert len(results) == 1
        assert results[0]["text"] == "Some chunk text"
        assert results[0]["score"] == pytest.approx(0.95)
        assert results[0]["source"] == "doc.pdf"

    @pytest.mark.asyncio
    async def test_search_with_doc_id_filter(self, service, mock_client):
        result_mock = MagicMock()
        result_mock.points = []
        mock_client.query_points.return_value = result_mock

        await service.search(query_vector=[0.0, 0.1, 0.2, 0.3], doc_id_filter="doc-123")

        call_kwargs = mock_client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_search_without_filter_passes_none(self, service, mock_client):
        result_mock = MagicMock()
        result_mock.points = []
        mock_client.query_points.return_value = result_mock

        await service.search(query_vector=[0.0, 0.1, 0.2, 0.3])

        call_kwargs = mock_client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is None

    @pytest.mark.asyncio
    async def test_search_handles_none_score(self, service, mock_client):
        point = MagicMock()
        point.score = None
        point.payload = {
            "text": "x",
            "section_title": "",
            "source_file": "",
            "doc_id": "",
            "chunk_index": 0,
        }
        result_mock = MagicMock()
        result_mock.points = [point]
        mock_client.query_points.return_value = result_mock

        results = await service.search(query_vector=[0.0, 0.1, 0.2, 0.3])
        assert results[0]["score"] is None

    # --- scroll_all_chunks ---

    @pytest.mark.asyncio
    async def test_scroll_all_chunks_returns_all(self, service, mock_client):
        point = MagicMock()
        point.payload = {
            "text": "chunk text",
            "doc_id": "d1",
            "chunk_index": 0,
            "section_title": "Intro",
            "source_file": "doc.pdf",
        }
        # First call returns a point + offset=None → stops loop
        mock_client.scroll.return_value = ([point], None)

        results = await service.scroll_all_chunks()

        assert len(results) == 1
        assert results[0]["text"] == "chunk text"

    @pytest.mark.asyncio
    async def test_scroll_paginates_until_done(self, service, mock_client):
        """Simulate two pages: first returns offset='page2', second returns offset=None."""
        point_a = MagicMock()
        point_a.payload = {
            "text": "a",
            "doc_id": "d1",
            "chunk_index": 0,
            "section_title": "",
            "source_file": "",
        }
        point_b = MagicMock()
        point_b.payload = {
            "text": "b",
            "doc_id": "d1",
            "chunk_index": 1,
            "section_title": "",
            "source_file": "",
        }

        mock_client.scroll.side_effect = [
            ([point_a], "page2"),
            ([point_b], None),
        ]

        results = await service.scroll_all_chunks()

        assert len(results) == 2
        assert mock_client.scroll.await_count == 2
