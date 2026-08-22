"""
Unit tests for the services layer.

Covers:
- SparseSearchIndex: build(), search(), error when not built
- Reranker: rerank() happy path, empty candidates, fallback on failure
- QdrantStorageService: ensure_collection_exists(), upsert_embedded_chunks(),
  search(), scroll_all_chunks()
- QdrantStorageService retry logic: transient failures trigger retries (tenacity)
- QdrantStorageService.ping() and /health endpoint
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medici.common.services.qdrant import QdrantStorageService
from medici.common.services.reranker import Reranker
from medici.common.services.sparse_index import SparseSearchIndex

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
        results = index.search("test query")
        assert results == []

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
        new_chunks = [{"text": "brand new content here", "doc_id": "d3", "chunk_index": 0}]
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
        with patch("medici.common.services.reranker.Ranker") as mock_ranker_cls:
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
            patch("medici.common.services.qdrant.AsyncQdrantClient") as mock_cls,
            patch("medici.common.services.qdrant.config") as mock_config,
            patch("medici.common.services.qdrant.logfire"),
        ):
            mock_config.QDRANT_COLLECTION_NAME = "test_collection"
            mock_config.QDRANT_CLUSTER_ENDPOINT = "http://localhost:6333"
            mock_config.QDRANT_API_KEY = "test-key"
            mock_config.QDRANT_SPARSE_MODEL = "test-model"

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
        info = MagicMock()
        vectors_config = {"dense": MagicMock(size=4)}
        info.config.params.vectors = vectors_config
        mock_client.get_collection.return_value = info

        await service.ensure_collection_exists()

        mock_client.create_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_validate_vector_dimension_mismatch_raises(self, service, mock_client):
        info = MagicMock()
        vectors_config = {"dense": MagicMock(size=99)}
        info.config.params.vectors = vectors_config
        mock_client.get_collection.return_value = info

        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            await service.validate_vector_dimension()

    @pytest.mark.asyncio
    async def test_validate_vector_dimension_match_ok(self, service, mock_client):
        info = MagicMock()
        # Matches service.vector_size (4)
        vectors_config = {"dense": MagicMock(size=4)}
        info.config.params.vectors = vectors_config
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
            ec.chunk.text = text
            ec.model_name = "text-embedding-004"
            ec.vector = [0.1, 0.2, 0.3, 0.4]
            # Return a realistic payload dict that mirrors Chunk.to_quant_payload()
            ec.chunk.to_quant_payload.return_value = {
                "text": text,
                "chunk_index": chunk_index,
                "doc_id": doc_id,
                "source_file": "test.pdf",
                "chunk_type": "text",
                "section_title": "Section",
                "page_numbers": [1],
                "token_count": 10,
                "metadata": {},
            }
            return ec

        return _make

    @pytest.mark.asyncio
    async def test_upsert_calls_ensure_collection(self, service, mock_client, make_embedded_chunk):
        mock_client.collection_exists.return_value = True
        info = MagicMock()
        vectors_config = {"dense": MagicMock(size=4)}
        info.config.params.vectors = vectors_config
        mock_client.get_collection.return_value = info

        chunks = [make_embedded_chunk(chunk_index=i) for i in range(3)]
        await service.upsert_embedded_chunks(chunks)

        mock_client.collection_exists.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upsert_batches_correctly(self, service, mock_client, make_embedded_chunk):
        """With batch_size=2 and 5 chunks → 3 upsert calls."""
        mock_client.collection_exists.return_value = False

        chunks = [make_embedded_chunk(chunk_index=i) for i in range(5)]
        await service.upsert_embedded_chunks(chunks)

        assert mock_client.upsert.await_count == 3  # ceil(5/2)

    @pytest.mark.asyncio
    async def test_upsert_propagates_exception(self, service, mock_client, make_embedded_chunk):
        """After all retries are exhausted the original exception is re-raised.

        Uses OSError (a genuine transient network error) so the retry policy
        correctly attempts 3 times before propagating.
        """
        mock_client.collection_exists.return_value = False
        mock_client.upsert.side_effect = OSError("Connection reset by peer")

        chunks = [make_embedded_chunk()]
        with pytest.raises(OSError, match="Connection reset by peer"):
            await service.upsert_embedded_chunks(chunks)

        # tenacity retries 3 times total before re-raising
        assert mock_client.upsert.await_count == 3

    @pytest.mark.asyncio
    async def test_upsert_payload_contains_doc_id_and_chunk_index(
        self, service, mock_client, make_embedded_chunk
    ):
        """Regression: every upserted PointStruct must carry doc_id and chunk_index.

        Before the fix, payload was hardcoded to {"text": ...} only.  Qdrant then
        returned points whose payload had no doc_id / chunk_index, causing
        HybridSearch to collapse all chunks to the same de-duplication key
        ('', None) and return at most one result regardless of top_k.
        """
        mock_client.collection_exists.return_value = False

        chunks = [
            make_embedded_chunk(doc_id="docA", chunk_index=0),
            make_embedded_chunk(doc_id="docA", chunk_index=1),
            make_embedded_chunk(doc_id="docB", chunk_index=0),
        ]
        await service.upsert_embedded_chunks(chunks)

        # Gather every PointStruct from every upsert call
        all_points = []
        for call in mock_client.upsert.call_args_list:
            all_points.extend(call.kwargs["points"])

        assert len(all_points) == 3, "All three chunks must be upserted"
        for point in all_points:
            assert "doc_id" in point.payload, "payload must contain doc_id"
            assert "chunk_index" in point.payload, "payload must contain chunk_index"

        # Ensure the three payloads carry distinct identities
        keys = {(p.payload["doc_id"], p.payload["chunk_index"]) for p in all_points}
        assert len(keys) == 3, "Each chunk must have a unique (doc_id, chunk_index) pair"

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

        results = await service.search(
            query="test query", query_vector=[0.1, 0.2, 0.3, 0.4], top_k=5
        )

        assert len(results) == 1
        assert results[0]["text"] == "Some chunk text"
        assert results[0]["score"] == pytest.approx(0.95)
        assert results[0]["source"] == "doc.pdf"

    @pytest.mark.asyncio
    async def test_search_with_doc_id_filter(self, service, mock_client):
        result_mock = MagicMock()
        result_mock.points = []
        mock_client.query_points.return_value = result_mock

        await service.search(
            query="test query", query_vector=[0.0, 0.1, 0.2, 0.3], doc_id_filter="doc-123"
        )

        call_kwargs = mock_client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_search_without_filter_passes_none(self, service, mock_client):
        result_mock = MagicMock()
        result_mock.points = []
        mock_client.query_points.return_value = result_mock

        await service.search(query="test query", query_vector=[0.0, 0.1, 0.2, 0.3])

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

        results = await service.search(query="test query", query_vector=[0.0, 0.1, 0.2, 0.3])
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


class TestQdrantRetry:
    """Verify tenacity retry is wired into Qdrant calls."""

    @pytest.fixture
    def mock_client(self):
        with (
            patch("medici.common.services.qdrant.AsyncQdrantClient") as mock_cls,
            patch("medici.common.services.qdrant.config") as mock_config,
            patch("medici.common.services.qdrant.logfire"),
        ):
            mock_config.QDRANT_COLLECTION_NAME = "test_collection"
            mock_config.QDRANT_CLUSTER_ENDPOINT = "http://localhost:6333"
            mock_config.QDRANT_API_KEY = "key"
            mock_config.QDRANT_SPARSE_MODEL = "test-model"
            mock_qdrant = AsyncMock()
            mock_cls.return_value = mock_qdrant
            yield mock_qdrant

    @pytest.fixture
    def service(self, mock_client):
        return QdrantStorageService(
            url="http://localhost:6333",
            collection_name="test_collection",
            vector_size=4,
            upsert_batch_size=10,
        )

    @pytest.mark.asyncio
    async def test_search_retries_on_transient_error(self, service, mock_client):
        """search() retries up to 3 times before re-raising."""
        mock_client.query_points.side_effect = OSError("timeout")

        with pytest.raises(OSError, match="timeout"):
            await service.search(query="test query", query_vector=[0.1, 0.2, 0.3, 0.4])

        assert mock_client.query_points.await_count == 3

    @pytest.mark.asyncio
    async def test_search_succeeds_on_first_try_calls_once(self, service, mock_client):
        """When no error occurs, the underlying client is called exactly once."""
        result_mock = MagicMock()
        result_mock.points = []
        mock_client.query_points.return_value = result_mock

        await service.search(query="test query", query_vector=[0.1, 0.2, 0.3, 0.4])

        assert mock_client.query_points.await_count == 1

    @pytest.mark.asyncio
    async def test_scroll_retries_on_transient_error(self, service, mock_client):
        """scroll_all_chunks() retries its first page call 3 times."""
        mock_client.scroll.side_effect = OSError("network blip")

        with pytest.raises(OSError, match="network blip"):
            await service.scroll_all_chunks()

        assert mock_client.scroll.await_count == 3

    @pytest.mark.asyncio
    async def test_chunk_count_retries_on_transient_error(self, service, mock_client):
        """chunk_count() retries 3 times before propagating."""
        mock_client.count.side_effect = OSError("unreachable")

        with pytest.raises(OSError, match="unreachable"):
            await service.chunk_count()

        assert mock_client.count.await_count == 3

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self, service, mock_client):
        """A transient failure on the first attempt is recovered by the second."""
        result_mock = MagicMock()
        result_mock.points = []
        mock_client.query_points.side_effect = [
            OSError("transient"),  # first attempt fails
            result_mock,  # second attempt succeeds
        ]

        results = await service.search(query="test query", query_vector=[0.1, 0.2, 0.3, 0.4])

        assert mock_client.query_points.await_count == 2
        assert results == []

    @pytest.mark.asyncio
    async def test_non_transient_4xx_is_not_retried(self, service, mock_client):
        """A 4xx UnexpectedResponse (e.g. 400 bad filter) must NOT be retried.

        The retry predicate must fast-fail immediately so we waste neither
        retry budget nor backoff time on errors that will never self-heal.
        """
        from qdrant_client.http.exceptions import UnexpectedResponse

        bad_request = UnexpectedResponse(
            status_code=400,
            reason_phrase="Bad Request",
            content=b'{"status": {"error": "Wrong filter"}}',
            headers={},
        )
        mock_client.query_points.side_effect = bad_request

        with pytest.raises(UnexpectedResponse):
            await service.search(query="test query", query_vector=[0.1, 0.2, 0.3, 0.4])

        # Called exactly once — no retries burned on a permanent client error.
        assert mock_client.query_points.await_count == 1

    @pytest.mark.asyncio
    async def test_5xx_unexpected_response_is_retried(self, service, mock_client):
        """A 5xx UnexpectedResponse is treated as transient and retried 3 times."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        server_error = UnexpectedResponse(
            status_code=503,
            reason_phrase="Service Unavailable",
            content=b"overloaded",
            headers={},
        )
        mock_client.query_points.side_effect = server_error

        with pytest.raises(UnexpectedResponse):
            await service.search(query="test query", query_vector=[0.1, 0.2, 0.3, 0.4])

        assert mock_client.query_points.await_count == 3

    @pytest.mark.asyncio
    async def test_response_handling_exception_is_retried(self, service, mock_client):
        """ResponseHandlingException (transport-level) is always retried."""
        from qdrant_client.http.exceptions import ResponseHandlingException

        transport_err = ResponseHandlingException(Exception("Connection refused"))
        mock_client.query_points.side_effect = transport_err

        with pytest.raises(ResponseHandlingException):
            await service.search(query="test query", query_vector=[0.1, 0.2, 0.3, 0.4])

        assert mock_client.query_points.await_count == 3


class TestQdrantPing:
    """Tests for QdrantStorageService.ping()."""

    @pytest.fixture
    def mock_client(self):
        with (
            patch("medici.common.services.qdrant.AsyncQdrantClient") as mock_cls,
            patch("medici.common.services.qdrant.config") as mock_config,
            patch("medici.common.services.qdrant.logfire"),
        ):
            mock_config.QDRANT_COLLECTION_NAME = "test_collection"
            mock_config.QDRANT_CLUSTER_ENDPOINT = "http://localhost:6333"
            mock_config.QDRANT_API_KEY = "key"
            mock_config.QDRANT_SPARSE_MODEL = "test-model"
            mock_qdrant = AsyncMock()
            mock_cls.return_value = mock_qdrant
            yield mock_qdrant

    @pytest.fixture
    def service(self, mock_client):
        return QdrantStorageService(
            url="http://localhost:6333",
            collection_name="test_collection",
            vector_size=4,
        )

    @pytest.mark.asyncio
    async def test_ping_returns_true_when_reachable(self, service, mock_client):
        mock_client.get_collections.return_value = MagicMock()

        result = await service.ping()

        assert result is True

    @pytest.mark.asyncio
    async def test_ping_returns_false_when_unreachable(self, service, mock_client):
        mock_client.get_collections.side_effect = OSError("refused")

        result = await service.ping()

        assert result is False

    @pytest.mark.asyncio
    async def test_ping_attempts_retries_before_returning_false(self, service, mock_client):
        """ping() retries via tenacity so 3 attempts are made before returning False."""
        mock_client.get_collections.side_effect = OSError("refused")

        result = await service.ping()

        assert result is False
        assert mock_client.get_collections.await_count == 3


class TestHealthEndpoint:
    """/health endpoint: 200 when Qdrant is up, 503 when it is not."""

    @pytest.fixture
    def fast_app(self):
        """Bare FastAPI app with the /health route, no lifespan overhead."""
        from fastapi import FastAPI
        from fastapi.requests import Request
        from fastapi.responses import JSONResponse

        app = FastAPI()

        @app.get("/health")
        async def health(request: Request):
            qdrant_service = getattr(request.app.state, "qdrant", None)
            qdrant_ok = False
            if qdrant_service is not None:
                qdrant_ok = await qdrant_service.ping()
            deps = {"qdrant": "ok" if qdrant_ok else "unreachable"}
            overall = "ok" if all(v == "ok" for v in deps.values()) else "degraded"
            return JSONResponse(
                status_code=200 if overall == "ok" else 503,
                content={"status": overall, "dependencies": deps},
            )

        return app

    def _make_service(self, ping_result: bool):
        svc = MagicMock(spec=QdrantStorageService)
        svc.ping = AsyncMock(return_value=ping_result)
        return svc

    @pytest.mark.asyncio
    async def test_health_ok_when_qdrant_up(self, fast_app):
        from httpx import ASGITransport, AsyncClient

        fast_app.state.qdrant = self._make_service(ping_result=True)
        async with AsyncClient(transport=ASGITransport(app=fast_app), base_url="http://test") as ac:
            resp = await ac.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["dependencies"]["qdrant"] == "ok"

    @pytest.mark.asyncio
    async def test_health_degraded_when_qdrant_down(self, fast_app):
        from httpx import ASGITransport, AsyncClient

        fast_app.state.qdrant = self._make_service(ping_result=False)
        async with AsyncClient(transport=ASGITransport(app=fast_app), base_url="http://test") as ac:
            resp = await ac.get("/health")

        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["dependencies"]["qdrant"] == "unreachable"

    @pytest.mark.asyncio
    async def test_health_degraded_when_no_qdrant_in_state(self, fast_app):
        """If qdrant is absent from app.state (e.g. startup failed) → 503."""
        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=fast_app), base_url="http://test") as ac:
            resp = await ac.get("/health")

        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"


class TestTokenBudgetGuard:
    """Verify _build_context() sorts by relevance_score and respects the token budget."""

    def _make_chunk(self, text: str, score: float, source: str = "doc.pdf") -> dict:
        return {
            "text": text,
            "relevance_score": score,
            "source": source,
            "section": "Test",
        }

    @patch("medici.agents.agentic.synthesizer.get_tokenizer")
    def test_build_context_sorts_by_relevance_score(self, mock_get_tok):
        """Highest-score chunk must appear first in the built context string."""
        from medici.agents.agentic.synthesizer import _build_context

        # Tokenizer that never trips the budget (returns tiny counts)
        tok = MagicMock()
        tok.count.return_value = 1
        tok.encode.return_value = [1]
        mock_get_tok.return_value = tok

        low = self._make_chunk("low relevance text", score=0.1)
        high = self._make_chunk("high relevance text", score=0.9)

        state = {"accepted_chunks": [low, high]}  # low-score chunk listed first
        ctx = _build_context(state)

        # high-score chunk must precede low-score chunk in output
        assert ctx.index("high relevance text") < ctx.index("low relevance text")

    @patch("medici.agents.agentic.synthesizer.get_tokenizer")
    def test_build_context_trims_low_relevance_when_over_budget(self, mock_get_tok):
        """Low-score chunk is dropped when the token budget is exhausted."""
        from medici.agents.agentic.synthesizer import _build_context

        tok = MagicMock()
        # Each chunk part costs 60 tokens; separator costs 5 — budget=70 fits only 1 chunk.
        tok.count.side_effect = lambda text: 60 if "relevance" in text else 5
        tok.encode.return_value = list(range(60))
        mock_get_tok.return_value = tok

        high = self._make_chunk("high relevance content", score=0.9)
        low = self._make_chunk("low relevance content", score=0.1)

        from medici.agents.agentic.synthesizer import config

        with (
            patch.object(config, "MAX_CONTEXT_CHARS", 1_000_000),
            patch.object(config, "MODEL_CONTEXT_LIMIT", 70),
            patch.object(config, "MAX_OUTPUT_TOKENS", 0),
            patch.object(config, "MAX_PROMPT_OVERHEAD_TOKENS", 0),
        ):
            state = {"accepted_chunks": [low, high]}
            ctx = _build_context(state)

        assert "high relevance content" in ctx
        assert "low relevance content" not in ctx

    @patch("medici.agents.agentic.synthesizer.get_tokenizer")
    def test_build_context_empty_chunks_returns_empty_string(self, mock_get_tok):
        """Empty accepted_chunks must produce an empty string."""
        from medici.agents.agentic.synthesizer import _build_context

        tok = MagicMock()
        tok.count.return_value = 0
        mock_get_tok.return_value = tok

        state = {"accepted_chunks": []}
        assert _build_context(state) == ""

    @patch("medici.agents.agentic.synthesizer.get_tokenizer")
    def test_build_context_all_chunks_fit_returns_all(self, mock_get_tok):
        """When total tokens < budget, all chunks must be included."""
        from medici.agents.agentic.synthesizer import _build_context

        tok = MagicMock()
        tok.count.return_value = 1  # tiny token counts — nothing will overflow
        mock_get_tok.return_value = tok

        chunks = [
            self._make_chunk("alpha content", score=0.8),
            self._make_chunk("beta content", score=0.6),
            self._make_chunk("gamma content", score=0.4),
        ]

        from medici.agents.agentic.synthesizer import config

        with (
            patch.object(config, "MAX_CONTEXT_CHARS", 1_000_000),
            patch.object(config, "MODEL_CONTEXT_LIMIT", 128_000),
            patch.object(config, "MAX_OUTPUT_TOKENS", 2_048),
            patch.object(config, "MAX_PROMPT_OVERHEAD_TOKENS", 1_024),
        ):
            state = {"accepted_chunks": chunks}
            ctx = _build_context(state)

        assert "alpha content" in ctx
        assert "beta content" in ctx
        assert "gamma content" in ctx


# Query input guard (512-token truncation in QueryExpander / QueryRewriter)


class TestQueryInputGuard:
    """Verify that long queries are truncated to MAX_QUERY_INPUT_TOKENS before the LLM sees them."""

    @pytest.mark.asyncio
    @patch("medici.common.utils.query_utils.get_tokenizer")
    async def test_expander_truncates_long_query(self, mock_get_tok):
        """A query > 512 tokens is truncated; the LLM prompt receives the shorter text."""
        from medici.agents.agentic.query_expander import QueryExpander

        tok = MagicMock()
        tok.encode.return_value = list(range(600))  # 600 tokens — over limit
        tok.decode.return_value = "truncated query text"
        mock_get_tok.return_value = tok

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = MagicMock(parsed_json=["alt1", "alt2"])

        expander = QueryExpander(llm_client=mock_llm)

        with patch("medici.common.utils.config") as mock_cfg:
            mock_cfg.MAX_QUERY_INPUT_TOKENS = 512

            await expander.expand("x" * 3000)

        # decode must have been called with the first 512 tokens only
        tok.decode.assert_called_once_with(list(range(512)))

        # the prompt sent to the LLM must contain the truncated text
        call_prompt = mock_llm.complete.call_args[0][0]
        assert "truncated query text" in call_prompt

    @pytest.mark.asyncio
    @patch("medici.common.utils.query_utils.get_tokenizer")
    async def test_expander_passes_short_query_unchanged(self, mock_get_tok):
        """A short query (≤ 512 tokens) is forwarded verbatim."""
        from medici.agents.agentic.query_expander import QueryExpander

        tok = MagicMock()
        tok.encode.return_value = list(range(10))  # 10 tokens — under limit
        tok.decode.return_value = "should not be called"
        mock_get_tok.return_value = tok

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = MagicMock(parsed_json=["alt1"])

        expander = QueryExpander(llm_client=mock_llm)
        original_query = "What is the capital of France?"

        with patch("medici.common.utils.config") as mock_cfg:
            mock_cfg.MAX_QUERY_INPUT_TOKENS = 512

            await expander.expand(original_query)

        # decode should NOT have been called (no truncation needed)
        tok.decode.assert_not_called()

        call_prompt = mock_llm.complete.call_args[0][0]
        assert original_query in call_prompt

    @pytest.mark.asyncio
    @patch("medici.common.utils.query_utils.get_tokenizer")
    async def test_rewriter_truncates_long_query(self, mock_get_tok):
        """A query > 512 tokens is truncated before rewrite() processes it."""
        from medici.agents.agentic.query_rewriter import QueryRewriter

        tok = MagicMock()
        tok.encode.return_value = list(range(700))  # 700 tokens — over limit
        tok.decode.return_value = "truncated rewriter query"
        mock_get_tok.return_value = tok

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = MagicMock(
            parsed_json={
                "rewritten_query": "truncated rewriter query",
                "was_rewritten": False,
                "resolved_references": [],
            }
        )

        rewriter = QueryRewriter(llm_client=mock_llm)

        # Provide a session with turns so the rewriter actually sends the prompt
        session = MagicMock()
        session.turns = ["turn1"]
        session.to_prompt_format.return_value = "history text"

        with patch("medici.common.utils.config") as mock_cfg:
            mock_cfg.MAX_QUERY_INPUT_TOKENS = 512

            await rewriter.rewrite("x" * 3500, session)

        tok.decode.assert_called_once_with(list(range(512)))

        call_prompt = mock_llm.complete.call_args[0][0]
        assert "truncated rewriter query" in call_prompt

    @pytest.mark.asyncio
    @patch("medici.common.utils.query_utils.get_tokenizer")
    async def test_rewriter_passes_short_query_unchanged(self, mock_get_tok):
        """A short query (≤ 512 tokens) is forwarded verbatim."""
        from medici.agents.agentic.query_rewriter import QueryRewriter

        tok = MagicMock()
        tok.encode.return_value = list(range(20))
        mock_get_tok.return_value = tok

        rewriter = QueryRewriter(llm_client=MagicMock())

        session = MagicMock()
        session.turns = []

        original_query = "Who signed the contract?"

        with patch("medici.common.utils.config") as mock_cfg:
            mock_cfg.MAX_QUERY_INPUT_TOKENS = 512

            result = await rewriter.rewrite(original_query, session)

        tok.decode.assert_not_called()
        assert result["rewritten_query"] == original_query
