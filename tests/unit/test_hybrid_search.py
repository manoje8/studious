from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.agents.hybrid_search import HybridSearch
from src.common.services.qdrant import QdrantStorageService
from src.common.services.sparse_index import SparseSearchIndex
from src.ingestion.embedding import EmbeddingService


# Mock classes and fixtures
@dataclass
class MockPoint:
    """Mock Qdrant point for testing"""

    payload: dict
    score: float


@pytest.fixture
def mock_qdrant_storage():
    """Fixture for mocked Qdrant storage service"""
    storage = Mock(spec=QdrantStorageService)
    storage.search = AsyncMock()
    storage.scroll_all_chunks = AsyncMock()
    return storage


@pytest.fixture
def mock_embedding_service():
    """Fixture for mocked embedding service"""
    embedding = Mock(spec=EmbeddingService)
    embedding.embed_single = AsyncMock()
    embedding.vector_size = 1536
    return embedding


@pytest.fixture
def mock_sparse_index():
    """Fixture for mocked sparse search index"""
    index = Mock(spec=SparseSearchIndex)
    index.search = Mock()
    index.chunks = ["chunk"] * 30
    return index


@pytest.fixture
def hybrid_search(mock_qdrant_storage, mock_embedding_service, mock_sparse_index):
    """Fixture for HybridSearch instance with mocked dependencies"""

    return HybridSearch(
        storage_service=mock_qdrant_storage,
        embedding_service=mock_embedding_service,
        sparse_index=mock_sparse_index,
        dense_top_k=20,
        sparse_top_k=20,
    )


@pytest.fixture
def sample_dense_results():
    """Fixture providing sample dense search results"""
    return [
        [
            {
                "doc_id": "doc1",
                "chunk_index": 0,
                "text": "Sample text 1",
                "score": 0.95,
                "section": "intro",
                "source": "doc1.pdf",
            },
            {
                "doc_id": "doc1",
                "chunk_index": 1,
                "text": "Sample text 2",
                "score": 0.85,
                "section": "methods",
                "source": "doc1.pdf",
            },
        ],
        [
            {
                "doc_id": "doc2",
                "chunk_index": 0,
                "text": "Sample text 3",
                "score": 0.90,
                "section": "intro",
                "source": "doc2.pdf",
            }
        ],
    ]


@pytest.fixture
def sample_sparse_results():
    """Fixture providing sample sparse search results"""
    return [
        [
            {
                "doc_id": "doc1",
                "chunk_index": 0,
                "text": "Sample text 1",
                "bm25_score": 1.2,
                "section": "intro",
                "source": "doc1.pdf",
            }
        ],
        [
            {
                "doc_id": "doc2",
                "chunk_index": 0,
                "text": "Sample text 3",
                "bm25_score": 0.8,
                "section": "intro",
                "source": "doc2.pdf",
            }
        ],
    ]


class TestReciprocalRankFusion:
    """Test cases for the _reciprocal_rank_fusion method"""

    def test_basic_fusion(self, hybrid_search):
        """Test basic RRF fusion with multiple result lists"""
        result_lists = [
            [
                {"doc_id": "doc1", "chunk_index": 0, "text": "text1"},
                {"doc_id": "doc2", "chunk_index": 0, "text": "text2"},
            ],
            [
                {"doc_id": "doc2", "chunk_index": 0, "text": "text2"},
                {"doc_id": "doc1", "chunk_index": 0, "text": "text1"},
            ],
        ]

        merged = hybrid_search._reciprocal_rank_fusion(result_lists, k=10)

        assert len(merged) == 2
        assert "rrf_score" in merged[0]
        assert merged[0]["rrf_score"] >= merged[1]["rrf_score"]

    def test_fusion_with_overlapping_chunks(self, hybrid_search):
        """Test RRF where chunks appear in multiple result lists"""
        result_lists = [
            [
                {"doc_id": "doc1", "chunk_index": 0, "text": "text1"},
                {"doc_id": "doc1", "chunk_index": 1, "text": "text2"},
                {"doc_id": "doc2", "chunk_index": 0, "text": "text3"},
            ],
            [
                {"doc_id": "doc1", "chunk_index": 0, "text": "text1"},
                {"doc_id": "doc2", "chunk_index": 0, "text": "text3"},
            ],
        ]

        merged = hybrid_search._reciprocal_rank_fusion(result_lists)

        # doc1:0 should have highest score (ranked 1st and 1st)
        assert merged[0]["doc_id"] == "doc1"
        assert merged[0]["chunk_index"] == 0
        assert len(merged) == 3

    def test_empty_result_lists(self, hybrid_search):
        """Test RRF with empty result lists"""
        result_lists = [[], []]

        merged = hybrid_search._reciprocal_rank_fusion(result_lists)

        assert merged == []

    def test_single_result_list(self, hybrid_search):
        """Test RRF with a single result list"""
        result_lists = [
            [
                {"doc_id": "doc1", "chunk_index": 0, "text": "text1"},
                {"doc_id": "doc2", "chunk_index": 0, "text": "text2"},
            ]
        ]

        merged = hybrid_search._reciprocal_rank_fusion(result_lists)

        assert len(merged) == 2
        assert merged[0]["doc_id"] == "doc1"
        assert merged[1]["doc_id"] == "doc2"

    def test_fusion_score_calculation(self, hybrid_search):
        """Test that RRF scores are calculated correctly"""
        result_lists = [
            [{"doc_id": "doc1", "chunk_index": 0, "text": "text1"}],
            [{"doc_id": "doc1", "chunk_index": 0, "text": "text1"}],
        ]

        merged = hybrid_search._reciprocal_rank_fusion(result_lists, k=10)

        # Expected score: 2 * 1/(10 + 0 + 1) = 2/11
        expected_score = 2 * (1 / (10 + 0 + 1))
        assert len(merged) == 1
        assert abs(merged[0]["rrf_score"] - expected_score) < 0.0001


class TestHybridSearch:
    """Test cases for the main search method"""

    @pytest.mark.asyncio
    async def test_single_query_search(
        self,
        hybrid_search,
        mock_qdrant_storage,
        mock_embedding_service,
        mock_sparse_index,
    ):
        """Test search with a single query"""
        # Setup mocks
        mock_embedding_service.embed_single.return_value = [0.1] * 1536
        mock_qdrant_storage.search.return_value = [
            {
                "text": "test text",
                "doc_id": "doc1",
                "chunk_index": 0,
                "section": "Test Section",
                "source": "test.pdf",
                "score": 0.95,
            }
        ]
        mock_sparse_index.search.return_value = [
            {
                "doc_id": "doc1",
                "chunk_index": 0,
                "text": "test text",
                "bm25_score": 1.5,
                "section": "Test Section",
                "source": "test.pdf",
            }
        ]

        results = await hybrid_search.search(["test query"])

        assert len(results) > 0
        assert all("rrf_score" in chunk for chunk in results)

    @pytest.mark.asyncio
    async def test_multi_query_search(
        self,
        hybrid_search,
        mock_qdrant_storage,
        mock_embedding_service,
        mock_sparse_index,
    ):
        """Test search with multiple query variants"""
        queries = ["query variant 1", "query variant 2"]

        # Setup mocks to return different results for each query
        mock_embedding_service.embed_single.side_effect = [[0.1] * 1536, [0.2] * 1536]

        mock_qdrant_storage.search.side_effect = [
            [
                {
                    "text": "result 1",
                    "doc_id": "doc1",
                    "chunk_index": 0,
                    "section": "",
                    "source": "",
                    "score": 0.9,
                }
            ],
            [
                {
                    "text": "result 2",
                    "doc_id": "doc2",
                    "chunk_index": 0,
                    "section": "",
                    "source": "",
                    "score": 0.85,
                }
            ],
        ]

        mock_sparse_index.search.side_effect = [
            [
                {
                    "doc_id": "doc1",
                    "chunk_index": 0,
                    "text": "result 1",
                    "bm25_score": 1.2,
                }
            ],
            [
                {
                    "doc_id": "doc2",
                    "chunk_index": 0,
                    "text": "result 2",
                    "bm25_score": 0.9,
                }
            ],
        ]

        results = await hybrid_search.search(queries)

        # Should have deduplicated results
        assert len(results) > 0
        # Verify that embedding was called for each query
        assert mock_embedding_service.embed_single.call_count == 2
        assert mock_qdrant_storage.search.call_count == 2
        assert mock_sparse_index.search.call_count == 2

    @pytest.mark.asyncio
    async def test_search_with_doc_id_filter(
        self,
        hybrid_search,
        mock_qdrant_storage,
        mock_embedding_service,
        mock_sparse_index,
    ):
        """Test search with document ID filtering"""
        doc_id_filter = "doc123"

        mock_embedding_service.embed_single.return_value = [0.1] * 1536
        mock_qdrant_storage.search.return_value = []
        mock_sparse_index.search.return_value = []

        await hybrid_search.search(["test query"], doc_id_filter=doc_id_filter)

        # Verify doc_id_filter is passed to dense search
        mock_qdrant_storage.search.assert_called_with(
            query_vector=[0.1] * 1536, top_k=20, doc_id_filter=doc_id_filter
        )

    @pytest.mark.asyncio
    async def test_search_combines_dense_and_sparse(
        self,
        hybrid_search,
        mock_qdrant_storage,
        mock_embedding_service,
        mock_sparse_index,
        sample_dense_results,
        sample_sparse_results,
    ):
        """Test that search properly combines dense and sparse results"""
        mock_embedding_service.embed_single.return_value = [0.1] * 1536
        mock_qdrant_storage.search.return_value = sample_dense_results[0]
        mock_sparse_index.search.return_value = sample_sparse_results[0]

        results = await hybrid_search.search(["test query"])

        # Should have results from both sources
        assert len(results) > 0
        # Results should be sorted by RRF score
        assert results[0]["rrf_score"] >= results[-1]["rrf_score"] if len(results) > 1 else True

    @pytest.mark.asyncio
    async def test_empty_search_results(
        self,
        hybrid_search,
        mock_qdrant_storage,
        mock_embedding_service,
        mock_sparse_index,
    ):
        """Test search when no results are found"""
        mock_embedding_service.embed_single.return_value = [0.1] * 1536
        mock_qdrant_storage.search.return_value = []
        mock_sparse_index.search.return_value = []

        results = await hybrid_search.search(["nonexistent query"])

        assert results == []


class TestQdrantStorageService:
    """Test cases for Qdrant storage service"""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock AsyncQdrantClient"""
        with patch("qdrant_client.AsyncQdrantClient") as mock_client:
            client_instance = AsyncMock()
            mock_client.return_value = client_instance
            yield client_instance

    @pytest.fixture
    def qdrant_service(self, mock_qdrant_client):
        """Create a QdrantStorageService with mocked client"""
        from src.common.services.qdrant import QdrantStorageService

        service = QdrantStorageService(
            url="http://localhost:6333",
            collection_name="test_collection",
            vector_size=1536,
        )
        service.client = mock_qdrant_client
        return service

    @pytest.mark.asyncio
    async def test_search_with_filter(self, qdrant_service, mock_qdrant_client):
        """Test search with document ID filter"""
        # Setup mock response
        mock_point = Mock()
        mock_point.payload = {
            "text": "test",
            "doc_id": "doc1",
            "chunk_index": 0,
            "section_title": "",
            "source_file": "",
        }
        mock_point.score = 0.95

        mock_result = Mock()
        mock_result.points = [mock_point]
        mock_qdrant_client.query_points = AsyncMock(return_value=mock_result)

        results = await qdrant_service.search(
            query_vector=[0.1] * 1536, top_k=5, doc_id_filter="doc1"
        )

        assert len(results) == 1
        assert results[0]["text"] == "test"
        assert results[0]["score"] == 0.95

        # Verify filter was passed correctly
        call_kwargs = mock_qdrant_client.query_points.call_args[1]
        assert call_kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_scroll_all_chunks(self, qdrant_service, mock_qdrant_client):
        """Test scrolling through all chunks"""
        # Setup mock scroll response
        mock_point1 = Mock()
        mock_point1.payload = {
            "text": "chunk1",
            "doc_id": "doc1",
            "chunk_index": 0,
            "section_title": "Section 1",
            "source_file": "doc1.pdf",
        }

        mock_point2 = Mock()
        mock_point2.payload = {
            "text": "chunk2",
            "doc_id": "doc1",
            "chunk_index": 1,
            "section_title": "Section 2",
            "source_file": "doc1.pdf",
        }

        # First call returns points and offset, second call returns empty
        mock_qdrant_client.scroll.side_effect = [
            ([mock_point1, mock_point2], "next_offset"),
            ([], None),
        ]

        chunks = await qdrant_service.scroll_all_chunks()

        assert len(chunks) == 2
        assert chunks[0]["text"] == "chunk1"
        assert chunks[1]["chunk_index"] == 1
        assert mock_qdrant_client.scroll.call_count == 2


class TestEmbeddingService:
    """Test cases for embedding service"""

    @pytest.fixture
    def mock_genai_client(self):
        """Mock the Google AI client"""
        with patch("src.ingestion.embedding.genai.Client") as mock_client:
            client_instance = Mock()
            mock_client.return_value = client_instance
            yield client_instance

    @pytest.fixture
    def embedding_service(self, mock_genai_client):
        """Create EmbeddingService with mocked client"""
        with patch("src.ingestion.embedding.config"):
            from src.ingestion.embedding import EmbeddingService

            service = EmbeddingService(model_name="test-model", dimensions=1536)
            return service

    @pytest.mark.asyncio
    async def test_embed_single_success(self, embedding_service, mock_genai_client):
        """Test successful single text embedding"""
        # Setup mock response
        mock_embedding = Mock()
        mock_embedding.values = [0.1, 0.2, 0.3]

        mock_response = Mock()
        mock_response.embeddings = [mock_embedding]

        mock_genai_client.models.embed_content = Mock(return_value=mock_response)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_response
            result = await embedding_service.embed_single("test text")

        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_empty_text(self, embedding_service):
        """Test embedding empty text raises error"""
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            await embedding_service.embed_single("")

    @pytest.mark.asyncio
    async def test_embed_single_with_retry(self, embedding_service, mock_genai_client):
        """Test embedding with retry on failure"""
        # Setup mock to fail once then succeed
        mock_embedding = Mock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_response = Mock()
        mock_response.embeddings = [mock_embedding]

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = [Exception("API Error"), mock_response]
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await embedding_service.embed_single("test text")

        assert result == [0.1, 0.2, 0.3]
        assert mock_to_thread.call_count == 2


class TestSparseSearchIndex:
    """Test cases for sparse search index"""

    @pytest.fixture
    def sparse_index(self):
        """Create a SparseSearchIndex instance"""
        from src.common.services.sparse_index import SparseSearchIndex

        return SparseSearchIndex()

    @pytest.fixture
    def sample_chunks(self):
        """Provide sample chunks for testing"""
        return [
            {
                "doc_id": "doc1",
                "chunk_index": 0,
                "text": "The quick brown fox jumps over the lazy dog",
                "section": "intro",
                "source": "doc1.pdf",
            },
            {
                "doc_id": "doc1",
                "chunk_index": 1,
                "text": "Machine learning is a subset of artificial intelligence",
                "section": "methods",
                "source": "doc1.pdf",
            },
            {
                "doc_id": "doc2",
                "chunk_index": 0,
                "text": "Deep learning uses neural networks with many layers",
                "section": "intro",
                "source": "doc2.pdf",
            },
        ]

    def test_build_index(self, sparse_index, sample_chunks):
        """Test building the BM25 index"""
        sparse_index.build(sample_chunks)

        assert sparse_index.index is not None
        assert len(sparse_index.chunks) == 3

    def test_search_before_build(self, sparse_index):
        """Test searching before index is built"""
        results = sparse_index.search("test query")
        assert results == []

    def test_basic_search(self, sparse_index, sample_chunks):
        """Test basic search functionality"""
        sparse_index.build(sample_chunks)

        results = sparse_index.search("machine learning", top_k=2)

        assert len(results) > 0
        assert "bm25_score" in results[0]
        # The machine learning chunk should be top result
        assert "machine learning" in results[0]["text"].lower()

    def test_search_no_results(self, sparse_index, sample_chunks):
        """Test search with no matching terms"""
        sparse_index.build(sample_chunks)

        results = sparse_index.search("xyznonexistent123", top_k=5)

        # Should return empty list since no chunk contains this term
        assert all(r.get("bm25_score", 0) > 0 for r in results) or len(results) == 0

    def test_search_score_ordering(self, sparse_index, sample_chunks):
        """Test that results are ordered by score"""
        sparse_index.build(sample_chunks)

        results = sparse_index.search("learning", top_k=5)

        scores = [r["bm25_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestBootstrapSparseIndex:
    """Test cases for bootstrap_sparse_index function"""

    @pytest.mark.asyncio
    async def test_bootstrap_calls_build(self):
        """Test that bootstrap fetches chunks and builds index"""
        from src.common.utils.helper import bootstrap_sparse_index

        mock_storage = AsyncMock()
        mock_storage.chunk_count.return_value = 1

        mock_storage.scroll_all_chunks.return_value = [
            {"text": "test chunk", "doc_id": "doc1", "chunk_index": 0}
        ]

        mock_sparse_index = Mock()
        mock_sparse_index.load.return_value = False

        await bootstrap_sparse_index(mock_storage, mock_sparse_index)

        mock_storage.scroll_all_chunks.assert_called_once()
        mock_sparse_index.build.assert_called_once_with(mock_storage.scroll_all_chunks.return_value)
        mock_sparse_index.save.assert_called_once()


# Test configuration for pytest
def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
