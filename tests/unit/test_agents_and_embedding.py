"""
Unit tests for the ingestion agents layer.

Covers:
- QueryExpander: expand() — happy path, prompt content, error propagation
- RetrievalAgent: retrieve(), retrieve_and_evaluate(), generate_refined_query()
- EmbeddedChunk: to_qdrant_point()
- EmbeddingService: embed_single(), embed_chunks() — batching, retries
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.query_expander import QueryExpander
from src.agents.retrieval import RetrievalAgent
from src.agents.agent_model import RetrievalDecision, RetrievalRound
from src.llm.base import LLMResponse
from src.ingestion.embedding import EmbeddedChunk, EmbeddingService
from src.ingestion.chunking.chunk import Chunk


def _base_state(**overrides) -> dict:
    state: dict = {
        "session_id": "sess-1",
        "user_id": "u1",
        "original_message": "What is RAG?",
        "effective_query": "What is RAG?",
        "current_query": "What is RAG?",
        "was_rewritten": False,
        "question_category": "factual",
        "sub_questions": ["What is RAG?"],
        "current_sub_question_idx": 0,
        "retrieval_round": 0,
        "max_retrieval_rounds": 3,
        "retrieval_history": [],
        "accepted_chunks": [],
        "final_answer": "",
        "sources": [],
        "doc_id_filter": None,
        "resolved_references": [],
    }
    state.update(overrides)
    return state


def _make_chunk(text="sample text", doc_id="d1", chunk_index=0) -> Chunk:
    return Chunk(
        text=text,
        chunk_index=chunk_index,
        doc_id=doc_id,
        source_file="test.pdf",
        chunk_type="fixed",
        token_count=10,
    )


# QueryExpander


class TestQueryExpander:
    """Tests for QueryExpander.expand()."""

    @pytest.fixture
    def mock_llm(self):
        m = MagicMock()
        response = MagicMock(spec=LLMResponse)
        response.parsed_json = [
            "Alternative phrasing one",
            "Alternative phrasing two",
            "Alternative phrasing three",
        ]
        m.complete = AsyncMock(return_value=response)
        return m

    @pytest.fixture
    def expander(self, mock_llm):
        return QueryExpander(llm_client=mock_llm)

    @pytest.mark.asyncio
    async def test_returns_list_of_strings(self, expander):
        result = await expander.expand("What is vector search?")
        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)

    @pytest.mark.asyncio
    async def test_includes_original_query(self, expander):
        original = "What is vector search?"
        result = await expander.expand(original)
        assert original in result

    @pytest.mark.asyncio
    async def test_total_queries_count(self, expander):
        """Original + 3 alternatives = 4 queries."""
        result = await expander.expand("test query")
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_original_query_is_first(self, expander):
        original = "first query"
        result = await expander.expand(original)
        assert result[0] == original

    @pytest.mark.asyncio
    async def test_llm_called_with_prompt_containing_query(self, expander, mock_llm):
        await expander.expand("my specific question")

        call_args = mock_llm.complete.call_args[0]
        assert "my specific question" in call_args[0]

    @pytest.mark.asyncio
    async def test_llm_called_once(self, expander, mock_llm):
        await expander.expand("query")
        mock_llm.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_propagates_llm_error(self, mock_llm):
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM failure"))
        expander = QueryExpander(llm_client=mock_llm)

        with pytest.raises(RuntimeError, match="LLM failure"):
            await expander.expand("query")

    @pytest.mark.asyncio
    async def test_handles_empty_alternatives(self, mock_llm):
        """When LLM returns empty list, only original is returned."""
        response = MagicMock(spec=LLMResponse)
        response.parsed_json = []
        mock_llm.complete = AsyncMock(return_value=response)

        expander = QueryExpander(llm_client=mock_llm)
        result = await expander.expand("original query")

        assert result == ["original query"]


# RetrievalAgent


class TestRetrievalAgent:
    """Tests for RetrievalAgent."""

    @pytest.fixture
    def mock_llm(self):
        m = MagicMock()
        response = MagicMock(spec=LLMResponse)
        response.parsed_json = {
            "scores": [0.9],
            "decision": "sufficient",
            "reasoning": "Very relevant chunks found",
            "refined_query": None,
        }
        response.text = "better search query"
        m.complete = AsyncMock(return_value=response)
        return m

    @pytest.fixture
    def mock_hybrid_search(self):
        m = MagicMock()
        m.search = AsyncMock(
            return_value=[
                {
                    "text": "RAG chunk",
                    "source": "doc.pdf",
                    "rrf_score": 0.9,
                    "doc_id": "d1",
                    "chunk_index": 0,
                }
            ]
        )
        return m

    @pytest.fixture
    def mock_reranker(self):
        m = MagicMock()
        m.rerank = AsyncMock(
            return_value=[
                {
                    "text": "RAG chunk",
                    "source": "doc.pdf",
                    "rrf_score": 0.9,
                    "doc_id": "d1",
                    "chunk_index": 0,
                }
            ]
        )
        return m

    @pytest.fixture
    def mock_query_expand(self):
        m = MagicMock()
        m.expand = AsyncMock(
            return_value=["What is RAG?", "Retrieval-Augmented Generation"]
        )
        return m

    @pytest.fixture
    def agent(self, mock_llm, mock_hybrid_search, mock_reranker, mock_query_expand):
        return RetrievalAgent(
            llm_client=mock_llm,
            hybrid_search=mock_hybrid_search,
            reranker=mock_reranker,
            query_expand=mock_query_expand,
        )

    # --- retrieve ---

    @pytest.mark.asyncio
    async def test_retrieve_returns_list(self, agent):
        results = await agent.retrieve("query", "original question")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_calls_query_expand(self, agent, mock_query_expand):
        await agent.retrieve("query", "original question")
        mock_query_expand.expand.assert_awaited_once_with("query")

    @pytest.mark.asyncio
    async def test_retrieve_calls_hybrid_search_with_expanded_queries(
        self, agent, mock_hybrid_search, mock_query_expand
    ):
        expanded = ["q1", "q2"]
        mock_query_expand.expand = AsyncMock(return_value=expanded)

        await agent.retrieve("query", "original", doc_id_filter="doc-1")

        mock_hybrid_search.search.assert_awaited_once_with(
            queries=expanded, doc_id_filter="doc-1"
        )

    @pytest.mark.asyncio
    async def test_retrieve_calls_reranker_with_candidates(
        self, agent, mock_reranker, mock_hybrid_search
    ):
        candidates = [{"text": "c1", "rrf_score": 0.8}]
        mock_hybrid_search.search = AsyncMock(return_value=candidates)

        await agent.retrieve("query", "original question")

        mock_reranker.rerank.assert_awaited_once_with(
            query="original question", candidates=candidates
        )

    # --- retrieve_and_evaluate ---

    @pytest.mark.asyncio
    async def test_retrieve_and_evaluate_returns_retrieval_round(self, agent):
        state = _base_state()
        result = await agent.retrieve_and_evaluate(
            query="query", original_question="question", state=state
        )
        assert isinstance(result, RetrievalRound)

    @pytest.mark.asyncio
    async def test_retrieve_and_evaluate_sets_decision(self, agent, mock_llm):
        mock_llm.complete = AsyncMock(
            return_value=MagicMock(
                parsed_json={
                    "scores": [0.9],
                    "decision": "sufficient",
                    "reasoning": "Good",
                    "refined_query": None,
                }
            )
        )
        state = _base_state()
        result = await agent.retrieve_and_evaluate("q", "q", state)
        assert result.decision == RetrievalDecision.SUFFICIENT

    @pytest.mark.asyncio
    async def test_retrieve_and_evaluate_empty_results_returns_refine(
        self, agent, mock_hybrid_search, mock_reranker
    ):
        """When no results come back, decision should be REFINE_QUERY."""
        mock_hybrid_search.search = AsyncMock(return_value=[])
        mock_reranker.rerank = AsyncMock(return_value=[])

        state = _base_state()
        result = await agent.retrieve_and_evaluate("q", "q", state)

        assert result.decision == RetrievalDecision.REFINE_QUERY
        assert result.chunk_retrieved == []

    @pytest.mark.asyncio
    async def test_retrieve_and_evaluate_populates_rrf_scores(self, agent, mock_llm):
        chunks = [
            {"text": "x", "rrf_score": 0.77, "source": "a.pdf"},
            {"text": "y", "rrf_score": 0.55, "source": "b.pdf"},
        ]
        agent.reranker.rerank = AsyncMock(return_value=chunks)
        mock_llm.complete = AsyncMock(
            return_value=MagicMock(
                parsed_json={
                    "scores": [0.9, 0.7],
                    "decision": "sufficient",
                    "reasoning": "Ok",
                    "refined_query": None,
                }
            )
        )

        state = _base_state()
        result = await agent.retrieve_and_evaluate("q", "q", state)

        assert result.relevance_score == [0.77, 0.55]

    # --- generate_refined_query ---

    def _make_retrieval_round(self, query="prev query", reasoning="not enough"):
        return RetrievalRound(
            query_used=query,
            chunk_retrieved=[],
            relevance_score=[],
            decision=RetrievalDecision.REFINE_QUERY,
            reasoning=reasoning,
        )

    @pytest.mark.asyncio
    async def test_generate_refined_query_returns_string(self, agent):
        rounds = [self._make_retrieval_round()]
        result = await agent.generate_refined_query(
            original_question="What is RAG?", previous_rounds=rounds
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_refined_query_calls_llm(self, agent, mock_llm):
        rounds = [self._make_retrieval_round()]
        await agent.generate_refined_query("question", previous_rounds=rounds)
        mock_llm.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_refined_query_prompt_contains_question(
        self, agent, mock_llm
    ):
        rounds = [self._make_retrieval_round()]
        await agent.generate_refined_query(
            "original question text", previous_rounds=rounds
        )
        prompt = mock_llm.complete.call_args[0][0]
        assert "original question text" in prompt


# EmbeddedChunk


class TestEmbeddedChunk:
    """Tests for EmbeddedChunk.to_qdrant_point()."""

    def test_to_qdrant_point_structure(self):
        chunk = _make_chunk()
        ec = EmbeddedChunk(chunk=chunk, vector=[0.1, 0.2, 0.3], model_name="embed-v1")

        point = ec.to_qdrant_point("test-point-id")

        assert point.id == "test-point-id"
        assert point.vector == [0.1, 0.2, 0.3]

    def test_to_qdrant_point_payload_has_text(self):
        chunk = _make_chunk(text="hello world")
        ec = EmbeddedChunk(chunk=chunk, vector=[0.0], model_name="m1")
        point = ec.to_qdrant_point("pid-1")

        assert point.payload["text"] == "hello world"

    def test_to_qdrant_point_payload_includes_embedding_model(self):
        chunk = _make_chunk()
        ec = EmbeddedChunk(chunk=chunk, vector=[0.5], model_name="text-embedding-004")
        point = ec.to_qdrant_point("pid-2")

        assert point.payload["embedding_model"] == "text-embedding-004"

    def test_to_qdrant_point_payload_has_doc_id(self):
        chunk = _make_chunk(doc_id="my-doc")
        ec = EmbeddedChunk(chunk=chunk, vector=[0.1], model_name="m1")
        point = ec.to_qdrant_point("pid-3")

        assert point.payload["doc_id"] == "my-doc"


# EmbeddingService


class TestEmbeddingService:
    """Tests for EmbeddingService.embed_single() and .embed_chunks()."""

    @pytest.fixture
    def mock_genai(self):
        with (
            patch("src.ingestion.embedding.genai") as mock_genai_module,
            patch("src.ingestion.embedding.config") as mock_config,
            patch("src.ingestion.embedding.logfire"),
        ):
            mock_config.PROJECT_ID = "test-proj"
            mock_config.LOCATION = "us-central1"
            mock_config.VECTOR_SIZE = 4

            mock_client = MagicMock()
            mock_genai_module.Client.return_value = mock_client
            mock_genai_module.types = MagicMock()

            yield mock_client

    @pytest.fixture
    def service(self, mock_genai):
        return EmbeddingService(
            model_name="text-embedding-004",
            dimensions=4,
            batch_size=2,
            max_retries=2,
            retry_delay=0.0,  # No real delay in tests
        )

    def _mock_embed_response(self, mock_client, vectors: list[list[float]]):
        """Configure mock_client.models.embed_content to return given vectors."""
        response = MagicMock()
        response.embeddings = [MagicMock(values=v) for v in vectors]
        mock_client.models.embed_content.return_value = response

    # --- embed_single ---

    @pytest.mark.asyncio
    async def test_embed_single_returns_list_of_floats(self, service, mock_genai):
        self._mock_embed_response(mock_genai, [[0.1, 0.2, 0.3, 0.4]])

        result = await service.embed_single("test text")

        assert isinstance(result, list)
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_embed_single_empty_text_raises(self, service):
        with pytest.raises(ValueError, match="Cannot embed empty text"):
            await service.embed_single("   ")

    @pytest.mark.asyncio
    async def test_embed_single_strips_whitespace(self, service, mock_genai):
        self._mock_embed_response(mock_genai, [[0.1, 0.2, 0.3, 0.4]])
        # Should strip, not raise
        result = await service.embed_single("  valid text  ")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_embed_single_retries_on_failure(self, service, mock_genai):
        """First call fails, second succeeds — retry logic is triggered."""
        response = MagicMock()
        response.embeddings = [MagicMock(values=[0.1, 0.2, 0.3, 0.4])]

        mock_genai.models.embed_content.side_effect = [
            RuntimeError("transient error"),
            response,
        ]

        result = await service.embed_single("text")
        assert result == [0.1, 0.2, 0.3, 0.4]
        assert mock_genai.models.embed_content.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_single_raises_after_max_retries(self, service, mock_genai):
        mock_genai.models.embed_content.side_effect = RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            await service.embed_single("text")

    # --- embed_chunks ---

    @pytest.mark.asyncio
    async def test_embed_chunks_returns_embedded_chunks(self, service, mock_genai):
        chunks = [_make_chunk(text=f"chunk {i}", chunk_index=i) for i in range(2)]
        self._mock_embed_response(
            mock_genai, [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        )

        result = await service.embed_chunks(chunks)

        assert len(result) == 2
        assert all(isinstance(r, EmbeddedChunk) for r in result)

    @pytest.mark.asyncio
    async def test_embed_chunks_empty_input_returns_empty(self, service):
        result = await service.embed_chunks([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_chunks_batches_correctly(self, service, mock_genai):
        """With batch_size=2 and 5 chunks, embed_content is called 3 times."""
        chunks = [_make_chunk(text=f"text{i}", chunk_index=i) for i in range(5)]

        # Each call returns the right number of embeddings
        def side_effect(*args, **kwargs):
            contents = kwargs.get("contents", args[1] if len(args) > 1 else [])
            n = len(contents) if isinstance(contents, list) else 1
            resp = MagicMock()
            resp.embeddings = [MagicMock(values=[0.1] * 4) for _ in range(n)]
            return resp

        mock_genai.models.embed_content.side_effect = side_effect

        result = await service.embed_chunks(chunks)

        assert len(result) == 5
        assert mock_genai.models.embed_content.call_count == 3  # ceil(5/2)

    @pytest.mark.asyncio
    async def test_embed_chunks_sets_model_name(self, service, mock_genai):
        chunks = [_make_chunk()]
        self._mock_embed_response(mock_genai, [[0.1, 0.2, 0.3, 0.4]])

        result = await service.embed_chunks(chunks)

        assert result[0].model_name == "text-embedding-004"

    @pytest.mark.asyncio
    async def test_embed_chunks_preserves_chunk_reference(self, service, mock_genai):
        chunk = _make_chunk(text="preserved", doc_id="d42")
        self._mock_embed_response(mock_genai, [[0.1, 0.2, 0.3, 0.4]])

        result = await service.embed_chunks([chunk])

        assert result[0].chunk is chunk

    @pytest.mark.asyncio
    async def test_vector_size_property(self, service):
        assert service.vector_size == 4
