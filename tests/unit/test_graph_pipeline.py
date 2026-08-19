"""
Unit tests for GraphPipeline (src/agents/graph/runner.py).

All external dependencies (graph, short_term_memory, semantic_cache, llm_clients)
are mocked — no real Redis, Postgres, or LLM calls are made.

Covers:
- chat() creates a new session when session_id is empty
- chat() reuses an existing session returned by short_term.get_session
- chat() creates a new session when get_session returns None
- chat() returns cache hit when semantic_cache.lookup returns a result
- chat() bypasses cache when semantic_cache is None
- chat() handles semantic_cache lookup failure gracefully
- chat() stores result in semantic_cache after successful graph run
- chat() handles semantic_cache store failure gracefully
- chat() aggregates token usage from multiple llm_clients
- chat() appends assistant turn to short_term memory after graph run
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from medici.agents.graph.runner import GraphPipeline
from medici.agents.memory.conversation_model import ConversationSession


def _make_session(session_id: str = "sess-1", user_id: str = "user-1") -> ConversationSession:
    return ConversationSession(session_id=session_id, user_id=user_id)


def _make_graph_result(answer: str = "Test answer") -> dict:
    return {
        "final_answer": answer,
        "sources": ["doc1"],
        "was_rewritten": False,
        "current_hop": 0,
    }


def _make_llm_client(model: str = "gemini", calls: int = 2, total_tokens: int = 500):
    client = MagicMock()
    client.reset_usage = MagicMock()
    client.usage_snapshot = MagicMock(
        return_value={"model": model, "calls": calls, "total_tokens": total_tokens}
    )
    return client


@pytest.fixture
def mock_graph():
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=_make_graph_result())
    return graph


@pytest.fixture
def mock_short_term():
    stm = MagicMock()
    stm.get_session = AsyncMock(return_value=None)
    stm.create_session = AsyncMock(return_value=_make_session())
    stm.append_turn = AsyncMock()
    return stm


@pytest.fixture
def pipeline(mock_graph, mock_short_term):
    return GraphPipeline(
        graph=mock_graph,
        short_term_memory=mock_short_term,
        semantic_cache=None,
        llm_clients=[],
    )


class TestSessionHandling:
    async def test_empty_session_id_creates_new_session(self, pipeline, mock_short_term):
        await pipeline.chat("hello", session_id="", user_id="u1")
        # create_session must be called (since get_session returns None)
        mock_short_term.create_session.assert_called_once()
        call_args = mock_short_term.create_session.call_args
        assert call_args[0][0] == "u1"
        assert "session_id" in call_args[1]
        assert call_args[1]["session_id"].startswith("u1_")

    async def test_existing_session_returned_by_get_session_is_reused(
        self, pipeline, mock_short_term
    ):
        existing = _make_session("existing-sess", "u1")
        mock_short_term.get_session.return_value = existing

        result = await pipeline.chat("hello", session_id="existing-sess", user_id="u1")

        mock_short_term.create_session.assert_not_called()
        assert result["session_id"] == "existing-sess"

    async def test_none_from_get_session_triggers_create_session(self, pipeline, mock_short_term):
        mock_short_term.get_session.return_value = None
        await pipeline.chat("hello", session_id="old-id", user_id="u1")
        mock_short_term.create_session.assert_called_once()

    async def test_append_turn_called_after_graph_run(self, pipeline, mock_short_term):
        mock_short_term.get_session.return_value = _make_session()
        await pipeline.chat("hello", session_id="s1", user_id="u1")
        assert mock_short_term.append_turn.call_count == 2


class TestResponseStructure:
    async def test_response_has_required_keys(self, pipeline, mock_short_term):
        mock_short_term.get_session.return_value = _make_session()
        result = await pipeline.chat("What is AI?", session_id="s1", user_id="u1")
        for key in ("answer", "session_id", "sources", "query_was_rewritten", "retrieval_hops"):
            assert key in result, f"Missing key: {key}"

    async def test_cache_hit_is_false_when_no_cache(self, pipeline, mock_short_term):
        mock_short_term.get_session.return_value = _make_session()
        result = await pipeline.chat("Q", session_id="s1", user_id="u1")
        assert result["cache_hit"] is False

    async def test_answer_matches_graph_output(self, mock_graph, mock_short_term):
        mock_graph.ainvoke.return_value = _make_graph_result(answer="Specific answer")
        mock_short_term.get_session.return_value = _make_session()
        p = GraphPipeline(mock_graph, mock_short_term)
        result = await p.chat("Q", session_id="s1", user_id="u1")
        assert result["answer"] == "Specific answer"


class TestSemanticCacheHit:
    async def test_cache_hit_returns_cached_answer(self, mock_graph, mock_short_term):
        from dataclasses import dataclass

        @dataclass
        class FakeCacheEntry:
            answer: str = "Cached answer"
            sources: list = None
            similarity: float = 0.95
            token_usage: dict = None

            def __post_init__(self):
                if self.sources is None:
                    self.sources = []
                if self.token_usage is None:
                    self.token_usage = {}

        cache = MagicMock()
        cache.lookup = AsyncMock(return_value=FakeCacheEntry())
        cache.store = AsyncMock()
        mock_short_term.get_session.return_value = _make_session()

        p = GraphPipeline(mock_graph, mock_short_term, semantic_cache=cache)
        result = await p.chat("Q", session_id="s1", user_id="u1")

        assert result["answer"] == "Cached answer"
        assert result["cache_hit"] is True
        # Graph should NOT have been called
        mock_graph.ainvoke.assert_not_called()

    async def test_cache_miss_runs_graph(self, mock_graph, mock_short_term):
        cache = MagicMock()
        cache.lookup = AsyncMock(return_value=None)
        cache.store = AsyncMock()
        mock_short_term.get_session.return_value = _make_session()

        p = GraphPipeline(mock_graph, mock_short_term, semantic_cache=cache)
        result = await p.chat("Q", session_id="s1", user_id="u1")

        assert result["cache_hit"] is False
        mock_graph.ainvoke.assert_called_once()

    async def test_cache_lookup_failure_runs_graph(self, mock_graph, mock_short_term):
        """Cache errors must not break the pipeline."""
        cache = MagicMock()
        cache.lookup = AsyncMock(side_effect=RuntimeError("Redis down"))
        cache.store = AsyncMock()
        mock_short_term.get_session.return_value = _make_session()

        p = GraphPipeline(mock_graph, mock_short_term, semantic_cache=cache)
        result = await p.chat("Q", session_id="s1", user_id="u1")

        # Pipeline still returns a result
        assert "answer" in result
        mock_graph.ainvoke.assert_called_once()

    async def test_cache_store_failure_does_not_raise(self, mock_graph, mock_short_term):
        """Cache store errors must not break the pipeline."""
        cache = MagicMock()
        cache.lookup = AsyncMock(return_value=None)
        cache.store = AsyncMock(side_effect=RuntimeError("Redis down"))
        mock_short_term.get_session.return_value = _make_session()

        p = GraphPipeline(mock_graph, mock_short_term, semantic_cache=cache)
        result = await p.chat("Q", session_id="s1", user_id="u1")
        assert "answer" in result


class TestTokenUsageAggregation:
    async def test_token_usage_aggregated_from_clients(self, mock_graph, mock_short_term):
        mock_short_term.get_session.return_value = _make_session()
        client1 = _make_llm_client("gemini", calls=3, total_tokens=300)
        client2 = _make_llm_client("groq", calls=2, total_tokens=200)

        p = GraphPipeline(mock_graph, mock_short_term, llm_clients=[client1, client2])
        result = await p.chat("Q", session_id="s1", user_id="u1")

        assert "gemini" in result["token_usage"]
        assert "groq" in result["token_usage"]

    async def test_reset_usage_called_on_all_clients(self, mock_graph, mock_short_term):
        mock_short_term.get_session.return_value = _make_session()
        client1 = _make_llm_client("gemini")
        client2 = _make_llm_client("groq")

        p = GraphPipeline(mock_graph, mock_short_term, llm_clients=[client1, client2])
        await p.chat("Q", session_id="s1", user_id="u1")

        client1.reset_usage.assert_called_once()
        client2.reset_usage.assert_called_once()
