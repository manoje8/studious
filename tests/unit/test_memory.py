"""
Unit tests for the memory subsystem.

Covers:
- ShortTermMemoryManager: trim_session()
- EpisodicMemoryManager: compress_session(), save_summary(), load_user_memory()
- QueryRewriter: rewrite() with episodic context
- ConversationSession / EpisodicSummary data models
"""

import pytest
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock

from src.agents.memory.conversation_model import (
    ConversationSession,
    EpisodicSummary,
)
from src.agents.memory.query_rewriter import QueryRewriter
from src.llm.base import LLMResponse

# Data Models


class TestEpisodicSummary:
    """Tests for the EpisodicSummary dataclass."""

    def test_creation_with_defaults(self):
        summary = EpisodicSummary(
            user_id="u1",
            session_id="s1",
            summary="Discussed RAG pipelines.",
        )
        assert summary.user_id == "u1"
        assert summary.topic_tags == []
        assert summary.turn_count == 0
        assert isinstance(summary.created_at, datetime)

    def test_creation_with_all_fields(self):
        now = datetime.now(UTC)
        summary = EpisodicSummary(
            user_id="u1",
            session_id="s1",
            summary="Discussed chunking strategies.",
            topic_tags=["chunking", "RAG"],
            turn_count=8,
            created_at=now,
        )
        assert summary.topic_tags == ["chunking", "RAG"]
        assert summary.turn_count == 8
        assert summary.created_at == now


class TestConversationSession:
    """Tests for ConversationSession memory helpers."""

    def test_add_turn(self):
        session = ConversationSession(session_id="s1", user_id="u1")
        session.add_turn("user", "Hello")
        session.add_turn("assistant", "Hi there!")

        assert len(session.turns) == 2
        assert session.turns[0].role == "user"
        assert session.turns[1].content == "Hi there!"

    def test_get_recent_turns(self):
        session = ConversationSession(session_id="s1", user_id="u1")
        for i in range(10):
            session.add_turn("user", f"Message {i}")

        recent = session.get_recent_turns(3)
        assert len(recent) == 3
        assert recent[0].content == "Message 7"

    def test_to_prompt_format(self):
        session = ConversationSession(session_id="s1", user_id="u1")
        session.add_turn("user", "What is RAG?")
        session.add_turn("assistant", "RAG stands for Retrieval-Augmented Generation.")

        prompt = session.to_prompt_format(n=2)
        assert "User: What is RAG?" in prompt
        assert "Assistant: RAG stands for" in prompt


# ShortTermMemoryManager — trim_session

# class TestShortTermTrimSession:
#     """Tests for ShortTermMemoryManager.trim_session()."""
#
#     @pytest.fixture
#     def manager(self):
#         with patch("src.agents.memory.short_term.redis") as mock_redis:
#             mock_client = MagicMock()
#             mock_client.setex = AsyncMock()
#             mock_redis.from_url.return_value = mock_client
#             mgr = ShortTermMemoryManager(redis_url="redis://fake:6379")
#             return mgr
#
#     def _make_session(self, turn_count: int) -> ConversationSession:
#         session = ConversationSession(session_id="s1", user_id="u1")
#         for i in range(turn_count):
#             role = "user" if i % 2 == 0 else "assistant"
#             session.add_turn(role, f"Message {i}")
#         return session
#
#     @pytest.mark.asyncio
#     async def test_no_trim_when_under_max(self, manager):
#         session = self._make_session(5)
#         overflow = await manager.trim_session(session, max_turns=10)
#
#         assert overflow == []
#         assert len(session.turns) == 5
#
#     @pytest.mark.asyncio
#     async def test_trim_returns_overflow(self, manager):
#         session = self._make_session(15)
#         overflow = await manager.trim_session(session, max_turns=10)
#
#         assert len(overflow) == 5
#         assert len(session.turns) == 10
#
#     @pytest.mark.asyncio
#     async def test_trim_keeps_most_recent(self, manager):
#         session = self._make_session(15)
#         await manager.trim_session(session, max_turns=10)
#
#         # The kept turns should be messages 5-14 (the last 10)
#         assert session.turns[0].content == "Message 5"
#         assert session.turns[-1].content == "Message 14"
#
#     @pytest.mark.asyncio
#     async def test_overflow_has_oldest(self, manager):
#         session = self._make_session(15)
#         overflow = await manager.trim_session(session, max_turns=10)
#
#         # Overflow should be messages 0-4 (the oldest 5)
#         assert overflow[0].content == "Message 0"
#         assert overflow[-1].content == "Message 4"
#
#     @pytest.mark.asyncio
#     async def test_trim_at_exact_max_no_overflow(self, manager):
#         session = self._make_session(10)
#         overflow = await manager.trim_session(session, max_turns=10)
#
#         assert overflow == []
#         assert len(session.turns) == 10


# ---------------------------------------------------------------------------
# EpisodicMemoryManager
# ---------------------------------------------------------------------------


# class TestEpisodicMemoryManager:
#     """Tests for EpisodicMemoryManager."""
#
#     @pytest.fixture
#     def mock_llm(self):
#         m = MagicMock()
#         response = MagicMock(spec=LLMResponse)
#         response.parsed_json = {
#             "summary": "User asked about RAG pipeline architecture and chunking.",
#             "topic_tags": ["RAG", "chunking", "architecture"],
#         }
#         m.complete = AsyncMock(return_value=response)
#         return m
#
#     @pytest.fixture
#     def mock_pool(self):
#         pool = MagicMock()
#         conn = AsyncMock()
#         cursor = AsyncMock()
#         cursor.fetchall = AsyncMock(return_value=[])
#         conn.execute = AsyncMock(return_value=cursor)
#         conn.__aenter__ = AsyncMock(return_value=conn)
#         conn.__aexit__ = AsyncMock(return_value=False)
#         pool.connection = MagicMock(return_value=conn)
#         return pool
#
#     @pytest.fixture
#     def episodic(self, mock_llm, mock_pool):
#         return EpisodicMemoryManager(llm_client=mock_llm, pool=mock_pool)
#
#     def _make_session(self, turn_count: int = 8) -> ConversationSession:
#         session = ConversationSession(session_id="s1", user_id="u1")
#         for i in range(turn_count):
#             role = "user" if i % 2 == 0 else "assistant"
#             session.add_turn(role, f"Turn {i}")
#         return session
#
#     # --- compress_session ---
#
#     @pytest.mark.asyncio
#     async def test_compress_returns_episodic_summary(self, episodic):
#         session = self._make_session()
#         result = await episodic.compress_session(session)
#
#         assert isinstance(result, EpisodicSummary)
#         assert result.user_id == "u1"
#         assert result.session_id == "s1"
#         assert result.turn_count == 8
#
#     @pytest.mark.asyncio
#     async def test_compress_summary_text(self, episodic):
#         session = self._make_session()
#         result = await episodic.compress_session(session)
#
#         assert "RAG" in result.summary
#         assert "chunking" in result.summary
#
#     @pytest.mark.asyncio
#     async def test_compress_topic_tags(self, episodic):
#         session = self._make_session()
#         result = await episodic.compress_session(session)
#
#         assert "RAG" in result.topic_tags
#         assert "architecture" in result.topic_tags
#
#     @pytest.mark.asyncio
#     async def test_compress_calls_llm(self, episodic, mock_llm):
#         session = self._make_session()
#         await episodic.compress_session(session)
#
#         mock_llm.complete.assert_awaited_once()
#         prompt = mock_llm.complete.call_args[0][0]
#         assert "compress" in prompt.lower() or "summary" in prompt.lower()
#
#     @pytest.mark.asyncio
#     async def test_compress_prompt_includes_transcript(self, episodic, mock_llm):
#         session = self._make_session(4)
#         await episodic.compress_session(session)
#
#         prompt = mock_llm.complete.call_args[0][0]
#         assert "Turn 0" in prompt
#         assert "Turn 3" in prompt
#
#     # --- save_summary ---
#
#     @pytest.mark.asyncio
#     async def test_save_executes_insert(self, episodic, mock_pool):
#         summary = EpisodicSummary(
#             user_id="u1",
#             session_id="s1",
#             summary="Test summary",
#             topic_tags=["test"],
#             turn_count=4,
#         )
#         await episodic.save_summary(summary)
#
#         conn = mock_pool.connection.return_value.__aenter__.return_value
#         conn.execute.assert_awaited_once()
#
#         call_args = conn.execute.call_args
#         sql = call_args[0][0]
#         params = call_args[0][1]
#         assert "INSERT" in sql
#         assert params["user_id"] == "u1"
#         assert params["summary"] == "Test summary"
#
#     # --- load_user_memory ---
#
#     @pytest.mark.asyncio
#     async def test_load_returns_empty_string_when_no_memories(self, episodic):
#         result = await episodic.load_user_memory("u1")
#         assert result == ""
#
#     @pytest.mark.asyncio
#     async def test_load_returns_formatted_memories(self, episodic, mock_pool):
#         rows = [
#             {
#                 "summary": "Discussed chunking strategies.",
#                 "topic_tags": ["chunking"],
#                 "created_at": datetime(2026, 6, 20, 10, 0, tzinfo=UTC),
#             },
#             {
#                 "summary": "Asked about re-ranking models.",
#                 "topic_tags": ["reranking", "models"],
#                 "created_at": datetime(2026, 6, 19, 15, 30, tzinfo=UTC),
#             },
#         ]
#
#         conn = mock_pool.connection.return_value.__aenter__.return_value
#         cursor = AsyncMock()
#         cursor.fetchall = AsyncMock(return_value=rows)
#         conn.execute = AsyncMock(return_value=cursor)
#
#         result = await episodic.load_user_memory("u1", limit=5)
#
#         assert "chunking" in result
#         assert "re-ranking" in result
#         assert "2026-06-20" in result
#
#     @pytest.mark.asyncio
#     async def test_load_passes_correct_params(self, episodic, mock_pool):
#         await episodic.load_user_memory("user-42", limit=3)
#
#         conn = mock_pool.connection.return_value.__aenter__.return_value
#         call_args = conn.execute.call_args[0]
#         params = call_args[1]
#         assert params["user_id"] == "user-42"
#         assert params["limit"] == 3
#
#     # --- setup ---
#
#     @pytest.mark.asyncio
#     async def test_setup_creates_table(self, episodic, mock_pool):
#         await episodic.setup()
#
#         conn = mock_pool.connection.return_value.__aenter__.return_value
#         # Should execute CREATE TABLE + 2 CREATE INDEX statements
#         assert conn.execute.await_count == 3
#
#         sqls = [call[0][0] for call in conn.execute.call_args_list]
#         assert any("CREATE TABLE" in sql for sql in sqls)
#         assert any("CREATE INDEX" in sql for sql in sqls)


# QueryRewriter — episodic context integration


class TestQueryRewriterWithEpisodicContext:
    """Tests for QueryRewriter.rewrite() with episodic context."""

    @pytest.fixture
    def mock_llm(self):
        m = MagicMock()
        response = MagicMock(spec=LLMResponse)
        response.parsed_json = {
            "rewritten_query": "What were the Q3 memory usage statistics?",
            "was_rewritten": True,
            "resolved_references": ["that report → Q3 memory usage report"],
        }
        m.complete = AsyncMock(return_value=response)
        return m

    @pytest.fixture
    def rewriter(self, mock_llm):
        return QueryRewriter(llm_client=mock_llm)

    def _make_session(self) -> ConversationSession:
        session = ConversationSession(session_id="s1", user_id="u1")
        session.add_turn("user", "Show me the Q3 memory usage report")
        session.add_turn("assistant", "Q3 memory usage was 84%.")
        return session

    @pytest.mark.asyncio
    async def test_rewrite_without_episodic(self, rewriter, mock_llm):
        session = self._make_session()
        result = await rewriter.rewrite("What about that report?", session)

        assert result["was_rewritten"] is True
        prompt = mock_llm.complete.call_args[0][0]
        # Should not contain long-term context section
        assert "Long-term context" not in prompt or "previous sessions" not in prompt

    @pytest.mark.asyncio
    async def test_no_session_returns_unchanged(self, rewriter):
        session = ConversationSession(session_id="s1", user_id="u1")
        result = await rewriter.rewrite("standalone question", session)

        assert result["rewritten_query"] == "standalone question"
        assert result["was_rewritten"] is False

    @pytest.mark.asyncio
    async def test_empty_session_returns_unchanged(self, rewriter):
        session = ConversationSession(session_id="s1", user_id="u1")
        result = await rewriter.rewrite("standalone question", session)

        assert result["rewritten_query"] == "standalone question"
        assert result["was_rewritten"] is False

    @pytest.mark.asyncio
    async def test_empty_episodic_context_omits_section(self, rewriter, mock_llm):
        session = self._make_session()
        await rewriter.rewrite("question", session)

        prompt = mock_llm.complete.call_args[0][0]
        assert "Long-term context" not in prompt
