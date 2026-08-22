"""
Integration tests: session_id round-trip seam.

Seam under test: GraphPipeline.chat() ↔ ShortTermMemoryManager.

The contract being verified:
  1. chat() calls short_term.get_session(session_id) — if None, creates a new one
  2. The session_id embedded into graph initial_state MUST equal the session_id
     returned from memory, not the caller's (possibly expired) session_id
  3. The session_id returned in chat()'s response dict must be the one stored
     in Redis (so the next call can find it)

These tests use AsyncMock at the Redis client layer so no real Redis is needed.
"""

import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from medici.agents.graph.runner import GraphPipeline
from medici.agents.memory.conversation_model import ConversationSession
from medici.agents.memory.short_term import ShortTermMemoryManager


def _make_session(user_id: str = "user-1") -> ConversationSession:
    return ConversationSession(session_id=str(uuid.uuid4()), user_id=user_id)


def _make_manager_with_mock_redis() -> tuple[ShortTermMemoryManager, AsyncMock]:
    """Return (manager, mock_redis_client) with redis patched at instantiation."""
    mock_redis = AsyncMock()
    with patch("medici.agents.memory.short_term.redis") as mock_redis_mod:
        mock_redis_mod.from_url.return_value = mock_redis
        manager = ShortTermMemoryManager(redis_url="redis://localhost:6379")
    return manager, mock_redis


@pytest.mark.integration
class TestShortTermMemoryManagerContract:
    """
    Verify that ShortTermMemoryManager's create/get/append contract is internally
    self-consistent. Uses AsyncMock at the redis client boundary.
    """

    @pytest.mark.asyncio
    async def test_create_session_returns_session_with_generated_id(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        mock_redis.setex = AsyncMock(return_value=True)

        session = await manager.create_session("user-42")

        assert isinstance(session, ConversationSession)
        assert session.session_id  # not empty
        assert session.user_id == "user-42"

    @pytest.mark.asyncio
    async def test_create_session_stores_in_redis_with_correct_key(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        mock_redis.setex = AsyncMock(return_value=True)

        session = await manager.create_session("user-1")

        mock_redis.setex.assert_awaited_once()
        call_args = mock_redis.setex.call_args
        key_used = call_args[0][0]
        # Key format must be "session:<session_id>" — what get_session() looks up
        assert key_used == f"session:{session.session_id}", (
            f"Redis key '{key_used}' does not match get_session lookup format. "
            "session_id round-trip will fail."
        )

    @pytest.mark.asyncio
    async def test_get_session_returns_none_for_unknown_id(self):
        manager, mock_redis = _make_manager_with_mock_redis()
        mock_redis.get = AsyncMock(return_value=None)

        result = await manager.get_session("nonexistent-session-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_session_reconstructs_stored_session(self):
        """get_session() must return a ConversationSession with the correct session_id."""
        manager, mock_redis = _make_manager_with_mock_redis()

        stored_session = _make_session("user-1")
        stored_data = json.dumps(
            {
                "session_id": stored_session.session_id,
                "user_id": stored_session.user_id,
                "turns": [],
            }
        ).encode()
        mock_redis.get = AsyncMock(return_value=stored_data)

        retrieved = await manager.get_session(stored_session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == stored_session.session_id
        assert retrieved.user_id == stored_session.user_id

    @pytest.mark.asyncio
    async def test_roundtrip_create_then_get_returns_same_session_id(self):
        """
        Core session_id contract: the session_id stored during create_session()
        must be recoverable via get_session() using the same id.
        This is the seam: runner.py uses the session_id from memory to key the
        LangGraph thread, so any inconsistency here breaks conversation continuity.
        """
        manager, mock_redis = _make_manager_with_mock_redis()

        captured_key = None
        captured_value = None

        async def capture_setex(key, ttl, value):
            nonlocal captured_key, captured_value
            captured_key = key
            captured_value = value

        mock_redis.setex = AsyncMock(side_effect=capture_setex)

        # Step 1: create
        session = await manager.create_session("user-1")
        original_id = session.session_id

        # Step 2: simulate get with the stored data
        mock_redis.get = AsyncMock(return_value=captured_value)
        retrieved = await manager.get_session(original_id)

        assert retrieved is not None
        assert retrieved.session_id == original_id, (
            "Round-trip failed: created session_id does not match retrieved session_id. "
            "GraphPipeline.chat() would inject a different id into graph state than "
            "what was stored in Redis."
        )


@pytest.mark.integration
class TestGraphPipelineSessionIdConsistency:
    """
    Verify that GraphPipeline.chat() correctly threads the session_id from
    ShortTermMemoryManager into the graph's initial_state and the response dict.
    """

    def _make_pipeline(self, short_term: ShortTermMemoryManager) -> GraphPipeline:
        """Build a GraphPipeline with a mock graph that records its invocation state."""
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "final_answer": "Test answer",
                "sources": [],
                "was_rewritten": False,
                "retrieval_round": 1,
            }
        )
        return GraphPipeline(graph=mock_graph, short_term_memory=short_term)

    @pytest.mark.asyncio
    async def test_response_session_id_matches_memory_session_id(self):
        """
        The session_id in chat()'s return value must match the session created
        by ShortTermMemoryManager, not the original caller-provided id.
        """
        manager, mock_redis = _make_manager_with_mock_redis()

        new_session = _make_session("user-1")
        # Simulate: no existing session (get returns None), then create a new one
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock(return_value=True)

        with patch.object(manager, "create_session", return_value=new_session) as mock_create:
            pipeline = self._make_pipeline(manager)
            result = await pipeline.chat(
                user_message="Hello",
                session_id="",  # empty → new session must be created
                user_id="user-1",
            )
            mock_create.assert_awaited_once()

        print(result)

        assert result["session_id"] == new_session.session_id, (
            f"chat() returned session_id={result['session_id']!r} but "
            f"memory stored session_id={new_session.session_id!r}. "
            "The client would receive an id that doesn't match what's in memory."
        )

    @pytest.mark.asyncio
    async def test_graph_thread_id_matches_memory_session_id(self):
        """
        The thread_id passed to LangGraph ainvoke must equal session.session_id.
        If they diverge, LangGraph uses a different checkpoint than the one
        associated with the Redis session, breaking conversation continuity.
        """
        manager, mock_redis = _make_manager_with_mock_redis()
        existing_session = _make_session("user-1")

        stored_data = json.dumps(
            {
                "session_id": existing_session.session_id,
                "user_id": existing_session.user_id,
                "turns": [],
            }
        ).encode()
        mock_redis.get = AsyncMock(return_value=stored_data)

        mock_graph = AsyncMock()
        captured_config = {}

        async def capture_invoke(state, config):
            captured_config.update(config)
            return {
                "final_answer": "Answer",
                "sources": [],
                "was_rewritten": False,
                "retrieval_round": 0,
            }

        mock_graph.ainvoke = AsyncMock(side_effect=capture_invoke)
        pipeline = GraphPipeline(graph=mock_graph, short_term_memory=manager)

        await pipeline.chat(
            user_message="What is RAG?",
            session_id=existing_session.session_id,
            user_id="user-1",
        )

        thread_id_used = captured_config.get("configurable", {}).get("thread_id")
        assert thread_id_used == existing_session.session_id, (
            f"LangGraph thread_id={thread_id_used!r} does not match "
            f"session.session_id={existing_session.session_id!r}. "
            "Checkpoint recovery will use the wrong session."
        )

    @pytest.mark.asyncio
    async def test_initial_state_session_id_matches_memory(self):
        """
        The session_id injected into initial_state must come from the memory
        object's session, not from the raw caller-supplied parameter.
        """
        manager, mock_redis = _make_manager_with_mock_redis()
        existing_session = _make_session("user-1")

        stored_data = json.dumps(
            {
                "session_id": existing_session.session_id,
                "user_id": existing_session.user_id,
                "turns": [],
            }
        ).encode()
        mock_redis.get = AsyncMock(return_value=stored_data)

        mock_graph = AsyncMock()
        captured_state = {}

        async def capture_invoke(state, config):
            captured_state.update(state)
            return {
                "final_answer": "Answer",
                "sources": [],
                "was_rewritten": False,
                "retrieval_round": 0,
            }

        mock_graph.ainvoke = AsyncMock(side_effect=capture_invoke)
        pipeline = GraphPipeline(graph=mock_graph, short_term_memory=manager)

        await pipeline.chat(
            user_message="Explain embeddings",
            session_id=existing_session.session_id,
            user_id="user-1",
        )

        state_session_id = captured_state.get("session_id")
        assert state_session_id == existing_session.session_id, (
            f"Graph state session_id={state_session_id!r} does not match "
            f"memory session_id={existing_session.session_id!r}."
        )
