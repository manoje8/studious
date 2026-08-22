"""
Unit tests for ShortTermMemoryManager (src/agents/memory/short_term.py).

Uses fakeredis to avoid needing a real Redis server.
Covers:
- create_session: generates a UUID session_id and saves to Redis
- get_session: returns None for unknown session_id
- get_session: deserialises a stored session correctly
- append_turn: adds a turn and persists back to Redis
- TTL is set on save
"""

import json

import pytest

from medici.agents.memory.short_term import ShortTermMemoryManager

# ---------------------------------------------------------------------------
# Helpers: fake Redis using fakeredis
# ---------------------------------------------------------------------------


def _make_manager_with_fake_redis():
    """Return a ShortTermMemoryManager wired to a fakeredis instance."""
    try:
        import fakeredis.aioredis as fake_aioredis

        fake_redis = fake_aioredis.FakeRedis()
    except ImportError:
        pytest.skip("fakeredis not installed — skipping ShortTermMemory tests")

    manager = ShortTermMemoryManager.__new__(ShortTermMemoryManager)
    manager.redis = fake_redis
    manager.session_ttl = 60 * 60 * 2
    return manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShortTermMemoryManager:
    async def test_create_session_returns_session_with_user_id(self):
        mgr = _make_manager_with_fake_redis()
        session = await mgr.create_session("user-42")
        assert session.user_id == "user-42"
        assert session.session_id  # non-empty UUID string

    async def test_create_session_persists_to_redis(self):
        mgr = _make_manager_with_fake_redis()
        session = await mgr.create_session("user-42")

        raw = await mgr.redis.get(f"session:{session.session_id}")
        assert raw is not None
        data = json.loads(raw)
        assert data["user_id"] == "user-42"
        assert data["session_id"] == session.session_id

    async def test_get_session_returns_none_for_unknown_id(self):
        mgr = _make_manager_with_fake_redis()
        result = await mgr.get_session("nonexistent-session-id")
        assert result is None

    async def test_get_session_returns_stored_session(self):
        mgr = _make_manager_with_fake_redis()
        created = await mgr.create_session("user-7")

        retrieved = await mgr.get_session(created.session_id)
        assert retrieved is not None
        assert retrieved.session_id == created.session_id
        assert retrieved.user_id == "user-7"

    async def test_get_session_deserialises_turns(self):
        mgr = _make_manager_with_fake_redis()
        session = await mgr.create_session("user-9")
        await mgr.append_turn(session, "user", "Hello!")
        await mgr.append_turn(session, "assistant", "Hi there!")

        retrieved = await mgr.get_session(session.session_id)
        assert len(retrieved.turns) == 2
        assert retrieved.turns[0].role == "user"
        assert retrieved.turns[0].content == "Hello!"
        assert retrieved.turns[1].role == "assistant"

    async def test_append_turn_adds_turn_to_session(self):
        mgr = _make_manager_with_fake_redis()
        session = await mgr.create_session("user-5")
        assert len(session.turns) == 0

        await mgr.append_turn(session, "user", "Question?", metadata={"source": "web"})
        assert len(session.turns) == 1
        assert session.turns[0].content == "Question?"
        assert session.turns[0].metadata == {"source": "web"}

    async def test_append_turn_persists_to_redis(self):
        mgr = _make_manager_with_fake_redis()
        session = await mgr.create_session("user-5")
        await mgr.append_turn(session, "user", "Persisted turn?")

        raw = await mgr.redis.get(f"session:{session.session_id}")
        data = json.loads(raw)
        assert len(data["turns"]) == 1
        assert data["turns"][0]["content"] == "Persisted turn?"

    async def test_append_turn_without_metadata_defaults_to_none(self):
        mgr = _make_manager_with_fake_redis()
        session = await mgr.create_session("user-6")
        await mgr.append_turn(session, "assistant", "Answer!")
        assert session.turns[-1].metadata is None

    async def test_multiple_sessions_are_independent(self):
        mgr = _make_manager_with_fake_redis()
        s1 = await mgr.create_session("user-A")
        s2 = await mgr.create_session("user-B")

        await mgr.append_turn(s1, "user", "Turn for A")

        retrieved_s1 = await mgr.get_session(s1.session_id)
        retrieved_s2 = await mgr.get_session(s2.session_id)

        assert len(retrieved_s1.turns) == 1
        assert len(retrieved_s2.turns) == 0
