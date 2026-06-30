import json
from uuid import uuid4

import logfire
import redis.asyncio as redis

from src.agents.memory.conversation_model import ConversationSession, ConversationTurn
from src.common.utils.config import config


class ShortTermMemoryManager:
    def __init__(self, redis_url: str = config.REDIS_URL):
        self.redis = redis.from_url(redis_url)
        self.session_ttl = 60 * 60 * 2  # 2 hours

    async def _save(self, session: ConversationSession) -> None:
        data = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "turns": [
                {"role": t.role, "content": t.content, "metadata": t.metadata}
                for t in session.turns
            ],
        }

        await self.redis.setex(f"session:{session.session_id}", self.session_ttl, json.dumps(data))

    async def get_session(self, session_id: str) -> ConversationSession | None:
        data = await self.redis.get(f"session:{session_id}")

        if not data:
            return None

        raw = json.loads(data)

        session = ConversationSession(session_id=raw["session_id"], user_id=raw["user_id"])

        for turn_data in raw["turns"]:
            session.turns.append(
                ConversationTurn(
                    role=turn_data["role"],
                    content=turn_data["content"],
                    metadata=turn_data.get("metadata", {}),
                )
            )

        return session

    async def create_session(self, user_id: str) -> ConversationSession:
        session = ConversationSession(session_id=str(uuid4()), user_id=user_id)

        await self._save(session)
        logfire.info(f"Created session: {session.session_id}")
        return session

    async def append_turn(
        self,
        session: ConversationSession,
        role: str,
        content: str,
        metadata: dict = None,
    ) -> None:
        session.add_turn(role, content, metadata)
        await self._save(session)
