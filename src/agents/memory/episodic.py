from src.agents.memory.conversation_model import ConversationSession


class EpisodicMemoryManager:
    def __init__(self, llm_client, db_client):
        self.llm = llm_client
        self.db = db_client

    async def compress_session(self, session: ConversationSession):
        pass

    async def load_user_memory(self, user_id: str, limit: int = 5):
        pass
