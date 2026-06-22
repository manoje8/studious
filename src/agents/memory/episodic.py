import logfire

from src.agents.memory.conversation_model import (
    ConversationSession,
    EpisodicSummary,
)


class EpisodicMemoryManager:
    """Manages long-term episodic memory stored in PostgreSQL.

    Compresses conversation sessions into concise summaries using an LLM,
    persists them in PostgreSQL, and retrieves recent episodic context
    for a user to inject into query rewriting and synthesis prompts.
    """

    def __init__(self, llm_client, pool):
        self.llm = llm_client
        self.pool = pool

    async def setup(self) -> None:
        """Create the episodic_memories table if it doesn't exist."""
        create_sql = """
            CREATE TABLE IF NOT EXISTS episodic_memories (
                id            SERIAL PRIMARY KEY,
                user_id       TEXT NOT NULL,
                session_id    TEXT NOT NULL,
                summary       TEXT NOT NULL,
                topic_tags    TEXT[] DEFAULT '{}',
                turn_count    INTEGER DEFAULT 0,
                created_at    TIMESTAMPTZ DEFAULT NOW()
            );
        """
        index_user_sql = """
            CREATE INDEX IF NOT EXISTS idx_episodic_user
            ON episodic_memories(user_id);
        """
        index_created_sql = """
            CREATE INDEX IF NOT EXISTS idx_episodic_created
            ON episodic_memories(created_at DESC);
        """

        async with self.pool.connection() as conn:
            await conn.execute(create_sql)
            await conn.execute(index_user_sql)
            await conn.execute(index_created_sql)

        logfire.info("Episodic memory table ready")

    async def compress_session(self, session: ConversationSession) -> EpisodicSummary:
        """Compress a conversation session into a concise episodic summary.

        Uses the LLM to distill the full conversation into a 2-3 sentence
        summary and extract key topic tags for later retrieval.
        """
        transcript = session.to_prompt_format(n=len(session.turns))

        prompt = f"""
        You are compressing a conversation into a concise episodic memory.

        Conversation transcript:
        {transcript}

        Task: Create a brief summary of this conversation that captures:
        1. What the user was asking about
        2. What key information was provided
        3. Any decisions or conclusions reached

        Respond in JSON only:
        {{
            "summary": "2-3 sentence summary of the conversation",
            "topic_tags": ["tag1", "tag2", "tag3"]
        }}
        """

        response = await self.llm.complete(prompt)
        result = response.parsed_json

        summary = EpisodicSummary(
            user_id=session.user_id,
            session_id=session.session_id,
            summary=result["summary"],
            topic_tags=result.get("topic_tags", []),
            turn_count=len(session.turns),
        )

        logfire.info(
            f"Compressed session {session.session_id} "
            f"({len(session.turns)} turns) into episodic summary"
        )

        return summary

    async def save_summary(self, summary: EpisodicSummary) -> None:
        """Persist an episodic summary to PostgreSQL."""
        insert_sql = """
            INSERT INTO episodic_memories
                (user_id, session_id, summary, topic_tags, turn_count, created_at)
            VALUES
                (%(user_id)s, %(session_id)s, %(summary)s,
                 %(topic_tags)s, %(turn_count)s, %(created_at)s)
        """

        async with self.pool.connection() as conn:
            await conn.execute(
                insert_sql,
                {
                    "user_id": summary.user_id,
                    "session_id": summary.session_id,
                    "summary": summary.summary,
                    "topic_tags": summary.topic_tags,
                    "turn_count": summary.turn_count,
                    "created_at": summary.created_at,
                },
            )

        logfire.info(
            f"Saved episodic summary for user {summary.user_id}, "
            f"session {summary.session_id}"
        )

    async def load_user_memory(self, user_id: str, limit: int = 5) -> str:
        """Load the most recent episodic summaries for a user.

        Returns a formatted string suitable for injection into LLM prompts.
        Returns an empty string if no episodic memories exist.
        """
        select_sql = """
            SELECT summary, topic_tags, created_at
            FROM episodic_memories
            WHERE user_id = %(user_id)s
            ORDER BY created_at DESC
            LIMIT %(limit)s
        """

        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                select_sql,
                {"user_id": user_id, "limit": limit},
            )
            rows = await cursor.fetchall()

        if not rows:
            return ""

        lines = []
        for row in rows:
            tags = ", ".join(row["topic_tags"]) if row["topic_tags"] else "general"
            timestamp = row["created_at"].strftime("%Y-%m-%d %H:%M")
            lines.append(f"[{timestamp} | Topics: {tags}] {row['summary']}")

        return "\n".join(lines)
