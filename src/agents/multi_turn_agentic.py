from src.agents.memory.episodic import EpisodicMemoryManager
from src.agents.memory.query_rewriter import QueryRewriter
from src.agents.memory.short_term import ShortTermMemoryManager


class MultiTurnAgenticRAGPipeline:
    def __init__(
        self,
        llm_client,
        rag_agent,
        short_term_memory: ShortTermMemoryManager,
        episodic_memory: EpisodicMemoryManager,
    ):
        self.rag = rag_agent
        self.rewriter = QueryRewriter(llm_client)
        self.short_term = short_term_memory
        self.episodic = episodic_memory

    async def chat(
        self,
        user_message: str,
        session_id: str,
        user_id: str,
        doc_id_filter: str | None = None,
    ):
        session = await self.short_term.get_session(session_id)

        if not session:
            await self.short_term.create_session(user_id)

        # Todo: Implement Episodic memory

        rewrite_result = await self.rewriter.rewrite(user_message, session)
        effective_query = rewrite_result["rewritten_query"]

        await self.short_term.append_turn(session, "user", user_message)

        rag_result = await self.rag.run(question=effective_query, doc_id_filter=doc_id_filter)

        await self.short_term.append_turn(
            session,
            "assistant",
            rag_result["answer"],
            metadata={
                "sources": rag_result["source"],
                "retrieval_rounds": rag_result["retrieval_rounds"],
                "original_query": user_message,
                "rewritten_query": effective_query,
                "was_written": rewrite_result["was_written"],
            },
        )

        if len(session.turns) > 10:
            # TODO: Implement episodic compression
            pass

        return {
            "answer": rag_result["answer"],
            "session_id": session.session_id,
            "sources": rag_result["sources"],
            "query_was_rewritten": rewrite_result["was_rewritten"],
            "resolved_references": rewrite_result["resolved_references"],
            "retrieval_rounds": rag_result["retrieval_rounds"],
        }
