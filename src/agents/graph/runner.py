import uuid

from src.agents.memory.conversation_model import ConversationSession


class GraphPipeline:
    def __init__(self, graph, short_term_memory):
        self.graph = graph
        self.short_term = short_term_memory

    async def chat(self, user_message: str, session_id: str, user_id: str) -> dict:
        if not session_id:
            session_id = f"{user_id}_{uuid.uuid4()}"

        session: ConversationSession = await self.short_term.get_session(session_id)
        if not session:
            session = await self.short_term.create_session(user_id)

        config = {"configurable": {"thread_id": session.session_id}}

        initial_state = {
            "session_id": session.session_id,
            "user_id": user_id,
            "original_message": user_message,
            "effective_query": user_message,
            "was_rewritten": False,
            "question_category": "",
            "sub_questions": [],
            "current_sub_question_idx": 0,
            "current_query": user_message,
            "retrieval_round": 0,
            "max_retrieval_rounds": 3,
            "retrieval_history": [],
            "accepted_chunks": [],
            "final_answer": "",
            "sources": [],
            "doc_id_filter": None,
            "episodic_context": "",
        }

        result = await self.graph.ainvoke(initial_state, config=config)

        return {
            "answer": result["final_answer"],
            "session_id": session_id,
            "sources": result["sources"],
            "query_was_rewritten": result["was_rewritten"],
            "retrieval_rounds": result["retrieval_round"],
        }
