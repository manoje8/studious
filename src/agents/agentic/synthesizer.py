from __future__ import annotations

from typing import Union

from src.agents.agent_model import AgentState
from src.agents.graph.state import State


def _get(state: Union[AgentState, dict], key: str, attr: str | None = None):
    """Unified accessor for both AgentState dataclass and LangGraph State dict."""
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, attr or key, None)


def _build_context(state: Union[AgentState, dict]) -> str:
    """Build context string regardless of state type."""
    if isinstance(state, AgentState):
        return state.all_retrieved_context

    chunks = state.get("accepted_chunks") or []
    return "\n\n---\n\n".join(
        f"[Source: {c.get('source', 'unknown')} | Section: {c.get('section', 'unknown')}]\n{c.get('text', '')}"
        for c in chunks
    )


class SynthesizerAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def synthesize(self, state: Union[State, dict]) -> str:
        question_category = state["question_category"] or ""
        retrieval_history = state["retrieval_history"] or []

        accepted_chunks = (
            state.accepted_chunks
            if isinstance(state, AgentState)
            else state.get("accepted_chunks")
        )

        if not accepted_chunks and question_category != "chitchat":
            return (
                "I could not find sufficient information in the documents "
                "to answer your question."
            )

        original_question = (
            state.original_question
            if isinstance(state, AgentState)
            else state.get("effective_query") or state.get("original_message", "")
        )

        context = _build_context(state)

        friendly_prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {retrieval_history}

        LATEST MESSAGE:
        "{original_question}"
        """

        prompt = f"""
        Answer the following question using only the provided context.
        Cite sources by referencing the section name in brackets.

        Question: {original_question}

        Context:
        {context}

        Rules:
        - Only use information from the provided context
        - If context is insufficient, say so explicitly
        - Cite every claim with [Section Name]
        - Be precise and direct
        """

        response = await self.llm.complete(
            prompt if question_category != "chitchat" else friendly_prompt
        )
        return response.text
