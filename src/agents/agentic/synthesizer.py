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
        """
        Main synthesis entry point that handles all question categories.
        Returns the final answer string.
        """

        question_category = _get(state, "question_category") or "factual"

        if question_category == "chitchat":
            return await self._synthesize_chitchat(state)
        elif question_category == "meta":
            return await self._synthesize_meta(state)
        elif question_category == "clarification":
            return await self._synthesize_clarification(state)
        elif question_category == "procedural":
            return await self._synthesize_procedural(state)
        elif question_category == "comparative":
            return await self._synthesize_comparative(state)
        elif question_category == "analytical":
            return await self._synthesize_analytical(state)
        elif question_category == "summarization":
            return await self._synthesize_summarization(state)
        else:
            return await self._synthesize_factual(state)

    async def direct_synthesize(self, state: Union[State, dict]) -> dict:
        """
        Handle simple factual queries without complex retrieval chain.
        Used in the 'direct_synthesize' graph node.
        """

        answer = await self._synthesize_factual(state)

        return {
            "final_answer": answer,
            "synthesis_metadata": {
                "mode": "direct",
                "category": _get(state, "question_category", "factual"),
                "retrieval_used": bool(_get(state, "accepted_chunks")),
            },
        }

    async def handle_simple_response(self, state: Union[State, dict]) -> dict:
        """
        Handle chitchat and meta queries without any retrieval.
        Used in the 'handle_simple_response' graph node.
        """

        category = _get(state, "question_category", "chitchat")

        if category == "chitchat":
            answer = await self._synthesize_chitchat(state)
        else:
            answer = await self._synthesize_meta(state)

        return {
            "final_answer": answer,
            "synthesis_metadata": {"category": category, "no_retrieval": True},
        }

    async def _synthesize_factual(self, state: Union[State, dict]) -> str:
        """Synthesize factual answers with strict source attribution."""
        original_question = self._get_question(state)
        context = _build_context(state)
        accepted_chunks = _get(state, "accepted_chunks") or []

        if not accepted_chunks:
            return (
                "I could not find sufficient information in the documents "
                "to answer your question. Please try rephrasing or "
                "providing more context."
            )

        prompt = f"""
        Answer the following question using only the provided context.
        You are an Enterprise AI Assistant focused on accuracy and precision.

        Question: {original_question}

        Context:
        {context}

        Instructions:
        1. Only use information explicitly stated in the provided context
        2. Cite every factual claim with the source section in brackets [Section Name]
        3. If the context is insufficient, clearly state what's missing
        4. Be direct and concise - avoid unnecessary elaboration
        5. If multiple sources conflict, acknowledge the discrepancy
        6. Format numerical data clearly with units when applicable

        Answer format:
        - Start with a direct answer to the question
        - Provide supporting details with citations
        - End with confidence level if information is incomplete
        """

        response = await self.llm.complete(prompt)
        return response.text

    async def _synthesize_chitchat(self, state: Union[State, dict]) -> str:
        """Handle casual conversation with friendly tone."""
        original_question = self._get_question(state)
        retrieval_history = _get(state, "retrieval_history") or []

        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Respond naturally to the user's message while maintaining professionalism.

        Conversation context:
        {self._format_history(retrieval_history)}

        User message: {original_question}

        Guidelines:
        - Be warm but professional
        - Keep responses concise
        - Transition naturally to work-related topics if appropriate
        - Don't pretend to have capabilities you don't have
        """

        response = await self.llm.complete(prompt)
        return response.text

    async def _synthesize_meta(self, state: Union[State, dict]) -> str:
        """Handle questions about the system's capabilities."""
        prompt = """
        You are an Enterprise AI Assistant. Answer questions about your capabilities
        honestly and clearly.

        Your capabilities:
        - Retrieve and analyze information from enterprise documents
        - Answer factual questions with source citations
        - Compare and analyze multiple entities or concepts
        - Summarize documents and conversations
        - Provide step-by-step procedural guidance
        - Remember conversation context within a session

        Limitations:
        - You can only access documents you've been given
        - You cannot browse the internet or access external systems
        - You cannot perform actions in other systems

        Respond helpfully and offer to assist with document-based questions.
        """

        response = await self.llm.complete(prompt)
        return response.text

    async def _synthesize_comparative(self, state: Union[State, dict]) -> str:
        """Synthesize comparative analysis with structured comparison."""
        original_question = self._get_question(state)
        context = _build_context(state)
        accepted_chunks = _get(state, "accepted_chunks") or []

        if len(accepted_chunks) < 2:
            return (
                "I need more information to make a meaningful comparison. "
                "Could you specify which aspects you'd like me to compare?"
            )

        prompt = f"""
        Compare and contrast based on the provided context.
        Provide a structured, balanced analysis.

        Question: {original_question}

        Context:
        {context}

        Structure your response as:
        1. Executive Summary (2-3 sentences)
        2. Detailed Comparison Table (if applicable)
           - Feature/Criterion | Entity A | Entity B
        3. Key Similarities (with citations)
        4. Key Differences (with citations)
        5. Recommendation or Conclusion (if appropriate)

        Rules:
        - Cite all claims with [Section Name]
        - Be objective and balanced
        - Acknowledge when data is incomplete
        - Use specific metrics/numbers when available
        """

        response = await self.llm.complete(prompt)
        return response.text

    async def _synthesize_analytical(self, state: Union[State, dict]) -> str:
        """Synthesize analytical answers with reasoning chains."""
        original_question = self._get_question(state)
        context = _build_context(state)

        prompt = f"""
        Provide a thorough analysis using chain-of-thought reasoning.
        Show your analytical process clearly.

        Question: {original_question}

        Context:
        {context}

        Structure your analysis:
        1. Key Findings (main insights from context)
        2. Analysis
           - Examine each relevant piece of evidence
           - Identify patterns and relationships
           - Consider implications and causality
        3. Reasoning Chain
           - Step-by-step logical progression
           - Address counterarguments
        4. Conclusion
           - Synthesize findings into clear answer
           - State confidence level
           - Note any assumptions made

        Rules:
        - Cite evidence with [Section Name]
        - Distinguish between facts and inferences
        - Acknowledge uncertainty
        - Be thorough but avoid speculation beyond evidence
        """

        response = await self.llm.complete(prompt)
        return response.text

    async def _synthesize_summarization(self, state: Union[State, dict]) -> str:
        """Synthesize summaries with hierarchical structure."""

        original_question = self._get_question(state)
        context = _build_context(state)
        retrieval_history = _get(state, "retrieval_history") or []

        if any(
            word in original_question.lower()
            for word in ["previous", "discussed", "conversation", "recap"]
        ):
            return await self._summarize_conversation(retrieval_history)

        prompt = f"""
        Create a comprehensive yet concise summary.

        Request: {original_question}

        Content to summarize:
        {context}

        Structure:
        1. Executive Summary (2-3 sentences)
        2. Key Points (bullet points with citations)
        3. Supporting Details (organized by theme)
        4. Conclusions or Next Steps

        Rules:
        - Preserve key facts, numbers, and dates
        - Cite sources with [Section Name]
        - Maintain original meaning - don't introduce new information
        - Be hierarchical: most important information first
        - Note any gaps in the source material
        """

        response = await self.llm.complete(prompt)
        return response.text

    async def _synthesize_clarification(self, state: Union[State, dict]) -> str:
        """Handle clarification requests by providing more detail."""

        original_question = self._get_question(state)
        context = _build_context(state)
        retrieval_history = _get(state, "retrieval_history") or []

        prompt = f"""
        The user is asking for clarification on a previous topic.
        Provide additional detail and explanation.

        Previous context:
        {self._format_history(retrieval_history)}

        Clarification needed: {original_question}

        Additional context:
        {context}

        Guidelines:
        - Reference what was previously discussed
        - Explain in more detail, using simpler terms if needed
        - Provide examples where helpful
        - Confirm understanding before elaborating
        - If the clarification requires information not available, say so
        """

        response = await self.llm.complete(prompt)
        return response.text

    async def _synthesize_procedural(self, state: Union[State, dict]) -> str:
        """Synthesize step-by-step procedural guidance."""

        original_question = self._get_question(state)
        context = _build_context(state)

        prompt = f"""
        Provide clear, actionable step-by-step guidance.

        Task: {original_question}

        Context:
        {context}

        Structure:
        1. Overview/Goal
        2. Prerequisites (if any)
        3. Step-by-Step Instructions
           - Number each step clearly
           - Include expected outcomes for each step
           - Note common pitfalls or alternatives
        4. Verification (how to confirm success)
        5. Troubleshooting (common issues and solutions)

        Rules:
        - Be precise and unambiguous
        - Cite sources for each major step [Section Name]
        - Indicate if steps are sequential or can be done in parallel
        - Include safety/security considerations if applicable
        - If the procedure is incomplete in the context, note what's missing
        """

        response = await self.llm.complete(prompt)
        return response.text

    async def _summarize_conversation(self, history: list) -> str:
        """Summarize the conversation history."""

        if not history:
            return "We haven't discussed anything yet in this session."

        prompt = f"""
        Summarize the following conversation:

        {self._format_history(history)}

        Provide:
        1. Main topics discussed
        2. Key questions asked and answers given
        3. Any decisions or conclusions reached
        4. Outstanding questions or next steps
        """

        response = await self.llm.complete(prompt)
        return response.text

    def _get_question(self, state: Union[State, dict]) -> str:
        """Extract the question from state regardless of type."""

        if isinstance(state, AgentState):
            return state.original_question or state.effective_query or ""

        return _get(state, "effective_query") or _get(state, "original_message") or ""

    def _format_history(self, history: list) -> str:
        """Format conversation history for prompts."""

        if not history:
            return "No previous conversation."

        formatted = []
        for i, turn in enumerate(history[-5:], 1):  # Last 5 turns
            if isinstance(turn, dict):
                user_msg = turn.get("user", turn.get("question", ""))
                asst_msg = turn.get("assistant", turn.get("answer", ""))
                formatted.append(f"{i}. User: {user_msg}\n   Assistant: {asst_msg}")
            else:
                formatted.append(f"{i}. {str(turn)}")

        return "\n".join(formatted)
