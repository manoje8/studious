from src.agents.agent_model import AgentState


class SynthesizerAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def synthesize(self, state: AgentState) -> str:
        if not state.accepted_chunks:
            return (
                "I could not find sufficient information in the documents "
                "to answer your question."
            )

        prompt = f"""
        Answer the following question using only the provided context.
        Cite sources by referencing the section name in brackets.

        Question: {state.original_question}

        Context:
        {state.all_retrieved_context}

        Rules:
        - Only use information from the provided context
        - If context is insufficient, say so explicitly
        - Cite every claim with [Section Name]
        - Be precise and direct
        """

        response = await self.llm.complete(prompt)
        return response.text
