class RouterAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def classify(self, question: str) -> dict:
        prompt = f"""
        Classify this question into exactly one category and respond in JSON only.

        Categories:
        - "factual": Single specific fact (what, when, who, how much)
        - "comparative": Comparing two or more things
        - "analytical": Requires reasoning across multiple pieces of information
        - "summarization": Wants an overview of a topic

        Question: {question}

        Respond with exactly:
        {{"category": "...", "needs_multi_retrieval": true/false, "reasoning": "..."}}
        """

        response = await self.llm.complete(prompt)
        return response.parsed_json
