class RouterAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def classify(self, question: str) -> dict:
        prompt = f"""
        Classify this question into exactly one category and respond in JSON only.

        Categories:
        - "factual": Single specific fact (what, when, who, how much, how many)
        - "comparative": Comparing two or more things (difference between X and Y, which is better)
        - "analytical": Requires reasoning across multiple pieces of information (why, how, what if, implications)
        - "summarization": Wants an overview, summary, or recap of a topic or conversation
        - "chitchat": ONLY greetings (hello, hi), farewells (goodbye), expressions of gratitude (thank you),
                       social pleasantries (how are you), or statements without information request.
                       Does NOT include: questions about previous conversation turns, requests for information,
                       or any query that requires retrieving or processing content.

        Special handling for conversation context:
        - "what did you say?" / "what was my previous question?" / "what did we discuss?" -> "factual" or "summarization"
        - "can you repeat that?" / "what did you mean by X?" -> "factual"
        - "tell me again about..." -> "factual" or "analytical" based on content

        Question: {question}

        Respond with exactly this JSON format:
        {{"category": "factual|comparative|analytical|summarization|chitchat", "needs_multi_retrieval": true/false, "reasoning": "brief explanation"}}
        """

        response = await self.llm.complete(prompt)
        return response.parsed_json
