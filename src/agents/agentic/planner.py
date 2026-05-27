import logfire


class PlannerAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def decompose(self, question: str, question_type: str) -> list[str]:
        if question_type == "factual":
            return [question]  # no decomposition needed

        prompt = f"""
        Break this question into 2-4 focused sub-questions.
        Each sub-question should be answerable from a single document section.
        Return only a JSON array of strings.

        Original question: {question}

        Example output: ["What was memory usage in Q3?", "What was memory usage in Q4?",
                         "What is the trend direction?"]
        """

        response = await self.llm.complete(prompt)
        sub_questions = response.parsed_json

        logfire.info(f"Planner decomposed into {len(sub_questions)} sub-questions")
        return sub_questions
