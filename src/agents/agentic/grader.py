import logfire


class GraderAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def grade_chunks(
        self, chunks: list[dict], original_question: str
    ) -> list[dict]:
        if not chunks:
            return []

        accepted = []

        for chunk in chunks:
            prompt = f"""
            Question: {original_question}

            Chunk text:
            {chunk['text'][:500]}

            Does this chunk contain information that directly helps answer the question?
            Respond with JSON only: {{"relevant": true/false, "reason": "..."}}
            """

            result = await self.llm.complete(prompt)
            grade = result.parsed_json

            if grade["relevant"]:
                chunk["grade_reason"] = grade["reason"]
                accepted.append(chunk)
            else:
                logfire.info(f"Grader rejected chunk: " f"{grade['reason']}")

        logfire.info(f"Grader accepted {len(accepted)}/{len(chunks)} chunks")

        return accepted
