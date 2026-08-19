import logfire

_VALID_NEXT_ACTIONS = frozenset({"rephrase", "new_sub_question", "sufficient"})


class PlannerAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def decompose(self, question: str, question_type: str) -> list[str]:
        """
        Generate a single seed sub-question for the first retrieval hop.

        For factual questions, the original question is returned as-is (no LLM call).
        For complex question types, the LLM picks the most important
        first retrieval target.
        """
        if question_type == "factual":
            return [question]  # no decomposition needed

        prompt = f"""\
You are a query planner for a document retrieval system.
Given a complex question, generate ONE focused sub-question that targets
the single most important piece of information needed to begin answering it.
This sub-question should be answerable from a single document section.
Return only a JSON array with exactly one string.

Original question: {question}

Example output: ["What was memory usage in Q3?"]
"""

        response = await self.llm.complete(prompt, stage_tag="planner", json_mode=True)
        sub_questions = response.parsed_json

        logfire.info(f"Planner generated seed sub-question: {sub_questions}")
        return sub_questions

    async def plan_next_hop(
        self,
        original_question: str,
        hop_questions: list[str],
        retrieved_context: str,
        question_category: str,
    ) -> dict:
        """
        Decide the next action after a retrieval hop.

        Examines what has been retrieved so far and chooses one of:
        - ``rephrase``: same information need, better retrieval wording
        - ``new_sub_question``: a different aspect/gap needs to be filled
        - ``sufficient``: enough information has been gathered

        Parameters
        ----------
        original_question : str
            The user's original question.
        hop_questions : list[str]
            Trail of sub-questions asked so far (most recent last).
        retrieved_context : str
            Structured, metadata-tagged raw chunk previews of everything
            retrieved across all hops.
        question_category : str
            The router's classification (factual, analytical, …).

        Returns
        -------
        dict
            ``{"next_action": "rephrase"|"new_sub_question"|"sufficient",
              "query": str | None, "reasoning": str}``
        """

        hop_trail = "\n".join(f"  Hop {i + 1}: {q}" for i, q in enumerate(hop_questions))

        prompt = f"""\
You are a retrieval planner deciding whether more document retrieval is needed.

Original question: {original_question}
Question category: {question_category}

Sub-questions asked so far:
{hop_trail}

Retrieved information so far:
{retrieved_context}

Based on the retrieved information, decide the next action:

1. "sufficient" — The retrieved chunks contain enough information to fully
   answer the original question. No more retrieval is needed.
2. "rephrase" — The last sub-question targeted the right information, but
   the retrieval phrasing was poor. Provide a better phrasing of the SAME
   intent to improve retrieval quality.
3. "new_sub_question" — There is a specific gap in the retrieved information.
   A different, focused sub-question is needed to fill it.

Respond with JSON only:
{{
    "next_action": "sufficient" | "rephrase" | "new_sub_question",
    "query": "the rephrased or new sub-question (null if sufficient)",
    "reasoning": "brief explanation of your decision"
}}
"""

        with logfire.span(
            "planner_next_hop",
            num_hops=len(hop_questions),
            question_category=question_category,
        ):
            response = await self.llm.complete(prompt, stage_tag="planner_hop", json_mode=True)

            result = response.try_parsed_json(default=None)

            if result is None:
                logfire.warning(
                    "plan_next_hop_parse_failure",
                    raw_text=response.raw_text[:300],
                    fallback_action="sufficient",
                )
                result = {
                    "next_action": "sufficient",
                    "query": None,
                    "reasoning": "parse failure – defaulting to sufficient",
                }

            next_action = result.get("next_action", "sufficient")
            if next_action not in _VALID_NEXT_ACTIONS:
                logfire.warning(
                    "plan_next_hop_invalid_action",
                    received=next_action,
                    fallback_action="sufficient",
                )
                result["next_action"] = "sufficient"

            logfire.info(
                "plan_next_hop_decision",
                next_action=result.get("next_action"),
                reasoning=(
                    result.get("reasoning", "")[:150] + "..."
                    if len(result.get("reasoning", "")) > 150
                    else result.get("reasoning", "")
                ),
                has_query=result.get("query") is not None,
            )

            return result
