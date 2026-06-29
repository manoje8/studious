import asyncio

import logfire

from src.agents.graph.state import State


class GraderAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def grade(self, state: State) -> dict:
        """Grades retrieval quality and returns a score for routing decisions."""

        query = state.get("effective_query", "original_message")
        sub_questions = state.get("sub_questions", [])
        accepted_chunks = state.get("accepted_chunks", [])
        # retrieval_history = state.get("retrieval_history", [])
        question_category = state.get("question_category", "factual")
        classification = state.get("classification", {})

        if not accepted_chunks:
            return {
                "retrieval_grade_score": 0.0,
                "grading_details": {
                    "reason": "No chunk to grade",
                    "accepted_count": 0,
                    "total_count": 0,
                },
                "needs_refinement": False,
            }

        graded_chunks, graded_results = await self._grade_chunks_batch(
            chunks=accepted_chunks, query=query, sub_questions=sub_questions
        )

        coverage_score = await self._calculate_coverage(
            graded_chunks=graded_chunks, sub_questions=sub_questions, query=query
        )

        completeness = await self._assess_completeness(
            query=query,
            graded_chunks=graded_chunks,
            question_category=question_category,
            retrieval_strategy=classification.get("retrieval_strategy", {}),
        )

        relevance_ratio = len(graded_chunks) / len(accepted_chunks) if accepted_chunks else 0

        overall_score = self._calculate_overall_score(
            relevance_ratio=relevance_ratio,
            coverage_score=coverage_score,
            completeness=completeness,
            question_category=question_category,
        )

        need_refinement = (
            overall_score < 0.4
            or len(graded_chunks) < 2
            or completeness.get("missing_critical_info", False)
        )

        logfire.info(
            f"Grader: score={overall_score:.2f},"
            f"chunks: {len(graded_chunks)}/{len(accepted_chunks)},"
            f"coverage: {coverage_score:.2f},"
            f"completeness: {completeness['level']},"
            f"needs refinement: {need_refinement}"
        )

        return {
            "retrieval_grade_score": overall_score,
            "grading_details": {
                "accepted_count": len(graded_chunks),
                "total_count": len(accepted_chunks),
                "relevance_ratio": relevance_ratio,
                "coverage_score": coverage_score,
                "completeness": completeness,
                "needs_refinement": need_refinement,
                "rejected_reasons": graded_results.get("rejected_reasons", []),
            },
            "accepted_chunks": graded_chunks,
            "needs_refinement": need_refinement,
        }

    async def _grade_chunks_batch(self, chunks: list[dict], query: str, sub_questions: list[str]):
        """
        Grade all chunks in parallel for efficiency.
        Returns accepted chunks and grading metadata.

        :param chunks:
        :param query:
        :param sub_questions:
        """
        if not chunks:
            return [], {"rejected_reasons": []}

        tasks = [self._grade_single_chunk(chunk, query, sub_questions) for chunk in chunks]

        results = await asyncio.gather(*tasks)

        accepted = []
        rejected_reasons = []

        for chunk, result in zip(chunks, results, strict=False):
            if result["relevant"]:
                chunk_copy = dict(chunk)
                chunk_copy["grade_reason"] = result["reason"]
                chunk_copy["relevance_score"] = result.get("score", 0.5)
                chunk_copy["answers_sub_questions"] = result.get("answers_sub_questions", [])
                accepted.append(chunk_copy)

            else:
                rejected_reasons.append(
                    {
                        "source": chunk.get("source", "unknown"),
                        "section": chunk.get("section", "unknown"),
                        "reason": result["reason"],
                        "preview": chunk.get("text", "")[:100],
                    }
                )
                logfire.warn(f"Grader rejected chunk: {result['reason'][:100]}")

        logfire.info(
            f"Grader accepted: {len(accepted)}/{len(chunks)} chunks "
            f"(rejected {len(rejected_reasons)})"
        )

        return accepted, {"rejected_reasons": rejected_reasons}

    async def _grade_single_chunk(self, chunk: dict, query: str, sub_questions: list[str]):
        """
        Grade a single chunk for relevance to the query and sub-questions.
        Returns detailed grading result.

        :param chunk:
        :param query:
        :param sub_questions:
        :return:
        """

        chunk_text = chunk.get("text", "")[:1000]
        source = chunk.get("source", "unknown")
        section = chunk.get("section", "unknown")

        sub_q_context = ""

        if sub_questions:
            # Limit to 5
            sub_q_context = "\nRelevant sub-questions:\n" + "\n".join(
                f"- {q}" for q in sub_questions[:5]
            )

        prompt = f"""
You are grading the relevance of a retrieved document chunk for answering a query.

Main question: {query}
{sub_q_context}

Chunk text (from [{source} | {section}]):
{chunk_text}

Evaluate whether this chunk contains information that:
1. Directly answers the main question or any sub-question
2. Provides necessary context for understanding the answer
3. Contains specific facts, numbers, or details relevant to the query

Respond with JSON only:
{{
    "relevant": true/false,
    "score": 0.0-1.0,
    "reason": "Brief explanation of why this chunk is relevant or not",
    "answers_sub_questions": ["list", "of", "sub-question", "indices", "this", "helps", "answer"],
    "information_type": "direct_answer|supporting_context|background|irrelevant",
    "key_information": ["list", "of", "key", "facts", "found"]
}}
"""
        try:
            result = await self.llm.complete(prompt)
            grade = result.parsed_json

            if "relevant" not in grade:
                return self._default_grade_result()

            return grade

        except Exception as e:
            logfire.error(f"Error grading chunk: {e}")
            return self._default_grade_result()

    async def _calculate_coverage(
        self, graded_chunks: list[dict], sub_questions: list[str], query: str
    ) -> float:
        """
        Calculate how well the graded chunks cover all sub-questions.
        Returns a coverage score from 0.0 to 1.0.

        :param graded_chunks:
        :param sub_questions:
        :param query:
        :return:
        """

        if not sub_questions:
            return 1.0 if graded_chunks else 0.0

        covered_indices = set()

        for chunk in graded_chunks:
            answers = chunk.get("answers_sub_questions", [])
            if isinstance(answers, list):
                for answer in answers:
                    if isinstance(answer, int) and 0 <= answer < len(sub_questions):
                        covered_indices.add(answer)
                    elif isinstance(answer, str):
                        for i, sq in enumerate(sub_questions):
                            if answer.lower() in sq.lower() or sq.lower() in answer.lower():
                                covered_indices.add(i)

        if len(covered_indices) < len(sub_questions) * 0.5:
            covered_indices = await self._llm_assess_coverage(graded_chunks, sub_questions, query)

        coverage_score = len(covered_indices) / len(sub_questions) if sub_questions else 1.0

        return min(coverage_score, 1.0)

    async def _llm_assess_coverage(
        self, graded_chunks: list[dict], sub_questions: list[str], query: str
    ):
        """Use LLM to assess which sub-questions are covered by chunks."""

        chunks_summary = "\n\n".join(
            f"Chunk {i} [{c.get('score', '?')}]: {c.get('text', '')[:200]}..."
            for i, c in enumerate(graded_chunks[:10])
        )

        sub_q_text = "\n".join(f"{i}: {q}" for i, q in enumerate(sub_questions))

        prompt = f"""
Given these sub-questions and retrieved chunks, determine which
sub-questions have sufficient information to answer them.

Sub-questions:
{sub_q_text}

Retrieved chunks:
{chunks_summary}

Return JSON: {{"covered_indices": [0, 2, 3]}}
Only include indices that have sufficient information.
"""
        try:
            result = await self.llm.complete(prompt)
            data = result.parsed_json
            return set(data.get("covered_indices", []))
        except Exception:
            return set()

    async def _assess_completeness(
        self,
        query: str,
        graded_chunks: list[dict],
        question_category: str,
        retrieval_strategy: dict,
    ):
        """
        Assess whether the retrieved information is complete enough
        to answer the question satisfactorily.

        :param query:
        :param graded_chunks:
        :param question_category:
        :param retrieval_strategy:
        :return:
        """

        if not graded_chunks:
            return {
                "level": "empty",
                "missing_critical_info": True,
                "reason": "No relevant chunks retrieved",
            }

        target_chunks = retrieval_strategy.get("target_chunks", 5)
        actual_chunks = len(graded_chunks)

        if actual_chunks >= target_chunks:
            return {
                "level": "sufficient",
                "missing_critical_info": False,
                "reason": f"Found {actual_chunks} relevant chunks (target: {target_chunks})",
            }

        if question_category in ("analytical", "procedural", "comparative"):
            return await self.deep_completeness_check(query, graded_chunks)

        if actual_chunks >= target_chunks * 0.5:
            return {
                "level": "adequate",
                "missing_critical_info": False,
                "reason": f"Adequate chunk Found {actual_chunks}",
            }

        return {
            "level": "insufficient",
            "missing_critical_info": True,
            "reason": f"Only {actual_chunks} chunk found, need atleast {target_chunks * 0.5}",
        }

    async def deep_completeness_check(self, query, graded_chunks):
        """Deep check for complex question types."""

        context = "\n\n".join(
            f"[{c.get('source', '?')} | {c.get('section', '?')}]\n{c.get('text', '')[:300]}"
            for c in graded_chunks[:8]
        )

        prompt = f"""
Assess whether the retrieved information is sufficient to fully answer
this question. Identify any critical gaps.

Question: {query}

Retrieved information:
{context}

Return JSON:
{{
    "sufficient": true/false,
    "missing_critical_info": true/false,
    "gaps": ["list", "of", "missing", "information"],
    "can_partially_answer": true/false,
    "confidence": 0.0-1.0
}}

"""
        try:
            result = await self.llm.complete(prompt)
            data = result.parsed_json

            return {
                "level": "sufficient" if data.get("sufficient") else "insufficient",
                "missing_critical_info": data.get("missing_critical_info", True),
                "gaps": data.get("gaps", []),
                "can_partially_answer": data.get("can_partially_answer", False),
                "confidence": data.get("confidence", 0.5),
            }
        except Exception:
            return {
                "level": "unknown",
                "missing_critical_info": False,
                "reason": "Could not assess completeness",
            }

    def _calculate_overall_score(
        self,
        relevance_ratio: float,
        coverage_score: float,
        completeness: dict,
        question_category: str,
    ):
        """
        Calculate overall retrieval quality score.
        Weighted based on question category.

        :param relevance_ratio:
        :param coverage_score:
        :param completeness:
        :param question_category:
        :return:
        """

        completeness_score = {
            "sufficient": 0.9,
            "adequate": 0.7,
            "insufficient": 0.3,
            "empty": 0.0,
            "unknown": 0.5,
        }.get(completeness.get("level", "unknown"), 0.5)

        if question_category in ("factual", "summarization"):
            weights = {"relevance": 0.5, "coverage": 0.2, "completeness": 0.3}
        elif question_category in ("comparative", "analytical"):
            weights = {"relevance": 0.3, "coverage": 0.4, "completeness": 0.3}
        elif question_category == "procedural":
            weights = {"relevance": 0.2, "coverage": 0.3, "completeness": 0.5}
        else:
            weights = {"relevance": 0.4, "coverage": 0.3, "completeness": 0.3}

        score = (
            weights["relevance"] * relevance_ratio
            + weights["coverage"] * coverage_score
            + weights["completeness"] * completeness_score
        )

        return round(score, 2)

    def _default_grade_result(self):
        """Default result when grading fails."""
        return {
            "relevant": False,
            "score": 0.0,
            "reason": "Grading failed - defaulting to not relevant",
            "answers_sub_questions": [],
            "information_type": "irrelevant",
            "key_information": [],
        }

    async def grade_chunks(self, chunks: list[dict], original_question: str) -> list[dict]:
        """
        Legacy method for simple per-chunk grading.
        Now delegates to the batch grader.
        """
        graded, _ = await self._grade_chunks_batch(
            chunks=chunks,
            query=original_question,
            sub_questions=[],
        )
        return graded
