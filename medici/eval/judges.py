"""
LLM-as-Judge Evaluators
=======================

Three complementary judges for end-to-end answer quality:

1. **Faithfulness**: Is the answer grounded in the retrieved context?
2. **Relevance**: Does the answer actually address the user's question?
3. **Hallucination**: Does the answer contain claims not in the context?

Each judge returns a score in [0, 1] and structured reasoning.  They are
designed to run independently so you can mix-and-match based on cost vs
thoroughness trade-offs.
"""

from __future__ import annotations

import logfire

from medici.common.llm.base import BaseLLM
from medici.eval.models import AnswerQualityMetrics, FactVerdict


async def judge_faithfulness(
    llm_client: BaseLLM,
    answer: str,
    context_chunks: list[dict],
    expected_facts: list[str],
) -> AnswerQualityMetrics:
    """Judge whether the answer is faithful to context and covers expected facts.

    Performs a single LLM call that evaluates:
    - Whether each expected fact appears in the answer
    - Whether each expected fact is supported by the retrieved context
    - An overall faithfulness score

    Returns
    -------
    AnswerQualityMetrics
        Populated with faithfulness_score, fact_verdicts, and reasoning.
    """
    if not expected_facts:
        return AnswerQualityMetrics(
            faithfulness_score=1.0,
            faithfulness_passed=True,
            reasoning="No expected facts to verify.",
        )

    context_text = "\n\n---\n\n".join(
        f"[Source: {c.get('source', '?')} | Section: {c.get('section', '?')}]\n"
        f"{c.get('text', '')[:600]}"
        for c in context_chunks[:10]
    )

    facts_block = "\n".join(f"  {i + 1}. {f}" for i, f in enumerate(expected_facts))

    prompt = f"""You are a strict faithfulness judge for a RAG system evaluation.

ANSWER UNDER TEST:
{answer[:2000]}

RETRIEVED CONTEXT (used to produce the answer):
{context_text[:4000]}

EXPECTED FACTS (claims the answer should make, supported by the context):
{facts_block}

TASK:
For each expected fact, determine:
  1. "in_answer": Is the fact's claim present or clearly implied in the ANSWER?
  2. "in_context": Is the fact's claim present or clearly implied in the CONTEXT?
  3. "supported": true only if BOTH in_answer AND in_context are true.

Respond with JSON only:
{{
  "fact_verdicts": [
    {{"fact_index": 1, "in_answer": true, "in_context": true, "supported": true, "note": "brief reason"}},
    ...
  ],
  "overall_faithful": true,
  "reasoning": "one-sentence summary"
}}

"overall_faithful" should be true only if ALL facts are supported.
"""

    with logfire.span(
        "eval_judge_faithfulness",
        num_facts=len(expected_facts),
        answer_length=len(answer),
        context_chunks_count=len(context_chunks),
    ):
        try:
            response = await llm_client.complete(
                prompt, max_tokens=1024, stage_tag="eval_faithfulness"
            )
            parsed = response.parsed_json
        except Exception as exc:
            logfire.warning("faithfulness_judge_failed", error=str(exc))
            return AnswerQualityMetrics(
                faithfulness_score=0.0,
                faithfulness_passed=False,
                reasoning=f"Judge call failed: {exc}",
                fact_verdicts=[FactVerdict(fact=f, note="judge error") for f in expected_facts],
            )

    verdicts_raw = parsed.get("fact_verdicts", [])
    fact_verdicts = []
    for i, fact in enumerate(expected_facts):
        entry = next(
            (v for v in verdicts_raw if v.get("fact_index") == i + 1),
            {},
        )
        fact_verdicts.append(
            FactVerdict(
                fact=fact,
                in_answer=entry.get("in_answer", False),
                in_context=entry.get("in_context", False),
                supported=entry.get("supported", False),
                note=entry.get("note", "no verdict returned"),
            )
        )

    supported_count = sum(1 for v in fact_verdicts if v.supported)
    total = len(expected_facts)
    score = supported_count / total if total > 0 else 0.0

    result = AnswerQualityMetrics(
        faithfulness_score=round(score, 4),
        faithfulness_passed=parsed.get("overall_faithful", False),
        reasoning=parsed.get("reasoning", ""),
        fact_verdicts=fact_verdicts,
    )

    logfire.info(
        "eval_faithfulness_result",
        score=result.faithfulness_score,
        passed=result.faithfulness_passed,
        supported=supported_count,
        total=total,
    )
    return result


async def judge_relevance(
    llm_client: BaseLLM,
    query: str,
    answer: str,
) -> float:
    """Score how well the answer addresses the user's question.

    Returns
    -------
    float
        Relevance score in [0, 1].  1.0 = perfectly relevant.
    """
    prompt = f"""You are a relevance judge for a RAG system.

USER QUESTION:
{query[:500]}

SYSTEM ANSWER:
{answer[:2000]}

Rate how well the answer addresses the question on a scale of 0.0 to 1.0:
- 1.0: Directly and completely answers the question
- 0.7-0.9: Mostly answers the question with minor gaps
- 0.4-0.6: Partially relevant but misses key aspects
- 0.1-0.3: Tangentially related but doesn't answer the question
- 0.0: Completely irrelevant

Respond with JSON only:
{{"relevance_score": 0.85, "reasoning": "brief explanation"}}
"""

    with logfire.span("eval_judge_relevance", query_length=len(query)):
        try:
            response = await llm_client.complete(prompt, max_tokens=256, stage_tag="eval_relevance")
            parsed = response.parsed_json
            score = float(parsed.get("relevance_score", 0.0))
            logfire.info("eval_relevance_result", score=score)
            return round(max(0.0, min(1.0, score)), 4)
        except Exception as exc:
            logfire.warning("relevance_judge_failed", error=str(exc))
            return 0.0


async def judge_hallucination(
    llm_client: BaseLLM,
    answer: str,
    context_chunks: list[dict],
) -> float:
    """Detect hallucinated content — claims in the answer not grounded in context.

    Returns
    -------
    float
        Hallucination score in [0, 1].  0.0 = no hallucination (good).
        1.0 = fully hallucinated (bad).
    """
    if not answer.strip() or not context_chunks:
        return 0.0

    context_text = "\n\n---\n\n".join(c.get("text", "")[:500] for c in context_chunks[:8])

    prompt = f"""You are a hallucination detector for a RAG system.

RETRIEVED CONTEXT:
{context_text[:3500]}

SYSTEM ANSWER:
{answer[:2000]}

TASK:
Identify claims in the ANSWER that are NOT supported by the CONTEXT.
Score the overall hallucination level:
- 0.0: Every claim in the answer is supported by the context
- 0.1-0.3: Minor unsupported details that don't affect accuracy
- 0.4-0.6: Some significant unsupported claims
- 0.7-0.9: Most claims lack context support
- 1.0: The answer is entirely fabricated

Respond with JSON only:
{{
  "hallucination_score": 0.1,
  "unsupported_claims": ["list of claims not in context"],
  "reasoning": "brief explanation"
}}
"""

    with logfire.span("eval_judge_hallucination", answer_length=len(answer)):
        try:
            response = await llm_client.complete(
                prompt, max_tokens=512, stage_tag="eval_hallucination"
            )
            parsed = response.parsed_json
            score = float(parsed.get("hallucination_score", 0.5))
            logfire.info(
                "eval_hallucination_result",
                score=score,
                unsupported_claims=len(parsed.get("unsupported_claims", [])),
            )
            return round(max(0.0, min(1.0, score)), 4)
        except Exception as exc:
            logfire.warning("hallucination_judge_failed", error=str(exc))
            return 0.5  # uncertain
