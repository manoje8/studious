"""
LLM-as-judge faithfulness checker.

Given a pipeline answer, the accepted context chunks, and a list of expected
facts, asks a cheap LLM call whether each fact is supported by both the answer
and the context.  Results are logged into the active Logfire span so they
appear alongside other pipeline telemetry.
"""

from __future__ import annotations

import logfire

from medici.common.llm.base import BaseLLM


async def judge_faithfulness(
    llm_client: BaseLLM,
    answer: str,
    context_chunks: list[dict],
    expected_facts: list[str],
) -> dict:
    """Run a single LLM-as-judge call for one evaluation case.

    Parameters
    ----------
    llm_client:
        Any ``BaseLLM`` subclass (Gemini, Groq, …).
    answer:
        The final answer produced by the pipeline.
    context_chunks:
        The ``accepted_chunks`` list from the graph's final state.
    expected_facts:
        Human-authored factual claims the answer should contain,
        drawn from the golden evaluation set.

    Returns
    -------
    dict
        ``{"faithful": bool, "score": float, "reasoning": str,
           "fact_verdicts": [{"fact": str, "supported": bool, "note": str}, …]}``
    """
    if not expected_facts:
        return {
            "faithful": True,
            "score": 1.0,
            "reasoning": "No expected facts to verify.",
            "fact_verdicts": [],
        }

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
  1. Is the fact's claim present or clearly implied in the ANSWER?
  2. Is the fact's claim supported by the RETRIEVED CONTEXT?

A fact is "supported" only if BOTH conditions hold.

Respond with JSON only:
{{
  "fact_verdicts": [
    {{"fact_index": 1, "supported": true, "note": "brief reason"}},
    ...
  ],
  "overall_faithful": true,
  "reasoning": "one-sentence summary"
}}

"overall_faithful" should be true only if ALL facts are supported.
"""

    with logfire.span(
        "faithfulness_judge",
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
            logfire.warning("faithfulness_judge_parse_failed", error=str(exc))
            return {
                "faithful": False,
                "score": 0.0,
                "reasoning": f"Judge call failed: {exc}",
                "fact_verdicts": [
                    {"fact": f, "supported": False, "note": "judge error"} for f in expected_facts
                ],
            }

    verdicts = parsed.get("fact_verdicts", [])
    supported_count = sum(1 for v in verdicts if v.get("supported", False))
    total = len(expected_facts)
    score = supported_count / total if total > 0 else 0.0

    fact_verdicts = []
    for i, fact in enumerate(expected_facts):
        verdict_entry = next(
            (v for v in verdicts if v.get("fact_index") == i + 1),
            {"supported": False, "note": "no verdict returned"},
        )
        fact_verdicts.append(
            {
                "fact": fact,
                "supported": verdict_entry.get("supported", False),
                "note": verdict_entry.get("note", ""),
            }
        )

    result = {
        "faithful": parsed.get("overall_faithful", False),
        "score": round(score, 3),
        "reasoning": parsed.get("reasoning", ""),
        "fact_verdicts": fact_verdicts,
    }

    logfire.info(
        "faithfulness_result",
        faithful=result["faithful"],
        score=result["score"],
        supported_facts=supported_count,
        total_facts=total,
    )

    return result
