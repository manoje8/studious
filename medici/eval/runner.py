"""
Evaluation Runner
=================

Async runner that executes golden-set cases against the live RAG pipeline,
collects retrieval metrics (Recall@K, Precision@K, MRR, NDCG@K) and answer
quality scores (faithfulness, relevance, hallucination), and produces a
structured :class:`EvalRunReport`.

This replaces the ad-hoc ``tests/eval/eval_runner.py`` with a properly
packaged module that can be imported and run programmatically.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import logfire

from medici.common.utils.config import config
from medici.eval.judges import (
    judge_faithfulness,
    judge_hallucination,
    judge_relevance,
)
from medici.eval.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from medici.eval.models import (
    AggregateMetrics,
    CaseResult,
    EvalRunReport,
    GoldenCase,
    GoldenSet,
    RetrievalMetrics,
)

if TYPE_CHECKING:
    from medici.agents.graph.runner import GraphPipeline
    from medici.common.llm.base import BaseLLM


async def run_single_case(
    case: GoldenCase,
    pipeline: GraphPipeline,
    judge_llm: BaseLLM,
    *,
    skip_faithfulness: bool = False,
    skip_relevance: bool = False,
    skip_hallucination: bool = False,
    retrieval_k: int = 5,
) -> CaseResult:
    """Execute one golden-set case through the full pipeline and score it."""

    with logfire.span(
        "eval_case",
        case_id=case.id,
        query=case.query[:80],
        category=case.category,
        difficulty=case.difficulty.value,
    ):
        t0 = time.monotonic()

        # Build initial state for direct graph invocation
        initial_state = {
            "session_id": f"eval_{case.id}",
            "user_id": "eval_runner",
            "original_message": case.query,
            "effective_query": case.query,
            "was_rewritten": False,
            "question_category": "",
            "hop_questions": [],
            "current_hop": 0,
            "max_hops": config.MAX_HOPS,
            "current_query": case.query,
            "retrieval_round": 0,
            "max_retrieval_rounds": config.MAX_RETRIEVAL_ROUND,
            "retrieval_history": [],
            "accepted_chunks": [],
            "hop_decision": "",
            "final_answer": "",
            "sources": [],
            "doc_id_filter": case.doc_id_filter,
            "episodic_context": "",
        }

        try:
            graph_config = {"configurable": {"thread_id": f"eval_{case.id}"}}
            final_state = await pipeline.graph.ainvoke(initial_state, config=graph_config)
            accepted_chunks = final_state.get("accepted_chunks", [])
            answer = final_state.get("final_answer", "")
            sources = final_state.get("sources", [])
        except Exception as exc:
            logfire.error("eval_case_error", case_id=case.id, error=str(exc))
            return CaseResult(
                case_id=case.id,
                query=case.query,
                category=case.category,
                difficulty=case.difficulty.value,
                pipeline_error=str(exc),
                elapsed_s=round(time.monotonic() - t0, 2),
            )

        # --- Retrieval metrics ---
        recall_result = recall_at_k(
            accepted_chunks,
            expected_chunk_ids=case.expected_chunk_ids,
            expected_chunks_text=case.expected_chunks_text,
            k=retrieval_k,
        )
        p_at_k = precision_at_k(
            accepted_chunks,
            expected_chunk_ids=case.expected_chunk_ids,
            expected_chunks_text=case.expected_chunks_text,
            k=retrieval_k,
        )
        mrr = mean_reciprocal_rank(
            accepted_chunks,
            expected_chunk_ids=case.expected_chunk_ids,
            expected_chunks_text=case.expected_chunks_text,
        )
        ndcg = ndcg_at_k(
            accepted_chunks,
            expected_chunk_ids=case.expected_chunk_ids,
            expected_chunks_text=case.expected_chunks_text,
            k=retrieval_k,
        )

        retrieval = RetrievalMetrics(
            recall_at_k=recall_result["recall"],
            precision_at_k=p_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg,
            k=retrieval_k,
            matched=recall_result["matched"],
            missed=recall_result["missed"],
            accepted_chunks_count=len(accepted_chunks),
        )

        logfire.info(
            "eval_retrieval_metrics",
            case_id=case.id,
            recall=retrieval.recall_at_k,
            precision=retrieval.precision_at_k,
            mrr=retrieval.mrr,
            ndcg=retrieval.ndcg_at_k,
        )

        # --- Answer quality judges ---
        from medici.eval.models import AnswerQualityMetrics

        answer_quality = AnswerQualityMetrics()

        if not skip_faithfulness and case.expected_facts:
            answer_quality = await judge_faithfulness(
                llm_client=judge_llm,
                answer=answer,
                context_chunks=accepted_chunks,
                expected_facts=case.expected_facts,
            )
        else:
            answer_quality.faithfulness_score = 1.0
            answer_quality.faithfulness_passed = True
            answer_quality.reasoning = "skipped"

        if not skip_relevance:
            answer_quality.relevance_score = await judge_relevance(
                llm_client=judge_llm,
                query=case.query,
                answer=answer,
            )

        if not skip_hallucination:
            answer_quality.hallucination_score = await judge_hallucination(
                llm_client=judge_llm,
                answer=answer,
                context_chunks=accepted_chunks,
            )

        elapsed = round(time.monotonic() - t0, 2)

        return CaseResult(
            case_id=case.id,
            query=case.query,
            category=case.category,
            difficulty=case.difficulty.value,
            answer_preview=answer[:300],
            retrieval=retrieval,
            answer_quality=answer_quality,
            elapsed_s=elapsed,
            sources=sources if isinstance(sources, list) else [],
        )


def _compute_aggregates(results: list[CaseResult]) -> AggregateMetrics:
    """Compute aggregate statistics from per-case results."""
    total = len(results)
    if total == 0:
        return AggregateMetrics()

    errors = sum(1 for r in results if r.pipeline_error)
    valid = [r for r in results if not r.pipeline_error]
    n_valid = len(valid) or 1  # avoid division by zero

    # Retrieval
    avg_recall = sum(r.retrieval.recall_at_k for r in valid) / n_valid
    avg_precision = sum(r.retrieval.precision_at_k for r in valid) / n_valid
    avg_mrr = sum(r.retrieval.mrr for r in valid) / n_valid
    avg_ndcg = sum(r.retrieval.ndcg_at_k for r in valid) / n_valid
    ret_pass = sum(1 for r in valid if r.retrieval.recall_at_k >= 0.5)

    # Answer quality
    avg_faith = sum(r.answer_quality.faithfulness_score for r in valid) / n_valid
    avg_rel = sum(r.answer_quality.relevance_score for r in valid) / n_valid
    avg_hall = sum(r.answer_quality.hallucination_score for r in valid) / n_valid
    faith_pass = sum(1 for r in valid if r.answer_quality.faithfulness_passed)

    # Latency
    latencies = sorted(r.elapsed_s for r in results)
    avg_lat = sum(latencies) / total
    p95_idx = int(total * 0.95)
    p95_lat = latencies[min(p95_idx, total - 1)]

    # Breakdown by category
    by_category: dict[str, dict[str, float]] = {}
    for r in valid:
        cat = r.category
        if cat not in by_category:
            by_category[cat] = {"count": 0, "recall": 0, "faithfulness": 0, "mrr": 0}
        by_category[cat]["count"] += 1
        by_category[cat]["recall"] += r.retrieval.recall_at_k
        by_category[cat]["faithfulness"] += r.answer_quality.faithfulness_score
        by_category[cat]["mrr"] += r.retrieval.mrr
    for cat in by_category:
        n = by_category[cat]["count"]
        by_category[cat]["recall"] = round(by_category[cat]["recall"] / n, 4)
        by_category[cat]["faithfulness"] = round(by_category[cat]["faithfulness"] / n, 4)
        by_category[cat]["mrr"] = round(by_category[cat]["mrr"] / n, 4)

    # Breakdown by difficulty
    by_difficulty: dict[str, dict[str, float]] = {}
    for r in valid:
        diff = r.difficulty
        if diff not in by_difficulty:
            by_difficulty[diff] = {"count": 0, "recall": 0, "faithfulness": 0, "mrr": 0}
        by_difficulty[diff]["count"] += 1
        by_difficulty[diff]["recall"] += r.retrieval.recall_at_k
        by_difficulty[diff]["faithfulness"] += r.answer_quality.faithfulness_score
        by_difficulty[diff]["mrr"] += r.retrieval.mrr
    for diff in by_difficulty:
        n = by_difficulty[diff]["count"]
        by_difficulty[diff]["recall"] = round(by_difficulty[diff]["recall"] / n, 4)
        by_difficulty[diff]["faithfulness"] = round(by_difficulty[diff]["faithfulness"] / n, 4)
        by_difficulty[diff]["mrr"] = round(by_difficulty[diff]["mrr"] / n, 4)

    return AggregateMetrics(
        total_cases=total,
        errors=errors,
        avg_recall_at_k=round(avg_recall, 4),
        avg_precision_at_k=round(avg_precision, 4),
        avg_mrr=round(avg_mrr, 4),
        avg_ndcg_at_k=round(avg_ndcg, 4),
        retrieval_pass_rate=round(ret_pass / n_valid, 4),
        avg_faithfulness=round(avg_faith, 4),
        avg_relevance=round(avg_rel, 4),
        avg_hallucination=round(avg_hall, 4),
        faithfulness_pass_rate=round(faith_pass / n_valid, 4),
        avg_latency_s=round(avg_lat, 2),
        p95_latency_s=round(p95_lat, 2),
        by_category=by_category,
        by_difficulty=by_difficulty,
    )


async def run_evaluation(
    pipeline: GraphPipeline,
    judge_llm: BaseLLM,
    golden_set: GoldenSet,
    *,
    skip_faithfulness: bool = False,
    skip_relevance: bool = False,
    skip_hallucination: bool = False,
    retrieval_k: int = 5,
    on_case_complete: callable | None = None,
) -> EvalRunReport:
    """Run a complete evaluation against a golden set.

    Parameters
    ----------
    pipeline:
        The compiled GraphPipeline to evaluate.
    judge_llm:
        LLM client for judge calls.
    golden_set:
        Golden evaluation cases.
    skip_faithfulness:
        Skip LLM faithfulness judging (faster, cheaper).
    skip_relevance:
        Skip relevance judging.
    skip_hallucination:
        Skip hallucination detection.
    retrieval_k:
        K value for Recall@K, Precision@K, NDCG@K.
    on_case_complete:
        Optional callback ``(CaseResult, index, total) -> None``.

    Returns
    -------
    EvalRunReport
        Complete evaluation report with per-case and aggregate metrics.
    """
    with logfire.span(
        "eval_run",
        total_cases=len(golden_set.cases),
        skip_faithfulness=skip_faithfulness,
    ):
        results: list[CaseResult] = []

        for i, case in enumerate(golden_set.cases):
            try:
                result = await run_single_case(
                    case=case,
                    pipeline=pipeline,
                    judge_llm=judge_llm,
                    skip_faithfulness=skip_faithfulness,
                    skip_relevance=skip_relevance,
                    skip_hallucination=skip_hallucination,
                    retrieval_k=retrieval_k,
                )
            except Exception as exc:
                result = CaseResult(
                    case_id=case.id,
                    query=case.query,
                    category=case.category,
                    difficulty=case.difficulty.value,
                    pipeline_error=str(exc),
                )

            results.append(result)

            if on_case_complete:
                on_case_complete(result, i + 1, len(golden_set.cases))

        aggregate = _compute_aggregates(results)

        report = EvalRunReport(
            config={
                "retrieval_k": retrieval_k,
                "skip_faithfulness": skip_faithfulness,
                "skip_relevance": skip_relevance,
                "skip_hallucination": skip_hallucination,
                "golden_set_version": golden_set.version,
                "total_cases": len(golden_set.cases),
            },
            aggregate=aggregate,
            results=results,
        )

        logfire.info(
            "eval_run_complete",
            **report.to_summary_dict(),
        )

        return report


def print_report(report: EvalRunReport, *, show_missed: bool = True) -> None:
    """Print a human-readable summary of an evaluation run."""
    agg = report.aggregate

    print("\n" + "=" * 82)
    print("  RAG EVALUATION REPORT")
    print(f"  Run: {report.run_id}  |  {report.timestamp}")
    print("=" * 82)
    print(f"  Total cases:       {agg.total_cases}")
    print(f"  Pipeline errors:   {agg.errors}")
    print()
    print("  RETRIEVAL METRICS")
    print(f"    Avg Recall@{report.config.get('retrieval_k', 5)}:   {agg.avg_recall_at_k:.4f}")
    print(f"    Avg Precision@{report.config.get('retrieval_k', 5)}: {agg.avg_precision_at_k:.4f}")
    print(f"    Avg MRR:          {agg.avg_mrr:.4f}")
    print(f"    Avg NDCG@{report.config.get('retrieval_k', 5)}:     {agg.avg_ndcg_at_k:.4f}")
    print(f"    Pass rate:        {agg.retrieval_pass_rate:.1%}")
    print()
    print("  ANSWER QUALITY")
    print(
        f"    Avg Faithfulness: {agg.avg_faithfulness:.4f}  (pass rate: {agg.faithfulness_pass_rate:.1%})"
    )
    print(f"    Avg Relevance:    {agg.avg_relevance:.4f}")
    print(f"    Avg Hallucination:{agg.avg_hallucination:.4f}  (lower is better)")
    print()
    print("  LATENCY")
    print(f"    Avg:              {agg.avg_latency_s:.1f}s")
    print(f"    P95:              {agg.p95_latency_s:.1f}s")
    print("=" * 82)

    # Per-case table
    hdr = (
        f"{'ID':<14} {'Cat':<12} {'Recall':>7} {'MRR':>6} {'Faith':>6} {'Hall':>6} {'Time':>6}  St"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in report.results:
        status = "ERR" if r.pipeline_error else ("\u2713" if r.passed else "\u2717")
        print(
            f"{r.case_id:<14} {r.category:<12} "
            f"{r.retrieval.recall_at_k:>7.3f} {r.retrieval.mrr:>6.3f} "
            f"{r.answer_quality.faithfulness_score:>6.3f} "
            f"{r.answer_quality.hallucination_score:>6.3f} "
            f"{r.elapsed_s:>5.1f}s  {status}"
        )

    # Category breakdown
    if agg.by_category:
        print(f"\n{'\u2500' * 50}")
        print("BY CATEGORY:")
        for cat, stats in sorted(agg.by_category.items()):
            print(
                f"  {cat:<16} n={stats['count']:<3.0f}  "
                f"recall={stats['recall']:.3f}  mrr={stats['mrr']:.3f}  "
                f"faith={stats['faithfulness']:.3f}"
            )

    # Missed retrievals
    if show_missed:
        missed_cases = [r for r in report.results if r.retrieval.missed]
        if missed_cases:
            print(f"\n{'\u2500' * 50}")
            print("MISSED RETRIEVALS:")
            for r in missed_cases:
                for m in r.retrieval.missed:
                    print(f"  {r.case_id}: [{m['type']}] {m['value'][:60]}")

    print()
