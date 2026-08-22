#!/usr/bin/env python3
"""
RAG Evaluation Harness — Golden-Set Runner
===========================================

Runs every query in ``golden_set.json`` through the full LangGraph pipeline,
checks **retrieval recall** (do expected chunks appear in accepted_chunks?) and
**faithfulness** (LLM-as-judge: are expected facts supported by the answer and
context?), then logs results into Logfire / LangSmith so regressions surface in
the same dashboards operators already use.

Usage
-----
    # Full run (retrieval + faithfulness)
    python tests/eval/eval_runner.py

    # Retrieval-only (no LLM judge — fast, no extra cost)
    python tests/eval/eval_runner.py --skip-faithfulness

    # CI mode: exit non-zero if thresholds are not met
    python tests/eval/eval_runner.py --exit-code \\
        --threshold-retrieval 0.70 \\
        --threshold-faithfulness 0.80
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import logfire

from medici.common.utils.config import config
from medici.common.utils.helper import has_internet

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN_SET_PATH = _EVAL_DIR / "golden_set.json"
_RESULTS_PATH = _EVAL_DIR / "eval_results.json"


async def _build_pipeline():
    """Construct the full GraphPipeline with live services.

    Mirrors the wiring in ``src/api/main.py`` so the evaluation tests the
    *real* pipeline, not a mock.
    """
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from medici.agents.agentic.grader import GraderAgent
    from medici.agents.agentic.planner import PlannerAgent
    from medici.agents.agentic.query_expander import QueryExpander
    from medici.agents.agentic.query_rewriter import QueryRewriter
    from medici.agents.agentic.router import RouterAgent
    from medici.agents.agentic.synthesizer import SynthesizerAgent
    from medici.agents.graph.graph import compile_graph_with_postgres
    from medici.agents.graph.runner import GraphPipeline
    from medici.agents.memory.short_term import ShortTermMemoryManager
    from medici.agents.retrieval import RetrievalAgent
    from medici.common.cache.embedding_cache import EmbeddingCache
    from medici.common.llm.gemini import GeminiClient
    from medici.common.llm.groq import GroqClient
    from medici.common.services.hybrid_search import HybridSearch
    from medici.common.services.qdrant import QdrantStorageService
    from medici.common.services.reranker import Reranker
    from medici.ingestion.embedding import EmbeddingService

    pool = AsyncConnectionPool(
        conninfo=config.POSTGRES_CONN_STRING,
        min_size=1,
        max_size=3,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await pool.open()
    await asyncio.wait_for(pool.wait(), timeout=15)

    gemini = GeminiClient(timeout_seconds=30, max_retries=2, model=config.GEMINI_MODEL)
    groq = GroqClient(timeout_seconds=30, max_retries=2)

    short_term = ShortTermMemoryManager(config.REDIS_URL)

    emb_cache: EmbeddingCache | None = None
    if config.EMBEDDING_CACHE_ENABLED:
        emb_cache = await EmbeddingCache.create(dsn=config.POSTGRES_CONN_STRING, max_entries=50_000)

    embedding_service = EmbeddingService(
        model_name=config.EMBEDDING_MODEL_NAME,
        dimensions=config.EMBEDDING_DIMENSIONS,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        cache=emb_cache,
    )

    storage = QdrantStorageService(
        url=config.QDRANT_CLUSTER_ENDPOINT,
        vector_size=embedding_service.vector_size,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )

    hybrid_search = HybridSearch(storage_service=storage, embedding_service=embedding_service)
    reranker = Reranker()
    query_expander = QueryExpander(gemini)
    retrieval_agent = RetrievalAgent(
        llm_client=groq,
        hybrid_search=hybrid_search,
        reranker=reranker,
        query_expand=query_expander,
    )

    graph = await compile_graph_with_postgres(
        pool=pool,
        short_term=short_term,
        rewriter=QueryRewriter(gemini),
        router=RouterAgent(groq),
        planner=PlannerAgent(gemini),
        retrieval_agent=retrieval_agent,
        grader=GraderAgent(groq),
        synthesizer=SynthesizerAgent(groq),
    )

    pipeline = GraphPipeline(
        graph,
        short_term_memory=short_term,
        semantic_cache=None,  # always bypass cache for eval
        llm_clients=[gemini, groq],
    )

    return pipeline, gemini, [pool, storage.client, short_term]


def _check_retrieval_recall(
    accepted_chunks: list[dict],
    expected_chunk_ids: list[str],
    expected_chunks_text: list[str],
) -> dict:
    """Check whether expected chunks appear in the pipeline's accepted_chunks.

    Supports two matching strategies:
      1. **Exact ID**: ``doc_id:chunk_index`` present in accepted_chunks
      2. **Text substring**: at least one accepted_chunk's text contains the
         expected substring (case-insensitive)

    Returns
    -------
    dict
        ``{"recall": float, "matched": [...], "missed": [...], "strategy": str}``
    """
    if not expected_chunk_ids and not expected_chunks_text:
        return {"recall": 1.0, "matched": [], "missed": [], "strategy": "none"}

    # Build lookup structures from accepted chunks
    chunk_ids_set = set()
    chunk_texts_lower: list[str] = []
    for c in accepted_chunks:
        cid = f"{c.get('doc_id', '')}:{c.get('chunk_index', '')}"
        chunk_ids_set.add(cid)
        chunk_texts_lower.append((c.get("text") or "").lower())

    matched = []
    missed = []

    # Strategy 1: exact chunk ID matching
    if expected_chunk_ids:
        for eid in expected_chunk_ids:
            if eid in chunk_ids_set:
                matched.append({"type": "id", "value": eid})
            else:
                missed.append({"type": "id", "value": eid})

    # Strategy 2: text-substring matching
    for substr in expected_chunks_text:
        substr_lower = substr.lower()
        if any(substr_lower in ct for ct in chunk_texts_lower):
            matched.append({"type": "text", "value": substr})
        else:
            missed.append({"type": "text", "value": substr})

    total = len(matched) + len(missed)
    recall = len(matched) / total if total > 0 else 0.0

    return {
        "recall": round(recall, 3),
        "matched": matched,
        "missed": missed,
        "strategy": "id+text" if expected_chunk_ids else "text",
    }


async def _run_single_case(
    case: dict,
    pipeline,
    judge_llm,
    skip_faithfulness: bool,
) -> dict:
    """Execute one golden-set case through the full pipeline."""

    case_id = case["id"]
    query = case["query"]

    with logfire.span(
        "eval_case",
        case_id=case_id,
        query=query[:80],
        category=case.get("category", "unknown"),
        difficulty=case.get("difficulty", "unknown"),
    ) as span:
        t0 = time.monotonic()

        # --- Run the full pipeline ---
        try:
            result = await pipeline.chat(
                user_message=query,
                session_id=f"eval_{case_id}",
                user_id="eval_runner",
            )
        except Exception as exc:
            logfire.error("eval_case_pipeline_error", case_id=case_id, error=str(exc))
            return {
                "case_id": case_id,
                "query": query,
                "pipeline_error": str(exc),
                "retrieval_pass": False,
                "retrieval_recall": 0.0,
                "faithfulness_pass": False,
                "faithfulness_score": 0.0,
                "elapsed_s": round(time.monotonic() - t0, 2),
            }

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # We need accepted_chunks from the pipeline's internal state.
        # The GraphPipeline.chat() returns a subset; to get accepted_chunks
        # we need to look at the graph's last checkpoint.  However, the
        # returned answer + sources are the primary interface, so we'll
        # infer retrieval from the answer text + sources for the ID check,
        # and rely on text-substring matching for recall.
        #
        # For a deeper integration we could modify chat() to also return
        # accepted_chunks, but that's out of scope for this harness.
        # Instead we call graph.ainvoke directly.
        # Actually — let's invoke the graph directly for accepted_chunks.

        # Re-invoke at graph level for accepted_chunks access
        initial_state = {
            "session_id": f"eval_deep_{case_id}",
            "user_id": "eval_runner",
            "original_message": query,
            "effective_query": query,
            "was_rewritten": False,
            "question_category": "",
            "hop_questions": [],
            "current_hop": 0,
            "max_hops": config.MAX_HOPS,
            "current_query": query,
            "retrieval_round": 0,
            "max_retrieval_rounds": config.MAX_RETRIEVAL_ROUND,
            "retrieval_history": [],
            "accepted_chunks": [],
            "hop_decision": "",
            "final_answer": "",
            "sources": [],
            "doc_id_filter": case.get("doc_id_filter"),
            "episodic_context": "",
        }

        try:
            graph_config = {"configurable": {"thread_id": f"eval_{case_id}"}}
            final_state = await pipeline.graph.ainvoke(initial_state, config=graph_config)
            accepted_chunks = final_state.get("accepted_chunks", [])
            answer = final_state.get("final_answer", answer)
        except Exception as exc:
            logfire.warning("eval_case_graph_fallback", case_id=case_id, error=str(exc))
            accepted_chunks = []

        # --- Retrieval recall ---
        retrieval_result = _check_retrieval_recall(
            accepted_chunks=accepted_chunks,
            expected_chunk_ids=case.get("expected_chunk_ids", []),
            expected_chunks_text=case.get("expected_chunks_text", []),
        )

        retrieval_pass = retrieval_result["recall"] >= 0.5  # at least half matched
        span.set_attributes(
            {
                "retrieval_recall": retrieval_result["recall"],
                "retrieval_pass": retrieval_pass,
                "retrieval_matched": len(retrieval_result["matched"]),
                "retrieval_missed": len(retrieval_result["missed"]),
                "accepted_chunks_count": len(accepted_chunks),
            }
        )

        logfire.info(
            "eval_retrieval_result",
            case_id=case_id,
            recall=retrieval_result["recall"],
            retrieval_pass=retrieval_pass,
            matched=len(retrieval_result["matched"]),
            missed=len(retrieval_result["missed"]),
        )

        # --- Faithfulness ---
        faithfulness_result = {
            "faithful": True,
            "score": 1.0,
            "reasoning": "skipped",
            "fact_verdicts": [],
        }

        if not skip_faithfulness and case.get("expected_facts"):
            from tests.eval.faithfulness_judge import judge_faithfulness

            faithfulness_result = await judge_faithfulness(
                llm_client=judge_llm,
                answer=answer,
                context_chunks=accepted_chunks,
                expected_facts=case["expected_facts"],
            )

        faithfulness_pass = faithfulness_result["faithful"]
        span.set_attributes(
            {
                "faithfulness_pass": faithfulness_pass,
                "faithfulness_score": faithfulness_result["score"],
            }
        )

        elapsed = round(time.monotonic() - t0, 2)

        return {
            "case_id": case_id,
            "query": query,
            "category": case.get("category", "unknown"),
            "difficulty": case.get("difficulty", "unknown"),
            "answer_preview": answer[:200],
            "accepted_chunks_count": len(accepted_chunks),
            "retrieval_pass": retrieval_pass,
            "retrieval_recall": retrieval_result["recall"],
            "retrieval_matched": retrieval_result["matched"],
            "retrieval_missed": retrieval_result["missed"],
            "faithfulness_pass": faithfulness_pass,
            "faithfulness_score": faithfulness_result["score"],
            "faithfulness_reasoning": faithfulness_result.get("reasoning", ""),
            "fact_verdicts": faithfulness_result.get("fact_verdicts", []),
            "sources": sources,
            "elapsed_s": elapsed,
        }


def _print_summary(results: list[dict], skip_faithfulness: bool) -> dict:
    """Print a human-readable summary table and return aggregate metrics."""
    total = len(results)
    if total == 0:
        print("\n⚠️  No evaluation cases to report.\n")
        return {"retrieval_rate": 0.0, "faithfulness_rate": 0.0}

    ret_pass = sum(1 for r in results if r["retrieval_pass"])
    faith_pass = sum(1 for r in results if r["faithfulness_pass"])
    errors = sum(1 for r in results if "pipeline_error" in r)

    avg_recall = sum(r["retrieval_recall"] for r in results) / total
    avg_faith = sum(r["faithfulness_score"] for r in results) / total
    avg_time = sum(r["elapsed_s"] for r in results) / total

    retrieval_rate = ret_pass / total
    faithfulness_rate = faith_pass / total

    print("\n" + "=" * 78)
    print("  RAG EVALUATION RESULTS")
    print("=" * 78)
    print(f"  Total cases:          {total}")
    print(f"  Pipeline errors:      {errors}")
    print(f"  Retrieval pass:       {ret_pass}/{total}  ({retrieval_rate:.1%})")
    print(f"  Avg retrieval recall: {avg_recall:.3f}")
    if not skip_faithfulness:
        print(f"  Faithfulness pass:    {faith_pass}/{total}  ({faithfulness_rate:.1%})")
        print(f"  Avg faithfulness:     {avg_faith:.3f}")
    print(f"  Avg latency:          {avg_time:.1f}s")
    print("=" * 78)

    # Per-case table
    print(f"\n{'ID':<12} {'Cat':<14} {'Ret':>5} {'Faith':>5} {'Time':>6}  Status")
    print("-" * 65)
    for r in results:
        ret_icon = "✅" if r["retrieval_pass"] else "❌"
        faith_icon = "✅" if r["faithfulness_pass"] else ("⏭️" if skip_faithfulness else "❌")
        status = "ERROR" if "pipeline_error" in r else "OK"
        print(
            f"{r['case_id']:<12} {r.get('category', '?'):<14} "
            f"{ret_icon:>5} {faith_icon:>5} "
            f"{r['elapsed_s']:>5.1f}s  {status}"
        )

    # Missed retrievals detail
    missed_cases = [r for r in results if r.get("retrieval_missed")]
    if missed_cases:
        print(f"\n{'─' * 40}")
        print("MISSED RETRIEVALS:")
        for r in missed_cases:
            for m in r["retrieval_missed"]:
                print(f"  {r['case_id']}: [{m['type']}] {m['value'][:60]}")

    print()

    return {"retrieval_rate": retrieval_rate, "faithfulness_rate": faithfulness_rate}


async def async_main(args: argparse.Namespace) -> int:
    """Orchestrate the evaluation run."""
    golden_path = Path(args.golden_set)
    if not golden_path.exists():
        print(f"❌ Golden set not found: {golden_path}")
        return 1

    with open(golden_path) as f:
        golden = json.load(f)

    cases = golden.get("cases", [])
    if not cases:
        print("❌ Golden set contains no cases.")
        return 1

    print(f"📋 Loaded {len(cases)} evaluation cases from {golden_path.name}")

    # Bootstrap pipeline
    print("🔧 Building pipeline (connecting to Qdrant, Postgres, Redis…)")
    closers = []
    try:
        pipeline, judge_llm, resources = await _build_pipeline()
        closers = resources
    except Exception as exc:
        print(f"❌ Pipeline setup failed: {exc}")
        return 1

    print(
        f"✅ Pipeline ready. Running evaluation ({'no faithfulness' if args.skip_faithfulness else 'full'})…\n"
    )

    # Run cases sequentially (avoids rate-limit issues)
    results = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['id']}: {case['query'][:50]}…", end="", flush=True)
        try:
            result = await _run_single_case(
                case=case,
                pipeline=pipeline,
                judge_llm=judge_llm,
                skip_faithfulness=args.skip_faithfulness,
            )
            icon = "✅" if result["retrieval_pass"] and result["faithfulness_pass"] else "❌"
            print(
                f"  {icon}  (recall={result['retrieval_recall']:.2f}, {result['elapsed_s']:.1f}s)"
            )
        except Exception as exc:
            result = {
                "case_id": case["id"],
                "query": case["query"],
                "pipeline_error": str(exc),
                "retrieval_pass": False,
                "retrieval_recall": 0.0,
                "faithfulness_pass": False,
                "faithfulness_score": 0.0,
                "elapsed_s": 0,
            }
            print(f"  💥 ERROR: {exc}")
        results.append(result)

    # Summary
    metrics = _print_summary(results, args.skip_faithfulness)

    # Write results JSON
    output_path = Path(args.output) if args.output else _RESULTS_PATH
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_cases": len(results),
                "metrics": metrics,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"📝 Detailed results written to {output_path}")

    # Log aggregate to Logfire
    logfire.info(
        "eval_run_complete",
        total_cases=len(results),
        retrieval_rate=metrics["retrieval_rate"],
        faithfulness_rate=metrics["faithfulness_rate"],
        errors=sum(1 for r in results if "pipeline_error" in r),
    )

    # Cleanup
    for resource in reversed(closers):
        try:
            if hasattr(resource, "close"):
                await resource.close()
            elif hasattr(resource, "aclose"):
                await resource.aclose()
        except Exception:
            pass

    # Exit code
    if args.exit_code:
        if metrics["retrieval_rate"] < args.threshold_retrieval:
            print(
                f"\n❌ FAIL: Retrieval rate {metrics['retrieval_rate']:.1%} "
                f"< threshold {args.threshold_retrieval:.1%}"
            )
            return 1
        if (
            not args.skip_faithfulness
            and metrics["faithfulness_rate"] < args.threshold_faithfulness
        ):
            print(
                f"\n❌ FAIL: Faithfulness rate {metrics['faithfulness_rate']:.1%} "
                f"< threshold {args.threshold_faithfulness:.1%}"
            )
            return 1
        print("\n✅ PASS: All thresholds met.")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Harness — golden-set retrieval & faithfulness testing",
    )
    parser.add_argument(
        "--golden-set",
        default=str(_GOLDEN_SET_PATH),
        help="Path to golden_set.json (default: tests/eval/golden_set.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write eval_results.json (default: tests/eval/eval_results.json)",
    )
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help="Skip LLM-as-judge faithfulness checks (faster, no extra LLM cost)",
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with code 1 if thresholds are not met (for CI gating)",
    )
    parser.add_argument(
        "--threshold-retrieval",
        type=float,
        default=0.70,
        help="Minimum retrieval pass rate (default: 0.70)",
    )
    parser.add_argument(
        "--threshold-faithfulness",
        type=float,
        default=0.80,
        help="Minimum faithfulness pass rate (default: 0.80)",
    )

    args = parser.parse_args()

    # Configure Logfire + LangSmith tracing (same as main.py)
    logfire.configure(service_name="studious-eval", send_to_logfire=has_internet())

    exit_code = asyncio.run(async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
