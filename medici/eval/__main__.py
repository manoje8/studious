"""
Evaluation CLI Entry Point
===========================

Run the full RAG evaluation harness from the command line::

    # Full evaluation (all judges)
    python -m medici.eval

    # Fast: retrieval metrics only (no LLM judges)
    python -m medici.eval --skip-faithfulness --skip-relevance --skip-hallucination

    # CI mode: exit non-zero if thresholds are not met
    python -m medici.eval --exit-code --threshold-recall 0.70 --threshold-faithfulness 0.80

    # Continuous monitoring
    python -m medici.eval --monitor --interval-hours 6

    # Show trend history
    python -m medici.eval --show-trends
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import logfire

from medici.common.utils.config import config
from medici.common.utils.helper import has_internet
from medici.eval.models import GoldenSet

_EVAL_DIR = Path(__file__).resolve().parent
_DEFAULT_GOLDEN_SET = Path("tests/eval/golden_set.json")
_DEFAULT_OUTPUT = Path("tests/eval/eval_results.json")


async def _build_pipeline():
    """
    Construct the full GraphPipeline with live services.

    Mirrors the wiring in ``medici/api/main.py`` so the evaluation tests
    the *real* pipeline, not a mock.
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
    groq = GroqClient(timeout_seconds=30, max_retries=2, model=config.GROQ_MODEL)

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


def _load_golden_set(path: Path) -> GoldenSet:
    """Load and validate a golden set JSON file."""
    with open(path) as f:
        raw = json.load(f)
    return GoldenSet.model_validate(raw)


async def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Run evaluation against the golden set."""
    from medici.eval.history import EvalHistory
    from medici.eval.runner import print_report, run_evaluation

    golden_path = Path(args.golden_set)
    if not golden_path.exists():
        print(f"Golden set not found: {golden_path}")
        return 1

    golden_set = _load_golden_set(golden_path)
    print(f"Loaded {len(golden_set.cases)} evaluation cases from {golden_path.name}")

    print("Building pipeline (connecting to Qdrant, Postgres, Redis…)")
    closers = []
    try:
        pipeline, judge_llm, resources = await _build_pipeline()
        closers = resources
    except Exception as exc:
        print(f"❌ Pipeline setup failed: {exc}")
        return 1

    mode_parts = []
    if args.skip_faithfulness:
        mode_parts.append("no faithfulness")
    if args.skip_relevance:
        mode_parts.append("no relevance")
    if args.skip_hallucination:
        mode_parts.append("no hallucination")
    mode = ", ".join(mode_parts) if mode_parts else "full"
    print(f"Pipeline ready. Running evaluation ({mode})…\n")

    def _progress(result, idx, total):
        icon = "[PASS] " if result.passed else "[FAIL] "
        err = f" ERROR: {result.pipeline_error}" if result.pipeline_error else ""
        print(
            f"  [{idx}/{total}] {result.case_id}: {icon}  "
            f"(recall={result.retrieval.recall_at_k:.2f}, "
            f"mrr={result.retrieval.mrr:.2f}, "
            f"{result.elapsed_s:.1f}s){err}"
        )

    report = await run_evaluation(
        pipeline=pipeline,
        judge_llm=judge_llm,
        golden_set=golden_set,
        skip_faithfulness=args.skip_faithfulness,
        skip_relevance=args.skip_relevance,
        skip_hallucination=args.skip_hallucination,
        retrieval_k=args.retrieval_k,
        on_case_complete=_progress,
    )

    print_report(report)

    # Save results
    output_path = Path(args.output) if args.output else _DEFAULT_OUTPUT
    with open(output_path, "w") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"Detailed results written to {output_path}")

    # Save to history
    history = EvalHistory()
    regressions = history.detect_regressions(report)
    history.append(report)

    if regressions:
        from medici.eval.monitor import RegressionAlert

        alert = RegressionAlert(regressions, report)
        print(f"\n{alert.format_message()}")

    # Cleanup
    for resource in reversed(closers):
        try:
            if hasattr(resource, "close"):
                await resource.close()
            elif hasattr(resource, "aclose"):
                await resource.aclose()
        except Exception:
            pass

    # Exit code checks
    if args.exit_code:
        agg = report.aggregate
        if agg.avg_recall_at_k < args.threshold_recall:
            print(
                f"\n FAIL: Avg recall {agg.avg_recall_at_k:.3f} "
                f"< threshold {args.threshold_recall:.3f}"
            )
            return 1
        if not args.skip_faithfulness and agg.avg_faithfulness < args.threshold_faithfulness:
            print(
                f"\n FAIL: Avg faithfulness {agg.avg_faithfulness:.3f} "
                f"< threshold {args.threshold_faithfulness:.3f}"
            )
            return 1
        print("\n PASS: All thresholds met.")

    return 0


async def _cmd_monitor(args: argparse.Namespace) -> int:
    """Run continuous monitoring."""
    from medici.eval.monitor import EvalMonitor

    golden_set = _load_golden_set(Path(args.golden_set))

    print("Building pipeline for continuous monitoring…")
    try:
        pipeline, judge_llm, _resources = await _build_pipeline()
    except Exception as exc:
        print(f"Pipeline setup failed: {exc}")
        return 1

    monitor = EvalMonitor(
        pipeline=pipeline,
        judge_llm=judge_llm,
        golden_set=golden_set,
        skip_faithfulness=args.skip_faithfulness,
        skip_relevance=args.skip_relevance,
        skip_hallucination=args.skip_hallucination,
    )

    print(f"Starting continuous monitor (interval={args.interval_hours}h)…\n")
    await monitor.run_loop(
        interval_hours=args.interval_hours,
        max_runs=args.max_runs,
    )

    return 0


def _cmd_trends(args: argparse.Namespace) -> int:
    """Show evaluation trend history."""
    from medici.eval.history import EvalHistory

    history = EvalHistory()
    trends = history.trend(window=args.window)

    if not trends["recall"]:
        print("No evaluation history found. Run an evaluation first.")
        return 0

    print(f"\n{'=' * 70}")
    print("  EVALUATION TRENDS (last {args.window} runs)")
    print(f"{'=' * 70}\n")

    for metric_name, points in trends.items():
        if not points:
            continue
        values = [p["value"] for p in points]
        latest = values[-1]
        avg = sum(values) / len(values)
        direction = (
            "↑"
            if len(values) > 1 and values[-1] > values[-2]
            else "↓"
            if len(values) > 1 and values[-1] < values[-2]
            else "→"
        )

        print(f"  {metric_name:<16} latest={latest:.4f}  avg={avg:.4f}  {direction}")

    print()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Medici RAG Evaluation Harness — comprehensive retrieval & answer quality testing",
        prog="python -m medici.eval",
    )

    # Mode selection
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Run in continuous monitoring mode",
    )
    parser.add_argument(
        "--show-trends",
        action="store_true",
        help="Show evaluation trend history and exit",
    )

    # Input / output
    parser.add_argument(
        "--golden-set",
        default=str(_DEFAULT_GOLDEN_SET),
        help="Path to golden_set.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write eval results JSON",
    )

    # Judge toggles
    parser.add_argument("--skip-faithfulness", action="store_true")
    parser.add_argument("--skip-relevance", action="store_true")
    parser.add_argument("--skip-hallucination", action="store_true")

    # Metrics config
    parser.add_argument("--retrieval-k", type=int, default=5, help="K for Recall/Precision/NDCG@K")

    # CI gating
    parser.add_argument("--exit-code", action="store_true", help="Exit 1 if thresholds not met")
    parser.add_argument("--threshold-recall", type=float, default=0.70)
    parser.add_argument("--threshold-faithfulness", type=float, default=0.80)

    # Monitor settings
    parser.add_argument("--interval-hours", type=float, default=6.0)
    parser.add_argument("--max-runs", type=int, default=None)

    # Trends
    parser.add_argument("--window", type=int, default=10, help="Trend window size")

    args = parser.parse_args()

    # Configure observability
    logfire.configure(service_name="medici-eval", send_to_logfire=has_internet())

    if args.show_trends:
        sys.exit(_cmd_trends(args))
    elif args.monitor:
        exit_code = asyncio.run(_cmd_monitor(args))
    else:
        exit_code = asyncio.run(_cmd_evaluate(args))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
