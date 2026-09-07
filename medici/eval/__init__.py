"""
Medici Evaluation Infrastructure
================================

Production-grade evaluation harness for continuous RAG quality monitoring.

Modules
-------
metrics
    Pure-function retrieval metrics (Recall@K, MRR, Precision@K, NDCG@K)
    and answer quality scorers (faithfulness, relevance, hallucination).
models
    Pydantic data models for evaluation cases, results, and run reports.
runner
    Async evaluation runner that executes golden-set cases against the
    live pipeline and collects all metrics.
monitor
    Continuous monitoring daemon that runs evaluations on a schedule
    and alerts on regressions.
judges
    LLM-as-judge implementations for faithfulness, relevance, and
    hallucination detection.
history
    Tracks evaluation results over time for trend analysis and
    regression detection.
"""

from medici.eval.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
]
