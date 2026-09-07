"""
Evaluation Data Models
======================

Pydantic models for structuring evaluation inputs (golden-set cases),
outputs (per-case results), and aggregate run reports.  These models
are the contract between the golden-set file format, the runner, the
judges, and the history tracker.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# Golden-Set Case


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GoldenCase(BaseModel):
    """A single evaluation case from the golden set."""

    id: str
    query: str
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_chunks_text: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    category: str = "factual"
    difficulty: Difficulty = Difficulty.EASY
    doc_id_filter: str | None = None
    tags: list[str] = Field(default_factory=list)


class GoldenSet(BaseModel):
    """Complete golden evaluation dataset."""

    cases: list[GoldenCase]
    version: str = "1.0"
    description: str = ""


# Retrieval Metrics


class RetrievalMetrics(BaseModel):
    """Per-case retrieval quality metrics."""

    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    k: int = 5
    matched: list[dict[str, str]] = Field(default_factory=list)
    missed: list[dict[str, str]] = Field(default_factory=list)
    accepted_chunks_count: int = 0


# Fact Verdict


class FactVerdict(BaseModel):
    """Judgment on a single expected fact."""

    fact: str
    in_answer: bool = False
    in_context: bool = False
    supported: bool = False
    note: str = ""


# Answer Quality Metrics


class AnswerQualityMetrics(BaseModel):
    """Per-case answer quality scores from LLM judges."""

    faithfulness_score: float = 0.0
    faithfulness_passed: bool = False
    relevance_score: float = 0.0
    hallucination_score: float = 0.0  # 0 = no hallucination, 1 = fully hallucinated
    fact_verdicts: list[FactVerdict] = Field(default_factory=list)
    reasoning: str = ""


# Per-Case Result


class CaseResult(BaseModel):
    """Complete result for a single evaluation case."""

    case_id: str
    query: str
    category: str = "unknown"
    difficulty: str = "unknown"
    answer_preview: str = ""
    retrieval: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    answer_quality: AnswerQualityMetrics = Field(default_factory=AnswerQualityMetrics)
    elapsed_s: float = 0.0
    pipeline_error: str | None = None
    sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            self.pipeline_error is None
            and self.retrieval.recall_at_k >= 0.5
            and self.answer_quality.faithfulness_passed
        )


# Aggregate Run Report


class AggregateMetrics(BaseModel):
    """Summary statistics across all cases in a run."""

    total_cases: int = 0
    errors: int = 0

    # Retrieval
    avg_recall_at_k: float = 0.0
    avg_precision_at_k: float = 0.0
    avg_mrr: float = 0.0
    avg_ndcg_at_k: float = 0.0
    retrieval_pass_rate: float = 0.0

    # Answer quality
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_hallucination: float = 0.0
    faithfulness_pass_rate: float = 0.0

    # Latency
    avg_latency_s: float = 0.0
    p95_latency_s: float = 0.0

    # Breakdown by category
    by_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_difficulty: dict[str, dict[str, float]] = Field(default_factory=dict)


class EvalRunReport(BaseModel):
    """Complete evaluation run output."""

    run_id: str = Field(default_factory=lambda: f"eval_{int(time.time())}")
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    config: dict[str, Any] = Field(default_factory=dict)
    aggregate: AggregateMetrics = Field(default_factory=AggregateMetrics)
    results: list[CaseResult] = Field(default_factory=list)

    def to_summary_dict(self) -> dict:
        """Flat dict suitable for Logfire / LangSmith logging."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_cases": self.aggregate.total_cases,
            "errors": self.aggregate.errors,
            "avg_recall": self.aggregate.avg_recall_at_k,
            "avg_mrr": self.aggregate.avg_mrr,
            "avg_faithfulness": self.aggregate.avg_faithfulness,
            "avg_hallucination": self.aggregate.avg_hallucination,
            "retrieval_pass_rate": self.aggregate.retrieval_pass_rate,
            "faithfulness_pass_rate": self.aggregate.faithfulness_pass_rate,
            "avg_latency_s": self.aggregate.avg_latency_s,
        }
