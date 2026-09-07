"""
Evaluation History Tracker
==========================

Persists evaluation run reports as JSONL for trend analysis and
regression detection.  Each line is a complete ``EvalRunReport``
serialized as JSON.

The tracker can:
- Append a new run report
- Load historical runs
- Detect regressions between the latest and previous runs
- Compute trend statistics over a rolling window

Storage location: ``{project_root}/.eval_history/runs.jsonl``
"""

from __future__ import annotations

import logging
from pathlib import Path

from medici.eval.models import EvalRunReport

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_DIR = Path(".eval_history")


class EvalHistory:
    """Append-only ledger of evaluation runs."""

    def __init__(self, history_dir: Path | None = None):
        self._dir = history_dir or _DEFAULT_HISTORY_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._runs_path = self._dir / "runs.jsonl"

    def append(self, report: EvalRunReport) -> None:
        """Append a run report to the history file."""
        with open(self._runs_path, "a") as f:
            f.write(report.model_dump_json() + "\n")
        logger.info("Saved eval run %s to history", report.run_id)

    def load_all(self) -> list[EvalRunReport]:
        """Load all historical runs, newest last."""
        if not self._runs_path.exists():
            return []

        runs = []
        for line_no, line in enumerate(self._runs_path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(EvalRunReport.model_validate_json(line))
            except Exception as exc:
                logger.warning("Skipping malformed history line %d: %s", line_no, exc)
        return runs

    def latest(self, n: int = 1) -> list[EvalRunReport]:
        """Return the N most recent runs."""
        all_runs = self.load_all()
        return all_runs[-n:]

    def detect_regressions(
        self,
        current: EvalRunReport,
        *,
        recall_threshold: float = 0.05,
        faithfulness_threshold: float = 0.05,
        mrr_threshold: float = 0.05,
        hallucination_threshold: float = 0.10,
    ) -> list[dict]:
        """Compare current run against the previous one and flag regressions.

        A regression is flagged when a metric drops by more than the
        specified threshold compared to the immediately preceding run.

        Returns
        -------
        list[dict]
            Each dict has: ``metric``, ``previous``, ``current``, ``delta``,
            ``threshold``, ``severity`` ("warning" or "critical").
        """
        previous_runs = self.latest(1)
        if not previous_runs:
            return []

        prev = previous_runs[-1].aggregate
        curr = current.aggregate
        regressions = []

        checks = [
            ("avg_recall_at_k", prev.avg_recall_at_k, curr.avg_recall_at_k, recall_threshold),
            ("avg_mrr", prev.avg_mrr, curr.avg_mrr, mrr_threshold),
            (
                "avg_faithfulness",
                prev.avg_faithfulness,
                curr.avg_faithfulness,
                faithfulness_threshold,
            ),
        ]

        for metric, prev_val, curr_val, thresh in checks:
            delta = curr_val - prev_val
            if delta < -thresh:
                severity = "critical" if abs(delta) > thresh * 2 else "warning"
                regressions.append(
                    {
                        "metric": metric,
                        "previous": round(prev_val, 4),
                        "current": round(curr_val, 4),
                        "delta": round(delta, 4),
                        "threshold": thresh,
                        "severity": severity,
                    }
                )

        # Hallucination increases are regressions (higher = worse)
        hall_delta = curr.avg_hallucination - prev.avg_hallucination
        if hall_delta > hallucination_threshold:
            severity = "critical" if hall_delta > hallucination_threshold * 2 else "warning"
            regressions.append(
                {
                    "metric": "avg_hallucination",
                    "previous": round(prev.avg_hallucination, 4),
                    "current": round(curr.avg_hallucination, 4),
                    "delta": round(hall_delta, 4),
                    "threshold": hallucination_threshold,
                    "severity": severity,
                }
            )

        return regressions

    def trend(
        self,
        window: int = 10,
    ) -> dict[str, list[dict]]:
        """Compute rolling-window trends for key metrics.

        Returns
        -------
        dict
            Mapping of metric name to list of
            ``{"run_id": str, "timestamp": str, "value": float}`` dicts.
        """
        runs = self.load_all()[-window:]

        trends: dict[str, list[dict]] = {
            "recall": [],
            "mrr": [],
            "faithfulness": [],
            "hallucination": [],
            "latency": [],
        }

        for run in runs:
            point = {"run_id": run.run_id, "timestamp": run.timestamp}
            trends["recall"].append({**point, "value": run.aggregate.avg_recall_at_k})
            trends["mrr"].append({**point, "value": run.aggregate.avg_mrr})
            trends["faithfulness"].append({**point, "value": run.aggregate.avg_faithfulness})
            trends["hallucination"].append({**point, "value": run.aggregate.avg_hallucination})
            trends["latency"].append({**point, "value": run.aggregate.avg_latency_s})

        return trends
