"""
Continuous Evaluation Monitor
=============================

Daemon-style runner that evaluates the RAG pipeline on a schedule and
alerts on regressions.  Designed to catch the silent quality degradation
that happens as the corpus grows or shifts over time.

Usage
-----
As a standalone process::

    python -m medici.eval.monitor --interval-hours 6

Programmatically::

    monitor = EvalMonitor(pipeline, judge_llm, golden_set)
    await monitor.run_once()
    # or:
    await monitor.run_loop(interval_hours=6)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import logfire

from medici.eval.history import EvalHistory
from medici.eval.models import EvalRunReport, GoldenSet
from medici.eval.runner import print_report, run_evaluation

if TYPE_CHECKING:
    from medici.agents.graph.runner import GraphPipeline
    from medici.common.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class RegressionAlert:
    """Encapsulates a detected regression for routing to alerts."""

    def __init__(self, regressions: list[dict], report: EvalRunReport):
        self.regressions = regressions
        self.report = report

    @property
    def has_critical(self) -> bool:
        return any(r["severity"] == "critical" for r in self.regressions)

    def format_message(self) -> str:
        lines = [
            f"⚠️  RAG Evaluation Regression Detected  (run: {self.report.run_id})",
            f"   Timestamp: {self.report.timestamp}",
            "",
        ]
        for r in self.regressions:
            icon = "🔴" if r["severity"] == "critical" else "🟡"
            lines.append(
                f"   {icon} {r['metric']}: {r['previous']:.4f} → {r['current']:.4f}  "
                f"(Δ = {r['delta']:+.4f}, threshold = {r['threshold']})"
            )
        return "\n".join(lines)


class EvalMonitor:
    """Continuous evaluation monitor with regression detection.

    Parameters
    ----------
    pipeline:
        The live RAG pipeline to evaluate.
    judge_llm:
        LLM client for quality judges.
    golden_set:
        The golden evaluation cases.
    history:
        Optional history tracker.  If None, uses the default location.
    alert_callback:
        Optional async callable invoked with a :class:`RegressionAlert`
        when regressions are detected.  Use this to send Slack/email/PagerDuty.
    """

    def __init__(
        self,
        pipeline: GraphPipeline,
        judge_llm: BaseLLM,
        golden_set: GoldenSet,
        *,
        history: EvalHistory | None = None,
        alert_callback: callable | None = None,
        skip_faithfulness: bool = False,
        skip_relevance: bool = True,
        skip_hallucination: bool = True,
    ):
        self.pipeline = pipeline
        self.judge_llm = judge_llm
        self.golden_set = golden_set
        self.history = history or EvalHistory()
        self.alert_callback = alert_callback
        self._skip_faithfulness = skip_faithfulness
        self._skip_relevance = skip_relevance
        self._skip_hallucination = skip_hallucination

    async def run_once(self) -> EvalRunReport:
        """Execute a single evaluation run with regression detection."""
        logger.info("Starting evaluation run (%d cases)", len(self.golden_set.cases))

        def _progress(result, idx, total):
            icon = "✓" if result.passed else "✗"
            logger.info(
                "  [%d/%d] %s: %s  (recall=%.2f, %.1fs)",
                idx,
                total,
                result.case_id,
                icon,
                result.retrieval.recall_at_k,
                result.elapsed_s,
            )

        report = await run_evaluation(
            pipeline=self.pipeline,
            judge_llm=self.judge_llm,
            golden_set=self.golden_set,
            skip_faithfulness=self._skip_faithfulness,
            skip_relevance=self._skip_relevance,
            skip_hallucination=self._skip_hallucination,
            on_case_complete=_progress,
        )

        # Check for regressions before saving (compare against last saved run)
        regressions = self.history.detect_regressions(report)

        # Save to history
        self.history.append(report)

        # Alert on regressions
        if regressions:
            alert = RegressionAlert(regressions, report)
            logger.warning(alert.format_message())

            logfire.warning(
                "eval_regression_detected",
                run_id=report.run_id,
                regressions=regressions,
                has_critical=alert.has_critical,
            )

            if self.alert_callback:
                try:
                    await self.alert_callback(alert)
                except Exception as exc:
                    logger.error("Alert callback failed: %s", exc)

        return report

    async def run_loop(
        self,
        interval_hours: float = 6.0,
        max_runs: int | None = None,
    ) -> None:
        """Run evaluations on a fixed schedule.

        Parameters
        ----------
        interval_hours:
            Hours between evaluation runs.
        max_runs:
            Stop after this many runs.  ``None`` = run forever.
        """
        run_count = 0
        interval_s = interval_hours * 3600

        logger.info(
            "Starting continuous evaluation monitor (interval=%sh, max_runs=%s)",
            interval_hours,
            max_runs or "unlimited",
        )

        while True:
            run_count += 1
            logger.info("\n" + "=" * 60)
            logger.info("Evaluation run #%d starting at %s", run_count, time.strftime("%H:%M:%S"))

            try:
                report = await self.run_once()
                print_report(report)
            except Exception as exc:
                logger.error("Evaluation run #%d failed: %s", run_count, exc)
                logfire.error(
                    "eval_monitor_run_failed",
                    run_count=run_count,
                    error=str(exc),
                )

            if max_runs and run_count >= max_runs:
                logger.info("Reached max_runs=%d, stopping monitor.", max_runs)
                break

            logger.info("Next evaluation in %.1f hours.", interval_hours)
            await asyncio.sleep(interval_s)
