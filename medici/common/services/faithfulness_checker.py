"""
Runtime Faithfulness Checker
============================
Post-synthesis gate that scores how well the generated answer is supported by
the retrieved context using the ``vectara/hallucination_evaluation_model``
(a lightweight cross-encoder NLI model fine-tuned specifically for RAG
hallucination detection).

Model card: https://huggingface.co/vectara/hallucination_evaluation_model

Scoring convention
------------------
The model returns a *factual consistency score* in [0, 1].
- Score close to **1.0** → the answer is fully grounded in the premise.
- Score close to **0.0** → the answer contains hallucinated / unsupported claims.

Gate behaviour (configurable via ``FAITHFULNESS_THRESHOLD`` env-var, default 0.5):
- ``score >= threshold`` → **pass** — answer is returned as-is.
- ``score <  threshold`` → **fail** — ``faithfulness_passed=False`` is set so the
  graph can optionally fall back or annotate the answer with a warning.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import logfire

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_MODEL_NAME = "vectara/hallucination_evaluation_model"

_tokenizer = None
_model = None
_load_lock = asyncio.Lock()


def _load_model_sync():
    """Load tokenizer + model synchronously (called once inside executor)."""
    global _tokenizer, _model  # noqa: PLW0603

    if _tokenizer is not None and _model is not None:
        return

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading faithfulness model '%s' …", _MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_NAME, trust_remote_code=True
        )
        _model.eval()
        logger.info("Faithfulness model loaded successfully.")
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Could not load faithfulness model '%s': %s — checker will be disabled.",
            _MODEL_NAME,
            exc,
        )
        _tokenizer = None
        _model = None


def _score_sync(premise: str, hypothesis: str) -> float:
    """
    Run NLI inference synchronously and return a float consistency score.

    Parameters
    ----------
    premise:
        The retrieved context that the answer should be grounded in.
    hypothesis:
        The synthesized answer to evaluate.

    Returns
    -------
    float
        Factual consistency score in [0, 1].
        Returns 1.0 (always-pass) if the model is unavailable.
    """
    if _model is None or _tokenizer is None:
        return 1.0  # fail-open: don't block if model unavailable

    try:
        import torch

        inputs = _tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = _model(**inputs)
            score = torch.softmax(outputs.logits, dim=1)[0][1].item()
        return float(score)
    except Exception as exc:  # pragma: no cover
        logger.warning("Faithfulness scoring failed: %s — returning 1.0 (pass).", exc)
        return 1.0


class FaithfulnessChecker:
    """
    Async façade around the Vectara hallucination evaluation model.

    Usage
    -----
    ::

        checker = FaithfulnessChecker(threshold=0.5)
        result  = await checker.check(premise=context, hypothesis=answer)
        # result → {"score": 0.83, "passed": True, "threshold": 0.5}

    Thread-safety
    -------------
    The underlying PyTorch model is **not** async-safe, so all inference runs
    inside ``asyncio``'s default ``ThreadPoolExecutor``.  The first call
    triggers model loading (also in the executor) via a shared ``asyncio.Lock``
    so only one coroutine initialises the model even under concurrent load.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._initialized = False

    async def _ensure_loaded(self) -> None:
        """Ensure the model is loaded exactly once, even under concurrency."""
        if self._initialized:
            return
        async with _load_lock:
            if not self._initialized:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, _load_model_sync)
                self._initialized = True

    async def check(self, premise: str, hypothesis: str) -> dict:
        """
        Score the faithfulness of *hypothesis* given *premise*.

        Parameters
        ----------
        premise:
            Concatenated retrieved context (the ground-truth evidence).
        hypothesis:
            The LLM-generated answer to evaluate.

        Returns
        -------
        dict with keys:
            - ``score``     (float): factual consistency score in [0, 1].
            - ``passed``    (bool):  ``score >= threshold``.
            - ``threshold`` (float): the configured gate threshold.
        """
        await self._ensure_loaded()

        loop = asyncio.get_event_loop()
        score: float = await loop.run_in_executor(None, _score_sync, premise, hypothesis)
        passed = score >= self.threshold

        logfire.info(
            "faithfulness_check",
            score=round(score, 4),
            threshold=self.threshold,
            passed=passed,
        )

        return {
            "score": round(score, 4),
            "passed": passed,
            "threshold": self.threshold,
        }

    async def check_from_state(self, state: dict) -> dict:
        """
        Convenience wrapper that extracts premise and hypothesis from a
        LangGraph ``State`` dict.

        The *premise* is built from ``accepted_chunks[].text`` (same chunks
        the synthesizer used).  The *hypothesis* is ``final_answer``.

        Returns the same dict as :meth:`check`, plus:
            - ``skipped`` (bool): ``True`` when there are no chunks or no
              answer, in which case ``score=1.0`` and ``passed=True``.
        """
        answer: str = state.get("final_answer", "").strip()
        chunks: list[dict] = state.get("accepted_chunks") or []

        if not answer or not chunks:
            logfire.info(
                "faithfulness_check_skipped",
                reason="no_answer_or_no_chunks",
                has_answer=bool(answer),
                num_chunks=len(chunks),
            )
            return {
                "score": 1.0,
                "passed": True,
                "threshold": self.threshold,
                "skipped": True,
            }

        # Build a concise premise (cap to avoid OOM on very long contexts)
        _MAX_PREMISE_CHARS = 3_000
        premise_parts: list[str] = []
        total_chars = 0
        for chunk in chunks:
            text = chunk.get("text", "")
            if total_chars + len(text) > _MAX_PREMISE_CHARS:
                remaining = _MAX_PREMISE_CHARS - total_chars
                if remaining > 0:
                    premise_parts.append(text[:remaining])
                break
            premise_parts.append(text)
            total_chars += len(text)

        premise = " ".join(premise_parts)

        result = await self.check(premise=premise, hypothesis=answer)
        result["skipped"] = False
        return result


@lru_cache(maxsize=1)
def get_faithfulness_checker(threshold: float = 0.5) -> FaithfulnessChecker:
    """
    Return a process-wide singleton ``FaithfulnessChecker``.

    The ``threshold`` parameter is baked in at first call — subsequent calls
    return the cached instance regardless of the passed threshold.
    Call :func:`get_faithfulness_checker.cache_clear()` in tests to reset.
    """
    return FaithfulnessChecker(threshold=threshold)
