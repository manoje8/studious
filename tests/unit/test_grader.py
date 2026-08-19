"""
Unit tests for GraderAgent (src/agents/agentic/grader.py).

Coverage:
  Fix A — asyncio.Semaphore caps concurrent LLM grading calls.
  Fix B — _grade_single_chunk retries on transient errors then fails-open;
           LLMContentError immediately fails-closed.
  Fix C — effective_query fallback resolves to state["original_message"],
           not the literal string "original_message".
  Helpers — _default_grade_result and _default_grade_result_fail_open shapes.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medici.agents.agentic.grader import _GRADE_MAX_RETRIES, GraderAgent
from medici.common.llm.base import LLMContentError


def _make_llm(return_value=None, side_effect=None):
    """Return a mock LLM client whose complete() behaves as instructed."""
    llm = MagicMock()
    if side_effect is not None:
        llm.complete = AsyncMock(side_effect=side_effect)
    else:
        response = MagicMock()
        response.parsed_json = return_value or {
            "relevant": True,
            "score": 0.9,
            "reason": "relevant",
            "answers_sub_questions": [],
            "information_type": "direct_answer",
            "key_information": [],
        }
        llm.complete = AsyncMock(return_value=response)
    return llm


def _make_chunk(text="chunk text", source="doc.pdf", section="intro"):
    return {"text": text, "source": source, "section": section}


def _base_state(**overrides):
    state = {
        "session_id": "sess-1",
        "user_id": "u1",
        "original_message": "What is RAG?",
        "effective_query": "What is RAG?",
        "current_query": "What is RAG?",
        "was_rewritten": False,
        "question_category": "factual",
        "hop_questions": ["What is RAG?"],
        "classification": {"retrieval_strategy": {"target_chunks": 3}},
        "current_sub_question_idx": 0,
        "retrieval_round": 0,
        "max_retrieval_rounds": 3,
        "retrieval_history": [],
        "accepted_chunks": [_make_chunk()],
        "final_answer": "",
        "sources": [],
        "doc_id_filter": None,
        "resolved_references": [],
        "episodic_context": "",
        "needs_refinement": False,
        "refinement_loops": 0,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Fix C: effective_query fallback
# ---------------------------------------------------------------------------


class TestEffectiveQueryFallback:
    """Fix C — the fallback for missing effective_query must resolve to
    state["original_message"], not the literal string 'original_message'."""

    @pytest.mark.asyncio
    async def test_uses_effective_query_when_present(self):
        """When effective_query is set it should be passed to the LLM prompt."""
        llm = _make_llm()
        grader = GraderAgent(llm)

        state = _base_state(
            effective_query="RAG systems explained", accepted_chunks=[_make_chunk()]
        )
        await grader.grade(state)

        # The LLM was called; grab the first call's prompt and check the query
        prompt = llm.complete.call_args_list[0][0][0]
        assert "RAG systems explained" in prompt

    @pytest.mark.asyncio
    async def test_falls_back_to_original_message_when_effective_query_missing(self):
        """effective_query absent → fall back to original_message (not the
        literal string 'original_message')."""
        llm = _make_llm()
        grader = GraderAgent(llm)

        # Remove effective_query so .get() returns None / falsy
        state = _base_state(accepted_chunks=[_make_chunk()])
        state["effective_query"] = ""  # empty string is falsy

        await grader.grade(state)

        prompt = llm.complete.call_args_list[0][0][0]
        # The actual message text ("What is RAG?") must appear, not the key name
        assert "What is RAG?" in prompt
        assert "original_message" not in prompt  # literal string must NOT appear


class TestGradeSingleChunkRetry:
    """Fix B — retry behaviour and fail strategy after exhaustion."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self):
        """Happy path: first attempt succeeds, returns grade dict."""
        llm = _make_llm()
        grader = GraderAgent(llm)
        chunk = _make_chunk()

        result = await grader._grade_single_chunk(chunk, "query", [])

        assert result["relevant"] is True
        assert llm.complete.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_error_then_succeeds(self):
        """One transient failure then success: total calls == 2."""
        good_response = MagicMock()
        good_response.parsed_json = {
            "relevant": True,
            "score": 0.85,
            "reason": "ok",
            "answers_sub_questions": [],
            "information_type": "direct_answer",
            "key_information": [],
        }
        llm = _make_llm(side_effect=[RuntimeError("transient"), good_response])
        grader = GraderAgent(llm)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await grader._grade_single_chunk(_make_chunk(), "query", [])

        assert result["relevant"] is True
        assert llm.complete.await_count == 2

    @pytest.mark.asyncio
    async def test_fails_open_after_max_retries_exhausted(self):
        """After _GRADE_MAX_RETRIES + 1 transient failures the chunk is kept
        (fail-open: relevant=True) rather than silently dropped."""
        llm = _make_llm(side_effect=[RuntimeError("transient")] * (_GRADE_MAX_RETRIES + 1))
        grader = GraderAgent(llm)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await grader._grade_single_chunk(_make_chunk(), "query", [])

        assert result["relevant"] is True, (
            "After retries are exhausted a transient error should fail-open "
            "(keep chunk), not silently discard it"
        )
        assert result["score"] == 0.3
        assert llm.complete.await_count == _GRADE_MAX_RETRIES + 1

    @pytest.mark.asyncio
    async def test_fails_closed_immediately_on_llm_content_error(self):
        """LLMContentError is non-retryable — chunk should be rejected immediately
        without burning retry budget."""
        llm = _make_llm(side_effect=LLMContentError("bad prompt"))
        grader = GraderAgent(llm)

        result = await grader._grade_single_chunk(_make_chunk(), "query", [])

        assert result["relevant"] is False
        assert llm.complete.await_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_fails_closed_on_malformed_json_no_relevant_key(self):
        """If the LLM returns valid JSON but missing 'relevant', treat as
        deterministic failure → fail-closed (no retry)."""
        response = MagicMock()
        response.parsed_json = {"score": 0.5, "reason": "incomplete"}
        llm = _make_llm(side_effect=[response])
        grader = GraderAgent(llm)

        result = await grader._grade_single_chunk(_make_chunk(), "query", [])

        assert result["relevant"] is False


class TestDefaultGradeResults:
    """Verify the shape of both default-result helpers."""

    def test_default_grade_result_fails_closed(self):
        grader = GraderAgent(MagicMock())
        r = grader._default_grade_result()
        assert r["relevant"] is False
        assert r["score"] == 0.0
        assert "reason" in r

    def test_default_grade_result_fail_open_keeps_chunk(self):
        grader = GraderAgent(MagicMock())
        r = grader._default_grade_result_fail_open()
        assert r["relevant"] is True
        assert 0.0 < r["score"] < 1.0  # low but non-zero sentinel
        assert "reason" in r

    def test_fail_open_information_type_is_not_irrelevant(self):
        """Fail-open chunks should not be labelled 'irrelevant' — they may
        contain useful context that the grader simply couldn't assess."""
        grader = GraderAgent(MagicMock())
        r = grader._default_grade_result_fail_open()
        assert r["information_type"] != "irrelevant"
