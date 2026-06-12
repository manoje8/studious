"""
Unit tests for LangGraph nodes (nodes.py) and conditional edges (edges.py).

Strategy:
- Each node function is pure async: given a State dict + injected dependencies,
  it returns a partial-state dict.  We mock the dependency objects so no I/O
  occurs and assert the exact keys/values that come back.
- Each edge function is synchronous: given a crafted State dict it returns a
  routing string.  We test every branch exhaustively.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.graph.nodes import (
    rewrite_query,
    route,
    plan,
    retrieve,
    refine_query,
    next_sub_question,
    grade,
    synthesize,
)
from src.agents.graph.edges import (
    route_after_classify,
    route_after_retrieve,
    route_after_next_sub_question,
)

# Helpers


def _base_state(**overrides) -> dict:
    """Return a minimal valid State dict, merging any overrides."""
    state = {
        "session_id": "sess-123",
        "user_id": "user-1",
        "original_message": "What is RAG?",
        "effective_query": "What is RAG?",
        "current_query": "What is RAG?",
        "was_rewritten": False,
        "question_category": "factual",
        "sub_questions": ["What is RAG?"],
        "current_sub_question_idx": 0,
        "retrieval_round": 0,
        "max_retrieval_rounds": 3,
        "retrieval_history": [],
        "accepted_chunks": [],
        "final_answer": "",
        "sources": [],
        "doc_id_filter": None,
        "resolved_references": [],
    }
    state.update(overrides)
    return state


# Node: rewrite_query


class TestRewriteQueryNode:
    """Tests for the rewrite_query node."""

    @pytest.fixture
    def mock_short_term(self):
        m = MagicMock()
        m.get_session = AsyncMock(return_value={"history": []})
        return m

    @pytest.fixture
    def mock_rewriter_rewritten(self):
        m = MagicMock()
        m.rewrite = AsyncMock(
            return_value={
                "rewritten_query": "What is Retrieval-Augmented Generation?",
                "was_rewritten": True,
                "resolved_references": ["RAG"],
            }
        )
        return m

    @pytest.fixture
    def mock_rewriter_unchanged(self):
        m = MagicMock()
        m.rewrite = AsyncMock(
            return_value={
                "rewritten_query": "What is RAG?",
                "was_rewritten": False,
                "resolved_references": [],
            }
        )
        return m

    @pytest.mark.asyncio
    async def test_returns_correct_keys(self, mock_short_term, mock_rewriter_rewritten):
        state = _base_state()
        result = await rewrite_query(
            state, short_term=mock_short_term, rewriter=mock_rewriter_rewritten
        )

        assert set(result.keys()) == {
            "current_query",
            "effective_query",
            "was_rewritten",
            "resolved_references",
        }

    @pytest.mark.asyncio
    async def test_rewritten_query_propagated(
        self, mock_short_term, mock_rewriter_rewritten
    ):
        state = _base_state()
        result = await rewrite_query(
            state, short_term=mock_short_term, rewriter=mock_rewriter_rewritten
        )

        assert result["current_query"] == "What is Retrieval-Augmented Generation?"
        assert result["effective_query"] == "What is Retrieval-Augmented Generation?"
        assert result["was_rewritten"] is True
        assert result["resolved_references"] == ["RAG"]

    @pytest.mark.asyncio
    async def test_unchanged_query(self, mock_short_term, mock_rewriter_unchanged):
        state = _base_state()
        result = await rewrite_query(
            state, short_term=mock_short_term, rewriter=mock_rewriter_unchanged
        )

        assert result["was_rewritten"] is False
        assert result["current_query"] == "What is RAG?"

    @pytest.mark.asyncio
    async def test_session_fetched_with_correct_id(
        self, mock_short_term, mock_rewriter_rewritten
    ):
        state = _base_state(session_id="my-session-42")
        await rewrite_query(
            state, short_term=mock_short_term, rewriter=mock_rewriter_rewritten
        )

        mock_short_term.get_session.assert_awaited_once_with("my-session-42")

    @pytest.mark.asyncio
    async def test_rewriter_called_with_original_message(
        self, mock_short_term, mock_rewriter_rewritten
    ):
        state = _base_state(original_message="Tell me about vector databases")
        await rewrite_query(
            state, short_term=mock_short_term, rewriter=mock_rewriter_rewritten
        )

        mock_rewriter_rewritten.rewrite.assert_awaited_once()
        call_args = mock_rewriter_rewritten.rewrite.call_args[0]
        assert call_args[0] == "Tell me about vector databases"


# ---------------------------------------------------------------------------
# Node: route
# ---------------------------------------------------------------------------


class TestRouteNode:
    """Tests for the route node."""

    @pytest.fixture
    def mock_router(self):
        m = MagicMock()
        m.classify = AsyncMock(return_value={"category": "factual"})
        return m

    @pytest.mark.asyncio
    async def test_returns_question_category(self, mock_router):
        state = _base_state()
        result = await route(state, router=mock_router)

        assert "question_category" in result
        assert result["question_category"] == "factual"

    @pytest.mark.asyncio
    async def test_router_called_with_effective_query(self, mock_router):
        state = _base_state(effective_query="How do embeddings work?")
        await route(state, router=mock_router)

        mock_router.classify.assert_awaited_once_with("How do embeddings work?")

    @pytest.mark.asyncio
    async def test_different_categories(self, mock_router):
        for category in ["factual", "analytical", "comparative"]:
            mock_router.classify = AsyncMock(return_value={"category": category})
            state = _base_state()
            result = await route(state, router=mock_router)
            assert result["question_category"] == category


# Node: plan


class TestPlanNode:
    """Tests for the plan node."""

    @pytest.fixture
    def mock_planner(self):
        m = MagicMock()
        m.decompose = AsyncMock(return_value=["sub-q1", "sub-q2", "sub-q3"])
        return m

    @pytest.mark.asyncio
    async def test_returns_planning_keys(self, mock_planner):
        state = _base_state()
        result = await plan(state, planner=mock_planner)

        assert set(result.keys()) == {
            "sub_questions",
            "current_sub_question_idx",
            "retrieval_round",
        }

    @pytest.mark.asyncio
    async def test_initialises_index_and_round_to_zero(self, mock_planner):
        state = _base_state()
        result = await plan(state, planner=mock_planner)

        assert result["current_sub_question_idx"] == 0
        assert result["retrieval_round"] == 0

    @pytest.mark.asyncio
    async def test_sub_questions_set(self, mock_planner):
        state = _base_state()
        result = await plan(state, planner=mock_planner)

        assert result["sub_questions"] == ["sub-q1", "sub-q2", "sub-q3"]

    @pytest.mark.asyncio
    async def test_planner_called_with_query_and_category(self, mock_planner):
        state = _base_state(
            effective_query="Compare FAISS and Qdrant", question_category="comparative"
        )
        await plan(state, planner=mock_planner)

        mock_planner.decompose.assert_awaited_once_with(
            "Compare FAISS and Qdrant", "comparative"
        )

    @pytest.mark.asyncio
    async def test_single_sub_question(self, mock_planner):
        mock_planner.decompose = AsyncMock(return_value=["only question"])
        state = _base_state()
        result = await plan(state, planner=mock_planner)

        assert len(result["sub_questions"]) == 1


# Node: retrieve


class TestRetrieveNode:
    """Tests for the retrieve node."""

    def _make_round_result(self, decision="sufficient", chunks=None):
        r = MagicMock()
        r.query_used = "effective query"
        r.decision = MagicMock()
        r.decision.value = decision
        r.reasoning = "enough context found"
        r.chunk_retrieved = chunks or [{"text": "chunk1", "source": "a.pdf"}]
        return r

    @pytest.fixture
    def mock_retrieval_agent(self):
        m = MagicMock()
        m.retrieve_and_evaluate = AsyncMock(return_value=self._make_round_result())
        return m

    @pytest.mark.asyncio
    async def test_appends_to_retrieval_history(self, mock_retrieval_agent):
        state = _base_state(retrieval_history=[], retrieval_round=0)
        result = await retrieve(state, retrieval_agent=mock_retrieval_agent)

        assert len(result["retrieval_history"]) == 1

    @pytest.mark.asyncio
    async def test_history_entry_has_correct_shape(self, mock_retrieval_agent):
        state = _base_state()
        result = await retrieve(state, retrieval_agent=mock_retrieval_agent)

        entry = result["retrieval_history"][0]
        assert "query" in entry
        assert "decision" in entry
        assert "reasoning" in entry
        assert "chunks" in entry

    @pytest.mark.asyncio
    async def test_increments_retrieval_round(self, mock_retrieval_agent):
        state = _base_state(retrieval_round=1)
        result = await retrieve(state, retrieval_agent=mock_retrieval_agent)

        assert result["retrieval_round"] == 2

    @pytest.mark.asyncio
    async def test_existing_history_preserved(self, mock_retrieval_agent):
        existing = [
            {"query": "old", "decision": "sufficient", "reasoning": "", "chunks": []}
        ]
        state = _base_state(retrieval_history=existing, retrieval_round=1)
        result = await retrieve(state, retrieval_agent=mock_retrieval_agent)

        assert len(result["retrieval_history"]) == 2
        assert result["retrieval_history"][0]["query"] == "old"

    @pytest.mark.asyncio
    async def test_retrieval_agent_receives_current_sub_question(
        self, mock_retrieval_agent
    ):
        state = _base_state(
            sub_questions=["q0", "q1"],
            current_sub_question_idx=1,
            current_query="q1",
        )
        await retrieve(state, retrieval_agent=mock_retrieval_agent)

        call_kwargs = mock_retrieval_agent.retrieve_and_evaluate.call_args[1]
        assert call_kwargs["original_question"] == "q1"


# Node: refine_query


class TestRefineQueryNode:
    """Tests for the refine_query node."""

    @pytest.fixture
    def mock_retrieval_agent(self):
        m = MagicMock()
        m.generate_refined_query = AsyncMock(return_value="refined search terms")
        return m

    @pytest.mark.asyncio
    async def test_returns_current_query_key(self, mock_retrieval_agent):
        state = _base_state(
            sub_questions=["q0"], current_sub_question_idx=0, retrieval_round=1
        )
        result = await refine_query(state, retrieval_agent=mock_retrieval_agent)

        assert "current_query" in result
        assert result["current_query"] == "refined search terms"

    @pytest.mark.asyncio
    async def test_passes_correct_sub_question(self, mock_retrieval_agent):
        state = _base_state(
            sub_questions=["q0", "q1"],
            current_sub_question_idx=1,
            retrieval_round=2,
        )
        await refine_query(state, retrieval_agent=mock_retrieval_agent)

        call_kwargs = mock_retrieval_agent.generate_refined_query.call_args[1]
        assert call_kwargs["original_question"] == "q1"
        assert call_kwargs["previous_rounds"] == 2


# Node: next_sub_question


class TestNextSubQuestionNode:
    """Tests for the next_sub_question node."""

    @pytest.mark.asyncio
    async def test_accumulates_accepted_chunks(self):
        existing = [{"text": "old chunk", "source": "x.pdf"}]
        new_chunk = {"text": "new chunk", "source": "y.pdf"}
        history = [
            {
                "query": "q",
                "decision": "sufficient",
                "reasoning": "",
                "chunks": [new_chunk],
            }
        ]

        state = _base_state(
            accepted_chunks=existing,
            retrieval_history=history,
            current_sub_question_idx=0,
            sub_questions=["q0", "q1"],
        )
        result = await next_sub_question(state)

        assert len(result["accepted_chunks"]) == 2
        assert result["accepted_chunks"][0] == existing[0]
        assert result["accepted_chunks"][1] == new_chunk

    @pytest.mark.asyncio
    async def test_increments_sub_question_index(self):
        history = [
            {"query": "q", "decision": "sufficient", "reasoning": "", "chunks": []}
        ]
        state = _base_state(
            retrieval_history=history,
            current_sub_question_idx=0,
            sub_questions=["q0", "q1"],
            accepted_chunks=[],
        )
        result = await next_sub_question(state)

        assert result["current_sub_question_idx"] == 1

    @pytest.mark.asyncio
    async def test_resets_retrieval_round(self):
        history = [
            {"query": "q", "decision": "sufficient", "reasoning": "", "chunks": []}
        ]
        state = _base_state(
            retrieval_history=history,
            current_sub_question_idx=0,
            sub_questions=["q0", "q1"],
            retrieval_round=2,
            accepted_chunks=[],
        )
        result = await next_sub_question(state)

        assert result["retrieval_round"] == 0

    @pytest.mark.asyncio
    async def test_sets_current_query_when_more_sub_questions_remain(self):
        history = [
            {"query": "q", "decision": "sufficient", "reasoning": "", "chunks": []}
        ]
        state = _base_state(
            retrieval_history=history,
            current_sub_question_idx=0,
            sub_questions=["q0", "q1"],
            accepted_chunks=[],
        )
        result = await next_sub_question(state)

        # next idx=1, sub_questions[1] = "q1"
        assert result["current_query"] == "q1"

    @pytest.mark.asyncio
    async def test_no_current_query_set_when_no_more_sub_questions(self):
        """When advancing past the last sub-question, current_query is not touched."""
        history = [
            {"query": "q", "decision": "sufficient", "reasoning": "", "chunks": []}
        ]
        state = _base_state(
            retrieval_history=history,
            current_sub_question_idx=0,
            sub_questions=["q0"],  # only one question
            accepted_chunks=[],
        )
        result = await next_sub_question(state)

        # next idx=1, but len(sub_questions)==1, so current_query should NOT be set
        assert "current_query" not in result

    @pytest.mark.asyncio
    async def test_handles_none_accepted_chunks(self):
        """accepted_chunks=None in state should not raise."""
        history = [
            {"query": "q", "decision": "sufficient", "reasoning": "", "chunks": []}
        ]
        state = _base_state(
            retrieval_history=history,
            current_sub_question_idx=0,
            sub_questions=["q0", "q1"],
            accepted_chunks=None,
        )
        result = await next_sub_question(state)

        assert isinstance(result["accepted_chunks"], list)


# Node: grade


class TestGradeNode:
    """Tests for the grade node."""

    @pytest.fixture
    def mock_grader(self):
        m = MagicMock()
        m.grade_chunks = AsyncMock(
            return_value=[{"text": "graded chunk", "source": "b.pdf"}]
        )
        return m

    @pytest.mark.asyncio
    async def test_returns_accepted_chunks_key(self, mock_grader):
        state = _base_state(
            accepted_chunks=[{"text": "raw chunk"}], effective_query="q"
        )
        result = await grade(state, grader=mock_grader)

        assert "accepted_chunks" in result

    @pytest.mark.asyncio
    async def test_graded_chunks_are_returned(self, mock_grader):
        state = _base_state(
            accepted_chunks=[{"text": "raw chunk"}], effective_query="q"
        )
        result = await grade(state, grader=mock_grader)

        assert result["accepted_chunks"] == [
            {"text": "graded chunk", "source": "b.pdf"}
        ]

    @pytest.mark.asyncio
    async def test_grader_called_with_chunks_and_query(self, mock_grader):
        chunks = [{"text": "chunk1"}, {"text": "chunk2"}]
        state = _base_state(accepted_chunks=chunks, effective_query="test query")
        await grade(state, grader=mock_grader)

        mock_grader.grade_chunks.assert_awaited_once_with(chunks, "test query")

    @pytest.mark.asyncio
    async def test_empty_chunks_returned_when_all_rejected(self, mock_grader):
        mock_grader.grade_chunks = AsyncMock(return_value=[])
        state = _base_state(
            accepted_chunks=[{"text": "irrelevant"}], effective_query="q"
        )
        result = await grade(state, grader=mock_grader)

        assert result["accepted_chunks"] == []


# Node: synthesize


class TestSynthesizeNode:
    """Tests for the synthesize node."""

    @pytest.fixture
    def mock_synthesizer(self):
        m = MagicMock()
        m.synthesize = AsyncMock(return_value="The answer is 42.")
        return m

    @pytest.mark.asyncio
    async def test_returns_final_answer_and_sources(self, mock_synthesizer):
        state = _base_state(
            accepted_chunks=[{"text": "context", "source": "doc.pdf"}],
            effective_query="q",
        )
        result = await synthesize(state, synthesizer=mock_synthesizer)

        assert "final_answer" in result
        assert "sources" in result

    @pytest.mark.asyncio
    async def test_answer_is_synthesizer_output(self, mock_synthesizer):
        state = _base_state(accepted_chunks=[], effective_query="q")
        result = await synthesize(state, synthesizer=mock_synthesizer)

        assert result["final_answer"] == "The answer is 42."

    @pytest.mark.asyncio
    async def test_sources_deduplicated(self, mock_synthesizer):
        chunks = [
            {"text": "a", "source": "doc.pdf"},
            {"text": "b", "source": "doc.pdf"},  # same source
            {"text": "c", "source": "other.pdf"},
        ]
        state = _base_state(accepted_chunks=chunks, effective_query="q")
        result = await synthesize(state, synthesizer=mock_synthesizer)

        assert len(result["sources"]) == 2
        assert set(result["sources"]) == {"doc.pdf", "other.pdf"}

    @pytest.mark.asyncio
    async def test_sources_empty_when_no_chunks(self, mock_synthesizer):
        state = _base_state(accepted_chunks=[], effective_query="q")
        result = await synthesize(state, synthesizer=mock_synthesizer)

        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_chunks_without_source_field_are_skipped(self, mock_synthesizer):
        chunks = [
            {"text": "a", "source": "doc.pdf"},
            {"text": "b"},  # no 'source' key
        ]
        state = _base_state(accepted_chunks=chunks, effective_query="q")
        result = await synthesize(state, synthesizer=mock_synthesizer)

        assert result["sources"] == ["doc.pdf"]

    @pytest.mark.asyncio
    async def test_synthesizer_receives_full_state(self, mock_synthesizer):
        state = _base_state(accepted_chunks=[{"text": "ctx", "source": "x.pdf"}])
        await synthesize(state, synthesizer=mock_synthesizer)

        mock_synthesizer.synthesize.assert_awaited_once_with(state)

    @pytest.mark.asyncio
    async def test_handles_none_accepted_chunks(self, mock_synthesizer):
        state = _base_state(accepted_chunks=None, effective_query="q")
        result = await synthesize(state, synthesizer=mock_synthesizer)

        assert result["sources"] == []


# Edge: route_after_classify


class TestRouteAfterClassify:
    """Tests for the route_after_classify edge."""

    def test_always_returns_plan(self):
        for category in ["factual", "analytical", "comparative", "unknown"]:
            state = _base_state(question_category=category)
            assert route_after_classify(state) == "plan"


# Edge: route_after_retrieve


class TestRouteAfterRetrieve:
    """Tests for the route_after_retrieve conditional edge — all branches."""

    def _state_with_last_decision(
        self, decision, round_no, max_rounds, sub_q_idx, total_sub_qs
    ):
        history = [{"query": "q", "decision": decision, "reasoning": "", "chunks": []}]
        return _base_state(
            retrieval_history=history,
            retrieval_round=round_no,
            max_retrieval_rounds=max_rounds,
            current_sub_question_idx=sub_q_idx,
            sub_questions=[f"q{i}" for i in range(total_sub_qs)],
        )

    # --- sufficient ---
    def test_sufficient_with_more_sub_questions_goes_to_next_sub_question(self):
        state = self._state_with_last_decision(
            "sufficient", 1, 3, sub_q_idx=0, total_sub_qs=2
        )
        assert route_after_retrieve(state) == "next_sub_question"

    def test_sufficient_on_last_sub_question_goes_to_grade(self):
        state = self._state_with_last_decision(
            "sufficient", 1, 3, sub_q_idx=1, total_sub_qs=2
        )
        assert route_after_retrieve(state) == "grade"

    def test_sufficient_single_sub_question_goes_to_grade(self):
        state = self._state_with_last_decision(
            "sufficient", 1, 3, sub_q_idx=0, total_sub_qs=1
        )
        assert route_after_retrieve(state) == "grade"

    # --- refine_query ---
    def test_refine_query_within_max_rounds_goes_to_refine(self):
        state = self._state_with_last_decision(
            "refine_query", round_no=1, max_rounds=3, sub_q_idx=0, total_sub_qs=1
        )
        assert route_after_retrieve(state) == "refine_query"

    def test_refine_query_at_max_rounds_falls_through_to_grade(self):
        # round_no == max_rounds → condition `round_no < max_rounds` is False
        state = self._state_with_last_decision(
            "refine_query", round_no=3, max_rounds=3, sub_q_idx=0, total_sub_qs=1
        )
        assert route_after_retrieve(state) == "grade"

    def test_refine_query_exceeds_max_rounds_with_more_sub_qs(self):
        state = self._state_with_last_decision(
            "refine_query", round_no=3, max_rounds=3, sub_q_idx=0, total_sub_qs=2
        )
        assert route_after_retrieve(state) == "next_sub_question"

    # --- expand_search ---
    def test_expand_search_with_more_sub_questions_goes_to_next_sub_question(self):
        state = self._state_with_last_decision(
            "expand_search", 1, 3, sub_q_idx=0, total_sub_qs=2
        )
        assert route_after_retrieve(state) == "next_sub_question"

    def test_expand_search_on_last_sub_question_goes_to_grade(self):
        state = self._state_with_last_decision(
            "expand_search", 1, 3, sub_q_idx=1, total_sub_qs=2
        )
        assert route_after_retrieve(state) == "grade"

    # --- unknown / fallback ---
    def test_unknown_decision_with_more_sub_questions_goes_to_next_sub_question(self):
        state = self._state_with_last_decision(
            "unknown_decision", 1, 3, sub_q_idx=0, total_sub_qs=2
        )
        assert route_after_retrieve(state) == "next_sub_question"

    def test_unknown_decision_on_last_sub_question_goes_to_grade(self):
        state = self._state_with_last_decision(
            "unknown_decision", 1, 3, sub_q_idx=0, total_sub_qs=1
        )
        assert route_after_retrieve(state) == "grade"


# Edge: route_after_next_sub_question
class TestRouteAfterNextSubQuestion:
    """Tests for the route_after_next_sub_question edge."""

    def test_always_returns_retrieve(self):
        state = _base_state(current_sub_question_idx=1, sub_questions=["q0", "q1"])
        assert route_after_next_sub_question(state) == "retrieve"
