"""
Unit tests for LangGraph nodes (nodes.py) and conditional edges (edges.py).

Strategy:
- Each node function is pure async: given a State dict + injected dependencies,
  it returns a partial-state dict.  We mock the dependency objects so no I/O
  occurs and assert the exact keys/values that come back.
- Each edge function is synchronous: given a crafted State dict it returns a
  routing string.  We test every branch exhaustively.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from medici.agents.graph.edges import (
    route_after_classify,
    route_after_grade,
    route_after_hop_check,
)
from medici.agents.graph.nodes import (
    grade,
    hop_check,
    plan,
    retrieve,
    rewrite_for_refinement,
    rewrite_query,
    route,
    synthesize,
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
        "hop_questions": ["What is RAG?"],
        "current_hop": 0,
        "max_hops": 4,
        "retrieval_round": 0,
        "max_retrieval_rounds": 3,
        "retrieval_history": [],
        "accepted_chunks": [],
        "hop_decision": "",
        "final_answer": "",
        "sources": [],
        "doc_id_filter": None,
        "resolved_references": [],
        "episodic_context": "",
    }
    state.update(overrides)
    return state


# Node: rewrite_query


class TestRewriteQueryNode:
    """Tests for the rewrite_query node."""

    @pytest.fixture
    def mock_short_term(self):
        m = MagicMock()
        session = MagicMock()
        session.turns = []
        session.to_prompt_format = MagicMock(return_value="")
        m.get_session = AsyncMock(return_value=session)
        m.append_turn = AsyncMock()
        m.create_session = AsyncMock(return_value=session)
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
            state,
            short_term=mock_short_term,
            rewriter=mock_rewriter_rewritten,
        )
        assert set(result.keys()) == {
            "current_query",
            "effective_query",
            "was_rewritten",
            "resolved_references",
        }

    @pytest.mark.asyncio
    async def test_rewritten_query_propagated(self, mock_short_term, mock_rewriter_rewritten):
        state = _base_state()
        result = await rewrite_query(
            state,
            short_term=mock_short_term,
            rewriter=mock_rewriter_rewritten,
        )
        assert result["current_query"] == "What is Retrieval-Augmented Generation?"
        assert result["effective_query"] == "What is Retrieval-Augmented Generation?"
        assert result["was_rewritten"] is True
        assert result["resolved_references"] == ["RAG"]

    @pytest.mark.asyncio
    async def test_unchanged_query(self, mock_short_term, mock_rewriter_unchanged):
        state = _base_state()
        result = await rewrite_query(
            state,
            short_term=mock_short_term,
            rewriter=mock_rewriter_unchanged,
        )
        assert result["was_rewritten"] is False
        assert result["current_query"] == "What is RAG?"

    @pytest.mark.asyncio
    async def test_session_fetched_with_correct_id(self, mock_short_term, mock_rewriter_rewritten):
        state = _base_state(session_id="my-session-42")
        await rewrite_query(
            state,
            short_term=mock_short_term,
            rewriter=mock_rewriter_rewritten,
        )
        mock_short_term.get_session.assert_awaited_once_with("my-session-42")

    @pytest.mark.asyncio
    async def test_rewriter_called_with_original_message(
        self, mock_short_term, mock_rewriter_rewritten
    ):
        state = _base_state(original_message="Tell me about vector databases")
        await rewrite_query(
            state,
            short_term=mock_short_term,
            rewriter=mock_rewriter_rewritten,
        )
        mock_rewriter_rewritten.rewrite.assert_awaited_once()
        call_args = mock_rewriter_rewritten.rewrite.call_args[0]
        assert call_args[0] == "Tell me about vector databases"


# Node: route


class TestRouteNode:
    """Tests for the route node."""

    @pytest.fixture
    def mock_router(self):
        m = MagicMock()
        m.classify = AsyncMock(return_value={"primary_category": "factual"})
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
        mock_router.classify.assert_awaited_once_with(
            "How do embeddings work?", conversation_history=[]
        )

    @pytest.mark.asyncio
    async def test_different_categories(self, mock_router):
        for category in ["factual", "analytical", "comparative"]:
            mock_router.classify = AsyncMock(return_value={"primary_category": category})
            state = _base_state()
            result = await route(state, router=mock_router)
            assert result["question_category"] == category


# Node: plan


class TestPlanNode:
    """Tests for the plan node."""

    @pytest.fixture
    def mock_planner(self):
        m = MagicMock()
        m.decompose = AsyncMock(return_value=["seed sub-question"])
        return m

    @pytest.mark.asyncio
    async def test_returns_planning_keys(self, mock_planner):
        state = _base_state()
        result = await plan(state, planner=mock_planner)
        assert set(result.keys()) == {"hop_questions", "current_hop", "retrieval_round"}

    @pytest.mark.asyncio
    async def test_initialises_hop_and_round_to_zero(self, mock_planner):
        state = _base_state()
        result = await plan(state, planner=mock_planner)
        assert result["current_hop"] == 0
        assert result["retrieval_round"] == 0

    @pytest.mark.asyncio
    async def test_hop_questions_set(self, mock_planner):
        state = _base_state()
        result = await plan(state, planner=mock_planner)
        assert result["hop_questions"] == ["seed sub-question"]

    @pytest.mark.asyncio
    async def test_planner_called_with_query_and_category(self, mock_planner):
        state = _base_state(
            effective_query="Compare FAISS and Qdrant", question_category="comparative"
        )
        await plan(state, planner=mock_planner)
        mock_planner.decompose.assert_awaited_once_with("Compare FAISS and Qdrant", "comparative")


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
        existing = [{"query": "old", "decision": "sufficient", "reasoning": "", "chunks": []}]
        state = _base_state(retrieval_history=existing, retrieval_round=1)
        result = await retrieve(state, retrieval_agent=mock_retrieval_agent)
        assert len(result["retrieval_history"]) == 2
        assert result["retrieval_history"][0]["query"] == "old"

    @pytest.mark.asyncio
    async def test_retrieval_agent_receives_last_hop_question(self, mock_retrieval_agent):
        state = _base_state(
            hop_questions=["q0", "q1"],
            current_query="q1",
        )
        await retrieve(state, retrieval_agent=mock_retrieval_agent)
        call_kwargs = mock_retrieval_agent.retrieve_and_evaluate.call_args[1]
        assert call_kwargs["original_question"] == "q1"


# Node: hop_check


class TestHopCheckNode:
    """Tests for the hop_check node."""

    @pytest.fixture
    def mock_planner_sufficient(self):
        m = MagicMock()
        m.plan_next_hop = AsyncMock(
            return_value={
                "next_action": "sufficient",
                "query": None,
                "reasoning": "All needed info retrieved",
            }
        )
        return m

    @pytest.fixture
    def mock_planner_rephrase(self):
        m = MagicMock()
        m.plan_next_hop = AsyncMock(
            return_value={
                "next_action": "rephrase",
                "query": "better phrased query",
                "reasoning": "Need better retrieval phrasing",
            }
        )
        return m

    @pytest.fixture
    def mock_planner_new_sub_q(self):
        m = MagicMock()
        m.plan_next_hop = AsyncMock(
            return_value={
                "next_action": "new_sub_question",
                "query": "what about aspect X?",
                "reasoning": "Missing info on aspect X",
            }
        )
        return m

    @pytest.mark.asyncio
    async def test_sufficient_sets_hop_decision(self, mock_planner_sufficient):
        history = [
            {"query": "q", "decision": "sufficient", "reasoning": "", "chunks": [{"text": "c1"}]}
        ]
        state = _base_state(retrieval_history=history, current_hop=0, max_hops=4)
        result = await hop_check(state, planner=mock_planner_sufficient)
        assert result["hop_decision"] == "sufficient"

    @pytest.mark.asyncio
    async def test_rephrase_sets_retrieve_again(self, mock_planner_rephrase):
        history = [{"query": "q", "decision": "refine_query", "reasoning": "", "chunks": []}]
        state = _base_state(retrieval_history=history, current_hop=0, max_hops=4)
        result = await hop_check(state, planner=mock_planner_rephrase)
        assert result["hop_decision"] == "retrieve_again"
        assert result["current_query"] == "better phrased query"
        assert result["current_hop"] == 1

    @pytest.mark.asyncio
    async def test_new_sub_question_sets_retrieve_again(self, mock_planner_new_sub_q):
        history = [
            {"query": "q", "decision": "sufficient", "reasoning": "", "chunks": [{"text": "c1"}]}
        ]
        state = _base_state(retrieval_history=history, current_hop=0, max_hops=4)
        result = await hop_check(state, planner=mock_planner_new_sub_q)
        assert result["hop_decision"] == "retrieve_again"
        assert result["current_query"] == "what about aspect X?"

    @pytest.mark.asyncio
    async def test_appends_to_hop_questions(self, mock_planner_new_sub_q):
        history = [{"query": "q", "decision": "sufficient", "reasoning": "", "chunks": []}]
        state = _base_state(
            retrieval_history=history,
            hop_questions=["q0"],
            current_hop=0,
            max_hops=4,
        )
        result = await hop_check(state, planner=mock_planner_new_sub_q)
        assert result["hop_questions"] == ["q0", "what about aspect X?"]

    @pytest.mark.asyncio
    async def test_accumulates_accepted_chunks(self, mock_planner_sufficient):
        existing = [{"text": "old chunk"}]
        new_chunk = {"text": "new chunk"}
        history = [{"query": "q", "decision": "sufficient", "reasoning": "", "chunks": [new_chunk]}]
        state = _base_state(
            accepted_chunks=existing,
            retrieval_history=history,
            current_hop=0,
            max_hops=4,
        )
        result = await hop_check(state, planner=mock_planner_sufficient)
        assert len(result["accepted_chunks"]) == 2

    @pytest.mark.asyncio
    async def test_budget_exhausted_skips_planner(self, mock_planner_new_sub_q):
        history = [{"query": "q", "decision": "sufficient", "reasoning": "", "chunks": []}]
        state = _base_state(retrieval_history=history, current_hop=4, max_hops=4)
        result = await hop_check(state, planner=mock_planner_new_sub_q)
        assert result["hop_decision"] == "exhausted"
        mock_planner_new_sub_q.plan_next_hop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resets_retrieval_round_on_new_hop(self, mock_planner_rephrase):
        history = [{"query": "q", "decision": "refine_query", "reasoning": "", "chunks": []}]
        state = _base_state(
            retrieval_history=history,
            retrieval_round=2,
            current_hop=0,
            max_hops=4,
        )
        result = await hop_check(state, planner=mock_planner_rephrase)
        assert result["retrieval_round"] == 0

    @pytest.mark.asyncio
    async def test_handles_none_accepted_chunks(self, mock_planner_sufficient):
        history = [{"query": "q", "decision": "sufficient", "reasoning": "", "chunks": []}]
        state = _base_state(accepted_chunks=None, retrieval_history=history)
        result = await hop_check(state, planner=mock_planner_sufficient)
        assert isinstance(result["accepted_chunks"], list)

    @pytest.mark.asyncio
    async def test_skip_grading_bypasses_planner(self, mock_planner_new_sub_q):
        history = [{"query": "q", "decision": "sufficient", "reasoning": "", "chunks": []}]
        state = _base_state(retrieval_history=history, current_hop=0, max_hops=4, skip_grading=True)
        result = await hop_check(state, planner=mock_planner_new_sub_q)
        assert result["hop_decision"] == "sufficient"
        mock_planner_new_sub_q.plan_next_hop.assert_not_awaited()


# Node: grade


class TestGradeNode:
    """Tests for the grade node."""

    @pytest.fixture
    def mock_grader(self):
        m = MagicMock()
        m.grade = AsyncMock(
            return_value={
                "retrieval_grade_score": 0.75,
                "grading_details": {
                    "accepted_count": 1,
                    "total_count": 1,
                    "relevance_ratio": 1.0,
                    "coverage_score": 1.0,
                    "completeness": {"level": "sufficient"},
                    "needs_refinement": False,
                    "rejected_reasons": [],
                },
                "accepted_chunks": [{"text": "graded chunk", "source": "b.pdf"}],
                "needs_refinement": False,
            }
        )
        return m

    @pytest.mark.asyncio
    async def test_grade_delegates_to_grader(self, mock_grader):
        state = _base_state(accepted_chunks=[{"text": "raw chunk"}], effective_query="q")
        await grade(state, grader=mock_grader)
        mock_grader.grade.assert_awaited_once_with(state)

    @pytest.mark.asyncio
    async def test_grade_returns_dict_with_required_keys(self, mock_grader):
        state = _base_state(accepted_chunks=[{"text": "raw chunk"}], effective_query="q")
        result = await grade(state, grader=mock_grader)
        assert "retrieval_grade_score" in result
        assert "accepted_chunks" in result
        assert "needs_refinement" in result

    @pytest.mark.asyncio
    async def test_grade_propagates_needs_refinement(self, mock_grader):
        mock_grader.grade = AsyncMock(
            return_value={
                "retrieval_grade_score": 0.2,
                "grading_details": {},
                "accepted_chunks": [],
                "needs_refinement": True,
            }
        )
        state = _base_state(accepted_chunks=[], effective_query="q")
        result = await grade(state, grader=mock_grader)
        assert result["needs_refinement"] is True


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
        state = _base_state(accepted_chunks=[{"text": "context", "source": "doc.pdf"}])
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
            {"text": "b", "source": "doc.pdf"},
            {"text": "c", "source": "other.pdf"},
        ]
        state = _base_state(accepted_chunks=chunks)
        result = await synthesize(state, synthesizer=mock_synthesizer)
        assert len(result["sources"]) == 2
        assert set(result["sources"]) == {"doc.pdf", "other.pdf"}

    @pytest.mark.asyncio
    async def test_sources_empty_when_no_chunks(self, mock_synthesizer):
        state = _base_state(accepted_chunks=[])
        result = await synthesize(state, synthesizer=mock_synthesizer)
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_chunks_without_source_field_are_skipped(self, mock_synthesizer):
        chunks = [{"text": "a", "source": "doc.pdf"}, {"text": "b"}]
        state = _base_state(accepted_chunks=chunks)
        result = await synthesize(state, synthesizer=mock_synthesizer)
        assert result["sources"] == ["doc.pdf"]

    @pytest.mark.asyncio
    async def test_synthesizer_receives_full_state(self, mock_synthesizer):
        state = _base_state(accepted_chunks=[{"text": "ctx", "source": "x.pdf"}])
        await synthesize(state, synthesizer=mock_synthesizer)
        mock_synthesizer.synthesize.assert_awaited_once_with(state)

    @pytest.mark.asyncio
    async def test_handles_none_accepted_chunks(self, mock_synthesizer):
        state = _base_state(accepted_chunks=None)
        result = await synthesize(state, synthesizer=mock_synthesizer)
        assert result["sources"] == []


# Edge: route_after_classify


class TestRouteAfterClassify:
    """Tests for the route_after_classify edge."""

    def test_chitchat_routes_to_simple_response(self):
        assert route_after_classify(_base_state(question_category="chitchat")) == "simple_response"

    def test_meta_routes_to_simple_response(self):
        assert route_after_classify(_base_state(question_category="meta")) == "simple_response"

    def test_simple_factual_routes_to_plan(self):
        state = _base_state(question_category="factual", classification={"complexity_level": 1})
        assert route_after_classify(state) == "plan"

    def test_analytical_routes_to_plan(self):
        assert route_after_classify(_base_state(question_category="analytical")) == "plan"

    def test_comparative_routes_to_plan(self):
        assert route_after_classify(_base_state(question_category="comparative")) == "plan"

    def test_procedural_routes_to_plan(self):
        assert route_after_classify(_base_state(question_category="procedural")) == "plan"

    def test_summarization_without_conversational_keywords_routes_to_plan(self):
        state = _base_state(
            question_category="summarization",
            original_message="Summarize the quarterly report",
        )
        assert route_after_classify(state) == "plan"

    def test_summarization_with_conversational_keywords_routes_to_synthesize(self):
        for keyword in ["previous", "discussed", "conversation", "recap", "summarize our"]:
            state = _base_state(
                question_category="summarization",
                original_message=f"Can you {keyword} what we talked about?",
            )
            assert route_after_classify(state) == "synthesize", (
                f"keyword '{keyword}' should route to synthesize"
            )


# Edge: route_after_hop_check


class TestRouteAfterHopCheck:
    """Tests for the route_after_hop_check conditional edge."""

    def test_retrieve_again_routes_to_retrieve(self):
        state = _base_state(hop_decision="retrieve_again")
        assert route_after_hop_check(state) == "retrieve"

    def test_sufficient_routes_to_grade(self):
        state = _base_state(hop_decision="sufficient")
        assert route_after_hop_check(state) == "grade"

    def test_exhausted_routes_to_grade(self):
        state = _base_state(hop_decision="exhausted")
        assert route_after_hop_check(state) == "grade"

    def test_unknown_decision_defaults_to_grade(self):
        state = _base_state(hop_decision="unknown_value")
        assert route_after_hop_check(state) == "grade"

    def test_global_budget_forces_grade(self):
        state = _base_state(hop_decision="retrieve_again", total_retrieval_steps=6)
        assert route_after_hop_check(state) == "grade"

    def test_within_budget_allows_retrieve(self):
        state = _base_state(hop_decision="retrieve_again", total_retrieval_steps=3)
        assert route_after_hop_check(state) == "retrieve"

    def test_skip_grading_routes_to_synthesize(self):
        state = _base_state(hop_decision="sufficient", skip_grading=True)
        assert route_after_hop_check(state) == "synthesize"


# Edge: route_after_grade


class TestRouteAfterGrade:
    """Tests for the route_after_grade conditional edge."""

    def test_needs_refinement_false_routes_to_synthesize(self):
        state = _base_state(needs_refinement=False, refinement_loops=0)
        assert route_after_grade(state) == "synthesize"

    def test_needs_refinement_true_within_budget_routes_to_rewrite(self):
        state = _base_state(needs_refinement=True, refinement_loops=0)
        assert route_after_grade(state) == "rewrite_for_refinement"

    def test_needs_refinement_true_budget_exhausted_routes_to_synthesize(self):
        state = _base_state(needs_refinement=True, refinement_loops=1)
        assert route_after_grade(state) == "synthesize"

    def test_missing_needs_refinement_defaults_to_synthesize(self):
        state = _base_state()
        assert route_after_grade(state) == "synthesize"

    def test_missing_refinement_loops_defaults_to_zero(self):
        state = _base_state(needs_refinement=True)
        assert route_after_grade(state) == "rewrite_for_refinement"


# Node: rewrite_for_refinement


class TestRewriteForRefinementNode:
    """Tests for the rewrite_for_refinement node."""

    @pytest.mark.asyncio
    async def test_increments_refinement_loops(self):
        result = await rewrite_for_refinement(_base_state(refinement_loops=0))
        assert result["refinement_loops"] == 1

    @pytest.mark.asyncio
    async def test_resets_retrieval_round_to_zero(self):
        result = await rewrite_for_refinement(_base_state(retrieval_round=3))
        assert result["retrieval_round"] == 0

    @pytest.mark.asyncio
    async def test_resets_current_hop_to_zero(self):
        result = await rewrite_for_refinement(_base_state(current_hop=2))
        assert result["current_hop"] == 0

    @pytest.mark.asyncio
    async def test_resets_total_retrieval_steps_to_zero(self):
        state = _base_state()
        state["total_retrieval_steps"] = 5
        result = await rewrite_for_refinement(state)
        assert result["total_retrieval_steps"] == 0

    @pytest.mark.asyncio
    async def test_clears_accepted_chunks(self):
        result = await rewrite_for_refinement(_base_state(accepted_chunks=[{"text": "old chunk"}]))
        assert result["accepted_chunks"] == []

    @pytest.mark.asyncio
    async def test_clears_retrieval_history(self):
        state = _base_state(
            retrieval_history=[
                {"query": "q", "decision": "sufficient", "reasoning": "", "chunks": []}
            ]
        )
        result = await rewrite_for_refinement(state)
        assert result["retrieval_history"] == []
