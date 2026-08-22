"""
Unit tests for PlannerAgent.

Covers:
- decompose() with question_type='factual' (short-circuit, no LLM call)
- decompose() with non-factual type (single seed sub-question)
- plan_next_hop() — sufficient / rephrase / new_sub_question decisions
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from medici.agents.agentic.planner import PlannerAgent


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete = AsyncMock()
    return llm


@pytest.fixture
def planner(mock_llm):
    return PlannerAgent(llm_client=mock_llm)


class TestPlannerDecompose:
    async def test_factual_returns_original_question_no_llm_call(self, planner, mock_llm):
        """Factual questions should short-circuit and return [question] without calling LLM."""
        question = "What is the capital of France?"
        result = await planner.decompose(question, question_type="factual")
        assert result == [question]
        mock_llm.complete.assert_not_called()

    async def test_non_factual_calls_llm_and_returns_seed_question(self, planner, mock_llm):
        seed_q = ["What was memory usage in Q3?"]
        mock_response = MagicMock()
        mock_response.parsed_json = seed_q
        mock_llm.complete.return_value = mock_response

        result = await planner.decompose(
            "Compare Q3 vs Q4 memory usage", question_type="comparative"
        )
        assert result == seed_q
        mock_llm.complete.assert_called_once()

    async def test_prompt_contains_question(self, planner, mock_llm):
        """Verify the question text appears in the LLM prompt."""
        mock_response = MagicMock()
        mock_response.parsed_json = ["sub1"]
        mock_llm.complete.return_value = mock_response

        question = "What are the implications of climate change?"
        await planner.decompose(question, question_type="summarization")

        call_args = mock_llm.complete.call_args
        prompt = call_args[0][0]
        assert question in prompt

    async def test_stage_tag_is_planner(self, planner, mock_llm):
        """Verify the planner stage_tag is passed to LLM."""
        mock_response = MagicMock()
        mock_response.parsed_json = ["sub1"]
        mock_llm.complete.return_value = mock_response

        await planner.decompose("Some question", question_type="procedural")

        call_args = mock_llm.complete.call_args
        assert call_args[1].get("stage_tag") == "planner" or (
            len(call_args[0]) > 1 and call_args[0][1] == "planner"
        )


class TestPlanNextHop:
    """Tests for PlannerAgent.plan_next_hop()."""

    async def test_returns_sufficient(self, planner, mock_llm):
        mock_response = MagicMock()
        mock_response.parsed_json = {
            "next_action": "sufficient",
            "query": None,
            "reasoning": "All info found",
        }

        mock_response.try_parsed_json.return_value = mock_response.parsed_json
        mock_llm.complete.return_value = mock_response

        result = await planner.plan_next_hop(
            original_question="What is RAG?",
            hop_questions=["What is RAG?"],
            retrieved_context="[Chunk 0] RAG is...",
            question_category="factual",
        )
        assert result["next_action"] == "sufficient"
        assert result["query"] is None

    async def test_returns_rephrase(self, planner, mock_llm):
        mock_response = MagicMock()
        mock_response.parsed_json = {
            "next_action": "rephrase",
            "query": "better phrasing of same question",
            "reasoning": "Poor retrieval quality",
        }
        mock_response.try_parsed_json.return_value = mock_response.parsed_json
        mock_llm.complete.return_value = mock_response

        result = await planner.plan_next_hop(
            original_question="Complex question",
            hop_questions=["sub q1"],
            retrieved_context="[Chunk 0] partial info",
            question_category="analytical",
        )
        assert result["next_action"] == "rephrase"
        assert result["query"] == "better phrasing of same question"

    async def test_returns_new_sub_question(self, planner, mock_llm):
        mock_response = MagicMock()
        mock_response.parsed_json = {
            "next_action": "new_sub_question",
            "query": "what about the missing aspect?",
            "reasoning": "Gap in coverage",
        }
        mock_response.try_parsed_json.return_value = mock_response.parsed_json
        mock_llm.complete.return_value = mock_response

        result = await planner.plan_next_hop(
            original_question="Compare X and Y",
            hop_questions=["what is X?"],
            retrieved_context="[Chunk 0] X is...",
            question_category="comparative",
        )
        assert result["next_action"] == "new_sub_question"
        assert result["query"] == "what about the missing aspect?"

    async def test_prompt_contains_original_question(self, planner, mock_llm):
        mock_response = MagicMock()
        mock_response.parsed_json = {"next_action": "sufficient", "query": None, "reasoning": "ok"}
        mock_llm.complete.return_value = mock_response

        await planner.plan_next_hop(
            original_question="My specific question",
            hop_questions=["hop1"],
            retrieved_context="context",
            question_category="factual",
        )
        prompt = mock_llm.complete.call_args[0][0]
        assert "My specific question" in prompt

    async def test_prompt_contains_hop_trail(self, planner, mock_llm):
        mock_response = MagicMock()
        mock_response.parsed_json = {"next_action": "sufficient", "query": None, "reasoning": "ok"}
        mock_llm.complete.return_value = mock_response

        await planner.plan_next_hop(
            original_question="Q",
            hop_questions=["hop1", "hop2"],
            retrieved_context="ctx",
            question_category="factual",
        )
        prompt = mock_llm.complete.call_args[0][0]
        assert "hop1" in prompt
        assert "hop2" in prompt

    async def test_stage_tag_is_planner_hop(self, planner, mock_llm):
        mock_response = MagicMock()
        mock_response.parsed_json = {"next_action": "sufficient", "query": None, "reasoning": "ok"}
        mock_llm.complete.return_value = mock_response

        await planner.plan_next_hop(
            original_question="Q",
            hop_questions=["q1"],
            retrieved_context="ctx",
            question_category="factual",
        )
        call_kwargs = mock_llm.complete.call_args[1]
        assert call_kwargs.get("stage_tag") == "planner_hop"
