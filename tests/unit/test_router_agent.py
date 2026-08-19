"""
Unit tests for RouterAgent.

All tests mock the LLM client so no network calls are made.
Covers:
- _build_context_prompt (empty, single turn, multi-turn, windowing)
- _parse_and_validate_response (happy path, missing fields, bad confidence, attribute errors)
- _create_fallback_response structure
- _apply_fallback_strategy (low confidence, procedural, chitchat)
- classify() happy path and exception path
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from medici.agents.agentic.router import RouterAgent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_valid_response() -> dict:
    return {
        "primary_category": "factual",
        "secondary_categories": [],
        "confidence_score": 0.9,
        "reasoning": "Simple factual query",
        "retrieval_strategy": {
            "needs_hybrid_search": True,
            "needs_multi_hop": False,
            "needs_re_ranking": True,
            "chunking_strategy": "medium",
            "confidence_threshold": 0.7,
            "max_retrieval_depth": 2,
            "target_chunks": 5,
        },
        "complexity_level": 1,
        "requires_citation": True,
        "requires_source_attribution": True,
        "suggested_model_temperature": 0.3,
    }


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete = AsyncMock()
    return llm


@pytest.fixture
def router(mock_llm):
    return RouterAgent(llm_client=mock_llm)


# ---------------------------------------------------------------------------
# _build_context_prompt
# ---------------------------------------------------------------------------


class TestBuildContextPrompt:
    def test_empty_history_returns_no_context_string(self, router):
        result = router._build_context_prompt([])
        assert "No previous conversation context" in result

    def test_none_history_returns_no_context_string(self, router):
        result = router._build_context_prompt(None)
        assert "No previous conversation context" in result

    def test_single_turn_included(self, router):
        history = [{"user": "hello", "assistant": "hi there"}]
        result = router._build_context_prompt(history)
        assert "hello" in result
        assert "hi there" in result

    def test_multi_turn_windowed(self, router):
        """Only the last context_window=5 turns should appear."""
        history = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(10)]
        result = router._build_context_prompt(history)
        # The first turns should NOT appear (window = 5)
        assert "q0" not in result
        assert "q9" in result

    def test_total_turns_reported(self, router):
        history = [{"user": "x", "assistant": "y"} for _ in range(3)]
        result = router._build_context_prompt(history)
        assert "Total turns: 3" in result

    def test_follow_up_note_present_when_history_exists(self, router):
        history = [{"user": "prev", "assistant": "answer"}]
        result = router._build_context_prompt(history)
        assert "follow-up" in result.lower() or "previous context" in result.lower()


# ---------------------------------------------------------------------------
# _parse_and_validate_response
# ---------------------------------------------------------------------------


class TestParseAndValidateResponse:
    def test_valid_response_returned_unchanged(self, router):
        mock_resp = MagicMock()
        mock_resp.parsed_json = _make_valid_response()
        result = router._parse_and_validate_response(mock_resp)
        assert result["primary_category"] == "factual"
        assert result["confidence_score"] == 0.9

    def test_missing_primary_category_returns_fallback(self, router):
        data = _make_valid_response()
        del data["primary_category"]
        mock_resp = MagicMock()
        mock_resp.parsed_json = data
        result = router._parse_and_validate_response(mock_resp)
        # Should fall back to factual
        assert result["primary_category"] == router.config["fallback_category"]

    def test_missing_confidence_score_returns_fallback(self, router):
        data = _make_valid_response()
        del data["confidence_score"]
        mock_resp = MagicMock()
        mock_resp.parsed_json = data
        result = router._parse_and_validate_response(mock_resp)
        assert result["primary_category"] == router.config["fallback_category"]

    def test_missing_retrieval_strategy_returns_fallback(self, router):
        data = _make_valid_response()
        del data["retrieval_strategy"]
        mock_resp = MagicMock()
        mock_resp.parsed_json = data
        result = router._parse_and_validate_response(mock_resp)
        assert "retrieval_strategy" in result

    def test_confidence_out_of_bounds_clamped(self, router):
        data = _make_valid_response()
        data["confidence_score"] = 1.5  # invalid
        mock_resp = MagicMock()
        mock_resp.parsed_json = data
        result = router._parse_and_validate_response(mock_resp)
        assert 0 <= result["confidence_score"] <= 1

    def test_attribute_error_on_parsed_json_returns_fallback(self, router):
        mock_resp = MagicMock()
        # Make parsed_json raise AttributeError
        type(mock_resp).parsed_json = property(
            lambda self: (_ for _ in ()).throw(AttributeError("no json"))
        )
        result = router._parse_and_validate_response(mock_resp)
        assert result["primary_category"] == router.config["fallback_category"]

    def test_type_error_on_parsed_json_returns_fallback(self, router):
        mock_resp = MagicMock()
        mock_resp.parsed_json = None  # iterating None raises TypeError
        result = router._parse_and_validate_response(mock_resp)
        assert result["primary_category"] == router.config["fallback_category"]


# ---------------------------------------------------------------------------
# _create_fallback_response
# ---------------------------------------------------------------------------


class TestCreateFallbackResponse:
    def test_fallback_has_required_keys(self, router):
        fb = router._create_fallback_response()
        assert "primary_category" in fb
        assert "retrieval_strategy" in fb
        assert "confidence_score" in fb

    def test_fallback_category_matches_config(self, router):
        fb = router._create_fallback_response()
        assert fb["primary_category"] == router.config["fallback_category"]

    def test_fallback_confidence_in_range(self, router):
        fb = router._create_fallback_response()
        assert 0 <= fb["confidence_score"] <= 1

    def test_fallback_retrieval_strategy_has_target_chunks(self, router):
        fb = router._create_fallback_response()
        assert "target_chunks" in fb["retrieval_strategy"]


# ---------------------------------------------------------------------------
# _apply_fallback_strategy
# ---------------------------------------------------------------------------


class TestApplyFallbackStrategy:
    def test_low_confidence_adds_requires_confirmation(self, router):
        data = _make_valid_response()
        data["confidence_score"] = 0.5  # below 0.7 threshold
        result = router._apply_fallback_strategy(data)
        assert result.get("requires_confirmation") is True
        assert "factual" in result.get("fallback_categories", [])

    def test_high_confidence_no_confirmation_flag(self, router):
        data = _make_valid_response()
        data["confidence_score"] = 0.95
        result = router._apply_fallback_strategy(data)
        assert result.get("requires_confirmation") is not True

    def test_procedural_upgrades_retrieval_depth(self, router):
        data = _make_valid_response()
        data["primary_category"] = "procedural"
        result = router._apply_fallback_strategy(data)
        assert result["retrieval_strategy"]["chunking_strategy"] == "large"
        assert result["retrieval_strategy"]["max_retrieval_depth"] == 3

    def test_chitchat_disables_retrieval(self, router):
        data = _make_valid_response()
        data["primary_category"] = "chitchat"
        result = router._apply_fallback_strategy(data)
        assert result["retrieval_strategy"]["target_chunks"] == 0
        assert result["requires_citation"] is False
        assert result["requires_source_attribution"] is False

    def test_factual_category_unchanged(self, router):
        data = _make_valid_response()
        data["primary_category"] = "factual"
        data["confidence_score"] = 0.9
        result = router._apply_fallback_strategy(data)
        assert result["primary_category"] == "factual"
        assert result["retrieval_strategy"]["target_chunks"] == 5


# ---------------------------------------------------------------------------
# classify() — full method
# ---------------------------------------------------------------------------


class TestClassify:
    async def test_classify_returns_valid_result(self, router, mock_llm):
        valid = _make_valid_response()
        mock_response = MagicMock()
        mock_response.parsed_json = valid
        mock_llm.complete.return_value = mock_response

        result = await router.classify("What is AI?")
        assert result["primary_category"] == "factual"

    async def test_classify_uses_conversation_history(self, router, mock_llm):
        valid = _make_valid_response()
        mock_response = MagicMock()
        mock_response.parsed_json = valid
        mock_llm.complete.return_value = mock_response

        history = [{"user": "previous q", "assistant": "previous a"}]
        result = await router.classify("Follow up?", conversation_history=history)
        assert "primary_category" in result
        # Verify LLM was called with history in prompt
        call_args = mock_llm.complete.call_args
        prompt_arg = call_args[0][0]
        assert "previous q" in prompt_arg

    async def test_classify_falls_back_on_llm_exception(self, router, mock_llm):
        mock_llm.complete.side_effect = RuntimeError("LLM down")
        result = await router.classify("Any question?")
        # Should return fallback, not raise
        assert result["primary_category"] == router.config["fallback_category"]

    async def test_classify_no_history(self, router, mock_llm):
        valid = _make_valid_response()
        mock_response = MagicMock()
        mock_response.parsed_json = valid
        mock_llm.complete.return_value = mock_response

        result = await router.classify("What is Python?", conversation_history=None)
        assert "retrieval_strategy" in result
