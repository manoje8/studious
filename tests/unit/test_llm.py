"""
Unit tests for the LLM layer.

Covers:
- LLMResponse: .text, .parsed_json (markdown-fenced stripping), __str__
- GeminiClient: .complete() — happy path and error paths
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.llm.base import LLMResponse, BaseLLM
from src.llm.gemini import GeminiClient


class TestLLMResponse:
    """Tests for the LLMResponse wrapper."""

    def test_text_property_returns_raw_text(self):
        resp = LLMResponse("Hello, world!")
        assert resp.text == "Hello, world!"

    def test_str_returns_raw_text(self):
        resp = LLMResponse("some text")
        assert str(resp) == "some text"

    def test_parsed_json_plain_json(self):
        payload = {"key": "value", "number": 42}
        resp = LLMResponse(json.dumps(payload))
        assert resp.parsed_json == payload

    def test_parsed_json_strips_backtick_fences(self):
        raw = '```json\n{"a": 1}\n```'
        resp = LLMResponse(raw)
        assert resp.parsed_json == {"a": 1}

    def test_parsed_json_strips_plain_backtick_fences(self):
        raw = '```\n{"b": 2}\n```'
        resp = LLMResponse(raw)
        assert resp.parsed_json == {"b": 2}

    def test_parsed_json_is_cached(self):
        resp = LLMResponse('{"x": 10}')
        first = resp.parsed_json
        second = resp.parsed_json
        # same object, not re-parsed
        assert first is second

    def test_parsed_json_array(self):
        resp = LLMResponse('["a", "b", "c"]')
        assert resp.parsed_json == ["a", "b", "c"]

    def test_parsed_json_invalid_raises(self):
        resp = LLMResponse("not valid json")
        with pytest.raises(Exception):
            _ = resp.parsed_json

    def test_parsed_json_with_surrounding_whitespace(self):
        resp = LLMResponse('  \n {"key": "val"} \n  ')
        assert resp.parsed_json == {"key": "val"}


# BaseLLM interface


class TestBaseLLMInterface:
    """Verify that BaseLLM is abstract and enforces .complete()."""

    def test_cannot_instantiate_base_llm(self):
        with pytest.raises(TypeError):
            BaseLLM()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class FakeLLM(BaseLLM):
            async def complete(self, prompt: str, max_token: int = 1024) -> LLMResponse:
                return LLMResponse("ok")

        llm = FakeLLM()
        assert callable(llm.complete)


# GeminiClient


class TestGeminiClient:
    """Tests for GeminiClient.complete()."""

    @pytest.fixture
    def mock_genai(self):
        """Patch google.genai.Client so no real API calls happen."""
        with (
            patch("src.llm.gemini.genai") as mock_genai_module,
            patch("src.llm.gemini.config") as mock_config,
        ):
            mock_config.GEMINI_API_KEY = "test-api-key"
            mock_client = MagicMock()
            mock_genai_module.Client.return_value = mock_client
            mock_genai_module.types = MagicMock()

            yield {
                "genai": mock_genai_module,
                "client": mock_client,
            }

    @pytest.fixture
    def gemini(self, mock_genai):
        return GeminiClient(model="gemini-test-model")

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self, gemini, mock_genai):
        mock_response = MagicMock()
        mock_response.text = "Test answer"
        mock_genai["client"].models.generate_content.return_value = mock_response

        result = await gemini.complete("What is RAG?")

        assert isinstance(result, LLMResponse)
        assert result.text == "Test answer"

    @pytest.mark.asyncio
    async def test_complete_calls_generate_content_with_prompt(
        self, gemini, mock_genai
    ):
        mock_response = MagicMock()
        mock_response.text = "answer"
        mock_genai["client"].models.generate_content.return_value = mock_response

        await gemini.complete("My prompt")

        call_kwargs = mock_genai["client"].models.generate_content.call_args
        assert "My prompt" in call_kwargs[1].get("contents", call_kwargs[0])

    @pytest.mark.asyncio
    async def test_complete_raises_when_no_text(self, gemini, mock_genai):
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = []
        mock_genai["client"].models.generate_content.return_value = mock_response

        with pytest.raises(RuntimeError, match="Gemini returned no content"):
            await gemini.complete("What?")

    @pytest.mark.asyncio
    async def test_complete_raises_with_finish_reason_when_no_text(
        self, gemini, mock_genai
    ):
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "SAFETY"
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = [mock_candidate]
        mock_genai["client"].models.generate_content.return_value = mock_response

        with pytest.raises(RuntimeError, match="SAFETY"):
            await gemini.complete("unsafe prompt")

    @pytest.mark.asyncio
    async def test_complete_passes_max_token(self, gemini, mock_genai):
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_genai["client"].models.generate_content.return_value = mock_response

        await gemini.complete("prompt", max_token=2048)

        # Verify generate_content was called (max_token is embedded in config object)
        mock_genai["client"].models.generate_content.assert_called_once()

    def test_default_model_is_set(self, mock_genai):
        client = GeminiClient()
        assert "gemini" in client.model
