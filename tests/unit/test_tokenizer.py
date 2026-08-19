"""
Unit tests for src/common/utils/tokenizer.py

Covers:
- TokenizerInterface: abstract, cannot be instantiated directly
- Tokenizer.encode: delegates to underlying tokenizer
- Tokenizer.encode: special token error path — retries with allowed_special="all"
- Tokenizer.decode: delegates to underlying tokenizer
- Tokenizer.count: returns len(encode(text))
- TikTokenTokenizer: instantiates for a valid model
- TikTokenTokenizer: raises ValueError for invalid model
- TikTokenTokenizer: encode/decode round-trip for simple text
- TikTokenTokenizer: count returns positive int for non-empty string
"""

from unittest.mock import MagicMock, patch

import pytest

from medici.common.utils.tokenizer import TikTokenTokenizer, Tokenizer, TokenizerInterface

# ---------------------------------------------------------------------------
# TokenizerInterface (ABC)
# ---------------------------------------------------------------------------


class TestTokenizerInterface:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            TokenizerInterface()  # type: ignore

    def test_count_uses_encode(self):
        class ConcreteTokenizer(TokenizerInterface):
            def encode(self, content: str) -> list[int]:
                return [1, 2, 3]

            def decode(self, tokens: list[int]) -> str:
                return "decoded"

        t = ConcreteTokenizer()
        assert t.count("anything") == 3


# ---------------------------------------------------------------------------
# Tokenizer (wraps a TokenizerInterface)
# ---------------------------------------------------------------------------


class TestTokenizer:
    @pytest.fixture
    def mock_inner(self):
        inner = MagicMock()
        inner.encode = MagicMock(return_value=[10, 20, 30])
        inner.decode = MagicMock(return_value="decoded text")
        return inner

    def test_encode_delegates_to_inner(self, mock_inner):
        t = Tokenizer("test-model", mock_inner)
        result = t.encode("hello")
        assert result == [10, 20, 30]
        mock_inner.encode.assert_called_once_with("hello")

    def test_decode_delegates_to_inner(self, mock_inner):
        t = Tokenizer("test-model", mock_inner)
        result = t.decode([10, 20, 30])
        assert result == "decoded text"
        mock_inner.decode.assert_called_once_with([10, 20, 30])

    def test_count_returns_length_of_encode(self, mock_inner):
        mock_inner.encode.return_value = [1, 2, 3, 4, 5]
        t = Tokenizer("test-model", mock_inner)
        assert t.count("five tokens") == 5

    def test_encode_retries_with_allowed_special_on_special_token_error(self, mock_inner):
        """When encode raises an error containing 'special token', retry with allowed_special='all'."""
        mock_inner.encode.side_effect = [
            ValueError("special token not allowed"),  # first attempt
            [99, 100],  # second attempt with allowed_special
        ]
        t = Tokenizer("test-model", mock_inner)
        result = t.encode("text with <|special|>")
        assert result == [99, 100]

    def test_encode_raises_on_non_special_token_error(self, mock_inner):
        """Non-special-token errors should propagate immediately."""
        mock_inner.encode.side_effect = ValueError("some other error")
        t = Tokenizer("test-model", mock_inner)
        with pytest.raises(ValueError, match="some other error"):
            t.encode("any text")

    def test_encode_raises_if_allowed_special_not_supported(self, mock_inner):
        """If the retry with allowed_special='all' raises TypeError, propagate it."""
        mock_inner.encode.side_effect = [
            ValueError("special token disallowed"),
            TypeError("unexpected kwarg"),
        ]
        t = Tokenizer("test-model", mock_inner)
        with pytest.raises(TypeError):
            t.encode("text <|end|>")


# ---------------------------------------------------------------------------
# TikTokenTokenizer
# ---------------------------------------------------------------------------


class TestTikTokenTokenizer:
    def test_instantiates_for_valid_model(self):
        t = TikTokenTokenizer("gpt-4o-mini")
        assert t.model_name == "gpt-4o-mini"

    def test_raises_value_error_for_invalid_model(self):
        with pytest.raises(ValueError, match="Invalid model"):
            TikTokenTokenizer("nonexistent-model-xyz-999")

    def test_raises_import_error_if_tiktoken_missing(self):
        with patch.dict("sys.modules", {"tiktoken": None}):
            with pytest.raises((ImportError, Exception)):
                # Re-import to trigger the import check
                import importlib

                import medici.common.utils.tokenizer as tok_mod

                importlib.reload(tok_mod)
                tok_mod.TikTokenTokenizer()

    def test_encode_returns_list_of_ints(self):
        t = TikTokenTokenizer()
        tokens = t.encode("Hello, world!")
        assert isinstance(tokens, list)
        assert all(isinstance(tok, int) for tok in tokens)

    def test_decode_returns_string(self):
        t = TikTokenTokenizer()
        tokens = t.encode("Hello")
        result = t.decode(tokens)
        assert isinstance(result, str)
        assert "Hello" in result

    def test_count_positive_for_non_empty(self):
        t = TikTokenTokenizer()
        count = t.count("Some text with several words")
        assert count > 0

    def test_count_zero_for_empty_string(self):
        t = TikTokenTokenizer()
        assert t.count("") == 0

    def test_encode_decode_roundtrip(self):
        t = TikTokenTokenizer()
        original = "Roundtrip test text"
        decoded = t.decode(t.encode(original))
        assert decoded == original
