"""Tests for mid-stream LLM fallback corruption guard.

Verifies that ``FallbackClient.stream_complete`` behaves correctly when
the primary provider fails mid-stream versus before any tokens are yielded.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from medici.common.llm.base import BaseLLM, LLMResponse
from medici.common.llm.fallback import (
    STREAM_BREAK_SENTINEL,
    FallbackClient,
    MidStreamFallbackError,
)

# Stub LLM implementations for testing


class _StubLLM(BaseLLM):
    """Minimal LLM stub that yields pre-configured tokens."""

    def __init__(self, name: str, tokens: list[str]):
        super().__init__()
        self._name = name
        self._tokens = tokens

    @property
    def model_name(self) -> str:
        return self._name

    async def _complete_impl(self, prompt, max_tokens, system_prompt=None, **kw) -> LLMResponse:
        return LLMResponse("".join(self._tokens))

    async def _stream_impl(
        self, prompt, max_tokens, system_prompt=None, **kw
    ) -> AsyncIterator[str]:
        for t in self._tokens:
            yield t


class _FailBeforeYieldLLM(BaseLLM):
    """Stub that raises *before* yielding any token."""

    def __init__(self, name: str = "fail-before"):
        super().__init__()
        self._name = name

    @property
    def model_name(self) -> str:
        return self._name

    async def _complete_impl(self, prompt, max_tokens, system_prompt=None, **kw) -> LLMResponse:
        raise RuntimeError("primary down")

    async def _stream_impl(
        self, prompt, max_tokens, system_prompt=None, **kw
    ) -> AsyncIterator[str]:
        raise RuntimeError("primary down")
        yield  # noqa: unreachable — makes this an async generator


class _FailMidStreamLLM(BaseLLM):
    """Stub that yields ``n_before_fail`` tokens, then raises."""

    def __init__(self, name: str = "fail-mid", n_before_fail: int = 3):
        super().__init__()
        self._name = name
        self._n = n_before_fail

    @property
    def model_name(self) -> str:
        return self._name

    async def _complete_impl(self, prompt, max_tokens, system_prompt=None, **kw) -> LLMResponse:
        raise RuntimeError("primary down")

    async def _stream_impl(
        self, prompt, max_tokens, system_prompt=None, **kw
    ) -> AsyncIterator[str]:
        for i in range(self._n):
            yield f"tok{i}"
        raise RuntimeError("connection reset mid-stream")


@pytest.mark.asyncio
async def test_fallback_used_when_primary_fails_before_any_tokens():
    """If the primary fails *before* yielding, fallback should stream normally."""
    primary = _FailBeforeYieldLLM()
    fallback = _StubLLM("fallback-model", ["hello", " world"])
    client = FallbackClient(primary=primary, fallback=fallback)

    tokens: list[str] = []
    async for tok in client.stream_complete("test prompt"):
        tokens.append(tok)

    assert tokens == ["hello", " world"]


@pytest.mark.asyncio
async def test_mid_stream_failure_raises_and_emits_sentinel():
    """If the primary fails *after* yielding tokens, we must NOT silently
    fall back.  Instead:
    1. The STREAM_BREAK_SENTINEL is yielded.
    2. A MidStreamFallbackError is raised.
    """
    primary = _FailMidStreamLLM(n_before_fail=3)
    fallback = _StubLLM("fallback-model", ["should", "not", "appear"])
    client = FallbackClient(primary=primary, fallback=fallback)

    tokens: list[str] = []
    with pytest.raises(MidStreamFallbackError) as exc_info:
        async for tok in client.stream_complete("test prompt"):
            tokens.append(tok)

    # First 3 real tokens, then the sentinel.
    assert tokens == ["tok0", "tok1", "tok2", STREAM_BREAK_SENTINEL]

    # Verify exception metadata.
    err = exc_info.value
    assert err.primary_model == "fail-mid"
    assert err.tokens_yielded == 3
    assert isinstance(err.cause, RuntimeError)


@pytest.mark.asyncio
async def test_mid_stream_error_does_not_contain_fallback_tokens():
    """Crucially, no tokens from the fallback LLM should appear in the
    collected output when the primary fails mid-stream."""
    primary = _FailMidStreamLLM(n_before_fail=1)
    fallback = _StubLLM("fallback-model", ["BAD1", "BAD2"])
    client = FallbackClient(primary=primary, fallback=fallback)

    tokens: list[str] = []
    with pytest.raises(MidStreamFallbackError):
        async for tok in client.stream_complete("test prompt"):
            tokens.append(tok)

    for tok in tokens:
        assert tok not in ("BAD1", "BAD2"), "Fallback tokens leaked into the stream!"


@pytest.mark.asyncio
async def test_successful_primary_stream():
    """Happy path — primary streams all tokens without error."""
    primary = _StubLLM("primary", ["A", "B", "C"])
    fallback = _StubLLM("fallback", ["X"])
    client = FallbackClient(primary=primary, fallback=fallback)

    tokens: list[str] = []
    async for tok in client.stream_complete("test prompt"):
        tokens.append(tok)

    assert tokens == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_empty_primary_falls_back():
    """If primary yields zero tokens (empty stream), it should fall back."""

    class _EmptyStreamLLM(BaseLLM):
        @property
        def model_name(self) -> str:
            return "empty"

        async def _complete_impl(self, prompt, max_tokens, system_prompt=None, **kw):
            return LLMResponse("")

        async def _stream_impl(self, prompt, max_tokens, system_prompt=None, **kw):
            return
            yield  # noqa

    primary = _EmptyStreamLLM()
    fallback = _StubLLM("fallback", ["ok"])
    client = FallbackClient(primary=primary, fallback=fallback)

    tokens: list[str] = []
    async for tok in client.stream_complete("test prompt"):
        tokens.append(tok)

    assert tokens == ["ok"]
