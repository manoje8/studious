import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import logfire
from google.api_core import exceptions as _gexc


class LLMParseError(Exception):
    def __init__(self, raw_text: str, original_error: Exception):
        self.raw_text = raw_text
        self.original_error = original_error
        super().__init__(f"Failed to parse LLM JSON: {original_error}")


class LLMTimeoutError(Exception):
    def __init__(self, timeout_seconds: int, retry_count: int):
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        super().__init__(f"LLM request timeout after {timeout_seconds}s (retries: {retry_count})")


class LLMContentError(Exception):
    def __init__(self, reason: str, details: dict | None = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(f"LLM content error: {reason}")


_NON_RETRYABLE_LLM_EXCEPTIONS = (
    _gexc.InvalidArgument,  # 400 – malformed prompt or generation config
    _gexc.PermissionDenied,  # 403 – quota exceeded or API key lacks access
    _gexc.Unauthenticated,  # 401 – invalid/missing API key
    _gexc.NotFound,  # 404 – model name does not exist
    LLMParseError,  # deterministic parse failure, same output every time
)


def _extract_outermost_json(text: str) -> str | None:
    """
    Return the substring of *text* that forms the first complete JSON
    object ``{…}`` or array ``[…]``, or ``None`` if none is found.

    Uses a character-by-character bracket-depth scanner instead of a greedy
    ``re.DOTALL`` regex.  The greedy approach is brittle when the model emits
    explanatory prose that contains ``{`` or ``[`` characters before the
    actual JSON payload - the regex would start the match at the wrong
    position and produce an invalid or truncated JSON string.

    The scanner correctly skips over:
    - String literals (so braces/brackets inside ``"..."`` are ignored)
    - Escape sequences (``\\"``, ``\\\\``, etc.) inside strings
    """
    OPENERS = {"{": "}", "[": "]"}
    for start, ch in enumerate(text):
        if ch not in OPENERS:
            continue
        closer = OPENERS[ch]
        depth = 0
        in_string = False
        i = start
        while i < len(text):
            c = text[i]
            if in_string:
                if c == "\\":
                    i += 2
                    continue
                if c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == ch:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
            i += 1
    return None


class LLMRequestContext:
    def __init__(
        self, prompt: str, max_tokens: int = 1024, system_prompt: str | None = None, **kwargs
    ):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.start_time = datetime.now(UTC)
        self.retry_count = 0
        self.kwargs = kwargs
        self._cost_estimate = None

    def estimate_cost(self, model: str) -> float:
        approx_tokens = len(self.prompt.split()) / 0.75
        return approx_tokens * 0.000001


class LLMResponse:
    def __init__(self, raw_text: str, metadata: dict | None = None):
        self.raw_text = raw_text
        self.metadata = metadata or {}
        self._parsed = None
        self._parse_error = None
        self._timestamp = datetime.now(UTC)
        self._token_usage = self.metadata.get("token_usage", {})

    @property
    def text(self) -> str:
        return self.raw_text

    @property
    def parsed_json(self):
        if self._parsed is None:
            text = self.raw_text.strip()

            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

            extracted = _extract_outermost_json(text)
            if extracted is not None:
                text = extracted

            try:
                self._parsed = json.loads(text)
            except json.JSONDecodeError as e:
                raise LLMParseError(self.raw_text, e) from e
        return self._parsed

    def try_parsed_json(self, default=None):
        try:
            return self.parsed_json
        except LLMParseError:
            return default

    def has_json(self):
        try:
            _ = self.parsed_json
            return True
        except LLMParseError:
            return False

    @property
    def __str__(self):
        return self.raw_text


class BaseLLM(ABC):
    def __init__(self, timeout_seconds: int = 30, max_retries: int = 2):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._call_count = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    @abstractmethod
    async def _complete_impl(
        self, prompt: str, max_tokens: int, system_prompt: str | None = None, **kwargs
    ) -> LLMResponse:
        pass

    async def _stream_impl(
        self,
        prompt: str,
        max_tokens: int,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Override in subclasses to provide real token-by-token streaming.

        The default implementation calls ``_complete_impl`` and yields the
        entire response as a single chunk — safe for clients that don't
        natively support streaming.
        """
        response = await self._complete_impl(
            prompt, max_tokens, system_prompt=system_prompt, **kwargs
        )
        yield response.text

    async def stream_complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        stage_tag: str = "unknown",
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Async-generator that yields text tokens as they arrive.

        Wraps ``_stream_impl`` with the same logfire span used by
        ``complete()`` so observability is preserved.
        """
        with logfire.span(
            "llm.stream_complete",
            model=self.model_name,
            stage=stage_tag,
            prompt_length=len(prompt),
            has_system_prompt=system_prompt is not None,
        ):
            async for token in self._stream_impl(
                prompt, max_tokens, system_prompt=system_prompt, **kwargs
            ):
                yield token

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        stage_tag: str = "unknown",
        system_prompt: str | None = None,
        json_mode: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """
        Call the LLM and return an :class:`LLMResponse`.
        """
        context = LLMRequestContext(
            prompt=prompt,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs,
        )

        for attempt in range(self.max_retries + 1):
            context.retry_count = attempt

            try:
                with logfire.span(
                    "llm.complete",
                    model=self.model_name,
                    stage=stage_tag,
                    attempt=attempt,
                    prompt_length=len(prompt),
                    has_system_prompt=system_prompt is not None,
                    json_mode=json_mode,
                ) as span:
                    response = await asyncio.wait_for(
                        self._complete_with_metadata(
                            prompt,
                            max_tokens,
                            context,
                            system_prompt=system_prompt,
                            json_mode=json_mode,
                            **kwargs,
                        ),
                        timeout=self.timeout_seconds,
                    )
                    usage = response.metadata.get("token_usage", {})
                    self._call_count += 1
                    self._total_tokens += usage.get("total", 0)
                    self._total_prompt_tokens += usage.get("prompt_tokens", 0)
                    self._total_completion_tokens += usage.get("completion_tokens", 0)
                    span.set_attributes(
                        {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total", 0),
                            "call_count": self._call_count,
                        }
                    )
                    return response

            except TimeoutError as err:
                if attempt == self.max_retries:
                    raise LLMTimeoutError(
                        timeout_seconds=self.timeout_seconds,
                        retry_count=attempt,
                    ) from err

                await asyncio.sleep(2**attempt)
            except Exception as e:
                if isinstance(e, _NON_RETRYABLE_LLM_EXCEPTIONS):
                    raise LLMContentError(
                        reason=str(e),
                        details=None,
                    ) from e
                if attempt == self.max_retries:
                    raise LLMContentError(
                        reason=str(e),
                        details=None,
                    ) from e
                await asyncio.sleep(2**attempt)

        raise RuntimeError("Unexpected failure in LLM request")

    async def _complete_with_metadata(
        self,
        prompt: str,
        max_tokens: int,
        context: LLMRequestContext,
        system_prompt: str | None = None,
        json_mode: bool = False,
        **kwargs,
    ) -> LLMResponse:
        response = await self._complete_impl(
            prompt, max_tokens, system_prompt=system_prompt, json_mode=json_mode, **kwargs
        )
        response.metadata.update(
            {
                "call_time": datetime.now(UTC).isoformat(),
                "retry_count": context.retry_count,
                "call_id": self._call_count + 1,
                "estimate_cost": context.estimate_cost(self.model_name),
            }
        )

        return response

    @property
    def total_calls(self) -> int:
        return self._call_count

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def usage_snapshot(self) -> dict:
        """
        Return a point-in-time snapshot of cumulative token usage for this client instance.

        Suitable for including in per-request response metadata after graph execution.
        Reset is intentionally NOT performed here — call reset_usage() if you need
        per-request isolation (e.g. when a single client is reused across many requests).
        """
        return {
            "model": self.model_name,
            "calls": self._call_count,
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_tokens,
        }

    def reset_usage(self) -> None:
        """Reset cumulative counters. Call before each request when sharing a client instance."""
        self._call_count = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
