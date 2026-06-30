import asyncio
import json
import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime


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


class LLMRequestContext:
    def __init__(self, prompt: str, max_tokens: int = 1024, **kwargs):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.start_time = datetime.now(UTC)
        self.retry_count = 0
        self.kwargs = kwargs
        self._cost_estimate = None

    # TODO: model -> cost per token here
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

            json_match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
            if json_match:
                text = json_match.group()
            try:
                self._parsed = json.loads(text.strip())
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

    @abstractmethod
    async def _complete_impl(self, prompt: str, max_token: int, **kwargs) -> LLMResponse:
        pass

    async def complete(self, prompt: str, max_tokens: int = 1024, **kwargs) -> LLMResponse:
        context = LLMRequestContext(
            prompt=prompt,
            max_tokens=max_tokens,
            **kwargs,
        )

        for attempt in range(self.max_retries + 1):
            context.retry_count = attempt

            try:
                response = await asyncio.wait_for(
                    self._complete_with_metadata(prompt, max_tokens, context, **kwargs),
                    timeout=self.timeout_seconds,
                )
                self._call_count += 1
                self._total_tokens += response.metadata.get("token_usage", {}).get("total", 0)
                return response

            except asyncio.TimeoutError as err:
                if attempt == self.max_retries:
                    raise LLMTimeoutError(
                        timeout_seconds=self.timeout_seconds,
                        retry_count=attempt,
                    ) from err

                await asyncio.sleep(2**attempt)
            except Exception as e:
                if attempt == self.max_retries:
                    raise LLMContentError(
                        reason=str(e),
                        details=None,
                    ) from e
                await asyncio.sleep(1)

        raise RuntimeError("Unexpected failure in LLM request")

    async def _complete_with_metadata(
        self, prompt: str, max_tokens: int, context: LLMRequestContext, **kwargs
    ) -> LLMResponse:
        response = await self._complete_impl(prompt, max_tokens, **kwargs)
        response.metadata.update(
            {
                "call_time": datetime.utcnow().isoformat(),
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

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
