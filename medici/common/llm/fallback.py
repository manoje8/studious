from collections.abc import AsyncIterator

import logfire

from medici.common.llm.base import BaseLLM, LLMResponse


class FallbackClient(BaseLLM):
    """
    A composite LLM client that attempts to use a primary client,
    and falls back to a secondary (fallback) client if the primary fails.
    """

    def __init__(self, primary: BaseLLM, fallback: BaseLLM):
        super().__init__()
        self.primary = primary
        self.fallback = fallback

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        stage_tag: str = "unknown",
        system_prompt: str | None = None,
        json_mode: bool = False,
        **kwargs,
    ) -> LLMResponse:
        try:
            return await self.primary.complete(
                prompt, max_tokens, stage_tag, system_prompt, json_mode=json_mode, **kwargs
            )
        except Exception as e:
            logfire.warning(
                f"Primary LLM ({self.primary.model_name}) failed for stage '{stage_tag}': {e}. "
                f"Falling back to {self.fallback.model_name}."
            )
            return await self.fallback.complete(
                prompt, max_tokens, stage_tag, system_prompt, json_mode=json_mode, **kwargs
            )

    async def stream_complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        stage_tag: str = "unknown",
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream from primary; on first error fall back to secondary."""
        try:
            yielded_any = False
            async for token in self.primary.stream_complete(
                prompt, max_tokens, stage_tag, system_prompt, **kwargs
            ):
                yielded_any = True
                yield token
            if not yielded_any:
                raise RuntimeError("Primary stream yielded no tokens")
        except Exception as e:
            logfire.warning(
                f"Primary LLM stream ({self.primary.model_name}) failed for stage "
                f"'{stage_tag}': {e}. Falling back to {self.fallback.model_name}."
            )
            async for token in self.fallback.stream_complete(
                prompt, max_tokens, stage_tag, system_prompt, **kwargs
            ):
                yield token

    async def _complete_impl(
        self, prompt: str, max_token: int, system_prompt: str | None = None, **kwargs
    ) -> LLMResponse:
        raise NotImplementedError("FallbackClient overrides `complete` directly.")

    @property
    def model_name(self) -> str:
        return f"{self.primary.model_name} (fallback: {self.fallback.model_name})"

    @property
    def total_calls(self) -> int:
        return self.primary.total_calls + self.fallback.total_calls

    @property
    def total_tokens(self) -> int:
        return self.primary.total_tokens + self.fallback.total_tokens

    def usage_snapshot(self) -> dict:
        p_usage = self.primary.usage_snapshot()
        f_usage = self.fallback.usage_snapshot()
        return {
            "model": self.model_name,
            "calls": p_usage["calls"] + f_usage["calls"],
            "prompt_tokens": p_usage.get("prompt_tokens", 0) + f_usage.get("prompt_tokens", 0),
            "completion_tokens": p_usage.get("completion_tokens", 0)
            + f_usage.get("completion_tokens", 0),
            "total_tokens": p_usage.get("total_tokens", 0) + f_usage.get("total_tokens", 0),
        }

    def reset_usage(self) -> None:
        self.primary.reset_usage()
        self.fallback.reset_usage()
