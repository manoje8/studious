from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from medici.common.llm.base import BaseLLM, LLMResponse
from medici.common.utils.config import config


class NvidiaClient(BaseLLM):
    def __init__(self, model: str = "meta/llama-3.3-70b-instruct", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.client = AsyncOpenAI(
            api_key=config.NVIDIA_API_KEY,
            base_url=kwargs.get("base_url", "https://integrate.api.nvidia.com/v1"),
            max_retries=self.max_retries,
            timeout=self.timeout_seconds,
        )

    @property
    def model_name(self) -> str:
        return self.model

    async def _complete_impl(
        self, prompt: str, max_tokens: int, system_prompt: str | None = None, **kwargs
    ) -> LLMResponse:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        extra_kwargs = {}
        if kwargs.get("json_mode", False):
            extra_kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.1),
            top_p=kwargs.get("top_p", 0.95),
            max_tokens=max_tokens,
            **extra_kwargs,
        )

        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("Nvidia returned empty response")

        content = response.choices[0].message.content

        token_usage = {}
        if response.usage:
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }

        return LLMResponse(content, {"token_usage": token_usage})

    async def _stream_impl(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        extra_kwargs = {}
        if kwargs.get("json_mode", False):
            extra_kwargs["response_format"] = {"type": "json_object"}

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.1),
            top_p=kwargs.get("top_p", 0.95),
            max_tokens=max_tokens,
            stream=True,
            **extra_kwargs,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
