from collections.abc import AsyncIterator

from langchain_groq import ChatGroq

from medici.common.llm.base import BaseLLM, LLMResponse
from medici.common.utils.config import config


class GroqClient(BaseLLM):
    def __init__(self, model: str = "llama-3.3-70b-versatile", max_tokens: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.client = ChatGroq(
            model=model,
            api_key=config.GROQ_API_KEY,
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=max_tokens,
            max_retries=kwargs.get("max_retries", 0),
            timeout=self.timeout_seconds,
        )

    @property
    def model_name(self) -> str:
        return self.model

    async def _complete_impl(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self.client
        if kwargs.get("json_mode", False):
            client = client.bind(response_format={"type": "json_object"})

        response = await client.ainvoke(messages)

        if not response or not response.content:
            raise RuntimeError("Groq returned empty response")

        token_usage = {}

        if hasattr(response, "response_metadata"):
            token_usage = {
                "prompt_tokens": response.response_metadata.get("token_usage", {}).get(
                    "prompt_tokens", 0
                ),
                "completion_tokens": response.response_metadata.get("token_usage", {}).get(
                    "completion_tokens", 0
                ),
                "total": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
            }

        return LLMResponse(response.content, {"token_usage": token_usage})

    async def _stream_impl(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Yield text tokens from Groq's astream."""
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async for chunk in self.client.astream(messages):
            if chunk.content:
                yield chunk.content
