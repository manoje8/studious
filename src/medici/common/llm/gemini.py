import asyncio
from collections.abc import AsyncIterator

import logfire
from google import genai

from medici.common.llm.base import BaseLLM, LLMResponse
from medici.common.utils.config import config


class GeminiClient(BaseLLM):
    def __init__(self, model: str = "gemini-3.5-flash", **kwargs):
        super().__init__(**kwargs)
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = model

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
        generate_config_kwargs: dict = {
            "temperature": kwargs.get("temperature", 0.1),
            "max_output_tokens": max_tokens,
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
        }
        if system_prompt:
            generate_config_kwargs["system_instruction"] = system_prompt
        if kwargs.get("json_mode", False):
            generate_config_kwargs["response_mime_type"] = "application/json"

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(**generate_config_kwargs),
            ),
        )

        if not response or not response.text:
            finish_reason = (
                response.candidates[0].finish_reason
                if response and response.candidates
                else "Unknown"
            )
            safety_rating = (
                response.candidates[0].safety_ratings if response and response.candidates else "N/A"
            )

            msg = (
                f"Gemini returned no content. finish_reason={finish_reason}, "
                f"safety_rating={safety_rating}"
            )
            logfire.error(msg)
            raise ValueError(msg)

        token_usage = {}

        if hasattr(response, "usage_metadata"):
            token_usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total": getattr(response.usage_metadata, "total_token_count", 0),
            }

        return LLMResponse(response.text, metadata={"token_usage": token_usage})

    async def _stream_impl(
        self,
        prompt: str,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Yield text tokens from Gemini's generate_content_stream."""
        generate_config_kwargs: dict = {
            "temperature": kwargs.get("temperature", 0.1),
            "max_output_tokens": max_tokens,
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
        }
        if system_prompt:
            generate_config_kwargs["system_instruction"] = system_prompt

        loop = asyncio.get_event_loop()

        import queue as _queue
        import threading

        q: _queue.Queue = _queue.Queue()
        _SENTINEL = object()

        def _producer():
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(**generate_config_kwargs),
                ):
                    if chunk.text:
                        q.put(chunk.text)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(_SENTINEL)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                logfire.error("GeminiClient stream error", error=str(item))
                break
            yield item
