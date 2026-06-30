import asyncio

import logfire
from google import genai

from src.common.llm.base import BaseLLM, LLMResponse
from src.common.utils.config import config


class GeminiClient(BaseLLM):
    def __init__(self, model: str = "gemini-2.5-flash-lite", **kwargs):
        super().__init__(**kwargs)
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = model

    @property
    def model_name(self) -> str:
        return self.model

    async def _complete_impl(self, prompt: str, max_tokens: int = 1024, **kwargs) -> LLMResponse:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=kwargs.get("temperature", 0.1),
                    max_output_tokens=max_tokens,
                    top_p=kwargs.get("top_p", 0.95),
                    top_k=kwargs.get("top_k", 40),
                ),
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
