import asyncio

from src.llm.base import BaseLLM, LLMResponse
from cerebras.cloud.sdk import AsyncCerebras

from src.utils.config import config


class CerebrasAI(BaseLLM):
    def __init__(self, model: str = "gpt-oss-120b", max_tokens: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.client = AsyncCerebras(
            api_key=config.CEREBRAS_API_KEY,
        )

    async def _complete_impl(
        self, prompt: str, max_token: int, **kwargs
    ) -> LLMResponse:
        loop = asyncio.get_event_loop()

        completion = await loop.run_in_executor(
            None,
            lambda: self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=max_token,
                top_p=kwargs.get("top_p", 0.95),
            ),
        )

        # TODO" Need to update the LLM response
        result = completion.choices[0].text

        if not result:
            raise ValueError("Returned no content")

        return LLMResponse(result, {"token_usage": 0})

    @property
    def model_name(self) -> str:
        return self.model
