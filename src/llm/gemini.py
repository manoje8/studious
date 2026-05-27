from google import genai

from src.llm.base import BaseLLM, LLMResponse
from utils.config import config


class GeminiClient(BaseLLM):
    def __init__(self, model: str = "gemini-1.5-pro"):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = model

    async def complete(self, prompt: str, max_token: int = 1024) -> LLMResponse:
        response = await self.client.models.generate_content(
            model=self.model, contents=prompt
        )

        return LLMResponse(response.text)
