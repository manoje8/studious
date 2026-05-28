from google import genai

from src.llm.base import BaseLLM, LLMResponse
from utils.config import config


class GeminiClient(BaseLLM):
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = model

    async def complete(self, prompt: str, max_token: int = 1024) -> LLMResponse:
        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )

        raw = response.text

        if not raw:
            finish_reason = (
                response.candidates[0].finish_reason
                if response.candidates
                else "unknown"
            )
            raise RuntimeError(
                f"Gemini returned no content. finish_reason={finish_reason}"
            )

        return LLMResponse(raw)
