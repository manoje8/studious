from src.llm.base import BaseLLM, LLMResponse
from langchain_groq import ChatGroq

from src.utils.config import config


class GroqClient(BaseLLM):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = ChatGroq(
            model=model, api_key=config.GROQ_API_KEY, temperature=0.1
        )

    async def complete(self, prompt: str, max_token: int = 1024) -> LLMResponse:
        message = [{"role": "user", "content": prompt}]
        response = await self.client.ainvoke(message)
        return LLMResponse(response.content)
