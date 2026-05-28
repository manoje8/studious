import json
import re
from abc import ABC, abstractmethod


class LLMResponse:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self._parsed = None

    @property
    def text(self) -> str:
        """Alias for raw_text — matches the usage pattern across agents."""
        return self.raw_text

    @property
    def parsed_json(self):
        if self._parsed is None:
            text = self.raw_text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            self._parsed = json.loads(text.strip())
        return self._parsed

    def __str__(self):
        return self.raw_text


class BaseLLM(ABC):
    @abstractmethod
    async def complete(self, prompt: str, max_token: int = 1024) -> LLMResponse:
        pass
