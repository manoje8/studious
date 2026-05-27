import json
from abc import ABC, abstractmethod


class LLMResponse:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self._parsed = None

    @property
    def parsed_json(self):
        if self._parsed is None:
            self._parsed = json.loads(self.raw_text)
        return self._parsed

    def __str__(self):
        return self.raw_text


class BaseLLM(ABC):
    @abstractmethod
    async def complete(self, prompt: str, max_token: int = 1024) -> LLMResponse:
        pass
