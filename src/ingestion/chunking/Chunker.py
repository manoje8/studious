from abc import ABC, abstractmethod
from typing import Any

from src.ingestion.chunking.chunk import Chunk


class Chunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Create a chunker instance."""

    @abstractmethod
    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        """hunk method definition."""
