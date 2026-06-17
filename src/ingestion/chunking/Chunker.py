from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, List

from src.ingestion.chunking.chunk import Chunk


class Chunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Create a chunker instance."""

    @abstractmethod
    def chunk(
        self, text: str, transform: Callable[[str], str] | None = None
    ) -> List[Chunk]:
        """hunk method definition."""
