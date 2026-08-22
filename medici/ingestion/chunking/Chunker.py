from abc import ABC, abstractmethod
from typing import Any

from medici.ingestion.chunking.chunk import Chunk


class Chunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Create a chunker instance."""

    @abstractmethod
    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        """Chunk method definition."""

    @abstractmethod
    def chunk_multimodal_items(
        self,
        items: list[dict[str, Any]],
        doc_id: str,
        source_file: str,
        start_index: int,
        storage=None,
    ):
        """Multimodal items definition"""
