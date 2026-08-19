"""Medici data ingestion module."""

from medici.ingestion.processor import Processor
from medici.ingestion.embedding import EmbeddingService, EmbeddedChunk

__all__ = [
    "Processor",
    "EmbeddingService",
    "EmbeddedChunk",
]
