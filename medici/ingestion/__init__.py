"""Medici data ingestion module."""

from medici.ingestion.embedding import EmbeddedChunk, EmbeddingService
from medici.ingestion.processor import Processor

__all__ = [
    "Processor",
    "EmbeddingService",
    "EmbeddedChunk",
]
