"""Shared data models used across Medici layers.

This module contains data classes that are referenced by both the ``common``
and ``ingestion`` packages, breaking what would otherwise be a circular
dependency between those layers.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client.http.models import PointStruct

from medici.ingestion.chunking.chunk import Chunk


@dataclass
class EmbeddedChunk:
    """A document chunk paired with its embedding vector."""

    chunk: Chunk
    vector: list[float]
    model_name: str

    def to_qdrant_point(self, point_id: str) -> PointStruct:
        payload = self.chunk.to_quant_payload()
        payload["embedding_model"] = self.model_name

        return PointStruct(id=point_id, vector=self.vector, payload=payload)
