"""
Shared data models used across Medici layers.

This module contains data classes that are referenced by both the ``common``
and ``ingestion`` packages, breaking what would otherwise be a circular
dependency between those layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client.http.models import PointStruct


@dataclass
class Chunk:
    text: str
    chunk_index: int
    doc_id: str
    source_file: str
    chunk_type: str
    section_title: str = ""
    page_numbers: list[int] = field(default_factory=list)
    block_types: list[str] = field(default_factory=list)
    token_count: int = 0
    parent_text: str = ""
    parent_token_count: int = 0
    parent_window_start: int = 0
    parent_window_end: int = 0
    metadata: dict = field(default_factory=dict)
    image_path: str = ""

    def to_quant_payload(self) -> dict:
        payload = {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "chunk_type": self.chunk_type,
            "content_type": self.chunk_type,
            "section_title": self.section_title,
            "page_numbers": self.page_numbers,
            "token_count": self.token_count,
            "parent_text": self.parent_text,
            "parent_token_count": self.parent_token_count,
            "metadata": self.metadata,
        }
        if self.image_path:
            payload["image_path"] = self.image_path
        return payload


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
