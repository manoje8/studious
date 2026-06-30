import asyncio
from dataclasses import dataclass

import logfire
from google import genai
from qdrant_client.http.models import PointStruct

from src.common.utils.config import config
from src.ingestion.chunking.chunk import Chunk


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vector: list[float]
    model_name: str

    def to_qdrant_point(self, point_id: str) -> PointStruct:
        payload = self.chunk.to_quant_payload()
        payload["embedding_model"] = self.model_name

        return PointStruct(id=point_id, vector=self.vector, payload=payload)


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "text-embedding-004",
        dimensions: int = 1356,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.client = genai.Client(
            vertexai=True, project=config.PROJECT_ID, location=config.LOCATION
        )
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @property
    def vector_size(self) -> int:
        return self.dimensions or config.VECTOR_SIZE

    async def embed_single(self, text: str) -> list[float]:
        """Embed one piece of text"""

        text = text.strip()

        if not text:
            raise ValueError("Cannot embed empty text")

        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.models.embed_content,
                    model=self.model_name,
                    contents=text,
                    config=genai.types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=self.dimensions,
                    ),
                )

                return response.embeddings[0].values
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logfire.error(f"Embedding failed after {self.max_retries} attempts: {e}")
                    raise
                logfire.warning(f"Embedding attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks in batches"""

        if not chunks:
            return []

        embedded = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for batch_num in range(total_batches):
            start = batch_num * self.batch_size
            end = start + self.batch_size

            batch = chunks[start:end]

            logfire.info(f"Embedding batch {batch_num + 1}/{total_batches}({len(batch)} chunks)")

            for attempt in range(self.max_retries):
                try:
                    batch_texts = [chunk.text for chunk in batch]
                    response = await asyncio.to_thread(
                        self.client.models.embed_content,
                        model=self.model_name,
                        contents=batch_texts,
                        config=genai.types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=self.dimensions,
                        ),
                    )

                    for chunk, emb in zip(batch, response.embeddings, strict=False):
                        embedded.append(
                            EmbeddedChunk(
                                chunk=chunk,
                                vector=emb.values,
                                model_name=self.model_name,
                            )
                        )
                    break

                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logfire.error(f"Batch {batch_num + 1} failed: {e}")
                        raise

                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        logfire.info(f"Embedding complete: {len(embedded)} vectors produced")

        return embedded
