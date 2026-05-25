import asyncio

import logfire
import vertexai
from dataclasses import dataclass

from qdrant_client.http.models import PointStruct
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

from utils.config import config
from src.ingestion.chunk import Chunk


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
        dimensions: int = 1536,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        logfire.configure(service_name="Embedding Service")
        vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
        self.model = TextEmbeddingModel.from_pretrained(model_name)
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

        loop = asyncio.get_event_loop()
        for attempt in range(self.max_retries):
            try:
                inputs = [TextEmbeddingInput(text, task_type="RETRIEVAL_QUERY")]
                embedding = await loop.run_in_executor(
                    None,
                    lambda: self.model.get_embeddings(
                        inputs, output_dimensionality=self.dimensions
                    ),
                )

                return embedding[0].values
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logfire.error(
                        f"Embedding failed after {self.max_retries} attempts: {e}"
                    )
                    raise
                logfire.warning(f"Embedding attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of chunks in batches"""

        if not chunks:
            return []

        embedded = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        loop = asyncio.get_event_loop()

        for batch_num in range(total_batches):
            start = batch_num * self.batch_size
            end = start + self.batch_size

            batch = chunks[start:end]

            logfire.info(
                f"Embedding batch {batch_num+1}/{total_batches}"
                f"({len(batch)} chunks)"
            )

            for attempt in range(self.max_retries):
                try:
                    inputs = [
                        TextEmbeddingInput(chunk.text, task_type="RETRIEVAL_DOCUMENT")
                        for chunk in batch
                    ]
                    embeddings = await loop.run_in_executor(
                        None,
                        lambda: self.model.get_embeddings(
                            texts=inputs, output_dimensionality=self.dimensions
                        ),
                    )
                    for chunk, emb in zip(batch, embeddings):
                        embedded.append(
                            EmbeddedChunk(
                                chunk=chunk,
                                vector=emb.values,
                                model_name=self.model_name,
                            )
                        )

                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logfire.error(f"Batch {batch_num + 1} failed: {e}")
                        raise

                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        logfire.info(f"Embedding complete: {len(embedded)} vectors produced")

        return embedded
