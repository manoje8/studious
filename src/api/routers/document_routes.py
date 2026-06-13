from pathlib import Path
from typing import Optional

from fastapi import APIRouter

from src.ingestion.processor import Processor
from src.utils.constants import ChunkingType


def create_document_routes():
    router = APIRouter(tags=["document"])

    processor = Processor()
    processor.config = {"parser": "google-vertexAI"}

    @router.post("/ingestion")
    async def ingestion(
        file_path: str | Path,
        parse_method: str,
        chunking_strategy: Optional[ChunkingType] = None,
        doc_id: str | None = None,
    ):
        return await processor.ingest_document(
            file_path=file_path,
            doc_id=doc_id,
            chunking_strategy=chunking_strategy,
            parse_method=parse_method,
        )

    return router
