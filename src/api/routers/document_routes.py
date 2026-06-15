from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.ingestion.processor import Processor
from src.utils.constants import ChunkingType, ParseMethod


class IngestionRequest(BaseModel):
    file_path: str | Path
    parse_method: ParseMethod
    chunking_strategy: Optional[ChunkingType] = None
    doc_id: str | None = None


def create_document_routes():
    router = APIRouter(tags=["document"])

    processor = Processor()
    processor.config = {"parser": "google-vertexAI"}

    @router.post("/ingestion")
    async def ingestion(body: IngestionRequest):
        return await processor.ingest_document(
            file_path=body.file_path,
            doc_id=body.doc_id,
            chunking_strategy=body.chunking_strategy,
            parse_method=body.parse_method,
        )

    return router
