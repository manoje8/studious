from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from src.common.utils.constants import ParseMethod
from src.ingestion.processor import Processor


class IngestionRequest(BaseModel):
    path: str | Path
    parse_method: ParseMethod
    doc_id: str | None = None


def create_document_routes():
    router = APIRouter(tags=["document"])

    processor = Processor()
    processor.config = {"parser": "google-vertexAI"}

    @router.post("/ingestion")
    async def ingestion(body: IngestionRequest):
        return await processor.ingest_document(
            file_path=body.path,
            doc_id=body.doc_id,
            parse_method=body.parse_method,
        )

    @router.post("/bulk-ingestion")
    async def bulk_ingestion(body: IngestionRequest):
        pass

    return router
