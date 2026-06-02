from pathlib import Path

from fastapi import APIRouter

from src.ingestion.processor import Processor


def create_document_routes(processor: Processor):
    router = APIRouter(tags=["document"])

    @router.post("/ingestion")
    async def ingestion(
        file_path: str | Path, chunking_strategy: str, doc_id: str | None = None
    ):
        return await processor.ingest_document(
            file_path=file_path, doc_id=doc_id, chunking_strategy=chunking_strategy
        )

    return router
