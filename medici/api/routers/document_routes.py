from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from medici.api.config_api import config_api
from medici.api.constants_api import ALLOWED_CONTENT_TYPES
from medici.api.deps import get_processor, require_auth
from medici.common.utils.constants import ParseMethod
from medici.common.utils.helper import supported_extensions_list
from medici.ingestion.processor import Processor

if TYPE_CHECKING:
    from medici.api.rate_limiter import RateLimiter

INGESTION_ROOT = Path(config_api.INGESTION_ROOT).resolve()
INGESTION_ROOT.mkdir(parents=True, exist_ok=True)


class IngestionRequest(BaseModel):
    path: str
    parse_method: ParseMethod
    doc_id: str | None = None


def create_document_routes(ingestion_limiter: RateLimiter | None = None):
    router = APIRouter(tags=["document"])
    ingest_deps = [Depends(ingestion_limiter)] if ingestion_limiter is not None else []

    @router.post("/ingestion", dependencies=ingest_deps)
    async def ingestion(
        file: UploadFile = File(...),
        parse_method: ParseMethod = Form(...),
        doc_id: str | None = Form(None),
        processor: Processor = Depends(get_processor),
        # _auth: dict = Depends(require_auth),
    ):
        if (
            parse_method == ParseMethod.GOOGLE_DOC_AI
            and file.content_type not in ALLOWED_CONTENT_TYPES
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported content type: {file.content_type}",
            )

        original_name = Path(file.filename or "upload")
        ext = original_name.suffix.lower()
        if ext not in supported_extensions_list():
            raise HTTPException(status_code=400, detail="Unsupported file extension")

        safe_id = uuid.uuid4().hex
        user_dir = INGESTION_ROOT
        user_dir.mkdir(parents=True, exist_ok=True)
        dest_path = user_dir / f"{safe_id}{ext}"

        size = 0
        with dest_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > config_api.MAX_UPLOAD_BYTES:
                    out.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File too large",
                    )
                out.write(chunk)

        return await processor.ingest_document(
            file_path=dest_path,
            doc_id=doc_id,
            parse_method=parse_method,
        )

    @router.post("/bulk-ingestion")
    async def bulk_ingestion(
        body: IngestionRequest,
        processor: Processor = Depends(get_processor),
        _auth: dict = Depends(require_auth),
    ):
        try:
            requested = Path(body.path).resolve()
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid path",
            ) from exc

        if not requested.is_relative_to(INGESTION_ROOT):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Path is outside the allowed ingestion root",
            )

        return await processor.ingest_documents(
            file_paths=[str(requested)], parse_method=body.parse_method
        )

    return router
