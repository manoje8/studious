import sys

import uvicorn
from fastapi import APIRouter, FastAPI

from src.agents.pipeline import Pipeline
from src.api.routers.document_routes import create_document_routes
from src.api.routers.query_router import create_query_routes
from src.ingestion.processor import Processor
from src.llm.gemini import GeminiClient
from src.utils.config import config
from src.utils.helper import check_env


def create_apps():
    app = FastAPI(title=config.PROJECT_NAME)

    processor = Processor()
    client = GeminiClient(model=config.GEMINI_MODEL)
    pipeline = Pipeline(llm_client=client, qdrant_url=config.QDRANT_CLUSTER_ENDPOINT)

    router = APIRouter()
    router.include_router(create_document_routes(processor))
    router.include_router(create_query_routes(pipeline))

    @app.get("/")
    def root():
        return "running"

    return app


def main():
    if not check_env():
        sys.exit(1)

    from multiprocessing import freeze_support

    freeze_support()

    app = create_apps()

    unicorn_config = {
        "app": app,
        "host": "0.0.0.0",
        "port": 8000,
    }

    uvicorn.run(**unicorn_config)


if __name__ == "__main__":
    main()
