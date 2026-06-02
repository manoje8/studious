import sys
from contextlib import asynccontextmanager

import logfire
import uvicorn
from fastapi import FastAPI

from src.agents.pipeline import Pipeline
from src.api.routers.document_routes import create_document_routes
from src.api.routers.query_router import create_query_routes
from src.ingestion.processor import Processor
from src.llm.groq import GroqClient
from src.utils.config import config
from src.utils.helper import check_env


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_apps():
    app = FastAPI(title=config.PROJECT_NAME)

    processor = Processor()
    processor.config = {"parser": "google-vertexAI"}
    client = GroqClient()
    pipeline = Pipeline(llm_client=client, qdrant_url=config.QDRANT_CLUSTER_ENDPOINT)

    @app.get("/")
    def root():
        return "running"

    app.include_router(create_document_routes(processor))
    app.include_router(create_query_routes(pipeline))

    return app


def main():
    logfire.configure(service_name=config.PROJECT_NAME)

    if not check_env():
        sys.exit(1)

    from multiprocessing import freeze_support

    freeze_support()

    app = create_apps()

    unicorn_config = {"app": app, "host": config.HOST, "port": config.PORT}

    uvicorn.run(**unicorn_config)


if __name__ == "__main__":
    main()
