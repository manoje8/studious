import asyncio
import contextlib
import sys
from contextlib import asynccontextmanager
from typing import Callable, Awaitable

import logfire
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.agents.agentic.grader import GraderAgent
from src.agents.agentic.planner import PlannerAgent
from src.agents.agentic.router import RouterAgent
from src.agents.agentic.synthesizer import SynthesizerAgent
from src.agents.graph.graph import compile_graph_with_postgres
from src.agents.graph.runner import GraphPipeline
from src.agents.hybrid_search import HybridSearch
from src.agents.memory.query_rewriter import QueryRewriter
from src.agents.memory.short_term import ShortTermMemoryManager
from src.agents.query_expander import QueryExpander
from src.agents.retrieval import RetrievalAgent
from src.api.routers.document_routes import create_document_routes
from src.api.routers.query_router import create_query_routes
from src.ingestion.embedding import EmbeddingService

from src.llm.gemini import GeminiClient
from src.llm.groq import GroqClient
from src.services.qdrant import QdrantStorageService
from src.services.reranker import Reranker
from src.services.sparse_index import SparseSearchIndex
from src.utils.config import config
from src.utils.helper import check_env, bootstrap_sparse_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = AsyncConnectionPool(
        conninfo=config.POSTGRES_CONN_STRING,
        min_size=2,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )

    closers: list[tuple[str, Callable[[], Awaitable[None]]]] = []
    rebuild_task: asyncio.Task | None = None

    try:
        await pool.open()
        await asyncio.wait_for(pool.wait(), timeout=10)
        closers.append(("postgres pool", pool.close))

        gemini_client = GeminiClient(timeout_seconds=30, max_retries=2)
        groq_client = GroqClient(timeout_seconds=30, max_retries=2)

        # episodic = EpisodicMemoryManager(llm_client=llm_client, pool=pool)
        # await episodic.setup()
        short_term = ShortTermMemoryManager(config.REDIS_URL)
        if hasattr(short_term, "aclose"):
            closers.append(("redis", short_term.aclose))

        embedding_service = EmbeddingService(
            model_name=config.EMBEDDING_MODEL_NAME,
            dimensions=config.EMBEDDING_DIMENSIONS,
            batch_size=config.EMBEDDING_BATCH_SIZE,
        )
        storage_service = QdrantStorageService(
            url=config.QDRANT_CLUSTER_ENDPOINT,
            vector_size=embedding_service.vector_size,
            collection_name=config.QDRANT_COLLECTION_NAME,
        )
        closers.append(("qdrant client", storage_service.client.close))

        sparse_index = SparseSearchIndex()

        hybrid_search = HybridSearch(
            storage_service=storage_service,
            embedding_service=embedding_service,
            sparse_index=sparse_index,
        )
        reranker = Reranker()
        query_expander = QueryExpander(gemini_client)
        retrieval_agent = RetrievalAgent(
            llm_client=groq_client,
            hybrid_search=hybrid_search,
            reranker=reranker,
            query_expand=query_expander,
        )

        graph = await compile_graph_with_postgres(
            pool=pool,
            short_term=short_term,
            rewriter=QueryRewriter(gemini_client),
            router=RouterAgent(groq_client),
            planner=PlannerAgent(gemini_client),
            retrieval_agent=retrieval_agent,
            grader=GraderAgent(groq_client),
            synthesizer=SynthesizerAgent(groq_client),
        )

        pipeline = GraphPipeline(graph, short_term_memory=short_term)
        app.state.pipeline = pipeline
        app.state.pool = pool

        rebuild_task = asyncio.create_task(
            bootstrap_sparse_index(storage_service, sparse_index)
        )
    except Exception:
        logfire.error("Startup failed; rolling back partially-initialized resources")
        for name, close in reversed(closers):
            with contextlib.suppress(Exception):
                await close()
        raise

    try:
        yield
    finally:
        app.state.pipeline = None
        if rebuild_task is not None:
            rebuild_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await rebuild_task
        for name, close in reversed(closers):
            try:
                await close()
            except Exception:
                logfire.error(f"Error closing {name} during shutdown")


def create_apps():
    app = FastAPI(title=config.PROJECT_NAME, lifespan=lifespan)

    def get_cors_origins():
        origins_str = config.CORS_ORIGINS
        if origins_str == "*":
            return ["*"]
        return [origin.strip() for origin in origins_str.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-New_Token"],
    )

    @app.get("/")
    def root():
        return "running"

    app.include_router(create_document_routes())
    app.include_router(create_query_routes())

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
