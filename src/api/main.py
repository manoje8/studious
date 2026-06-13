import sys
from contextlib import asynccontextmanager

import logfire
import uvicorn
from fastapi import FastAPI
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

# from src.llm.gemini import GeminiClient
from src.llm.groq import GroqClient
from src.services.qdrant import QdrantStorageService
from src.services.reranker import Reranker
from src.services.sparse_index import SparseSearchIndex
from src.utils.config import config
from src.utils.helper import check_env, bootstrap_sparse_index

pipeline: GraphPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    pool = AsyncConnectionPool(
        conninfo=config.POSTGRES_CONN_STRING,
        min_size=2,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )

    await pool.open()
    await pool.wait()

    # llm_client = GeminiClient()
    llm_client = GroqClient()
    short_term = ShortTermMemoryManager(config.REDIS_URL)

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
    sparse_index = SparseSearchIndex()

    logfire.info("Rebuilding sparse index...")
    await bootstrap_sparse_index(storage_service, sparse_index)

    hybrid_search = HybridSearch(
        storage_service=storage_service,
        embedding_service=embedding_service,
        sparse_index=sparse_index,
    )
    reranker = Reranker()
    query_expander = QueryExpander(llm_client)
    retrieval_agent = RetrievalAgent(
        llm_client=llm_client,
        hybrid_search=hybrid_search,
        reranker=reranker,
        query_expand=query_expander,
    )

    graph = await compile_graph_with_postgres(
        pool=pool,
        short_term=short_term,
        rewriter=QueryRewriter(llm_client),
        router=RouterAgent(llm_client),
        planner=PlannerAgent(llm_client),
        retrieval_agent=retrieval_agent,
        grader=GraderAgent(llm_client),
        synthesizer=SynthesizerAgent(llm_client),
    )

    pipeline = GraphPipeline(graph, short_term_memory=short_term)
    app.state.pipeline = pipeline
    app.state.pool = pool

    yield
    app.state.pipeline = None
    await pool.close()


def create_apps():
    app = FastAPI(title=config.PROJECT_NAME, lifespan=lifespan)

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
