import asyncio
import contextlib
import sys
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import logfire
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.agents.agentic.grader import GraderAgent
from src.agents.agentic.planner import PlannerAgent
from src.agents.agentic.query_expander import QueryExpander
from src.agents.agentic.query_rewriter import QueryRewriter
from src.agents.agentic.router import RouterAgent
from src.agents.agentic.synthesizer import SynthesizerAgent
from src.agents.graph.graph import compile_graph_with_postgres
from src.agents.graph.runner import GraphPipeline
from src.agents.memory.short_term import ShortTermMemoryManager
from src.agents.retrieval import RetrievalAgent
from src.api.auth import auth_handler
from src.api.config_api import config_api as _config_api
from src.api.rate_limiter import AsyncRateLimiterBackend, RateLimiter
from src.api.routers.document_routes import create_document_routes
from src.api.routers.query_router import create_query_routes
from src.common.cache.embedding_cache import EmbeddingCache
from src.common.cache.semantic_cache import SemanticQueryCache
from src.common.llm.fallback import FallbackClient
from src.common.llm.gemini import GeminiClient
from src.common.llm.groq import GroqClient
from src.common.services.faithfulness_checker import get_faithfulness_checker
from src.common.services.hybrid_search import HybridSearch
from src.common.services.qdrant import QdrantStorageService
from src.common.services.reranker import Reranker
from src.common.utils.config import config
from src.common.utils.helper import check_env, has_internet
from src.common.utils.tokenizer import TikTokenTokenizer
from src.ingestion.embedding import EmbeddingService
from src.ingestion.processor import Processor
from src.knowledge_graph.extractor import KGExtractor
from src.knowledge_graph.retriever import KGRetriever
from src.knowledge_graph.store import KGStore


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

    try:
        _config_api.validate()

        await pool.open()
        await asyncio.wait_for(pool.wait(), timeout=10)
        closers.append(("postgres pool", pool.close))

        gemini_client = GeminiClient(timeout_seconds=30, max_retries=2, model=config.GEMINI_MODEL)
        groq_client = GroqClient(timeout_seconds=30, max_retries=2)

        primary_groq_fallback_gemini = FallbackClient(primary=groq_client, fallback=gemini_client)
        primary_gemini_fallback_groq = FallbackClient(primary=gemini_client, fallback=groq_client)

        # episodic = EpisodicMemoryManager(llm_client=llm_client, pool=pool)
        # await episodic.setup()
        short_term = ShortTermMemoryManager(config.REDIS_URL)
        if hasattr(short_term, "aclose"):
            closers.append(("redis", short_term.aclose))

        emb_cache: EmbeddingCache | None = None
        if config.EMBEDDING_CACHE_ENABLED:
            emb_cache = await EmbeddingCache.create(
                dsn=config.POSTGRES_CONN_STRING, max_entries=50_000
            )

        embedding_service = EmbeddingService(
            model_name=config.EMBEDDING_MODEL_NAME,
            dimensions=config.EMBEDDING_DIMENSIONS,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            cache=emb_cache,
        )

        storage_service = QdrantStorageService(
            url=config.QDRANT_CLUSTER_ENDPOINT,
            vector_size=embedding_service.vector_size,
            collection_name=config.QDRANT_COLLECTION_NAME,
        )
        closers.append(("qdrant client", storage_service.client.close))

        tokenizer = TikTokenTokenizer()

        hybrid_search = HybridSearch(
            storage_service=storage_service, embedding_service=embedding_service
        )
        reranker = Reranker()
        query_expander = QueryExpander(primary_gemini_fallback_groq)
        retrieval_agent = RetrievalAgent(
            llm_client=primary_groq_fallback_gemini,
            hybrid_search=hybrid_search,
            reranker=reranker,
            query_expand=query_expander,
        )

        faithfulness_checker = None
        if config.FAITHFULNESS_ENABLED:
            faithfulness_checker = get_faithfulness_checker(threshold=config.FAITHFULNESS_THRESHOLD)
            logfire.info(
                "FaithfulnessChecker enabled",
                model="vectara/hallucination_evaluation_model",
                threshold=config.FAITHFULNESS_THRESHOLD,
            )
        else:
            logfire.info("FaithfulnessChecker disabled (FAITHFULNESS_ENABLED=false)")

        kg_retriever = None
        kg_extractor = None
        kg_store = None
        if config.KG_ENABLED:
            kg_store = KGStore(pool)
            await kg_store.setup()
            kg_extractor = KGExtractor(
                llm_client=primary_gemini_fallback_groq,
                batch_size=config.KG_EXTRACTION_BATCH_SIZE,
                min_confidence=config.KG_MIN_CONFIDENCE,
            )
            kg_retriever = KGRetriever(
                llm_client=primary_gemini_fallback_groq,
                kg_store=kg_store,
                storage_service=storage_service,
                max_hops=config.KG_MAX_HOPS,
                max_kg_chunks=config.KG_MAX_CHUNKS,
            )
            logfire.info(
                "Knowledge Graph layer enabled",
                max_hops=config.KG_MAX_HOPS,
                max_chunks=config.KG_MAX_CHUNKS,
            )
        else:
            logfire.info("Knowledge Graph layer disabled (KG_ENABLED=false)")

        processor = Processor(
            tokenizer,
            embedding_service,
            storage_service,
            kg_extractor=kg_extractor,
            kg_store=kg_store,
        )

        graph = await compile_graph_with_postgres(
            pool=pool,
            short_term=short_term,
            rewriter=QueryRewriter(primary_gemini_fallback_groq),
            router=RouterAgent(primary_groq_fallback_gemini),
            planner=PlannerAgent(primary_gemini_fallback_groq),
            retrieval_agent=retrieval_agent,
            grader=GraderAgent(primary_groq_fallback_gemini),
            synthesizer=SynthesizerAgent(primary_groq_fallback_gemini),
            faithfulness_checker=faithfulness_checker,
            kg_retriever=kg_retriever,
        )

        semantic_cache: SemanticQueryCache | None = None
        if config.SEMANTIC_CACHE_ENABLED:
            semantic_cache = SemanticQueryCache(
                redis_url=config.REDIS_URL,
                embedding_fn=embedding_service.embed_single,
                similarity_threshold=config.SEMANTIC_CACHE_THRESHOLD,
                ttl_seconds=config.SEMANTIC_CACHE_TTL_SECONDS,
                max_entries=config.SEMANTIC_CACHE_MAX_ENTRIES,
            )
            closers.append(("semantic cache", semantic_cache.aclose))
            logfire.info("SemanticQueryCache enabled", threshold=config.SEMANTIC_CACHE_THRESHOLD)

        pipeline = GraphPipeline(
            graph,
            short_term_memory=short_term,
            semantic_cache=semantic_cache,
            llm_clients=[gemini_client, groq_client],
        )
        app.state.pipeline = pipeline
        app.state.pool = pool
        app.state.qdrant = storage_service
        app.state.processor = processor

        app.state.rate_limiter = None
        if _config_api.RATE_LIMIT_ENABLED:
            rl_backend = AsyncRateLimiterBackend(redis_url=config.REDIS_URL)
            app.state.rate_limiter = rl_backend
            closers.append(("rate limiter", rl_backend.aclose))
            logfire.info(
                "Rate limiting enabled",
                query_limit=f"{_config_api.RATE_LIMIT_QUERY_MAX_REQUESTS}/{_config_api.RATE_LIMIT_QUERY_WINDOW_SECONDS}s",
                ingestion_limit=f"{_config_api.RATE_LIMIT_INGESTION_MAX_REQUESTS}/{_config_api.RATE_LIMIT_INGESTION_WINDOW_SECONDS}s",
                login_limit=f"{_config_api.RATE_LIMIT_LOGIN_MAX_REQUESTS}/{_config_api.RATE_LIMIT_LOGIN_WINDOW_SECONDS}s",
            )
        else:
            logfire.info("Rate limiting disabled (RATE_LIMIT_ENABLED=false)")

    except Exception:
        logfire.error("Startup failed; rolling back partially-initialized resources")
        for _name, close in reversed(closers):
            with contextlib.suppress(Exception):
                await close()
        raise

    try:
        yield
    finally:
        app.state.pipeline = None
        app.state.processor = None
        for name, close in reversed(closers):
            try:
                await close()
            except Exception:
                logfire.error(f"Error closing {name} during shutdown")


_query_limiter = RateLimiter(
    scope="query",
    max_requests=_config_api.RATE_LIMIT_QUERY_MAX_REQUESTS,
    window_seconds=_config_api.RATE_LIMIT_QUERY_WINDOW_SECONDS,
)
_login_limiter = RateLimiter(
    scope="login",
    max_requests=_config_api.RATE_LIMIT_LOGIN_MAX_REQUESTS,
    window_seconds=_config_api.RATE_LIMIT_LOGIN_WINDOW_SECONDS,
)


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
        expose_headers=["X-New_Token", "Cache-Control", "Content-Type"],
    )

    @app.get("/")
    def root():
        return "server running"

    @app.get("/health", tags=["observability"])
    async def health(request: Request):
        qdrant_ok = False
        qdrant_service: QdrantStorageService | None = getattr(request.app.state, "qdrant", None)
        if qdrant_service is not None:
            qdrant_ok = await qdrant_service.ping()

        deps = {"qdrant": "ok" if qdrant_ok else "unreachable"}
        overall = "ok" if all(v == "ok" for v in deps.values()) else "degraded"

        status_code = 200 if overall == "ok" else 503
        return JSONResponse(
            status_code=status_code,
            content={"status": overall, "service": config.PROJECT_NAME, "dependencies": deps},
        )

    app.include_router(
        create_document_routes(
            ingestion_limiter=RateLimiter(
                scope="ingestion",
                max_requests=_config_api.RATE_LIMIT_INGESTION_MAX_REQUESTS,
                window_seconds=_config_api.RATE_LIMIT_INGESTION_WINDOW_SECONDS,
            )
        )
    )
    app.include_router(
        create_query_routes(
            query_limiter=_query_limiter,
        )
    )

    @app.get("/auth-status")
    async def get_auth_status():
        if not auth_handler.accounts:
            guest_token = auth_handler.create_access_token(
                user_name="guest", role="guest", metadata={"auth_mode": False}
            )

            return {
                "auth_configured": False,
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": False,
                "message": "Authentication is disabled. Using guest access.",
            }

        return {
            "auth_configured": True,
            "auth_mode": True,
        }

    @app.post("/login", dependencies=[Depends(_login_limiter)])
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        if not auth_handler.accounts:
            guest_token = auth_handler.create_access_token(
                user_name="guest", role="guest", metadata={"auth_mode": False}
            )

            return {
                "auth_configured": False,
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": False,
                "message": "Authentication is disabled. Using guest access.",
            }

        user_name = form_data.username

        if not auth_handler.verify_password(user_name, form_data.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect credentials"
            )

        user_token = auth_handler.create_access_token(
            user_name=user_name, role="user", metadata={"auth_mode": True}
        )

        return {
            "access_token": user_token,
            "token_type": "bearer",
            "auth_mode": True,
        }

    return app


def main():
    logfire.configure(service_name=config.PROJECT_NAME, send_to_logfire=has_internet())

    if not check_env():
        sys.exit(1)

    from multiprocessing import freeze_support

    freeze_support()

    app = create_apps()
    unicorn_config = {"app": app, "host": config.HOST, "port": config.PORT}
    uvicorn.run(**unicorn_config)


if __name__ == "__main__":
    main()
