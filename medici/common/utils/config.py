import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Medici")
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "localhost")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS")
    WORKERS = int(os.getenv("WORKERS", 1))

    PROJECT_ID: str = os.getenv("PROJECT_ID")
    LOCATION: str = os.getenv("LOCATION")
    GCP_DOC_AI_LOCATION: str = os.getenv("GCP_DOC_AI_LOCATION")
    GCP_DOC_AI_PROCESSOR_ID: str = os.getenv("GCP_DOC_AI_PROCESSOR_ID")
    GCP_RAW_BUCKET: str = os.getenv("GCP_RAW_BUCKET")
    GCP_PROCESSED_BUCKET: str = os.getenv("GCP_PROCESSED_BUCKET")
    GCP_BUCKET_PREFIX: str = os.getenv("GCP_BUCKET_PREFIX", "")
    VPC_CONNECTOR: str = os.getenv("VPC_CONNECTOR")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY")
    NVIDIA_MODEL: str = os.getenv("NVIDIA_MODEL", "llama-3.3-70b-instruct")

    CEREBRAS_API_KEY: str = os.getenv("CEREBRAS_API_KEY")
    CEREBRAS_MODEL: str = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")

    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_docs")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT: str = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_DENSE_MODEL: str = os.getenv(
        "QDRANT_DENSE_MODEL", "sentence-transformers/all-mpnet-base-v2"
    )
    QDRANT_SPARSE_MODEL: str = os.getenv("QDRANT_SPARSE_MODEL", "Qdrant/bm25")

    LOGFIRE_TOKEN: str = os.getenv("LOGFIRE_TOKEN")
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT")

    MAX_PAGE_PER_PARSE: int = int(os.getenv("MAX_PAGE_PER_PARSE", 20))

    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", 768))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 100))

    VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 768))
    MAX_RETRIEVAL_ROUND = int(os.getenv("MAX_RETRIEVAL_ROUND", 1))
    MAX_HOPS = int(os.getenv("MAX_HOPS", 4))
    MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 100_000))
    USE_PARENT_CONTEXT: bool = os.getenv("USE_PARENT_CONTEXT", "true").lower() == "true"
    SKIP_EXPANSION_CATEGORIES: frozenset[str] = frozenset(
        c.strip().lower()
        for c in os.getenv("SKIP_EXPANSION_CATEGORIES", "factual").split(",")
        if c.strip()
    )

    # Token-budget guard
    MODEL_CONTEXT_LIMIT: int = int(os.getenv("MODEL_CONTEXT_LIMIT", 128_000))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", 2_048))
    MAX_PROMPT_OVERHEAD_TOKENS: int = int(os.getenv("MAX_PROMPT_OVERHEAD_TOKENS", 1_024))
    MAX_QUERY_INPUT_TOKENS: int = int(os.getenv("MAX_QUERY_INPUT_TOKENS", 512))

    CACHE_DIR = Path(".cache/doc_parser")
    CACHE_MANIFEST = CACHE_DIR / "manifest.json"

    EMBEDDING_CACHE_ENABLED: bool = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    EMBEDDING_CACHE_DIR: Path = Path(os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings"))
    EMBEDDING_CACHE_MAX_ENTRIES: int = int(os.getenv("EMBEDDING_CACHE_MAX_ENTRIES", 50_000))

    SEMANTIC_CACHE_ENABLED: bool = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
    SEMANTIC_CACHE_TTL_SECONDS: int = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", 3600))
    SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", 0.88))
    SEMANTIC_CACHE_MAX_ENTRIES: int = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", 500))

    SYNTHESIS_MAX_TOKENS_FACTUAL: int = int(os.getenv("SYNTHESIS_MAX_TOKENS_FACTUAL", 1024))
    SYNTHESIS_MAX_TOKENS_ANALYTICAL: int = int(os.getenv("SYNTHESIS_MAX_TOKENS_ANALYTICAL", 2048))
    SYNTHESIS_MAX_TOKENS_COMPARATIVE: int = int(os.getenv("SYNTHESIS_MAX_TOKENS_COMPARATIVE", 2048))
    SYNTHESIS_MAX_TOKENS_PROCEDURAL: int = int(os.getenv("SYNTHESIS_MAX_TOKENS_PROCEDURAL", 2048))
    SYNTHESIS_MAX_TOKENS_SUMMARIZATION: int = int(
        os.getenv("SYNTHESIS_MAX_TOKENS_SUMMARIZATION", 1536)
    )
    SYNTHESIS_MAX_TOKENS_CHITCHAT: int = int(os.getenv("SYNTHESIS_MAX_TOKENS_CHITCHAT", 512))
    SYNTHESIS_MAX_TOKENS_CLARIFICATION: int = int(
        os.getenv("SYNTHESIS_MAX_TOKENS_CLARIFICATION", 1024)
    )
    SYNTHESIS_MAX_TOKENS_META: int = int(os.getenv("SYNTHESIS_MAX_TOKENS_META", 512))

    # Docling
    TABLE_MODE = os.getenv("TABLE_MODE", "fast")
    DO_TABLES: bool = os.getenv("DO_TABLES", "true").lower() == "true"
    DO_OCR: bool = os.getenv("DO_OCR", "false").lower() == "true"

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    POSTGRES_CONN_STRING = os.getenv("POSTGRES_CONN_STRING")

    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")
    STORAGE_BASE_DIR: str = os.getenv("STORAGE_BASE_DIR", "./data")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 64))
    FLASHRANK_CACHE_DIR: str = os.getenv("FLASHRANK_CACHE_DIR", "/flashrank")

    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", 100_000_00))
    MIN_RERANK_SCORE = 0.1

    # Post-synthesis faithfulness gate (NLI model)
    FAITHFULNESS_ENABLED: bool = os.getenv("FAITHFULNESS_ENABLED", "true").lower() == "true"
    FAITHFULNESS_THRESHOLD: float = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.5"))
    FAITHFULNESS_MODEL: str = os.getenv(
        "FAITHFULNESS_MODEL", "vectara/hallucination_evaluation_model"
    )


config = Config()
