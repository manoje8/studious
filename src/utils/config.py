import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "studious")
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "localhost")

    PROJECT_ID: str = os.getenv("PROJECT_ID")
    LOCATION: str = os.getenv("")
    GCP_DOC_AI_LOCATION: str = os.getenv("GCP_DOC_AI_LOCATION")
    GCP_DOC_AI_PROCESSOR_ID: str = os.getenv("GCP_DOC_AI_PROCESSOR_ID")
    GCP_RAW_BUCKET: str = os.getenv("GCP_RAW_BUCKET")
    GCP_PROCESSED_BUCKET: str = os.getenv("GCP_PROCESSED_BUCKET")
    VPC_CONNECTOR: str = os.getenv("VPC_CONNECTOR")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_docs")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT: str = os.getenv("QDRANT_CLUSTER_ENDPOINT")

    LOGFIRE_TOKEN: str = os.getenv("LOGFIRE_TOKEN")
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT")

    MAX_PAGE_PER_PARSE: int = int(os.getenv("MAX_PAGE_PER_PARSE", "20"))

    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", 1356))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 100))

    VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 1356))
    MAX_RETRIEVAL_ROUND = int(os.getenv("MAX_RETRIEVAL_ROUND", 1))

    CACHE_DIR = Path(".cache/doc_parser")
    CACHE_MANIFEST = CACHE_DIR / "manifest.json"


config = Config()
