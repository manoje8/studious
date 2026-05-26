import logfire

from src.agents.hybrid_search import HybridSearch
from src.ingestion.embedding import EmbeddingService
from src.services.qdrant import QdrantStorageService
from src.services.sparse_index import SparseSearchIndex


class Pipeline:
    def __init__(self, llm_client, redis_url, db_client, qdrant_url):
        logfire.configure(service_name="RAG Pipeline")
        embedding_service = EmbeddingService()
        storage_service = QdrantStorageService(url=qdrant_url)
        sparse_index = SparseSearchIndex()

        hybrid_search = HybridSearch(
            storage_service=storage_service,
            embedding_service=embedding_service,
            sparse_index=sparse_index,
        )

        logfire.info(f"Hybrid search: {hybrid_search.search(['Example'])}")

        # Sparse Index search
        # Hybrid search
        # rerank
        # Query Expander
        # Retrieval
        # RAG pipeline
        # multi turn agent
