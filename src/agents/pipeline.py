import logfire

from src.agents.hybrid_search import HybridSearch
from src.agents.query_expander import QueryExpander
from src.agents.retrieval import RetrievalAgent
from src.ingestion.embedding import EmbeddingService
from src.services.qdrant import QdrantStorageService
from src.services.reranker import Reranker
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

        rerank = Reranker()

        query_expander = QueryExpander(llm_client)

        retriever = RetrievalAgent(
            llm_client=llm_client,
            hybrid_search=hybrid_search,
            reranker=rerank,
            query_expand=query_expander,
        )

        logfire.info(f"Retriever: {str(retriever)}")

        # Sparse Index search
        # Hybrid search
        # rerank
        # Query Expander
        # Retrieval
        # RAG pipeline
        # multi turn agent
