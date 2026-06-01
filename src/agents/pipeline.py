import asyncio

import logfire

from src.agents.agentic.agentic import AgenticRAG
from src.agents.hybrid_search import HybridSearch
from src.agents.query_expander import QueryExpander
from src.agents.retrieval import RetrievalAgent
from src.ingestion.embedding import EmbeddingService
from src.llm.gemini import GeminiClient
from src.services.qdrant import QdrantStorageService
from src.services.reranker import Reranker
from src.services.sparse_index import SparseSearchIndex
from src.utils.config import config


class Pipeline:
    # TODO: redis_url, db_client in params
    def __init__(self, llm_client, qdrant_url):
        logfire.configure(service_name="RAG Pipeline")
        embedding_service = EmbeddingService(model_name=config.EMBEDDING_MODEL_NAME)
        storage_service = QdrantStorageService(url=qdrant_url)
        sparse_index = SparseSearchIndex()

        hybrid_search = HybridSearch(
            storage_service=storage_service,
            embedding_service=embedding_service,
            sparse_index=sparse_index,
        )

        rerank = Reranker()

        query_expander = QueryExpander(llm_client)

        retriever = RetrievalAgent(
            llm_client=llm_client,
            hybrid_search=hybrid_search,
            reranker=rerank,
            query_expand=query_expander,
        )

        self.rag = AgenticRAG(
            llm_client=llm_client,
            embedding_service=embedding_service,
            storage_service=storage_service,
            retrieval_agent=retriever,
        )

    async def chat(
        self, message: str, session_id: str, user_id: str, max_rounds: int = 1
    ) -> dict:
        return await self.rag.run(question=message, max_rounds=max_rounds)

    # multi turn agent


async def main():
    client = GeminiClient(model=config.GEMINI_MODEL)
    pipeline = Pipeline(llm_client=client, qdrant_url=config.QDRANT_CLUSTER_ENDPOINT)
    result = await pipeline.chat("Explain design pattern", "535", "455")
    print(result.get("answer"))


if __name__ == "__main__":
    asyncio.run(main())
