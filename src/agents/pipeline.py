import asyncio

import logfire

from src.agents.agentic.agentic import AgenticRAG
from src.agents.hybrid_search import HybridSearch
from src.agents.memory.episodic import EpisodicMemoryManager
from src.agents.memory.short_term import ShortTermMemoryManager
from src.agents.multi_turn_agentic import MultiTurnAgenticRAGPipeline
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
    def __init__(
        self,
        llm_client,
        qdrant_url,
        redis_url: str = config.REDIS_URL,
        db_client: str | None = None,
    ):
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

        rag = AgenticRAG(
            llm_client=llm_client,
            retrieval_agent=retriever,
        )

        self.pipeline = MultiTurnAgenticRAGPipeline(
            llm_client=llm_client,
            rag_agent=rag,
            short_term_memory=ShortTermMemoryManager(redis_url),
            episodic_memory=EpisodicMemoryManager(llm_client, db_client),
        )

    async def chat(self, message: str, session_id: str, user_id: str) -> dict:
        return await self.pipeline.chat(
            user_message=message, session_id=session_id, user_id=user_id
        )


async def main():
    # groq_client = GroqClient(model="llama-3.3-70b-versatile")
    google_client = GeminiClient()
    pipeline = Pipeline(
        llm_client=google_client, qdrant_url=config.QDRANT_CLUSTER_ENDPOINT
    )
    result = await pipeline.chat(
        "Explain Structural design pattern",
        "f4725d32-4cc9-4497-99ce-2aeb992855e5",
        "455",
    )
    print(result.get("answer"))


if __name__ == "__main__":
    logfire.configure(service_name="Studious")
    asyncio.run(main())
