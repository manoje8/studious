from typing import Optional

from fastapi import APIRouter, Depends

from src.agents.graph.runner import GraphPipeline
from src.api.deps import get_pipeline


def create_query_routes(api_key: Optional[str] = None, top_k: int = 60):
    router = APIRouter(tags=["query"])

    @router.post("/query")
    async def create_query(
        question: str,
        session_id: str,
        user_id: str,
        pipeline: GraphPipeline = Depends(get_pipeline),
    ):
        return await pipeline.chat(
            user_message=question, session_id=session_id, user_id=user_id
        )

    return router
