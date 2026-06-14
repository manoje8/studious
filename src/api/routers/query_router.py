from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.agents.graph.runner import GraphPipeline
from src.api.deps import get_pipeline


class QueryRequest(BaseModel):
    question: str
    user_id: str
    session_id: str | None = None
    is_multi_retriever: bool = False


def create_query_routes(api_key: Optional[str] = None, top_k: int = 60):
    router = APIRouter(tags=["query"])

    @router.post("/query")
    async def create_query(
        body: QueryRequest,
        pipeline: GraphPipeline = Depends(get_pipeline),
    ):
        return await pipeline.chat(
            user_message=body.question,
            session_id=body.session_id,
            user_id=body.user_id,
        )

    return router
