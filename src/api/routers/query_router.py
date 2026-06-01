from typing import Optional

from fastapi import APIRouter


def create_query_routes(rag, api_key: Optional[str] = None, top_k: int = 60):
    router = APIRouter(tags=["query"])

    @router.post("/query")
    async def create_query(question: str):
        return await rag.chat(question, max_rounds=1)
