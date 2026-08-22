from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import logfire
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from medici.agents.graph.runner import GraphPipeline
from medici.api.deps import get_pipeline

if TYPE_CHECKING:
    from medici.api.rate_limiter import RateLimiter


class QueryRequest(BaseModel):
    question: str
    user_id: str
    session_id: str | None = None
    is_multi_retriever: bool = False


async def _event_generator(
    pipeline: GraphPipeline,
    question: str,
    session_id: str | None,
    user_id: str,
) -> AsyncIterator[str]:
    """
    Convert ``GraphPipeline.chat_stream`` dicts into SSE-formatted bytes.

    Each SSE event is a ``data: <json>\\n\\n`` line.  The stream ends after
    the ``done`` or ``error`` event type is yielded by the pipeline.
    """
    try:
        async for event in pipeline.chat_stream(
            user_message=question,
            session_id=session_id or "",
            user_id=user_id,
        ):
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("done", "error"):
                break
    except Exception as exc:
        logfire.error("SSE generator unhandled error", error=str(exc))
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"


def create_query_routes(
    api_key: str | None = None, top_k: int = 60, query_limiter: RateLimiter | None = None
):
    router = APIRouter(tags=["query"])
    route_deps = [Depends(query_limiter)] if query_limiter is not None else []

    @router.post("/query", dependencies=route_deps)
    async def create_query(
        body: QueryRequest,
        pipeline: GraphPipeline = Depends(get_pipeline),
        # _auth: dict = Depends(require_auth),
    ):
        result = await pipeline.chat(
            user_message=body.question,
            session_id=body.session_id,
            user_id=body.user_id,
        )
        return {
            "answer": result["answer"],
            "session_id": result["session_id"],
            "sources": result["sources"],
            "images": result.get("images", []),
            "query_was_rewritten": result["query_was_rewritten"],
            "cache_hit": result.get("cache_hit", False),
            "cache_similarity": result.get("cache_similarity"),
            "token_usage": result.get("token_usage", {}),
        }

    @router.post("/query/stream", dependencies=route_deps)
    async def stream_query(
        body: QueryRequest,
        pipeline: GraphPipeline = Depends(get_pipeline),
        # _auth: dict = Depends(require_auth),
    ):
        """Server-Sent Events endpoint for streamed query responses.

        Each event is a JSON object with a ``type`` field:

        * ``progress``  — pipeline stage notification (node name + label)
        * ``token``     — single answer token from the LLM
        * ``done``      — stream complete; carries full result metadata
        * ``error``     — unrecoverable error; stream ends

        **Example client (JavaScript)**::

            const evtSource = new EventSource('/query/stream');
            evtSource.onmessage = ({ data }) => {
                const ev = JSON.parse(data);
                if (ev.type === 'token') appendToAnswer(ev.content);
                if (ev.type === 'done')  evtSource.close();
            };
        """
        return StreamingResponse(
            _event_generator(
                pipeline=pipeline,
                question=body.question,
                session_id=body.session_id,
                user_id=body.user_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable Nginx response buffering
            },
        )

    return router
