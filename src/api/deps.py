from fastapi import HTTPException, Request

from src.agents.graph.runner import GraphPipeline


def get_pipeline(request: Request) -> GraphPipeline:
    pipeline = request.app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline
