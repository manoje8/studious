from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from medici.agents.graph.runner import GraphPipeline
from medici.api.auth import auth_handler
from medici.ingestion.processor import Processor

_bearer = HTTPBearer(auto_error=False)


def get_pipeline(request: Request) -> GraphPipeline:
    pipeline = request.app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


def get_processor(request: Request) -> Processor:
    processor = request.app.state.processor
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")
    return processor


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """
    FastAPI dependency: validate Bearer JWT; return decoded payload.

    Raises HTTP 401 when the header is absent or the token is invalid/expired.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "bearer"},
        )
    return auth_handler.validate_token(credentials.credentials)


def require_admin(token_payload: dict = Depends(require_auth)) -> dict:
    """
    FastAPI dependency: require a valid JWT **with role='admin'**.

    Raises HTTP 403 when the authenticated user's role is not 'admin'.
    """
    if token_payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return token_payload
