"""
Sliding-window rate limiter backed by Redis.

Each limit is expressed as ``max_requests`` within a rolling ``window_seconds``
window.  The key is built from a *scope* tag, the endpoint path, and an
identity string (user-id from a validated JWT or the client's IP address).

The limiter stores the rate-limit state in the app's ``state.rate_limiter``
object, which is an ``AsyncRateLimiterBackend`` instance wired up in the
application lifespan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import logfire
import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

if TYPE_CHECKING:
    pass

_bearer = HTTPBearer(auto_error=False)


class AsyncRateLimiterBackend:
    """
    Redis-backed sliding-window rate limiter.

    Implements the *sorted-set* sliding window algorithm:
    - Each request adds a member ``{timestamp}:{uuid}`` with score = timestamp.
    - Members older than the window are removed.
    - If the count ≥ limit → reject (HTTP 429).

    All operations are executed inside a single Lua script to make them atomic.
    """

    _LUA_SCRIPT = """
local key      = KEYS[1]
local now      = tonumber(ARGV[1])
local window   = tonumber(ARGV[2])
local limit    = tonumber(ARGV[3])
local member   = ARGV[4]
local ttl      = window + 1

-- remove expired members
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)

local count = redis.call('ZCARD', key)

if count >= limit then
    -- return remaining window time for Retry-After header
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = math.ceil((tonumber(oldest[2]) + window) - now)
    return {0, retry_after}
end

redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, ttl)
return {1, 0}
"""

    def __init__(self, redis_url: str) -> None:
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, encoding="utf-8", decode_responses=True
        )
        self._script: aioredis.client.Script | None = None

    async def _get_script(self) -> aioredis.client.Script:
        if self._script is None:
            self._script = self._redis.register_script(self._LUA_SCRIPT)
        return self._script

    async def check(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Returns ``(allowed, retry_after_seconds)``.

        - ``allowed`` is False when the limit is exceeded.
        - ``retry_after_seconds`` is the number of seconds until the caller
          may try again (0 when allowed).
        """
        now = time.time()
        member = f"{now:.6f}"

        try:
            script = await self._get_script()
            result = await script(
                keys=[key],
                args=[now, window_seconds, max_requests, member],
            )
            allowed = bool(result[0])
            retry_after = int(result[1])
            return allowed, retry_after
        except Exception as exc:
            logfire.warn(
                "Rate-limiter Redis error – failing open",
                error=str(exc),
                key=key,
            )
            return True, 0

    async def aclose(self) -> None:
        await self._redis.aclose()


@dataclass(unsafe_hash=True)
class RateLimiter:
    """
    FastAPI dependency factory that enforces a sliding-window rate limit.

    Parameters
    ----------
    scope:
        A short label for the limit group (e.g. ``"query"``, ``"ingestion"``).
        Combined with the endpoint path to form the Redis key prefix.
    max_requests:
        Maximum number of requests allowed within ``window_seconds``.
    window_seconds:
        Length of the sliding window in seconds.
    """

    scope: str
    max_requests: int
    window_seconds: int
    _instances: list[RateLimiter] = field(default_factory=list, init=False, repr=False, hash=False)

    async def __call__(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> None:
        backend: AsyncRateLimiterBackend | None = getattr(request.app.state, "rate_limiter", None)

        if backend is None:
            return

        identity = _resolve_identity(request, credentials)
        redis_key = f"rl:{self.scope}:{identity}"

        allowed, retry_after = await backend.check(
            key=redis_key,
            max_requests=self.max_requests,
            window_seconds=self.window_seconds,
        )

        logfire.debug(
            "Rate-limit check",
            scope=self.scope,
            identity=identity,
            allowed=allowed,
            retry_after=retry_after,
        )

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded. "
                    f"Max {self.max_requests} requests per {self.window_seconds}s. "
                    f"Retry after {retry_after}s."
                ),
                headers={"Retry-After": str(retry_after)},
            )


def _resolve_identity(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
) -> str:
    """
    Derive the rate-limit identity from the request.

    Priority:
    1. ``user_id`` claim from a validated JWT (so authenticated users share a
       per-user bucket, not per-IP).
    2. The client's IP address (for unauthenticated endpoints).
    """
    if credentials is not None:
        try:
            from medici.api.auth import auth_handler

            payload = auth_handler.validate_token(credentials.credentials)
            return f"user:{payload['user_name']}"
        except HTTPException:
            pass  # fall through to IP-based identity

    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"

    return f"ip:{client_ip}"
