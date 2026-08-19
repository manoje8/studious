"""
Unit tests for src.api.rate_limiter.

All Redis calls are mocked so these tests run without a real Redis server.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from medici.api.rate_limiter import AsyncRateLimiterBackend, RateLimiter, _resolve_identity


@pytest.fixture()
def backend(monkeypatch):
    """Return an AsyncRateLimiterBackend with a mocked Redis connection."""
    with patch("medici.api.rate_limiter.aioredis.from_url") as mock_from_url:
        mock_redis = AsyncMock()
        mock_from_url.return_value = mock_redis
        b = AsyncRateLimiterBackend(redis_url="redis://localhost:6379")
        b._redis = mock_redis
        yield b, mock_redis


class TestAsyncRateLimiterBackend:
    @pytest.mark.asyncio
    async def test_allowed_when_under_limit(self, backend):
        b, mock_redis = backend
        script_mock = AsyncMock(return_value=[1, 0])
        b._script = script_mock

        allowed, retry_after = await b.check(
            "rl:test:ip:127.0.0.1", max_requests=10, window_seconds=60
        )

        assert allowed is True
        assert retry_after == 0

    @pytest.mark.asyncio
    async def test_rejected_when_over_limit(self, backend):
        b, mock_redis = backend
        script_mock = AsyncMock(return_value=[0, 30])
        b._script = script_mock

        allowed, retry_after = await b.check(
            "rl:test:ip:127.0.0.1", max_requests=10, window_seconds=60
        )

        assert allowed is False
        assert retry_after == 30

    @pytest.mark.asyncio
    async def test_fails_open_on_redis_error(self, backend):
        """When Redis is unavailable, the limiter should fail open (allow)."""
        b, mock_redis = backend
        script_mock = AsyncMock(side_effect=ConnectionError("Redis down"))
        b._script = script_mock

        allowed, retry_after = await b.check(
            "rl:test:ip:127.0.0.1", max_requests=10, window_seconds=60
        )

        assert allowed is True
        assert retry_after == 0

    @pytest.mark.asyncio
    async def test_script_called_with_correct_keys(self, backend):
        b, mock_redis = backend
        script_mock = AsyncMock(return_value=[1, 0])
        b._script = script_mock

        key = "rl:query:user:alice"
        await b.check(key, max_requests=5, window_seconds=30)

        call_kwargs = script_mock.call_args
        assert call_kwargs.kwargs["keys"] == [key]
        argv = call_kwargs.kwargs["args"]
        assert float(argv[1]) == 30  # window_seconds
        assert int(argv[2]) == 5  # max_requests

    @pytest.mark.asyncio
    async def test_aclose(self, backend):
        b, mock_redis = backend
        mock_redis.aclose = AsyncMock()
        await b.aclose()
        mock_redis.aclose.assert_awaited_once()


class TestResolveIdentity:
    def _make_request(self, client_host: str = "1.2.3.4", headers: dict | None = None):
        req = MagicMock()
        req.client.host = client_host
        req.headers = headers or {}
        return req

    def test_ip_identity_no_credentials(self):
        req = self._make_request("10.0.0.1")
        identity = _resolve_identity(req, credentials=None)
        assert identity == "ip:10.0.0.1"

    def test_ip_identity_from_x_forwarded_for(self):
        req = self._make_request(headers={"X-Forwarded-For": "192.168.1.100, 10.0.0.1"})
        identity = _resolve_identity(req, credentials=None)
        assert identity == "ip:192.168.1.100"

    def test_user_identity_with_valid_token(self):
        req = self._make_request()
        creds = MagicMock()
        creds.credentials = "valid_token"

        with patch("medici.api.auth.auth_handler") as mock_auth:
            mock_auth.validate_token.return_value = {"user_name": "alice"}
            identity = _resolve_identity(req, credentials=creds)

        assert identity == "user:alice"

    def test_falls_back_to_ip_on_invalid_token(self):
        req = self._make_request("5.5.5.5")
        creds = MagicMock()
        creds.credentials = "bad_token"

        with patch("medici.api.auth.auth_handler") as mock_auth:
            mock_auth.validate_token.side_effect = HTTPException(status_code=401)
            identity = _resolve_identity(req, credentials=creds)

        assert identity == "ip:5.5.5.5"

    def test_unknown_ip_when_no_client(self):
        req = MagicMock()
        req.client = None
        req.headers = {}
        identity = _resolve_identity(req, credentials=None)
        assert identity == "ip:unknown"


class TestRateLimiter:
    def _make_request_with_backend(self, backend_instance):
        req = MagicMock()
        req.client.host = "127.0.0.1"
        req.headers = {}
        req.app.state.rate_limiter = backend_instance
        return req

    @pytest.mark.asyncio
    async def test_allows_request_under_limit(self):
        limiter = RateLimiter(scope="query", max_requests=10, window_seconds=60)

        mock_backend = AsyncMock(spec=AsyncRateLimiterBackend)
        mock_backend.check = AsyncMock(return_value=(True, 0))

        req = self._make_request_with_backend(mock_backend)
        # Should not raise
        await limiter(request=req, credentials=None)
        mock_backend.check.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_429_when_limit_exceeded(self):
        limiter = RateLimiter(scope="query", max_requests=10, window_seconds=60)

        mock_backend = AsyncMock(spec=AsyncRateLimiterBackend)
        mock_backend.check = AsyncMock(return_value=(False, 45))

        req = self._make_request_with_backend(mock_backend)

        with pytest.raises(HTTPException) as exc_info:
            await limiter(request=req, credentials=None)

        assert exc_info.value.status_code == 429
        assert exc_info.value.headers["Retry-After"] == "45"

    @pytest.mark.asyncio
    async def test_passes_when_no_backend_initialised(self):
        """Rate limiter must fail open when app.state.rate_limiter is None."""
        limiter = RateLimiter(scope="query", max_requests=10, window_seconds=60)

        req = MagicMock()
        req.client.host = "127.0.0.1"
        req.headers = {}
        req.app.state.rate_limiter = None

        # Should not raise even without a backend
        await limiter(request=req, credentials=None)

    @pytest.mark.asyncio
    async def test_redis_key_format(self):
        limiter = RateLimiter(scope="ingestion", max_requests=5, window_seconds=30)

        mock_backend = AsyncMock(spec=AsyncRateLimiterBackend)
        mock_backend.check = AsyncMock(return_value=(True, 0))

        req = self._make_request_with_backend(mock_backend)
        await limiter(request=req, credentials=None)

        call_args = mock_backend.check.call_args
        assert call_args.kwargs["key"].startswith("rl:ingestion:")
        assert call_args.kwargs["max_requests"] == 5
        assert call_args.kwargs["window_seconds"] == 30
