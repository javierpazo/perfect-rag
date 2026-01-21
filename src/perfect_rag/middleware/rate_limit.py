"""Rate limiting middleware."""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog
from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting."""

    tokens: float
    last_update: float
    requests_count: int = 0
    tokens_used: int = 0


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    burst_multiplier: float = 1.5


class RateLimiter:
    """Token bucket rate limiter with sliding window.

    Supports:
    - Requests per minute (RPM)
    - Tokens per minute (TPM)
    - Per-user and global limits
    - Burst allowance
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

        # Default limits
        self.default_rpm = self.settings.rate_limit_rpm
        self.default_tpm = self.settings.rate_limit_tpm

        # Buckets by identifier (user_id, api_key, or IP)
        self._buckets: dict[str, RateLimitBucket] = {}
        self._bucket_lock = asyncio.Lock()

        # Global bucket
        self._global_bucket = RateLimitBucket(
            tokens=self.settings.rate_limit_global_rpm,
            last_update=time.time(),
        )

        # Cleanup interval (remove old buckets)
        self._cleanup_interval = 300  # 5 minutes

    def _get_bucket(
        self,
        identifier: str,
        rpm: int | None = None,
        tpm: int | None = None,
    ) -> RateLimitBucket:
        """Get or create a rate limit bucket for identifier."""
        if identifier not in self._buckets:
            self._buckets[identifier] = RateLimitBucket(
                tokens=rpm or self.default_rpm,
                last_update=time.time(),
            )
        return self._buckets[identifier]

    def _refill_bucket(
        self,
        bucket: RateLimitBucket,
        max_tokens: int,
        refill_rate: float,
    ) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - bucket.last_update

        # Calculate tokens to add (tokens per second)
        tokens_to_add = elapsed * (refill_rate / 60.0)
        bucket.tokens = min(max_tokens * 1.5, bucket.tokens + tokens_to_add)  # Allow burst
        bucket.last_update = now

    async def check_rate_limit(
        self,
        identifier: str,
        rpm: int | None = None,
        tpm: int | None = None,
        tokens_required: int = 1,
    ) -> tuple[bool, dict[str, Any]]:
        """Check if request is allowed under rate limits.

        Args:
            identifier: User ID, API key, or IP address
            rpm: Custom requests per minute limit
            tpm: Custom tokens per minute limit
            tokens_required: Estimated tokens for this request

        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        async with self._bucket_lock:
            rpm = rpm or self.default_rpm
            tpm = tpm or self.default_tpm

            bucket = self._get_bucket(identifier, rpm, tpm)
            self._refill_bucket(bucket, rpm, rpm)

            # Check if we have enough tokens
            if bucket.tokens < 1:
                retry_after = int((1 - bucket.tokens) / (rpm / 60.0)) + 1
                return False, {
                    "allowed": False,
                    "limit": rpm,
                    "remaining": 0,
                    "reset": retry_after,
                    "retry_after": retry_after,
                }

            # Consume token
            bucket.tokens -= 1
            bucket.requests_count += 1
            bucket.tokens_used += tokens_required

            return True, {
                "allowed": True,
                "limit": rpm,
                "remaining": int(bucket.tokens),
                "reset": 60,
            }

    async def check_global_limit(self) -> tuple[bool, dict[str, Any]]:
        """Check global rate limit."""
        async with self._bucket_lock:
            self._refill_bucket(
                self._global_bucket,
                self.settings.rate_limit_global_rpm,
                self.settings.rate_limit_global_rpm,
            )

            if self._global_bucket.tokens < 1:
                return False, {"allowed": False, "reason": "global_limit"}

            self._global_bucket.tokens -= 1
            return True, {"allowed": True}

    def get_usage_stats(self, identifier: str) -> dict[str, Any]:
        """Get usage stats for an identifier."""
        bucket = self._buckets.get(identifier)
        if not bucket:
            return {"requests_count": 0, "tokens_used": 0}

        return {
            "requests_count": bucket.requests_count,
            "tokens_used": bucket.tokens_used,
            "remaining_requests": int(bucket.tokens),
        }

    async def cleanup_old_buckets(self) -> int:
        """Remove old inactive buckets."""
        async with self._bucket_lock:
            now = time.time()
            old_buckets = [
                k for k, v in self._buckets.items()
                if now - v.last_update > self._cleanup_interval
            ]

            for key in old_buckets:
                del self._buckets[key]

            return len(old_buckets)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        app,
        rate_limiter: RateLimiter | None = None,
        exclude_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json"]

    def _get_identifier(self, request: Request) -> str:
        """Get rate limit identifier from request."""
        # Try to get user ID from state (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        # Try API key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key[:16]}"  # Use prefix for privacy

        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        return f"ip:{client_ip}"

    def _get_custom_limits(self, request: Request) -> tuple[int | None, int | None]:
        """Get custom rate limits from request state (set by auth)."""
        rpm = getattr(request.state, "rate_limit_rpm", None)
        tpm = getattr(request.state, "rate_limit_tpm", None)
        return rpm, tpm

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Skip excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)

        identifier = self._get_identifier(request)
        rpm, tpm = self._get_custom_limits(request)

        # Check global limit first
        global_allowed, _ = await self.rate_limiter.check_global_limit()
        if not global_allowed:
            logger.warning("Global rate limit exceeded")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Server is overloaded. Please try again later.",
                headers={"Retry-After": "60"},
            )

        # Check per-user/key/IP limit
        allowed, info = await self.rate_limiter.check_rate_limit(
            identifier=identifier,
            rpm=rpm,
            tpm=tpm,
        )

        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                identifier=identifier,
                retry_after=info.get("retry_after"),
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(info.get("retry_after", 60)),
                    "X-RateLimit-Limit": str(info.get("limit", 60)),
                    "X-RateLimit-Remaining": str(info.get("remaining", 0)),
                    "X-RateLimit-Reset": str(info.get("reset", 60)),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(info.get("limit", 60))
        response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
        response.headers["X-RateLimit-Reset"] = str(info.get("reset", 60))

        return response


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
