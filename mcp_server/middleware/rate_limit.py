"""Weighted token-bucket rate limiter.

Used by :class:`mcp_server.middleware.request_logger.ASGIRequestLoggerMiddleware`
to admission-control MCP requests. Pure in-memory, per-IP, with a periodic
eviction sweep so the bucket map stays bounded under sustained floods.
"""

from __future__ import annotations

import asyncio
import time

__all__ = ["TokenBucketRateLimiter"]


class TokenBucketRateLimiter:
    """In-memory weighted token bucket per effective client IP."""

    def __init__(self, rate: float = 20.0, capacity: float = 500.0) -> None:
        self.rate = rate
        self.capacity = capacity
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str, cost: float = 1.0) -> bool:
        now = time.time()
        async with self._lock:
            if len(self._buckets) > 10_000:
                cutoff = now - 3600
                self._buckets = {k: v for k, v in self._buckets.items() if v[1] > cutoff}
                if len(self._buckets) > 10_000:
                    self._buckets = dict(
                        sorted(
                            self._buckets.items(),
                            key=lambda item: item[1][1],
                            reverse=True,
                        )[:5000]
                    )
            tokens, last_time = self._buckets.get(client_id, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last_time) * self.rate)
            if tokens >= cost:
                self._buckets[client_id] = (tokens - cost, now)
                return True
            self._buckets[client_id] = (tokens, now)
            return False
