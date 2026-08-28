from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any


class LocalMetricsTracker:
    """In-memory thread-safe metrics and logging tracker for the MCP server."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._total_requests: int = 0
        self._tool_counts: dict[str, int] = defaultdict(int)
        self._tool_latencies: dict[str, list[float]] = defaultdict(list)
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._errors: dict[str, int] = defaultdict(int)
        self._start_time: float = time.time()

    async def record(
        self, tool: str, duration_ms: float, cache_hit: bool = False, is_error: bool = False
    ) -> None:
        async with self._lock:
            self._total_requests += 1
            self._tool_counts[tool] += 1
            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

            if is_error:
                self._errors[tool] += 1

            lat_list = self._tool_latencies[tool]
            lat_list.append(duration_ms)
            if len(lat_list) > 1000:
                self._tool_latencies[tool] = lat_list[-500:]

    async def get_stats(self) -> dict[str, Any]:
        async with self._lock:
            uptime = round(time.time() - self._start_time, 1)
            total_cache_ops = self._cache_hits + self._cache_misses
            hit_rate = round((self._cache_hits / total_cache_ops) * 100.0, 1) if total_cache_ops > 0 else 0.0

            per_tool_stats: dict[str, dict[str, Any]] = {}
            for tool, count in self._tool_counts.items():
                lats = sorted(self._tool_latencies[tool])
                if lats:
                    p50 = round(lats[len(lats) // 2], 1)
                    p95 = round(lats[int(len(lats) * 0.95)], 1)
                    avg = round(sum(lats) / len(lats), 1)
                else:
                    p50 = p95 = avg = 0.0

                per_tool_stats[tool] = {
                    "calls": count,
                    "errors": self._errors[tool],
                    "avg_ms": avg,
                    "p50_ms": p50,
                    "p95_ms": p95,
                }

            return {
                "uptime_seconds": uptime,
                "total_requests": self._total_requests,
                "cache_hit_rate_percent": hit_rate,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "tools": per_tool_stats,
            }


metrics = LocalMetricsTracker()
