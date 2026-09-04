"""Engine lifespan + pool-stats logger.

Extracted from :mod:`mcp_server.engine.pool_factory`. Owns the
:func:`mcp_lifespan` FastMCP lifespan handler (eager pool init, warm
search, periodic pool-stats logging) and the :func:`pool_stats_logger`
background task.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import chess
from mcp.server.mcpserver import MCPServer

from core.engines.pool import AnalyzerPool

from mcp_server.config import get_mcp_settings
from mcp_server.engine.pool_lifecycle import create_analyzer_pool
from mcp_server.metrics import metrics
from mcp_server.tcp_analyzer import TCPAnalyzerPool


log = logging.getLogger("chessy_mcp.engine.lifespan")


async def pool_stats_logger(
    pool: AnalyzerPool | TCPAnalyzerPool,
    interval_s: float,
) -> None:
    """Emit a structured pool-stats log line every ``interval_s`` seconds."""
    while True:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
        try:
            qsize = pool._pool._q.qsize()  # type: ignore[attr-defined]
            alive = pool._pool._alive_count  # type: ignore[attr-defined]
            target = pool._pool._target_size  # type: ignore[attr-defined]
            stats = await metrics.get_stats()
            log.info(
                "pool_stats queue_depth=%d alive=%d target=%d "
                "uptime_s=%s total=%s hit_rate_pct=%s tools=%s",
                qsize,
                alive,
                target,
                stats["uptime_seconds"],
                stats["total_requests"],
                stats["cache_hit_rate_percent"],
                {k: v["calls"] for k, v in stats["tools"].items()},
            )
        except Exception as exc:
            log.warning("pool_stats log iteration failed (continuing): %s", exc)


@asynccontextmanager
async def mcp_lifespan(server: MCPServer) -> AsyncGenerator[dict[str, Any]]:
    """Initialize the Stockfish pool at startup, tear it down at exit.

    Replaces the lazy-init path with eager startup so the first user request
    doesn't pay the TCP handshake / UCI isready round-trips. The pool is
    shared with every tool via the ``lifespan_context["pool"]`` indirection.
    """
    cfg = get_mcp_settings()
    cpu = os.cpu_count() or 8
    pool_size = cfg.pool_size if cfg.pool_size is not None else min(cpu, 4)
    pool: AnalyzerPool | TCPAnalyzerPool = await create_analyzer_pool(cfg, pool_size=pool_size)

    warmup_board = chess.Board()
    try:
        await asyncio.gather(*[pool.evaluate(warmup_board, depth=2) for _ in range(pool_size)])
        log.info("Pool warm-search complete (%d workers primed)", pool_size)
    except Exception as exc:
        log.warning("Pool warm-search failed (non-fatal): %s", exc)

    stats_task: asyncio.Task[None] | None = None
    stats_interval = float(cfg.pool_stats_interval_s)
    if stats_interval > 0:
        stats_task = asyncio.create_task(
            pool_stats_logger(pool, stats_interval), name="pool-stats-logger"
        )

    try:
        # Phase 31: also yield a ToolContext so tools can fetch dependencies
        # via ``ctx.request_context.lifespan_context["ctx"]`` instead of
        # reaching back into module globals. Built once here, frozen for
        # the lifetime of the process.
        from mcp_server.core.context import ToolContext

        ctx = ToolContext.from_engine_layer(
            engine=pool,
            settings=cfg,
        )
        yield {"pool": pool, "settings": cfg, "pool_size": pool_size, "ctx": ctx}
    finally:
        if stats_task is not None:
            stats_task.cancel()
            try:
                await stats_task
            except (asyncio.CancelledError, Exception):
                pass
        log.info("Shutting down analyzer pool (%d engines)", pool_size)
        await pool.close()


# Back-compat shims.
_pool_stats_logger = pool_stats_logger
_mcp_lifespan = mcp_lifespan
