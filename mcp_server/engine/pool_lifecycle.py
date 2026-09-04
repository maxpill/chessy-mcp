"""Analyzer pool lifecycle — creation, lookup, and capability probes.

Extracted from :mod:\`mcp_server.engine.pool_factory\`. The pool itself
lives in :mod:\`core.engines.pool\` (subprocess pool) or
:mod:\`mcp_server.tcp_analyzer\` (TCP pool); this module owns the
:func:\`_create_analyzer_pool\` factory, the lifespan-aware lookup
:func:\`_get_analyzer_pool\`, the runtime capability probe
:func:\`_pool_supports_root_moves\`, and the dispatcher
:func:\`_eval_via_analyzer_or_pool\`.

The :func:\`close_analyzer_pool\` teardown is also here.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import TYPE_CHECKING, Any, cast

import chess

from core.engines.pool import AnalyzerPool

from mcp_server.config import MCPSettings, get_mcp_settings
from mcp_server.tcp_analyzer import TCPAnalyzerPool

if TYPE_CHECKING:
    from mcp.server.mcpserver import Context


log = logging.getLogger("chessy_mcp.engine.lifecycle")


_POOL_SUPPORTS_ROOT_MOVES: dict[type, bool] = {}


async def create_analyzer_pool(
    cfg: MCPSettings,
    *,
    pool_size: int,
) -> AnalyzerPool | TCPAnalyzerPool:
    """Single source of truth for analyzer pool creation."""
    threads = max(1, cfg.threads_per_worker)
    if cfg.host and cfg.port:
        pool: AnalyzerPool | TCPAnalyzerPool = await TCPAnalyzerPool.create(
            cfg.host,
            cfg.port,
            size=pool_size,
            name="stockfish",
            threads=threads,
            hash_mb=cfg.hash_mb,
            show_wdl=cfg.show_wdl,
            syzygy_path=cfg.syzygy_path or None,
        )
        log.info(
            "TCP analyzer pool ready: %d engines @ %s:%d (threads=%d hash=%dMB wdl=%s syzygy=%s ponder=%s)",
            pool_size,
            cfg.host,
            cfg.port,
            threads,
            cfg.hash_mb,
            cfg.show_wdl,
            cfg.syzygy_path or "(none)",
            cfg.ponder_enabled,
        )
        pool._mcp_ponder_enabled = cfg.ponder_enabled  # type: ignore[attr-defined]
        return pool
    pool = await AnalyzerPool.create(
        stockfish_path(),
        size=pool_size,
        depth=20,
        threads=threads,
        hash_mb=cfg.hash_mb,
        show_wdl=cfg.show_wdl,
        syzygy_path=cfg.syzygy_path or None,
    )
    log.info(
        "Subprocess analyzer pool ready: %d engines @ %s",
        pool_size,
        stockfish_path(),
    )
    return pool


async def get_analyzer_pool(ctx: "Context | None" = None) -> AnalyzerPool | TCPAnalyzerPool:
    """Fetch the live analyzer pool from the FastMCP lifespan context.

    Falls back to the legacy lazy-init path when called outside a request
    (e.g. tests that don't go through the FastMCP runner) so existing test
    setups keep working without rewriting every fixture.
    """
    if ctx is not None:
        ls = ctx.request_context.lifespan_context
        pool = ls.get("pool")
        if pool is not None:
            return pool
    from mcp_server import server

    state = server
    async with state._pool_lock:
        if state._analyzer_pool is None:
            mcp_cfg = get_mcp_settings()
            cpu = os.cpu_count() or 8
            pool_size = mcp_cfg.pool_size if mcp_cfg.pool_size is not None else min(cpu, 4)
            state._analyzer_pool = await create_analyzer_pool(mcp_cfg, pool_size=pool_size)
        return state._analyzer_pool


async def close_analyzer_pool() -> None:
    """Gracefully close all engine workers in the pool. Idempotent."""
    from mcp_server import server

    state = server
    async with state._pool_lock:
        if state._analyzer_pool is not None:
            await state._analyzer_pool.close()
            state._analyzer_pool = None
    state._evaluate_semaphore = None


def pool_supports_root_moves(pool: object) -> bool:
    """Memoized runtime check for the ``root_moves`` keyword on pool.evaluate."""
    cls = type(pool)
    cached = _POOL_SUPPORTS_ROOT_MOVES.get(cls)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(pool.evaluate)  # type: ignore[attr-defined]
        supports = "root_moves" in sig.parameters
    except (TypeError, ValueError, AttributeError):
        supports = False
    _POOL_SUPPORTS_ROOT_MOVES[cls] = supports
    return supports


async def eval_via_analyzer_or_pool(
    analyzer: object | None,
    pool: AnalyzerPool | TCPAnalyzerPool,
    b: chess.Board,
    *,
    depth: int,
    reuse_tt: bool,
    root_moves: list[chess.Move] | None = None,
) -> Any:
    """Single eval call routed through analyzer (TT-accumulating) or pool."""
    if analyzer is not None:
        return await analyzer.evaluate(  # type: ignore[attr-defined]
            b, depth=depth, reuse_tt=reuse_tt, root_moves=root_moves
        )
    if pool_supports_root_moves(pool):
        return await pool.evaluate(b, depth=depth, root_moves=root_moves)  # type: ignore[arg-type]
    return await pool.evaluate(b, depth=depth)  # type: ignore[arg-type]


def stockfish_path() -> str:
    """Local import shim — delegates to :mod:\`mcp_server.engine.identity\`."""
    from mcp_server.engine.identity import stockfish_path as _impl

    return _impl()


# Back-compat shims — tests monkeypatch these names on the legacy
# ``mcp_server.engine.pool_factory`` module, so the symbols must remain
# reachable under their original underscore-prefixed identifiers.
_create_analyzer_pool = create_analyzer_pool
_get_analyzer_pool = get_analyzer_pool
_pool_supports_root_moves = pool_supports_root_moves
_eval_via_analyzer_or_pool = eval_via_analyzer_or_pool
