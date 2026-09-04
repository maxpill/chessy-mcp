"""Ponder-warming background tasks.

Extracted from :mod:`mcp_server.engine.pool_factory`. Owns the
provenance-preserving :func:`ponder_warm_cache` background eval and
the :func:`maybe_ponder_warm` dispatcher that schedules it after a
successful :func:`cached_evaluator.evaluate_game_position_cached` call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import chess

from core.engines.pool import AnalyzerPool

from mcp_server.cache import eval_cache_key
from mcp_server.tcp_analyzer import TCPAnalyzerPool


log = logging.getLogger("chessy_mcp.engine.ponder")


_background_tasks: set[asyncio.Task[Any]] = set()


async def ponder_warm_cache(
    pool: AnalyzerPool | TCPAnalyzerPool,
    predicted_board: chess.Board,
    depth: int,
    history_complete: str,
) -> None:
    """Background cache warmer that preserves the predicted board's move stack."""
    try:
        board = predicted_board.copy(stack=True)
        if board.is_game_over(claim_draw=False):
            return
        # Lazy lookup of the cached evaluator keeps the legacy
        # ``monkeypatch.setattr("mcp_server.engine.pool_factory._evaluate_game_position_cached", ...)``
        # test pattern working: the patch lands on the re-exported symbol
        # in pool_factory, and we look it up at call time.
        from mcp_server.engine import pool_factory as _pool_factory
        from mcp_server.engine.cached_evaluator import cache

        ckey = eval_cache_key(
            board,
            depth,
            engine_version=getattr(pool, "engine_version", None),
            history_completeness=history_complete,
        )
        if (await cache.get_eval(ckey)) is not None:
            return
        await _pool_factory._evaluate_game_position_cached(
            board,
            depth,
            pool,
            requested_depth=depth,
            history_complete=history_complete,
        )
    except Exception as exc:
        log.debug("ponder pre-eval failed: %s", exc)


def maybe_ponder_warm(
    pool: AnalyzerPool | TCPAnalyzerPool,
    board: chess.Board,
    best_move_uci: str | None,
    depth: int,
    ponder_enabled: bool,
    history_complete: str,
) -> None:
    """Schedule a provenance-preserving background pre-evaluation."""
    if not ponder_enabled or not best_move_uci:
        return
    try:
        next_board = board.copy(stack=True)
        next_board.push_uci(best_move_uci)
        if next_board.is_game_over(claim_draw=False):
            return
        task = asyncio.create_task(
            ponder_warm_cache(pool, next_board, depth, history_complete),
            name="ponder-warm",
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except Exception:
        pass


# Back-compat shims.
_ponder_warm_cache = ponder_warm_cache
_maybe_ponder_warm = maybe_ponder_warm
