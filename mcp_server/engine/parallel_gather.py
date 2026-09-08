"""Parallel-position gatherer (audit P1 fix).

Extracted from :mod:`mcp_server.engine.pool_factory`. Owns
:func:`gather_evaluate_positions_bounded`: evaluates N positions
partitioned across the engine pool with TT-reuse per slice, bounded by
the module-level semaphore from
:mod:`mcp_server.engine.cached_evaluator`.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import chess

from core.engines.pool import AnalyzerPool

from mcp_server.engine.cached_evaluator import (
    get_evaluate_semaphore,
)
from mcp_server.models import MCPEval
from mcp_server.tcp_analyzer import TCPAnalyzerPool


log = logging.getLogger("chessy_mcp.engine.parallel_gather")


async def gather_evaluate_positions_bounded(
    positions: list[chess.Board],
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
    *,
    requested_depth: int,
    history_complete: str = "complete",
    short_circuit: Any = None,
) -> list[tuple[MCPEval, bool]]:
    """Evaluate N positions partitioned across the pool with TT reuse per slice.

    2026-09-08 audit Bug 8: ``short_circuit`` is an optional callable
    ``(chess.Board, int) -> MCPEval | None`` that receives the
    position AND its index in the caller's list. When it returns a
    non-None MCPEval, that position is short-circuited (no engine
    call). Used by ``analyze_game`` for the final position of a PGN
    that ended with a 50-move-rule draw — the engine call there is
    wasted because the player will claim rather than play.
    """
    if not positions:
        return []
    sem = await get_evaluate_semaphore()

    pool_target: int
    try:
        pool_target = pool._pool._target_size  # type: ignore[attr-defined]
    except AttributeError:
        pool_target = 4
    k = max(1, min(pool_target, len(positions)))

    chunk = math.ceil(len(positions) / k)
    slices: list[list[tuple[int, chess.Board]]] = [
        [(i, positions[i]) for i in range(start, min(start + chunk, len(positions)))]
        for start in range(0, len(positions), chunk)
    ]
    slices = slices[:k]

    async def _run_slice(
        slice_items: list[tuple[int, chess.Board]],
    ) -> list[tuple[int, tuple[MCPEval, bool]]]:
        # Lazy lookup of the cached evaluator preserves the legacy
        # ``monkeypatch.setattr("mcp_server.engine.pool_factory._evaluate_game_position_cached", ...)``
        # test pattern: the patch lands on the pool_factory re-export and we
        # read it through ``_pool_factory._evaluate_game_position_cached`` at
        # call time, so the patched callable is what actually runs.
        from mcp_server.engine import pool_factory as _pool_factory

        eval_cached = _pool_factory._evaluate_game_position_cached
        async with sem:
            if hasattr(pool, "_pool"):

                async def _on_worker(
                    analyzer: Any,
                ) -> list[tuple[int, tuple[MCPEval, bool]]]:
                    out: list[tuple[int, tuple[MCPEval, bool]]] = []
                    for j, (idx, b) in enumerate(slice_items):
                        if short_circuit is not None:
                            mc = short_circuit(b, idx)
                            if mc is not None:
                                out.append((idx, (mc, True)))
                                continue
                        r, hit = await eval_cached(
                            b,
                            depth,
                            pool,
                            requested_depth=requested_depth,
                            reuse_tt=(j > 0),
                            analyzer=analyzer,
                            history_complete=history_complete,
                        )
                        out.append((idx, (r, hit)))
                    return out

                return await pool._pool.run(_on_worker)  # type: ignore[attr-defined]

            out = []
            for j, (idx, b) in enumerate(slice_items):
                if short_circuit is not None:
                    mc = short_circuit(b, idx)
                    if mc is not None:
                        out.append((idx, (mc, True)))
                        continue
                r, hit = await eval_cached(
                    b,
                    depth,
                    pool,
                    requested_depth=requested_depth,
                    reuse_tt=False,
                    history_complete=history_complete,
                )
                out.append((idx, (r, hit)))
            return out

    slice_results = await asyncio.gather(*[_run_slice(s) for s in slices if s])
    out: list[tuple[MCPEval, bool]] = [None] * len(positions)  # type: ignore[list-item]
    for slice_result in slice_results:
        for idx, item in slice_result:
            out[idx] = item
    return out


# Back-compat shim.
_gather_evaluate_positions_bounded = gather_evaluate_positions_bounded
