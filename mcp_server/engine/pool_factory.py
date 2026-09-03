"""Stockfish pool factory, lifespan wiring, and the cached evaluation pipeline.

Extracted from ``mcp_server.server``. This module owns:

- The :func:`_create_analyzer_pool` factory that picks subprocess vs TCP based on
  ``MCPSettings``.
- The :func:`_mcp_lifespan` FastMCP lifespan handler (eager pool init, warm
  search, periodic pool-stats logging).
- The cached single-position evaluator :func:`_evaluate_game_position_cached`
  with rule status, terminal short-circuit, multi-tier cache, SingleFlight
  coalescing, and ponder-warming.
- The :func:`_gather_evaluate_positions_bounded` parallel-position helper used
  by ``analyze_game``.
- The runtime ``root_moves`` capability probe :func:`_pool_supports_root_moves`.

Symbols are re-exported through ``mcp_server.server`` so the test suite keeps
working with its current import paths. ``monkeypatch.setattr(server_module,
\"_evaluate_game_position_cached\", ...)\"`` MUST keep working — every public
binding is also assigned as a module attribute on ``mcp_server.server``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import chess
from mcp.server.mcpserver import Context, MCPServer

from core.engines.pool import AnalyzerPool

from mcp_server.actions import build_best_action, build_legal_actions
from mcp_server.cache import (
    MultiTierCache,
    SingleFlight,
    eval_cache_key,
)
from mcp_server.config import MCPSettings, get_mcp_settings
from mcp_server.engine.eval_pipeline import (
    build_terminal_mcpeval,
    evaluate_zeroing_post_state,
)
from mcp_server.engine.identity import build_identity, stockfish_path
from mcp_server.metrics import metrics
from mcp_server.models import MCPEval
from mcp_server.rules import evaluate_rule_status
from mcp_server.tcp_analyzer import TCPAnalyzerPool
from mcp_server.urls import lichess_urls

__all__ = [
    "_create_analyzer_pool",
    "_eval_via_analyzer_or_pool",
    "_evaluate_game_position_cached",
    "_gather_evaluate_positions_bounded",
    "_get_analyzer_pool",
    "_get_evaluate_semaphore",
    "_maybe_ponder_warm",
    "_mcp_lifespan",
    "_ponder_warm_cache",
    "_pool_stats_logger",
    "_pool_supports_root_moves",
    "cache",
    "close_analyzer_pool",
    "single_flight",
]


log = logging.getLogger("chessy_mcp.engine")


# Module-level cache + SingleFlight — shared across every tool entry point.
cache: MultiTierCache = MultiTierCache(l1_size=50_000)
_cache = cache
single_flight: SingleFlight[Any] = SingleFlight()
_single_flight = single_flight


# P1 audit fix: bound concurrent evaluate calls so analyze_game at depth 30
# cannot self-inflict PoolBusy by spawning hundreds of simultaneous waiters.
# The semaphore is created lazily on first call so it always belongs to the
# live event loop (pytest-asyncio's per-function event loop means a module-
# level asyncio.Semaphore() would be bound to whichever loop runs first and
# explode on subsequent loops).
_evaluate_semaphore: asyncio.Semaphore | None = None
_evaluate_semaphore_lock = asyncio.Lock()


async def _get_evaluate_semaphore() -> asyncio.Semaphore:
    state = _state()
    async with state._evaluate_semaphore_lock:
        if state._evaluate_semaphore is None:
            cfg = get_mcp_settings()
            state._evaluate_semaphore = asyncio.Semaphore(max(1, cfg.max_concurrent_evaluates))
        return state._evaluate_semaphore


def _state():
    """Lazy import of :mod:`mcp_server.server` to break the circular dependency.

    The module-level pool / semaphore globals live on ``server`` because the
    test suite mutates ``server_module._analyzer_pool`` directly to install
    mock analyzers; ``pool_factory`` must read/write those exact bindings so
    monkey-patching at the server module level reaches the live state.
    """
    from mcp_server import server

    return server


async def close_analyzer_pool() -> None:
    """Gracefully close all engine workers in the pool.

    Idempotent: safe to call from tests and from lifespan teardown. Also
    drops the cached evaluate semaphore so a fresh one is lazily created on
    the next request — keeps pytest-asyncio's per-function event loop happy.
    """
    state = _state()
    async with state._pool_lock:
        if state._analyzer_pool is not None:
            await state._analyzer_pool.close()
            state._analyzer_pool = None
    state._evaluate_semaphore = None


async def _create_analyzer_pool(
    cfg: MCPSettings,
    *,
    pool_size: int,
) -> AnalyzerPool | TCPAnalyzerPool:
    """Single source of truth for analyzer pool creation.

    Used by both the FastMCP lifespan (``_mcp_lifespan``) and the lazy-init
    fallback in :func:`_get_analyzer_pool` so the two paths cannot drift in
    their UCI kwargs.
    """
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
        depth=20,  # default ceiling for per-call depth; tools clamp to caller-supplied value
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


async def _get_analyzer_pool(
    ctx: Context | None = None,
) -> AnalyzerPool | TCPAnalyzerPool:
    """Fetch the live analyzer pool from the FastMCP lifespan context.

    Falls back to the legacy lazy-init path when called outside a request
    (e.g. tests that don't go through the FastMCP runner), so existing test
    setups keep working without rewriting every fixture.
    """
    if ctx is not None:
        ls = ctx.request_context.lifespan_context
        pool = ls.get("pool")
        if pool is not None:
            return pool
    state = _state()
    async with state._pool_lock:
        if state._analyzer_pool is None:
            mcp_cfg = get_mcp_settings()
            import os

            cpu = os.cpu_count() or 8
            pool_size = mcp_cfg.pool_size if mcp_cfg.pool_size is not None else min(cpu, 4)
            state._analyzer_pool = await _create_analyzer_pool(mcp_cfg, pool_size=pool_size)
        return state._analyzer_pool


async def _pool_stats_logger(pool: AnalyzerPool | TCPAnalyzerPool, interval_s: float) -> None:
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
async def _mcp_lifespan(server: MCPServer) -> AsyncGenerator[dict[str, Any]]:
    """Initialize the Stockfish pool at startup, tear it down at exit.

    Replaces the lazy-init path with eager startup so the first user request
    doesn't pay the TCP handshake / UCI isready round-trips. The pool is
    shared with every tool via the ``lifespan_context[\"pool\"]`` indirection.

    Side jobs at startup:
      * Apply UCI options: ShowWDL (when enabled) + SyzygyPath (when set).
      * Warm-search: one depth=2 eval per worker so UCI isready completes
        and the engine is primed (saves ~120ms on the first real request).
      * Periodic structured pool-stats logging every
        ``CHESS_MCP_POOL_STATS_INTERVAL_S`` seconds (queue depth, alive count,
        cache hit rate). Set to 0 to disable.
    """
    cfg = get_mcp_settings()
    import os

    cpu = os.cpu_count() or 8
    pool_size = cfg.pool_size if cfg.pool_size is not None else min(cpu, 4)
    pool: AnalyzerPool | TCPAnalyzerPool = await _create_analyzer_pool(cfg, pool_size=pool_size)

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
            _pool_stats_logger(pool, stats_interval), name="pool-stats-logger"
        )

    try:
        yield {"pool": pool, "settings": cfg, "pool_size": pool_size}
    finally:
        if stats_task is not None:
            stats_task.cancel()
            try:
                await stats_task
            except (asyncio.CancelledError, Exception):
                pass
        log.info("Shutting down analyzer pool (%d engines)", pool_size)
        await pool.close()


async def _ponder_warm_cache(
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
        ckey = eval_cache_key(
            board,
            depth,
            engine_version=getattr(pool, "engine_version", None),
            history_completeness=history_complete,
        )
        if (await cache.get_eval(ckey)) is not None:
            return
        await _evaluate_game_position_cached(
            board,
            depth,
            pool,
            requested_depth=depth,
            history_complete=history_complete,
        )
    except Exception as exc:
        log.debug("ponder pre-eval failed: %s", exc)


_background_tasks: set[asyncio.Task[Any]] = set()


def _maybe_ponder_warm(
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
            _ponder_warm_cache(pool, next_board, depth, history_complete),
            name="ponder-warm",
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except Exception:
        pass


_POOL_SUPPORTS_ROOT_MOVES: dict[type, bool] = {}


def _pool_supports_root_moves(pool: object) -> bool:
    """Memoized runtime check for the ``root_moves`` keyword on pool.evaluate."""
    cls = type(pool)
    cached = _POOL_SUPPORTS_ROOT_MOVES.get(cls)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(pool.evaluate)  # type: ignore[attr-defined]
        supports = "root_moves" in sig.parameters
    except (TypeError, ValueError):
        supports = False
    _POOL_SUPPORTS_ROOT_MOVES[cls] = supports
    return supports


async def _eval_via_analyzer_or_pool(
    analyzer: object | None,
    pool: AnalyzerPool | TCPAnalyzerPool,
    b: chess.Board,
    *,
    depth: int,
    reuse_tt: bool,
    root_moves: list[chess.Move] | None = None,
) -> Any:
    """Run a single eval call.

    When ``analyzer`` is given, use it directly (skips pool acquire round-trip
    and lets the caller control ``reuse_tt`` for TT accumulation across
    calls). When ``analyzer`` is None, route through ``pool.evaluate`` which
    acquires a fresh worker — ``reuse_tt`` is ignored on this path.
    """
    if analyzer is not None:
        return await analyzer.evaluate(  # type: ignore[attr-defined]
            b, depth=depth, reuse_tt=reuse_tt, root_moves=root_moves
        )
    if _pool_supports_root_moves(pool):
        return await pool.evaluate(b, depth=depth, root_moves=root_moves)  # type: ignore[arg-type]
    return await pool.evaluate(b, depth=depth)  # type: ignore[arg-type]


async def _evaluate_game_position_cached(
    b: chess.Board,
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
    requested_depth: int | None = None,
    history_complete: str | bool = "incomplete",
    reuse_tt: bool = False,
    analyzer: object | None = None,
) -> tuple[MCPEval, bool]:
    """Evaluate a single board state with rule status, terminal short-circuits, and multi-tier cache."""
    req_d = requested_depth if requested_depth is not None else depth
    history_state = (
        ("complete" if history_complete else "incomplete")
        if isinstance(history_complete, bool)
        else history_complete
    )
    canonical_fen_str = b.fen()
    url, img = lichess_urls(canonical_fen_str)

    rule_status = evaluate_rule_status(b, history_complete=history_state)
    if rule_status.terminal is not None:
        return (
            build_terminal_mcpeval(
                rule_status=rule_status,
                board=b,
                requested_depth=req_d,
                pool=pool,
            ),
            True,
        )

    ckey = eval_cache_key(
        b,
        depth,
        engine_version=getattr(pool, "engine_version", None),
        history_completeness=history_state,
    )
    cached = await cache.get_eval(ckey)
    if cached is not None:
        return cached.model_copy(update={"requested_depth": req_d}), True

    async def _compute_pos() -> MCPEval:
        ev = await _eval_via_analyzer_or_pool(analyzer, pool, b, depth=depth, reuse_tt=reuse_tt)

        # U-02 (2026-09-01): at halfmove>=100, the root cp/mate can be
        # "polluted" by draw awareness — a winning zeroing capture like
        # Kxe2 in K+R vs R at halfmove=100 reports a tiny cp because the
        # draw is on the table, even though the post-state is K+R vs K
        # (a forced win). Re-evaluate the post-state of the engine's best
        # zeroing move and surface it to the action policy.
        zeroing_best_cp_arg: int | None = None
        zeroing_best_mate_arg: int | None = None
        if ev.best_move and b.halfmove_clock >= 100 and not b.is_game_over():
            try:
                bm_obj = chess.Move.from_uci(ev.best_move.lower())
                if bm_obj in b.legal_moves:
                    is_zeroing = b.is_capture(bm_obj) or (
                        b.piece_type_at(bm_obj.from_square) == chess.PAWN
                    )
                    if is_zeroing:
                        b_after = b.copy(stack=True)
                        b_after.push(bm_obj)
                        if not b_after.is_game_over(claim_draw=False):
                            try:
                                post_ev = await pool.evaluate(b_after, depth=depth)
                                mover_sign = 1 if b.turn == chess.WHITE else -1
                                if post_ev.mate is not None:
                                    mover_mate = mover_sign * post_ev.mate
                                    if mover_mate > 0:
                                        zeroing_best_mate_arg = mover_mate
                                elif post_ev.cp is not None:
                                    mover_cp = mover_sign * post_ev.cp
                                    if mover_cp > 0:
                                        zeroing_best_cp_arg = mover_cp
                            except Exception:
                                pass
            except Exception:
                pass

        # Rule-aware root best-move check (P0 audit fix):
        # If at halfmove 149, or halfmove >= 100 with winning score:
        # Check if the raw best move walks into 75-move draw or concedes a
        # claim while another move preserves the win.
        if ev.best_move and (b.halfmove_clock == 149 or b.halfmove_clock >= 100):
            try:
                bm_obj = chess.Move.from_uci(ev.best_move)
                b_after = b.copy(stack=True)
                if bm_obj in b.legal_moves:
                    b_after.push(bm_obj)
                    is_bm_75 = b_after.is_seventyfive_moves()
                    is_bm_conceded = b.halfmove_clock >= 100 and b_after.is_fifty_moves()

                    if is_bm_75 or is_bm_conceded:
                        override_move: chess.Move | None = None
                        override_is_mate = False
                        for cand in b.legal_moves:
                            b_sub = b.copy(stack=True)
                            b_sub.push(cand)
                            if b_sub.is_checkmate():
                                override_move = cand
                                override_is_mate = True
                                break
                            elif not b_sub.is_seventyfive_moves() and (
                                b.is_capture(cand)
                                or b.piece_type_at(cand.from_square) == chess.PAWN
                            ):
                                override_move = cand
                                break

                        if override_move is not None:
                            if override_is_mate:
                                ev.best_move = override_move.uci()
                                ev.mate = 1
                                ev.cp = None
                                ev.pv = [override_move.uci()]
                                ev.depth = depth
                            else:
                                try:
                                    override_eval = await pool.evaluate(
                                        b, depth=depth, root_moves=[override_move]
                                    )
                                    if (
                                        override_eval.best_move
                                        and override_eval.best_move.lower()
                                        == override_move.uci().lower()
                                    ):
                                        ev.best_move = override_eval.best_move
                                        ev.cp = override_eval.cp
                                        ev.mate = override_eval.mate
                                        ev.pv = override_eval.pv
                                        ev.depth = override_eval.depth
                                except Exception:
                                    pass
            except Exception:
                pass

        mcp_eval = MCPEval.from_eval(
            ev,
            canonical_fen_str,
            board=b,
            requested_depth=req_d,
            history_complete=history_state,
            zeroing_move_best_score=zeroing_best_cp_arg,
            zeroing_move_best_mate=zeroing_best_mate_arg,
        )
        identity = build_identity(pool)
        mcp_eval = mcp_eval.model_copy(
            update={
                "build_sha": identity["build_sha"],
                "engine_config": identity["engine_config"],
            }
        )
        # U-13 (2026-09-01): also stamp the build identity into the
        # nested engine_eval sub-dict so a caller reading just the
        # sub-dict (e.g. for telemetry) gets the same provenance.
        if mcp_eval.engine_eval is not None:
            mcp_eval = mcp_eval.model_copy(
                update={
                    "engine_eval": {
                        **mcp_eval.engine_eval,
                        "build_sha": identity["build_sha"],
                        "engine_config": identity["engine_config"],
                    }
                }
            )
        await cache.set_eval(ckey, mcp_eval)
        _maybe_ponder_warm(
            pool,
            b,
            mcp_eval.best_move,
            depth,
            ponder_enabled=getattr(pool, "_mcp_ponder_enabled", False),
            history_complete=history_state,
        )
        return mcp_eval

    res = cast(MCPEval, await single_flight.do(ckey, _compute_pos))
    return res.model_copy(update={"requested_depth": req_d}), False


async def _gather_evaluate_positions_bounded(
    positions: list[chess.Board],
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
    *,
    requested_depth: int,
    history_complete: str = "complete",
) -> list[tuple[MCPEval, bool]]:
    """Evaluate N positions partitioned across the pool with TT reuse per slice."""
    if not positions:
        return []
    sem = await _get_evaluate_semaphore()

    pool_target: int
    try:
        pool_target = pool._pool._target_size  # type: ignore[attr-defined]
    except AttributeError:
        pool_target = 4
    k = max(1, min(pool_target, len(positions)))

    chunk = math.ceil(len(positions) / k)
    slices: list[list[chess.Board]] = [
        list(positions[i : i + chunk]) for i in range(0, len(positions), chunk)
    ]
    slices = slices[:k]

    async def _run_slice(slice_positions: list[chess.Board]) -> list[tuple[MCPEval, bool]]:
        async with sem:
            if hasattr(pool, "_pool"):

                async def _on_worker(analyzer: object) -> list[tuple[MCPEval, bool]]:
                    out: list[tuple[MCPEval, bool]] = []
                    for j, b in enumerate(slice_positions):
                        r, hit = await _evaluate_game_position_cached(
                            b,
                            depth,
                            pool,
                            requested_depth=requested_depth,
                            reuse_tt=(j > 0),
                            analyzer=analyzer,
                            history_complete=history_complete,
                        )
                        out.append((r, hit))
                    return out

                return await pool._pool.run(_on_worker)  # type: ignore[attr-defined]

            out: list[tuple[MCPEval, bool]] = []
            for b in slice_positions:
                r, hit = await _evaluate_game_position_cached(
                    b,
                    depth,
                    pool,
                    requested_depth=requested_depth,
                    reuse_tt=False,
                    history_complete=history_complete,
                )
                out.append((r, hit))
            return out

    slice_results = await asyncio.gather(*[_run_slice(s) for s in slices if s])
    out: list[tuple[MCPEval, bool]] = [None] * len(positions)  # type: ignore[list-item]
    cursor = 0
    for slice_result in slice_results:
        for j, item in enumerate(slice_result):
            out[cursor + j] = item
        cursor += len(slice_result)
    return out
