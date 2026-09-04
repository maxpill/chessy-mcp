"""Cached single-position evaluator.

Extracted from :mod:`mcp_server.engine.pool_factory`. Owns
:func:`evaluate_game_position_cached` — the per-position entry point
that ties rule status, terminal short-circuits, multi-tier cache,
SingleFlight coalescing, the U-02 zeroing-post-state re-eval, the P0
75-move / 50-move rule-aware override, identity stamping, and ponder
warming into one place.

The :func:`evaluate_zeroing_post_state` helper in
:mod:`mcp_server.engine.zeroing_post_state` (lifted from
:mod:`mcp_server.engine.eval_pipeline`) is now an explicit
co-conspirator of the U-02 path.

Module-level state lives here:

  * :data:`cache` — :class:`MultiTierCache`, shared with every tool.
  * :data:`single_flight` — :class:`SingleFlight`, coalesces in-flight
    cache misses.
  * :data:`evaluate_semaphore` — async bound on concurrent evaluate calls.
    Created lazily on first call so it always belongs to the live event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import chess

from core.engines.pool import AnalyzerPool

from mcp_server.cache import MultiTierCache, SingleFlight, eval_cache_key
from mcp_server.engine.identity import build_identity
from mcp_server.engine.pool_lifecycle import (
    eval_via_analyzer_or_pool,
)
from mcp_server.engine.ponder import maybe_ponder_warm
from mcp_server.engine.eval_pipeline import build_terminal_mcpeval
from mcp_server.engine.zeroing_post_state import evaluate_zeroing_post_state
from mcp_server.models import MCPEval
from mcp_server.rules import evaluate_rule_status
from mcp_server.tcp_analyzer import TCPAnalyzerPool


log = logging.getLogger("chessy_mcp.engine.cached_evaluator")


# Module-level cache + SingleFlight — shared across every tool entry point.
cache: MultiTierCache = MultiTierCache(l1_size=50_000)
_cache = cache
single_flight: SingleFlight[Any] = SingleFlight()
_single_flight = single_flight


async def get_evaluate_semaphore() -> asyncio.Semaphore:
    """Lazily create the evaluate-semaphore bound to the live event loop.

    Reads/writes ``server._evaluate_semaphore`` directly so the legacy
    ``monkeypatch.setattr("mcp_server.server._evaluate_semaphore", None)``
    test pattern keeps working.
    """
    from mcp_server import server

    state = server
    async with state._evaluate_semaphore_lock:
        if state._evaluate_semaphore is None:
            from mcp_server.config import get_mcp_settings

            cfg = get_mcp_settings()
            state._evaluate_semaphore = asyncio.Semaphore(max(1, cfg.max_concurrent_evaluates))
        return state._evaluate_semaphore


async def evaluate_game_position_cached(
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
        ev = await eval_via_analyzer_or_pool(analyzer, pool, b, depth=depth, reuse_tt=reuse_tt)
        zeroing_best = await _maybe_zeroing_best_override(b, ev, depth, pool)
        _apply_rule_aware_best_move_override(b, ev, depth, pool)
        mcp_eval = MCPEval.from_eval(
            ev,
            canonical_fen_str,
            board=b,
            requested_depth=req_d,
            history_complete=history_state,
            zeroing_move_best_score=zeroing_best.cp,
            zeroing_move_best_mate=zeroing_best.mate,
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
        maybe_ponder_warm(
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


async def _maybe_zeroing_best_override(
    b: chess.Board,
    ev: Any,
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
) -> Any:
    """U-02 audit fix (2026-09-01): at halfmove>=100, the root cp/mate can
    be "polluted" by draw awareness — re-eval the post-state of the
    engine's best zeroing move to surface the winning score."""
    if not (ev.best_move and b.halfmove_clock >= 100 and not b.is_game_over()):
        return _ZeroingNoop()
    return await evaluate_zeroing_post_state(b, ev.best_move, depth, pool)


class _ZeroingNoop:
    cp: int | None = None
    mate: int | None = None


def _apply_rule_aware_best_move_override(
    b: chess.Board,
    ev: Any,
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
) -> None:
    """P0 audit fix: at halfmove 149, or halfmove >= 100 with a winning
    score, override the engine's best move if it walks into 75-move draw
    or concedes a claim when another move preserves the win. Mutates
    ``ev`` in place (matches the original inline behavior)."""
    if not (ev.best_move and (b.halfmove_clock == 149 or b.halfmove_clock >= 100)):
        return
    try:
        bm_obj = chess.Move.from_uci(ev.best_move)
        b_after = b.copy(stack=True)
        if bm_obj not in b.legal_moves:
            return
        b_after.push(bm_obj)
        is_bm_75 = b_after.is_seventyfive_moves()
        is_bm_conceded = b.halfmove_clock >= 100 and b_after.is_fifty_moves()

        if not (is_bm_75 or is_bm_conceded):
            return

        override_move, override_is_mate = _pick_override_move(b)
        if override_move is None:
            return
        if override_is_mate:
            ev.best_move = override_move.uci()
            ev.mate = 1
            ev.cp = None
            ev.pv = [override_move.uci()]
            ev.depth = depth
        else:
            _apply_override_eval(ev, override_move, b, depth, pool)
    except Exception:
        log.debug("rule-aware best-move override failed (continuing)")


def _pick_override_move(b: chess.Board) -> tuple[chess.Move | None, bool]:
    """Find a move that doesn't walk into 75-move / 50-move draw."""
    override_move: chess.Move | None = None
    override_is_mate = False
    for cand in b.legal_moves:
        b_sub = b.copy(stack=True)
        b_sub.push(cand)
        if b_sub.is_checkmate():
            override_move = cand
            override_is_mate = True
            break
        if not b_sub.is_seventyfive_moves() and (
            b.is_capture(cand) or b.piece_type_at(cand.from_square) == chess.PAWN
        ):
            override_move = cand
            break
    return override_move, override_is_mate


def _apply_override_eval(
    ev: Any,
    override_move: chess.Move,
    b: chess.Board,
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
) -> None:
    """Synchronous — schedules a background override eval and updates ``ev``."""
    try:
        task = asyncio.create_task(_eval_override(pool, b, depth, override_move))
        task.add_done_callback(
            lambda t: _maybe_apply_override_result(t.result(), ev, override_move)
        )
    except Exception:
        log.debug("override eval failed (continuing)")


async def _eval_override(
    pool: AnalyzerPool | TCPAnalyzerPool,
    b: chess.Board,
    depth: int,
    override_move: chess.Move,
) -> Any:
    return await pool.evaluate(b, depth=depth, root_moves=[override_move])


def _maybe_apply_override_result(
    override_eval: Any,
    ev: Any,
    override_move: chess.Move,
) -> None:
    if override_eval.best_move and override_eval.best_move.lower() == override_move.uci().lower():
        ev.best_move = override_eval.best_move
        ev.cp = override_eval.cp
        ev.mate = override_eval.mate
        ev.pv = override_eval.pv
        ev.depth = override_eval.depth


# Back-compat shims.
_evaluate_game_position_cached = evaluate_game_position_cached
_get_evaluate_semaphore = get_evaluate_semaphore
