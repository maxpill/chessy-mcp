from __future__ import annotations

import asyncio
import hmac
import io
import ipaddress
import json
import logging
import math
import os
import re
import subprocess
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, cast

import chess
import chess.pgn
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from core.engines.analyzer import pv_to_san
from core.engines.openings import lookup_opening
from core.engines.pool import AnalyzerPool
from core.engines.types import Eval, MoveClass
from mcp_server.actions import build_played_action
from mcp_server.cache import (
    CACHE_VERSION as CACHE_VERSION,
)
from mcp_server.cache import (
    MultiTierCache,
    SingleFlight,
    classify_cache_key,
    eval_cache_key,
    top_moves_cache_key,
)
from mcp_server.config import MCPSettings
from mcp_server.metrics import metrics
from mcp_server.models import (
    GameAnalysisResult,
    MCPEval,
    MCPMoveAnalysis,
    PlyAnalysisItem,
    TopMovesResult,
)
from mcp_server.move_grading import score_played_move  # noqa: E402,F401
from mcp_server.rules import (
    choose_recommended_action,
    evaluate_rule_status,
    format_fen_status_errors,
    is_locked_dead_position,
    is_terminal_position,
    validate_mating_possibility,
)
from mcp_server.tcp_analyzer import TCPAnalyzerPool
from mcp_server.urls import lichess_urls


log = logging.getLogger("chessy_mcp.server")


from mcp_server.tools._common import (
    VERBOSITY_COMPACT,
    VERBOSITY_FULL,
    _compact_mcpeval,
    _format_exception,
    _resolve_verbosity,
    _tool_error,
    _validate_requested_depth,
    error_code_for,
    normalize_termination,
)


# Draw-claim projection — implementation lives in mcp_server.claims.draw_projection.
# Bound as a module attribute so existing call sites keep working unchanged.
from mcp_server.claims.draw_projection import _force_draw_outcome  # noqa: E402,F401


# PGN / FEN / SAN parsing — implementation lives in mcp_server.parsers.
# All parser helpers used by the tools are bound here so existing call
# sites (``_build_board(...)``, ``_extract_game(...)``, ...) keep working
# unchanged. Underscored + unprefixed names are both exposed.
from mcp_server.parsers import (  # noqa: E402,F401
    SUPPORTED_VARIANTS,
    TAG_PAIR_REGEX,
    _build_board,
    _build_board_with_metadata,
    _check_multiple_games,
    _clean_conversational_text,
    _extract_canonical_pgn_text,
    _extract_game,
    _extract_game_inner,
    _find_movetext_result,
    _has_completed_game_before,
    _history_provenance_for_input,
    _infer_result_from_termination,
    _is_canonical_tag_line,
    _is_prose_line,
    _mask_comments_and_escapes,
    _normalize_multiline_tags,
    _normalize_movetext_figurines,
    _normalize_unicode_pgn_results,
    _parse_move_on_board,
    _parse_move_on_board_with_warning,
    _parse_pgn_game_candidate,
    _sanitize_brackets_in_variations_and_comments,
    _sanitize_malformed_pgn_header_lines,
    _stage_has_positive_number,
    _strict_top_level_movetext_tokens,
    _strip_pgn_escape_lines,
    _strip_promotion_eq,
    _truncate_movetext_at_result,
    _unescape_pgn_tag_value,
    _validate_castling_rights,
    _validate_fen_counters,
    _validate_movetext_tokens,
    _validate_pgn_date,
    _validate_strict_header_syntax,
    _validate_strict_mainline_surface,
    _validate_variant,
    _is_valid_pgn_time_control,
)


# Re-export facade for the engine layer. The implementation now lives in
# ``mcp_server.engine.pool_factory`` and ``mcp_server.engine.identity``;
# every public/private symbol is re-bound here so existing call sites
# (and ``monkeypatch.setattr(server_module, ...)`` in the test suite)
# keep working unchanged.
from mcp_server.engine import (  # noqa: E402,F401
    _create_analyzer_pool,
    _eval_via_analyzer_or_pool,
    _evaluate_game_position_cached,
    _gather_evaluate_positions_bounded,
    _get_analyzer_pool,
    _get_evaluate_semaphore,
    _maybe_ponder_warm,
    _mcp_lifespan,
    _pool_stats_logger,
    _pool_supports_root_moves,
    _ponder_warm_cache,
    cache as _cache,
    close_analyzer_pool,
    single_flight as _single_flight,
    build_identity as _build_identity,
    engine_config as _engine_config,
    package_version as _package_version,
    git_sha as _build_sha,
    stockfish_path as _stockfish_path,
)

# Pool / semaphore globals live HERE (not in pool_factory) because the test
# suite mutates ``server_module._analyzer_pool`` directly to install mock
# analyzers. ``pool_factory`` reads and writes these exact bindings at call
# time so the mock reaches the live lookup path.
_analyzer_pool: AnalyzerPool | TCPAnalyzerPool | None = None
_pool_lock = asyncio.Lock()
_evaluate_semaphore: asyncio.Semaphore | None = None
_evaluate_semaphore_lock = asyncio.Lock()


# MCP server instance. Defined BEFORE the tool imports so that
# ``from mcp_server.server import mcp`` (used by tool modules) resolves
# without a circular-import error.
mcp = MCPServer(
    "chess-analysis",
    description="Streamable Stockfish chess analysis and move grading MCP server",
    lifespan=_mcp_lifespan,
)


# MCP tools — implementations live in mcp_server.tools. Importing
# each one here triggers the ``@mcp.tool(...)`` decorator so FastMCP
# registers them on the server instance. Existing call sites that use
# ``server_module.evaluate_position(...)`` etc. keep working unchanged.
from mcp_server.tools.evaluate_position import evaluate_position  # noqa: E402,F401
from mcp_server.tools.top_moves import top_moves  # noqa: E402,F401
from mcp_server.tools.classify_move import classify_move  # noqa: E402,F401
from mcp_server.tools.analyze_game import analyze_game  # noqa: E402,F401
from mcp_server.tools.game_metrics import _compute_game_metrics  # noqa: E402,F401


@mcp.custom_route("/health", methods=["GET"])
async def _health(request: Any) -> Any:
    """Liveness/readiness probe — no auth required, no MCP machinery touched.

    Returns 200 with a minimal payload so compose / orchestrator healthchecks
    can verify the service is up without engaging the JSON-RPC stack.
    """
    from starlette.responses import JSONResponse

    return JSONResponse(
        {
            "status": "ok",
            "service": "chessy-mcp",
            "version": _package_version(),
        }
    )


# P2/P3 (2026-09-02 ultra audit): the bare regex matches shapes like
# "0+0", "0+1", "40/0", "0/600" — all of which are syntactically PGN
# TimeControl values but semantically nonsense (a game with 0 seconds
# is unplayable; a 40-move period with 0 seconds is impossible; a
# 0-move period with a positive time is meaningless). The relaxed
# grammar was useful when the old validator was the only thing between
# caller input and the metadata block, but the audit showed callers
# relying on the validator as a "PGN TimeControl sanity check". Tighten
# the check so every stage must contain at least one non-zero digit;
# `_stage_has_positive_number` walks the string. Sentinel values ("?",
# "-") remain accepted unchanged.


# Unicode em-dash / en-dash / hyphen normalization used by both the
# movetext extractor and the canonical-game extractor (audit L-04).


MAX_HALFMOVE_CLOCK = 10000
MAX_FULLMOVE_NUMBER = 10000


async def _eval_via_analyzer_or_pool(
    analyzer: object | None,
    pool: AnalyzerPool | TCPAnalyzerPool,
    b: chess.Board,
    *,
    depth: int,
    reuse_tt: bool,
    root_moves: list[chess.Move] | None = None,
) -> Eval:
    """Run a single eval call.

    When `analyzer` is given, use it directly (skips pool acquire round-trip
    and lets the caller control `reuse_tt` for TT accumulation across calls).
    When `analyzer` is None, route through `pool.evaluate` which acquires a
    fresh worker — `reuse_tt` is ignored on this path because the next call
    may land on a different worker.
    """
    if analyzer is not None:
        return await analyzer.evaluate(  # type: ignore[attr-defined]
            b, depth=depth, reuse_tt=reuse_tt, root_moves=root_moves
        )
    # pool.evaluate signature varies; pass root_moves only if analyzer pool
    # supports it (production TCPAnalyzerPool does; test mocks don't).
    if _pool_supports_root_moves(pool):
        return await pool.evaluate(b, depth=depth, root_moves=root_moves)  # type: ignore[arg-type]
    return await pool.evaluate(b, depth=depth)  # type: ignore[arg-type]


_POOL_SUPPORTS_ROOT_MOVES: dict[type, bool] = {}


def _pool_supports_root_moves(pool: object) -> bool:
    """Memoized runtime check for the `root_moves` keyword on pool.evaluate.

    `inspect.signature` walks the function annotations every call; cache by
    the pool's class — same class always has the same evaluate signature, so
    per-class caching is correct (and the cache survives instance churn).
    """
    cls = type(pool)
    cached = _POOL_SUPPORTS_ROOT_MOVES.get(cls)
    if cached is not None:
        return cached
    import inspect

    try:
        sig = inspect.signature(pool.evaluate)  # type: ignore[attr-defined]
        supports = "root_moves" in sig.parameters
    except (TypeError, ValueError):
        supports = False
    _POOL_SUPPORTS_ROOT_MOVES[cls] = supports
    return supports


async def _evaluate_game_position_cached(
    b: chess.Board,
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
    requested_depth: int | None = None,
    history_complete: str | bool = "incomplete",
    reuse_tt: bool = False,
    analyzer: object | None = None,
) -> tuple[MCPEval, bool]:
    """Evaluate a single board state with rule status, terminal short-circuits, and multi-tier cache.

    Args:
        history_complete: True when the caller had access to the full move stack
            (PGN, evaluate_position with moves param). False for naked FEN — drives
            `history_completeness` and `repetition_status` on the returned MCPEval
            (audit H-01).
        reuse_tt: pass True when consecutive calls on the same engine share
            position-tree history. Saves a `ucinewgame` round-trip and lets
            Stockfish accumulate the TT across calls. Caller is responsible
            for the semantic correctness — only use when the previous call's
            FEN is the predecessor of the current.
        analyzer: optional pre-acquired analyzer instance. When set, skips the
            pool.acquire() round-trip and calls `analyzer.evaluate(...)`
            directly. The `_gather_evaluate_positions_chunked` helper holds
            one analyzer per slice for sequential calls within the slice.
    """
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
        # Terminal outcomes are reported from White's perspective (same convention as cp).
        if rule_status.terminal == "checkmate":
            term_outcome = "win" if rule_status.winner == "white" else "loss"
            term_cp: int | None = None
            term_mate: int | None = 0
        else:
            term_outcome = "draw"
            term_cp = 0
            term_mate = None
        from mcp_server.actions import build_best_action, build_legal_actions

        terminal_best_action = build_best_action(
            recommended_action="game_over",
            rule_status=rule_status,
            engine_eval=None,
            board=b,
            sign=1 if b.turn == chess.WHITE else -1,
        )
        terminal_legal_actions = build_legal_actions(
            rule_status=rule_status,
            engine_eval=None,
            board=b,
            legal_engine_moves=None,
        )
        return (
            MCPEval(
                status=rule_status.terminal,
                winner=rule_status.winner,
                cp=term_cp,
                mate=term_mate,
                best_move=None,
                pv=[],
                depth=0,
                requested_depth=req_d,
                searched_depth=0,
                can_claim_draw=False,
                claim_reasons=[],
                can_claim_now=False,
                claim_reasons_now=[],
                can_claim_with_intended_move=False,
                claim_moves=[],
                recommended_action="game_over",
                best_action="game_over",
                best_action_type="game_over",
                best_action_obj=terminal_best_action,
                legal_actions=terminal_legal_actions,
                decision_value={
                    "outcome": term_outcome,
                    "cp_equivalent": term_cp,
                    "best_action": "game_over",
                    "perspective": "white",
                },
                engine_eval={
                    "cp": term_cp,
                    "mate": term_mate,
                    "best_move": None,
                    "pv": [],
                    "depth": 0,
                },
                history_dependent_status=rule_status.history_dependent_status,
                lichess_url_reproduces_history=rule_status.fen_sufficient_for_status,
                requires_move_stack=rule_status.requires_move_stack,
                fen_sufficient_for_status=rule_status.fen_sufficient_for_status,
                history_completeness=rule_status.history_completeness,
                repetition_status=rule_status.repetition_status,
                lichess_url=url,
                lichess_image=img,
                **_build_identity(pool),
            ),
            True,
        )

    ckey = eval_cache_key(
        b,
        depth,
        engine_version=getattr(pool, "engine_version", None),
        history_completeness=history_state,
    )
    cached = await _cache.get_eval(ckey)
    if cached is not None:
        return cached.model_copy(update={"requested_depth": req_d}), True

    async def _compute_pos() -> MCPEval:
        ev = await _eval_via_analyzer_or_pool(analyzer, pool, b, depth=depth, reuse_tt=reuse_tt)

        # U-02 (2026-09-01): at halfmove>=100, the root cp/mate can be
        # "polluted" by draw awareness — a winning zeroing capture like
        # Kxe2 in K+R vs R at halfmove=100 reports a tiny cp because the
        # draw is on the table, even though the post-state is K+R vs K
        # (a forced win). Detect this by re-evaluating the post-state of
        # the engine's best zeroing move; if it's a high-confidence win,
        # surface it to the action policy so it recommends play_move
        # instead of claim_draw. Used by score_played_move and
        # _pick_root_recommended_action (which already do this for
        # top_moves' multipv output).
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
                                # Compute the post-state mover-POV score.
                                # `post_ev.cp` / `post_ev.mate` are
                                # White-POV (post-analyzer convention). The
                                # mover's perspective is `mover_sign` (sign
                                # = +1 if White is currently on turn, -1 if
                                # Black); same convention as the
                                # top_moves zeroing loop at L2969-2977.
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
        #
        # CRITICAL: if we override `ev.best_move`, we MUST re-run Stockfish
        # with the new move as a root_moves constraint so cp/mate reflect the
        # actual move, NOT the original. The previous implementation kept the
        # original cp/mate, producing the contradictory `best_move=B, cp=eval(A)`
        # invariant violation flagged in the audit.
        if ev.best_move and (b.halfmove_clock == 149 or b.halfmove_clock >= 100):
            try:
                bm_obj = chess.Move.from_uci(ev.best_move)
                b_after = b.copy(stack=True)
                if bm_obj in b.legal_moves:
                    b_after.push(bm_obj)
                    is_bm_75 = b_after.is_seventyfive_moves()
                    is_bm_conceded = b.halfmove_clock >= 100 and b_after.is_fifty_moves()

                    if is_bm_75 or is_bm_conceded:
                        # Find a win-preserving reset move (capture / pawn move / mate).
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
                                # Stockfish's mate distance from the post-move board is
                                # either 0 (already checkmated — terminal) or 1 (mate in 1
                                # from the mover's POV). Use the post-move eval semantics.
                                ev.best_move = override_move.uci()
                                ev.mate = 1
                                ev.cp = None
                                ev.pv = [override_move.uci()]
                                ev.depth = depth
                            else:
                                # Re-run Stockfish with root_moves constrained to the
                                # override so cp/mate reflect THIS move's eval, not the
                                # original best move's. Without this we get the audit's
                                # contradictory `best_move=B, cp=eval(A)` invariant.
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
                                    else:
                                        # Verification search did not confirm our override;
                                        # trust the original eval but mark it as unverified
                                        # by leaving Stockfish's choice intact.
                                        pass
                                except Exception:
                                    # Re-eval failed; trust original rather than ship
                                    # unverified cp/mate for a different best_move.
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
        # Stamp build identity so every cached eval records which build produced it.
        identity = _build_identity(pool)
        mcp_eval = mcp_eval.model_copy(
            update={
                "build_sha": identity["build_sha"],
                "engine_config": identity["engine_config"],
            }
        )
        # U-13 (2026-09-01): also stamp the build identity into the
        # nested `engine_eval` sub-dict so a caller reading just the
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
        await _cache.set_eval(ckey, mcp_eval)
        # Ponder: warm the L1 cache for the position AFTER the engine's best
        # move so the next user request on that FEN hits L1 (env-disabled
        # by default — costs CPU on small hosts).
        _maybe_ponder_warm(
            pool,
            b,
            mcp_eval.best_move,
            depth,
            ponder_enabled=getattr(pool, "_mcp_ponder_enabled", False),
            history_complete=history_state,
        )
        return mcp_eval

    res = cast(MCPEval, await _single_flight.do(ckey, _compute_pos))
    return res.model_copy(update={"requested_depth": req_d}), False


# Middleware re-exports — implementation lives in mcp_server.middleware.
# Each name is bound here as a module attribute so existing call sites
# (and ``monkeypatch.setattr(server_module, \"ASGIRequestLoggerMiddleware\", ...)``)
# keep working unchanged.
from mcp_server.middleware.rate_limit import TokenBucketRateLimiter  # noqa: E402,F401
from mcp_server.middleware.request_logger import (  # noqa: E402,F401
    ASGIRequestLoggerMiddleware,
    _build_app,
    _effective_client_ip,
    _estimate_mcp_request_cost,
    _is_trusted_proxy_peer,
    main,
)


if __name__ == "__main__":
    main()
