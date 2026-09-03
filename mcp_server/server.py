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
    score_played_move,
)
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


mcp = MCPServer(
    "chess-analysis",
    description="Streamable Stockfish chess analysis and move grading MCP server",
    lifespan=_mcp_lifespan,
)


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


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def evaluate_position(
    fen: str,
    moves: list[str] | None = None,
    depth: int = 14,
    strict: bool = False,
    verbosity: str | None = None,
    ctx: Context | None = None,
) -> MCPEval:
    """Evaluate a chess position with Stockfish.

    Args:
        fen: FEN or PGN string for the position (or position before `moves` are replayed).
        moves: Optional UCI or SAN moves to replay onto the position first.
        depth: Stockfish search depth (default 14, clamped 1-30).
        strict: When True, reject non-canonical SAN syntax or move numbers (default False).
        verbosity: "full" (default, every field) or "compact" (strips Lichess URLs,
            images, decision_value/engine_eval duplication). Use compact when the
            caller is an LLM and you want to minimize context spend (audit M-05).

    Returns:
         Eval with cp (from White's perspective), mate (from White's perspective),
         best_move (UCI), pv (principal variation), and Lichess board URLs.
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="evaluate_position")
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    try:
        verbosity_mode = _resolve_verbosity(verbosity)
        board, input_fen, canonical_fen, fen_was_canonicalized = _build_board_with_metadata(
            fen, moves or [], strict=strict
        )
        pool = await _get_analyzer_pool(ctx)
        # History completeness is derived from whether the caller had the move
        # stack. Naked FEN (no moves) cannot detect threefold repetition;
        # we MUST report `repetition_status="unknown"` for the audit H-01 fix.
        # When moves were supplied, the move stack is complete and we can
        # answer threefold claims definitively.
        history_complete = _history_provenance_for_input(fen, moves)
        res, is_hit = await _evaluate_game_position_cached(
            board,
            depth,
            pool,
            requested_depth=raw_requested_depth,
            history_complete=history_complete,
        )
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, cache_hit=is_hit)
        # L-06: surface input vs canonical FEN. Canonicalization describes
        # parser normalization of the supplied FEN only; replayed suffix moves
        # are reflected in canonical_fen but do not make the input noncanonical.
        result = res.model_copy(
            update={
                "requested_depth": raw_requested_depth,
                "input_fen": input_fen,
                "canonical_fen": canonical_fen,
                "fen_was_canonicalized": fen_was_canonicalized,
            }
        )
        if verbosity_mode == VERBOSITY_COMPACT:
            result = _compact_mcpeval(result)
        return result
    except ToolError:
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, is_error=True)
        msg = str(exc)
        code = error_code_for(msg)
        raise _tool_error(code=code, message=msg, tool="evaluate_position", input=fen) from exc
    except Exception as exc:
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="evaluate_position") from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def top_moves(
    fen: str,
    moves: list[str] | None = None,
    n: int = 3,
    depth: int = 14,
    strict: bool = False,
    verbosity: str | None = None,
    ctx: Context | None = None,
) -> TopMovesResult:
    """Get the top N candidate moves for a position, ranked best first.

    Args:
        fen: FEN or PGN string for the position.
        moves: Optional UCI or SAN moves to replay onto the position first.
        n: Number of candidates to return (default 3, clamped 1-20).
        depth: Stockfish search depth (default 14, clamped 1-30).
        strict: When True, reject non-canonical SAN syntax or move numbers (default False).

    Returns:
        TopMovesResult object with `status`, `winner`, `recommended_action`,
        `best_action_obj` (typed discriminated union per audit 10.2),
        `legal_actions` (typed list of legal actions), and a `result` array
        of candidate MCPEval objects ranked best first.

        IMPORTANT (audit C-02 / H-03):
          Each candidate in `result` represents a `play_move` action. Its
          `best_move`, `pv`, and engine `cp`/`mate` retain the root MultiPV
          action value and notation frame, so PV[0] is the candidate move and
          a mating candidate may retain Stockfish's root mate distance (e.g. 1).
          The candidate `canonical_fen`, terminal status, winner, rule fields,
          and `post_position` describe the board AFTER that candidate is played.
          Automatic terminal draws normalize candidate `cp` to 0. Draw-claim
          actions are reported separately via outer `best_action_obj` and
          `legal_actions`; they are not mixed into the MultiPV candidate list.

        For terminal positions (checkmate, stalemate, insufficient material,
        repetition, 75-move rule), returns TopMovesResult with status and
        empty `result: []`.
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="top_moves")
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    raw_requested_n = n
    clamped_n = max(1, min(n, 20))
    n = clamped_n
    try:
        verbosity_mode = _resolve_verbosity(verbosity)
        board, _input_fen, canonical_fen, fen_was_canonicalized = _build_board_with_metadata(
            fen, moves or [], strict=strict
        )
        # evaluate_position with explicit moves has full history; naked FEN doesn't.
        history_complete = _history_provenance_for_input(fen, moves)
        rule_status = evaluate_rule_status(board, history_complete=history_complete)
        pool = await _get_analyzer_pool(ctx)
        engine_name_str = getattr(pool, "engine_version", getattr(pool, "name", "Stockfish"))
        legal_move_count = board.legal_moves.count()

        if rule_status.terminal is not None:
            await metrics.record("top_moves", (time.time() - t0) * 1000, cache_hit=True)
            # Build a typed game_over best_action
            from mcp_server.actions import build_best_action, build_legal_actions

            best_action_obj = build_best_action(
                recommended_action=rule_status.recommended_action,
                rule_status=rule_status,
                engine_eval=None,
                board=board,
                sign=1 if board.turn == chess.WHITE else -1,
            )
            legal_actions = build_legal_actions(
                rule_status=rule_status,
                engine_eval=None,
                board=board,
                legal_engine_moves=None,
            )
            return TopMovesResult(
                status=rule_status.terminal,
                winner=rule_status.winner,
                recommended_action="game_over",
                can_claim_draw=False,
                claim_reasons=[],
                can_claim_now=False,
                claim_reasons_now=[],
                can_claim_with_intended_move=False,
                claim_moves=[],
                best_action_obj=best_action_obj,
                legal_actions=legal_actions,
                history_completeness=rule_status.history_completeness,
                repetition_status=rule_status.repetition_status,
                requested_depth=raw_requested_depth,
                searched_depth=0,
                requested_n=raw_requested_n,
                clamped_n=clamped_n,
                returned_n=0,
                legal_move_count=legal_move_count,
                canonical_fen=canonical_fen,
                fen_was_canonicalized=fen_was_canonicalized,
                engine="Stockfish",
                engine_version=engine_name_str,
                **_build_identity(pool),
                result=[],
            )

        cache_key = top_moves_cache_key(
            board,
            depth,
            n=n,
            engine_version=getattr(pool, "engine_version", None),
            history_completeness=history_complete,
        )

        # sign = mover's perspective sign (White=+1, Black=-1). Used below for
        # both the cache-hit and the freshly-computed paths to decide whether a
        # candidate is winning FOR the side-to-move (cp is White-POV).
        sign = 1 if board.turn == chess.WHITE else -1

        from mcp_server.actions import build_best_action, build_legal_actions

        def _pick_root_recommended_action(items: list[MCPEval]) -> str:
            if not items:
                return rule_status.recommended_action
            best = items[0]
            # U-01 (2026-09-01): mate must take precedence over cp. When
            # Stockfish finds a forced mate it sometimes still emits a
            # saturated cp=±20000; per chess convention, mate wins.
            # Use post_state_* when available (audit B-04 / B-05).
            eff_mate = best.post_state_mate if best.post_state_mate is not None else best.mate
            eff_cp = best.post_state_cp if best.post_state_cp is not None else best.cp
            if eff_mate is not None:
                mover_score: int | None = sign * eff_mate * 1000
            elif eff_cp is not None:
                mover_score = sign * eff_cp
            else:
                mover_score = None
            mate_for_mover = sign * eff_mate if eff_mate is not None else None
            # AUDIT B-04: also surface the best post-state value across all
            # zeroing candidates (capture or pawn move) so the policy can
            # prefer play_move over claim_draw when a zeroing move wins.
            # The post-state cp/mate is attached to each item by the fresh
            # path (audit B-05); the cache-hit path inherits the same data
            # because items are persisted with their post_state_* fields.
            zeroing_best_cp: int | None = None
            zeroing_best_mate: int | None = None
            for item in items:
                if not item.best_move:
                    continue
                try:
                    bm = chess.Move.from_uci(item.best_move)
                except Exception:
                    continue
                if not (board.is_capture(bm) or board.piece_type_at(bm.from_square) == chess.PAWN):
                    continue
                # Prefer the re-evaluated post-state value when present
                # (draw-pollution guard, audit B-04); fall back to the
                # multipv value otherwise.
                eff_cp = item.post_state_cp if item.post_state_cp is not None else item.cp
                eff_mate = item.post_state_mate if item.post_state_mate is not None else item.mate
                if eff_mate is not None:
                    mover_mate = sign * eff_mate
                    if mover_mate > 0 and (
                        zeroing_best_mate is None or mover_mate > zeroing_best_mate
                    ):
                        zeroing_best_mate = mover_mate
                elif eff_cp is not None:
                    mover_cp = sign * eff_cp
                    if zeroing_best_cp is None or mover_cp > zeroing_best_cp:
                        zeroing_best_cp = mover_cp
            return choose_recommended_action(
                board,
                can_claim_now=rule_status.can_claim_now,
                can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
                mover_score=mover_score,
                mate_for_mover=mate_for_mover,
                zeroing_move_best_score=zeroing_best_cp,
                zeroing_move_best_mate=zeroing_best_mate,
            )

        cached = await _cache.get_top_moves(cache_key)
        if cached is not None and len(cached) >= n:
            await metrics.record("top_moves", (time.time() - t0) * 1000, cache_hit=True)
            items = [
                c.model_copy(update={"requested_depth": raw_requested_depth}) for c in cached[:n]
            ]
            # Apply compact verbosity to cached candidates too (audit M-05)
            if verbosity_mode == VERBOSITY_COMPACT:
                items = [_compact_mcpeval(c) for c in items]
            root_rec_action = _pick_root_recommended_action(items)
            best_action_obj = build_best_action(
                recommended_action=root_rec_action,
                rule_status=rule_status,
                engine_eval=items[0] if items else None,
                board=board,
                sign=sign,
            )
            legal_actions = build_legal_actions(
                rule_status=rule_status,
                engine_eval=items[0] if items else None,
                board=board,
                legal_engine_moves=list(items),
            )
            return TopMovesResult(
                status="active",
                winner=None,
                recommended_action=root_rec_action,
                can_claim_draw=rule_status.can_claim_draw,
                claim_reasons=rule_status.claim_reasons,
                claim_move=rule_status.claim_move,
                can_claim_now=rule_status.can_claim_now,
                claim_reasons_now=rule_status.claim_reasons_now,
                can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
                claim_moves=rule_status.claim_moves,
                best_action_obj=best_action_obj,
                legal_actions=legal_actions,
                history_completeness=rule_status.history_completeness,
                repetition_status=rule_status.repetition_status,
                requested_depth=raw_requested_depth,
                searched_depth=depth,
                requested_n=raw_requested_n,
                clamped_n=clamped_n,
                returned_n=len(items),
                legal_move_count=legal_move_count,
                canonical_fen=canonical_fen,
                fen_was_canonicalized=fen_was_canonicalized,
                engine="Stockfish",
                engine_version=engine_name_str,
                **_build_identity(pool),
                result=items,
            )

        async def _compute() -> list[MCPEval]:
            # MultiPV search. Stockfish returns the top-N lines with multipv=N;
            # line 1 (multipv=1) is by definition the engine's canonical best,
            # same as a standalone `evaluate_position` would return. No need
            # for a redundant single-PV pre-search (was costing ~25% of
            # `top_moves` wall time at depth 14).
            res_list: list[MCPEval] = []
            results = await pool.top_moves(board, n=n, depth=depth)
            # AUDIT B-04: when a draw claim is available (immediately or with
            # an intended zeroing move), the root MultiPV cp/mate of a zeroing
            # move can be "polluted" by the engine seeing the draw on the
            # table (e.g. K+R vs K at halfmove=100 reports a tiny cp). We
            # re-evaluate zeroing moves' post-state ONLY when the multipv
            # output looks suspect — i.e. no explicit mate AND a non-positive
            # cp for the mover. Stockfish multipv is authoritative in every
            # other case (no draw on the table, or the engine already gave a
            # clearly winning cp/mate); re-evaluating otherwise just costs an
            # extra engine call without changing the answer. The candidate's
            # reported cp/mate remains the multipv value so ranking and
            # back-compat consumers see the same numbers they did before.
            needs_post_eval = bool(
                rule_status.can_claim_now or rule_status.can_claim_with_intended_move
            )
            zeroing_best_cp: int | None = None
            zeroing_best_mate: int | None = None
            for r in results:
                b_cand = board.copy(stack=True)
                cand_san_val: str | None = None
                cand_post_terminal: str | None = None
                cand_winner: str | None = None
                cand_can_claim_now = False
                cand_can_claim_draw = False
                cand_claim_reasons: list[str] = []
                cand_claim_reasons_now: list[str] = []
                cand_claim_moves: list[str] = []
                # Default to the root rule_status; the post-state branch below
                # refines it. Used by the best_action_obj build below as a
                # fallback when `r.best_move` is missing or fails to parse.
                cand_rule = rule_status
                # Track the post-state cp/mate for the action policy without
                # mutating the candidate's reported values.
                post_state_cp: int | None = None
                post_state_mate: int | None = None

                if r.best_move:
                    try:
                        bm_obj = chess.Move.from_uci(r.best_move.lower())
                        if bm_obj in board.legal_moves:
                            cand_san_val = board.san(bm_obj)
                            is_zeroing = board.is_capture(bm_obj) or (
                                board.piece_type_at(bm_obj.from_square) == chess.PAWN
                            )
                            b_cand.push(bm_obj)
                            # AUDIT B-04: re-evaluate zeroing-move post-state
                            # when the multipv output looks draw-polluted. We
                            # only do this when there's no explicit mate AND
                            # the multipv cp is non-positive for the mover (a
                            # winning move at halfmove=100 should at least
                            # show cp>0; if it doesn't, the engine is treating
                            # the draw as the value of the move and the
                            # post-state is what really matters). The
                            # post-state values feed the action policy
                            # decision; they DO NOT overwrite the candidate's
                            # reported cp/mate (B-05 / C-02 contract).
                            # The post-state re-eval is a draw-pollution guard
                            # (audit B-04 / U-08): when the multipv says the
                            # zeroing move is no better than the draw
                            # (cp<=0 or None), the post-state is what really
                            # matters — the engine is treating the draw as
                            # the value of the move. We do NOT re-evaluate
                            # for strongly positive multipv (the engine has
                            # a clear opinion and a re-eval would only add
                            # cost). The post_state_cp/mate are surfaced on
                            # the wire for client inspection (U-08) — they
                            # are None when no re-eval happened, which is
                            # the honest contract: "no refined post-state
                            # value" rather than fabricating one.
                            multipv_suspect = r.mate is None and (r.cp is None or r.cp <= 0)
                            if (
                                needs_post_eval
                                and is_zeroing
                                and not b_cand.is_game_over(claim_draw=False)
                                and multipv_suspect
                            ):
                                try:
                                    post_ev = await pool.evaluate(b_cand, depth=depth)
                                    if post_ev.mate is not None:
                                        post_state_mate = post_ev.mate
                                    elif post_ev.cp is not None:
                                        post_state_cp = post_ev.cp
                                except Exception:
                                    pass
                            cand_sign = 1 if b_cand.turn == chess.WHITE else -1
                            cand_mover_score: int | None
                            if r.mate is not None:
                                cand_mover_score = cand_sign * r.mate * 1000
                            elif r.cp is not None:
                                cand_mover_score = cand_sign * r.cp
                            else:
                                cand_mover_score = None
                            cand_mate_for_mover = cand_sign * r.mate if r.mate is not None else None
                            cand_rule = evaluate_rule_status(
                                b_cand,
                                mover_score=cand_mover_score,
                                mate_for_mover=cand_mate_for_mover,
                                history_complete=history_complete,
                            )
                            cand_post_terminal = cand_rule.terminal
                            cand_winner = cand_rule.winner
                            cand_can_claim_now = cand_rule.can_claim_now
                            cand_can_claim_draw = cand_rule.can_claim_draw
                            cand_claim_reasons = cand_rule.claim_reasons
                            cand_claim_reasons_now = cand_rule.claim_reasons_now
                            cand_claim_moves = cand_rule.claim_moves
                            # Track best zeroing post-state value for the
                            # action policy below. Sign is mover-POV so we
                            # compare apples to apples. Use the re-evaluated
                            # post-state values when available; fall back to
                            # multipv otherwise (audit B-04 guard).
                            eff_cp = post_state_cp if post_state_cp is not None else r.cp
                            eff_mate = post_state_mate if post_state_mate is not None else r.mate
                            if (
                                needs_post_eval
                                and is_zeroing
                                and (eff_mate is not None or eff_cp is not None)
                            ):
                                mover_sign = 1 if board.turn == chess.WHITE else -1
                                if eff_mate is not None:
                                    mover_mate = mover_sign * eff_mate
                                    if mover_mate > 0 and (
                                        zeroing_best_mate is None or mover_mate > zeroing_best_mate
                                    ):
                                        zeroing_best_mate = mover_mate
                                else:
                                    mover_cp = mover_sign * (eff_cp or 0)
                                    if zeroing_best_cp is None or mover_cp > zeroing_best_cp:
                                        zeroing_best_cp = mover_cp
                    except Exception:
                        pass

                identity = _build_identity(pool)
                # Candidate's reported cp/mate stays at the multipv value so
                # ranking and back-compat callers see the same numbers they
                # did before. Re-evaluated post-state values feed only the
                # action policy decision (audit B-04 / B-05 separation).
                post_eval_for_candidate = Eval(
                    cp=r.cp,
                    mate=r.mate,
                    best_move=r.best_move,
                    pv=r.pv,
                    depth=r.depth,
                )
                # Audit C-03 (2026-09-01 adversarial probe): the candidate's
                # outer action type is the type of move it represents —
                # `play_move` (a candidate IS a play_move action) or
                # `game_over` (the post-state is terminal). The post-state's
                # `rule_status.recommended_action` can be a claim (e.g. after
                # Qb1 the opponent can claim draw) but that is the OPPONENT's
                # perspective, not the candidate's. Reassign `best_action` /
                # `best_action_type` / `best_action_obj` to the candidate's
                # own action type so each candidate reads as a self-consistent
                # play_move or game_over unit. The post-state's recommendation
                # is preserved in `post_position.recommended_action`.
                cand_recommended_action = (
                    "game_over" if cand_post_terminal is not None else "play_move"
                )
                from mcp_server.actions import build_best_action as _build_ba

                if cand_post_terminal is not None:
                    outcome = (
                        "draw"
                        if cand_post_terminal != "checkmate"
                        else ("win" if cand_winner == "white" else "loss")
                    )
                    cand_best_action_obj: dict[str, Any] = {
                        "type": "game_over",
                        "outcome": outcome,
                        "reason": cand_post_terminal,
                    }
                else:
                    # Use the root `board` (not b_cand) for SAN lookup: the
                    # candidate's `best_move` is a legal move AT THE ROOT, not
                    # after it has been played. Passing b_cand would make
                    # `bm in board.legal_moves` False and silently drop SAN.
                    cand_best_action_obj = _build_ba(
                        recommended_action="play_move",
                        rule_status=cand_rule,
                        engine_eval=r,
                        board=board,
                        sign=sign,
                    )
                mcp_eval = MCPEval.from_eval(
                    post_eval_for_candidate,
                    b_cand.fen(),
                    board=b_cand,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                    pv_board=board,
                ).model_copy(
                    update={
                        "build_sha": identity["build_sha"],
                        "engine_config": identity["engine_config"],
                        "post_terminal_status": cand_post_terminal,
                        "candidate_san": cand_san_val,
                        "post_can_claim_draw": cand_can_claim_draw,
                        "post_can_claim_now": cand_can_claim_now,
                        "post_claim_reasons": cand_claim_reasons,
                        "post_claim_moves": cand_claim_moves,
                        "recommended_action": cand_recommended_action,
                        "best_action": cand_recommended_action,
                        "best_action_type": cand_recommended_action,
                        "best_action_obj": cand_best_action_obj,
                        "post_state_cp": post_state_cp,
                        "post_state_mate": post_state_mate,
                        "post_position": {
                            "status": cand_post_terminal or "active",
                            "winner": cand_winner if cand_post_terminal == "checkmate" else None,
                            "can_claim_now": cand_can_claim_now,
                            "can_claim_draw": cand_can_claim_draw,
                            "claim_reasons": cand_claim_reasons_now or cand_claim_reasons,
                            "recommended_action": getattr(
                                cand_rule, "recommended_action", "play_move"
                            ),
                        },
                    }
                )
                res_list.append(mcp_eval)

            def _candidate_rank_key(eval_item: MCPEval) -> float:
                # U-01 (2026-09-01): mate for the mover must outrank any
                # finite-cp win, and the ordering must NOT depend on `n`
                # (the number of candidates requested). The previous rank
                # key had two failure modes:
                #   1. cp was returned unclamped, so a saturated cp=+20000
                #      candidate outranked a mate-in-1 candidate (9999).
                #   2. The rank key preferred the multipv cp of zeroing
                #      moves over the mate branch, so a non-mating capture
                #      could rank above a mating move.
                # Chess-correct total order for the side-to-move is:
                #   delivered mate (terminal) > forced mate for mover
                #     > finite-cp win (clamped to mate ceiling)
                #     > draw  > finite-cp loss > forced mate against mover.
                # We clamp cp to ±MATE_RANK_CEILING so any saturated
                # sentinel (cp=±20000, syzygy fallback, depth=0 win) cannot
                # outrank a forced mate. We always sort (the previous gate
                # `halfmove>=100 or has_terminal_cand` let Stockfish's
                # raw MultiPV order leak through for the >99% case, where
                # a forced mate could still be in slot 2+ at shallow depth).
                MATE_RANK_CEILING = 9999.0
                MATE_VALUE = 10000.0

                # Terminal checks first — these are the strongest signals
                # regardless of cp/mate.
                if eval_item.post_terminal_status == "checkmate":
                    # Candidate delivered mate. Always ranks above any
                    # non-mate candidate (mate=1 is the canonical best).
                    return MATE_VALUE
                if eval_item.post_terminal_status in (
                    "stalemate",
                    "insufficient_material",
                    "seventyfive_moves",
                    "fivefold_repetition",
                    "dead_position",
                ):
                    return 0.0

                # Mate branch BEFORE cp branch (U-01): a mate-in-1 must
                # outrank any finite-cp win. Use the post-state mate when
                # available (audit B-05 — re-eval can refine the multipv
                # mate; falls back to multipv when no re-eval happened).
                eff_mate = (
                    eval_item.post_state_mate
                    if eval_item.post_state_mate is not None
                    else eval_item.mate
                )
                if eff_mate is not None:
                    mover_mate = sign * eff_mate
                    if mover_mate > 0:
                        # Forced mate for mover: shorter is better.
                        return MATE_VALUE - abs(mover_mate)
                    # Forced mate against mover: longer is "less bad",
                    # but always below the floor for any finite cp.
                    return -MATE_VALUE + abs(mover_mate)

                # Cp branch: clamped to the mate ceiling so a saturated
                # cp=±20000 cannot outrank a forced mate. Use post-state
                # cp when available for zeroing moves that were re-eval'd
                # (audit B-04 draw-pollution guard); otherwise use the
                # multipv cp.
                eff_cp = (
                    eval_item.post_state_cp if eval_item.post_state_cp is not None else eval_item.cp
                )
                if eff_cp is not None:
                    mover_cp = sign * eff_cp
                    # Clamp so finite-cp wins never exceed the mate ceiling.
                    if mover_cp > MATE_RANK_CEILING:
                        return MATE_RANK_CEILING
                    if mover_cp < -MATE_RANK_CEILING:
                        return -MATE_RANK_CEILING
                    return float(mover_cp)

                return 0.0

            # Always sort (U-01): n-invariance requires a stable chess-correct
            # ordering regardless of halfmove / terminal state. Removing the
            # gate does not change behavior for positions where Stockfish's
            # raw order already matches the chess-correct order; it just
            # fixes the cases where it doesn't.
            res_list.sort(key=_candidate_rank_key, reverse=True)

            # Persist zeroing-move findings on the cache so the cache-hit path
            # below reuses the same policy decision without re-searching.
            await _cache.set_top_moves(cache_key, res_list)
            return res_list

        sf_key = f"{cache_key}:n={n}"
        res = cast(list[MCPEval], await _single_flight.do(sf_key, _compute))
        await metrics.record("top_moves", (time.time() - t0) * 1000, cache_hit=False)
        items = [c.model_copy(update={"requested_depth": raw_requested_depth}) for c in res[:n]]
        if verbosity_mode == VERBOSITY_COMPACT:
            items = [_compact_mcpeval(c) for c in items]
        root_rec_action = _pick_root_recommended_action(items)
        best_action_obj = build_best_action(
            recommended_action=root_rec_action,
            rule_status=rule_status,
            engine_eval=items[0] if items else None,
            board=board,
            sign=sign,
        )
        legal_actions = build_legal_actions(
            rule_status=rule_status,
            engine_eval=items[0] if items else None,
            board=board,
            legal_engine_moves=list(items),
        )
        return TopMovesResult(
            status="active",
            winner=None,
            recommended_action=root_rec_action,
            can_claim_draw=rule_status.can_claim_draw,
            claim_reasons=rule_status.claim_reasons,
            claim_move=rule_status.claim_move,
            can_claim_now=rule_status.can_claim_now,
            claim_reasons_now=rule_status.claim_reasons_now,
            can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
            claim_moves=rule_status.claim_moves,
            best_action_obj=best_action_obj,
            legal_actions=legal_actions,
            history_completeness=rule_status.history_completeness,
            repetition_status=rule_status.repetition_status,
            requested_depth=raw_requested_depth,
            searched_depth=depth,
            requested_n=raw_requested_n,
            clamped_n=clamped_n,
            returned_n=len(items),
            legal_move_count=legal_move_count,
            canonical_fen=canonical_fen,
            fen_was_canonicalized=fen_was_canonicalized,
            engine="Stockfish",
            engine_version=engine_name_str,
            **_build_identity(pool),
            result=items,
        )
    except ToolError:
        await metrics.record("top_moves", (time.time() - t0) * 1000, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record("top_moves", (time.time() - t0) * 1000, is_error=True)
        msg = str(exc)
        code = error_code_for(msg)
        raise _tool_error(code=code, message=msg, tool="top_moves", input=fen) from exc
    except Exception as exc:
        await metrics.record("top_moves", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="top_moves") from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def classify_move(
    fen: str,
    move: str | None = None,
    moves: list[str] | None = None,
    depth: int = 14,
    action_type: Literal["play_move", "claim_draw", "claim_draw_with_intended_move"] = "play_move",
    strict: bool = False,
    ctx: Context | None = None,
) -> MCPMoveAnalysis:
    """Grade a played move against Stockfish's best alternative.

    Grades: 'best', 'good', 'inaccuracy', 'mistake', 'blunder'. Note that `move_class`
    is derived from `effective_loss` (win probability impact & position context, e.g.
    decisive advantage saturation), NOT directly from raw `centipawn_loss`. Also returns
    centipawn loss, mate distance loss, and evals before/after the move.

    Args:
        fen: FEN or PGN string for the position BEFORE `move`.
        move: The move to grade in UCI (e.g. "e2e4") or SAN (e.g. "e4", "Bxf3", "O-O").
            Required for `play_move` and `claim_draw_with_intended_move`; optional for
            `claim_draw` (the claim outcome does not depend on any specific move).
        moves: Optional UCI or SAN moves to replay onto the position first.
        depth: Stockfish search depth (default 14, clamped 1-30).
        action_type: Intended chess action ('play_move', 'claim_draw', 'claim_draw_with_intended_move').
        strict: When True, reject non-canonical SAN syntax or move numbers (default False).

    Returns:
        MoveAnalysis with move_class, centipawn_loss, effective_loss, eval_before, eval_after,
        best_move_san, best_line_san, and played_line_san.
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="classify_move")
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    try:
        if action_type not in {"play_move", "claim_draw", "claim_draw_with_intended_move"}:
            raise ValueError(f"INVALID_ACTION_TYPE: {action_type}")
        # P2 (2026-09-02 ultra audit): request-shape validation must run BEFORE
        # board-state validation. The audit found that `claim_draw` with a
        # supplied `move` argument on a non-claimable board returned
        # "draw cannot be claimed now" — a board-state error — instead of
        # the structural "claim_draw must not include a move" error. The
        # structural error is consistent regardless of position and lets
        # callers distinguish bad input from bad state. Same applies to
        # `play_move` / `claim_draw_with_intended_move` missing a move.
        # R5 (2026-09-02 round-5 super-deep audit): the move parameter
        # must be a string. A non-string (int, list, None-as-empty) used
        # to fall through to `move.strip()` and produce a confusing
        # AttributeError. Validate type first so the rejection is a clean
        # INVALID_INPUT message.
        if move is not None and not isinstance(move, str):
            raise ValueError(f"INVALID_INPUT: 'move' must be a string, got {type(move).__name__}.")
        if action_type == "claim_draw":
            if move is not None and move.strip() and move.strip() != "(none)":
                if strict:
                    raise ValueError(
                        f"STRICT_SAN_ERROR: action_type='claim_draw' must not "
                        f"include a `move` argument; got {move!r}. Pass move=None "
                        f"or omit the parameter."
                    )
                # Lenient mode: still record that the caller passed a
                # meaningless argument (per U-12 invariant — B-02 audit).
                # We surface this as a syntax_warning later via the response.
        else:
            if move is None or not move.strip():
                raise ValueError(
                    "MISSING_MOVE: 'move' is required for action_type='play_move' "
                    "and action_type='claim_draw_with_intended_move'"
                )
        board = _build_board(fen, moves or [], strict=strict)
        history_complete = _history_provenance_for_input(fen, moves)
        rule_before = evaluate_rule_status(board, history_complete=history_complete)
        # AUDIT B-01/B-02/B-03: for `claim_draw`, the dummy `move` argument must
        # not be parsed/executed; the claim outcome is purely procedural. Accept
        # `move=None` (or any string) but never push the move onto the board
        # when classifying a draw claim. `claim_draw_with_intended_move` still
        # requires a real intended move because the move IS the claim.
        if action_type == "claim_draw":
            chess_move: chess.Move | None = None
            syntax_warn: str | None = None
            if move is not None and move.strip() and move.strip() != "(none)":
                # P2 (2026-09-02 ultra audit): lenient mode still warns when
                # the caller passes a meaningless `move` argument to
                # `claim_draw`. Strict mode rejects outright (above). The
                # warning makes the structural mismatch observable without
                # breaking the claim.
                syntax_warn = (
                    f"action_type='claim_draw' ignores supplied move argument "
                    f"{move!r} (the claim outcome is purely procedural)."
                )
            # P2 (2026-09-02 ultra audit): terminal-state handling must
            # happen before action-specific claim validation so every
            # action on a finished board returns the same GAME_ALREADY_OVER
            # error, not a position-dependent ILLEGAL_ACTION variant.
            if is_terminal_position(board):
                raise ValueError(
                    f"GAME_ALREADY_OVER: Position '{board.fen()}' is already game over; "
                    f"no further actions can be taken on a finished game."
                )
            if not rule_before.can_claim_now:
                raise ValueError("ILLEGAL_ACTION: draw cannot be claimed now")
        else:
            assert move is not None and move.strip()  # shape validated above
            chess_move, syntax_warn = _parse_move_on_board_with_warning(board, move, strict=strict)
            if (
                action_type == "claim_draw_with_intended_move"
                and chess_move.uci() not in rule_before.intended_claim_ucis
            ):
                raise ValueError("ILLEGAL_ACTION: intended move does not create a legal draw claim")
        pool = await _get_analyzer_pool(ctx)

        # Cache key uses an empty/dummy move for claim_draw so the same
        # underlying position/action always maps to one cache entry, regardless
        # of the dummy `move` the caller passed (audit B-02 invariant).
        cache_move_uci = chess_move.uci() if chess_move is not None else ""
        cache_key = classify_cache_key(
            board,
            cache_move_uci,
            depth,
            action_type=action_type,
            engine_version=getattr(pool, "engine_version", None),
            history_completeness=history_complete,
        )

        cached = await _cache.get_classify(cache_key)
        if cached is not None:
            await metrics.record("classify_move", (time.time() - t0) * 1000, cache_hit=True)
            eval_bef = cached.eval_before.model_copy(
                update={"requested_depth": raw_requested_depth}
            )
            eval_aft = cached.eval_after.model_copy(update={"requested_depth": raw_requested_depth})
            return cached.model_copy(
                update={
                    "eval_before": eval_bef,
                    "eval_after": eval_aft,
                    "syntax_warning": syntax_warn,
                }
            )

        # Build played_san / board_after defensively: for claim_draw they are
        # NOT derived from any chess move because the claim is procedural.
        if chess_move is not None:
            played_san = board.san(chess_move)
            board_after = board.copy(stack=True)
            board_after.push(chess_move)
        else:
            played_san = None
            board_after = board.copy(stack=True)

        async def _compute() -> MCPMoveAnalysis:
            pool = await _get_analyzer_pool(ctx)

            if (
                chess_move is not None
                and hasattr(pool, "classify_move")
                and type(pool)
                not in (
                    AnalyzerPool,
                    TCPAnalyzerPool,
                )
            ):
                result = await pool.classify_move(board, chess_move, depth=depth)
                return MCPMoveAnalysis.from_analysis(
                    result,
                    fen_before=board.fen(),
                    fen_after=board_after.fen(),
                    played_san=played_san,
                    board_before=board,
                    board_after=board_after,
                    syntax_warning=None,
                    action_type=action_type,
                    history_complete=history_complete,
                )

            eval_before, _ = await _evaluate_game_position_cached(
                board,
                depth,
                pool,
                requested_depth=raw_requested_depth,
                history_complete=history_complete,
            )

            # AUDIT B-02/B-03: for draw-claim actions, the post-state is the
            # position AFTER the claim is granted, not after the supplied
            # (irrelevant) move is played. Re-evaluate the same root board so
            # the resulting `eval_after` reflects the draw outcome (cp=0,
            # outcome=draw) regardless of any dummy move the caller passed.
            if action_type in ("claim_draw", "claim_draw_with_intended_move"):
                eval_after, _ = await _evaluate_game_position_cached(
                    board,
                    depth,
                    pool,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                )
                # The claim outcome is a draw; force cp=0 and outcome=draw so
                # every downstream caller sees a consistent post-claim state
                # independent of the dummy move.
                eval_after = _force_draw_outcome(eval_after)
            else:
                # Correctness first: eval_after must describe the immediate
                # post-move position. Reusing the root PV tail or root score
                # can misstate finite-depth CP and mate distance. Engine/cache
                # layers remain responsible for performance reuse.
                eval_after, _ = await _evaluate_game_position_cached(
                    board_after,
                    depth,
                    pool,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                )

            if chess_move is not None:
                score = score_played_move(
                    board,
                    chess_move,
                    eval_before,
                    eval_after,
                    board_after,
                    action_type=action_type,
                )
            else:
                # claim_draw without a move: pass a placeholder Move and the
                # post-claim board (= root board). score_played_move still
                # consults rule_before.can_claim_now and the post-claim eval,
                # so the dummy Move here is purely structural and never
                # affects the score.
                placeholder = next(iter(board.legal_moves), None)
                if placeholder is None:
                    raise ValueError("ILLEGAL_ACTION: no legal moves; cannot evaluate claim")
                score = score_played_move(
                    board,
                    placeholder,
                    eval_before,
                    eval_after,
                    board_after,
                    action_type=action_type,
                )

            # Candidate Verification Search (Opera Morphy invariant enforcement):
            # If played move matched eval_before.best_move, but grading would produce mistake/blunder,
            # run a deeper verification search so eval_before is updated with the true best candidate.
            #
            # P1 audit fix: this branch used to be FAIL-OPEN — when the deeper
            # search threw any exception, the code silently flipped move_class to
            # BEST, effective_loss=0. That makes a buggy engine produce honest
            # answers and a buggy harness produce lies. The fixed behavior:
            #   - if verification succeeds and finds a better move, regrade.
            #   - if verification succeeds and confirms our move, lock to BEST.
            #   - if verification FAILS, do NOT silently overwrite grading;
            #     mark classification_verified=False so callers see the
            #     unverified result instead of a fabricated "best".
            verification_attempted = False
            # Audit P0/P1 (2026-09-01 adversarial probe): the verification
            # block is for `play_move` only. Draw-claim actions classify the
            # CLAIM, not the supplied move; the move may coincidentally match
            # `eval_before.best_move` (e.g. `claim_draw + Qc8#` where the
            # engine's best IS the mating move the player is refusing to play).
            # In that case the depth+4 verification correctly confirms the
            # move is the engine's best legal attempt — but that's irrelevant
            # to grading the CLAIM. Allowing the "else" branch to overwrite
            # `move_class=BEST, effective_loss=0` here violates the invariant
            # `is_best_action==False AND best outcome==win AND played
            # outcome==draw ⇒ effective_loss > 0`. Skip the whole block for
            # claim actions; the score from `score_played_move` is final.
            if (
                action_type == "play_move"
                and chess_move is not None
                and (
                    chess_move.uci().lower() == (eval_before.best_move or "").lower()
                    and score.move_class in (MoveClass.MISTAKE, MoveClass.BLUNDER)
                    and not score.missed_draw_claim
                    and not score.conceded_draw_claim
                )
            ):
                try:
                    # Cache the depth+4 verification result via the same
                    # L1/L2 path as any other eval. Previously this went
                    # straight to pool.evaluate, bypassing the cache — every
                    # classify_move that hit this verification path paid the
                    # full uncached depth+4 cost. Now the depth+4 result is
                    # cached like any other eval.
                    verify_eval_result, _verify_hit = await _evaluate_game_position_cached(
                        board,
                        depth + 4,
                        pool,
                        requested_depth=raw_requested_depth + 4,
                        history_complete=history_complete,
                    )
                    verify_ev: Eval = Eval(
                        cp=verify_eval_result.cp,
                        mate=verify_eval_result.mate,
                        best_move=verify_eval_result.best_move,
                        pv=verify_eval_result.pv,
                        depth=verify_eval_result.searched_depth or (depth + 4),
                    )
                    verification_attempted = True
                    if (
                        verify_ev.best_move
                        and verify_ev.best_move.lower() != chess_move.uci().lower()
                    ):
                        # Verification discovered a better move! Update eval_before
                        eval_before = MCPEval.from_eval(
                            verify_ev,
                            board.fen(),
                            board=board,
                            requested_depth=raw_requested_depth,
                            history_complete=history_complete,
                        )
                        score = score_played_move(
                            board,
                            chess_move,
                            eval_before,
                            eval_after,
                            board_after,
                            action_type=action_type,
                        )
                    else:
                        # Played move is confirmed as the best legal attempt.
                        score.move_class = MoveClass.BEST
                        score.effective_loss = 0
                        score.is_best_engine_move = True
                except Exception:
                    # Verification FAILED — leave the original grading intact
                    # and mark the response unverified rather than fabricating
                    # a BEST verdict we cannot prove (audit P1 fix).
                    verification_attempted = True

            best_san: str | None = None
            if score.is_best_engine_move and chess_move is not None:
                best_san = played_san
            elif eval_before.best_move:
                try:
                    bm = chess.Move.from_uci(eval_before.best_move.lower())
                    if bm in board.legal_moves:
                        best_san = board.san(bm)
                except Exception:
                    pass

            best_line_san = pv_to_san(board, eval_before.pv) if eval_before.pv else best_san
            played_continuation: str | None = None
            if eval_after.pv and not board_after.is_game_over() and chess_move is not None:
                played_continuation = pv_to_san(board_after, eval_after.pv)

            played_line_san = played_san
            if played_continuation and played_san is not None:
                played_line_san = f"{played_san} {played_continuation}"

            verified = True
            if (
                action_type == "play_move"
                and score.best_action != "play_move"
                and score.is_best_action
                and not score.action_equivalent
            ):
                verified = False
            if (
                score.effective_loss
                and score.effective_loss > 0
                and (not score.loss_kind or score.loss_kind == "none")
            ):
                verified = False
            # P1 audit fix: verification failure must NOT silently downgrade
            # grading. If we tried to verify but couldn't reach a conclusion,
            # the response must be marked unverified.
            if verification_attempted and score.move_class in (
                MoveClass.MISTAKE,
                MoveClass.BLUNDER,
            ):
                verified = False

            played_uci = chess_move.uci() if chess_move is not None else ""
            mcp_analysis = MCPMoveAnalysis(
                played=played_uci,
                played_san=played_san,
                move_class=score.move_class,
                is_engine_best=score.is_best_engine_move,
                is_best_engine_move=score.is_best_engine_move,
                centipawn_loss=score.centipawn_loss,
                mate_distance_loss=score.mate_distance_loss,
                raw_centipawn_loss=score.raw_centipawn_loss,
                raw_centipawn_delta=score.raw_centipawn_delta,
                effective_loss=score.effective_loss,
                loss_kind=score.loss_kind,
                engine_cp_loss=score.engine_cp_loss,
                mate_distance_penalty=score.mate_distance_penalty,
                outcome_penalty=score.outcome_penalty,
                rule_action_penalty=score.rule_action_penalty,
                eval_before=eval_before,
                eval_after=eval_after,
                best_move_san=best_san,
                best_line_san=best_line_san,
                best_line_san_truncated=bool(eval_before.pv and len(eval_before.pv) > 6),
                played_line_san=played_line_san,
                played_continuation_san=played_continuation,
                syntax_warning=None,
                action_type=action_type,
                best_action=score.best_action,
                is_best_action=score.is_best_action,
                action_equivalent=score.action_equivalent,
                played_action_obj=build_played_action(
                    action_type,
                    move_uci=played_uci,
                    move_san=played_san,
                    rule_status=rule_before,
                    cp=eval_after.cp,
                    mate=eval_after.mate,
                ),
                best_action_obj=eval_before.best_action_obj,
                missed_draw_claim=score.missed_draw_claim,
                conceded_draw_claim=score.conceded_draw_claim,
                claim_reason=score.claim_reason,
                claim_move=score.claim_move,
                can_claim_now=score.can_claim_now,
                can_claim_with_intended_move=score.can_claim_with_intended_move,
                claim_moves=score.claim_moves,
                classification_verified=verified,
            )
            await _cache.set_classify(cache_key, mcp_analysis)
            return mcp_analysis

        res = cast(MCPMoveAnalysis, await _single_flight.do(cache_key, _compute))
        await metrics.record("classify_move", (time.time() - t0) * 1000, cache_hit=False)
        return res.model_copy(update={"syntax_warning": syntax_warn})
    except ToolError:
        await metrics.record("classify_move", (time.time() - t0) * 1000, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record("classify_move", (time.time() - t0) * 1000, is_error=True)
        msg = str(exc)
        code = error_code_for(msg)
        raise _tool_error(code=code, message=msg, tool="classify_move", input=move) from exc
    except Exception as exc:
        await metrics.record("classify_move", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="classify_move") from exc


def _compute_game_metrics(
    positions: list[chess.Board],
    moves: list[chess.Move],
    evals: list[MCPEval],
) -> tuple[
    float | None,  # white_acc
    float | None,  # black_acc
    float | None,  # white_acpl
    float | None,  # black_acpl
    float | None,  # white_raw_acpl
    float | None,  # black_raw_acpl
    float | None,  # white_average_effective_loss
    float | None,  # black_average_effective_loss
    tuple[int, int, int],  # white blunders, mistakes, inaccuracies
    tuple[int, int, int],  # black blunders, mistakes, inaccuracies
    list[PlyAnalysisItem],  # turning_points
]:
    """Calculate ACPL, accuracy %, mistakes, and turning points from position evaluations."""
    white_cpls: list[int] = []
    black_cpls: list[int] = []
    white_raw_cpls: list[int] = []
    black_raw_cpls: list[int] = []
    white_eff_losses: list[int] = []
    black_eff_losses: list[int] = []
    white_accs: list[float] = []
    black_accs: list[float] = []
    white_blunders = white_mistakes = white_inaccuracies = 0
    black_blunders = black_mistakes = black_inaccuracies = 0
    turning_points: list[PlyAnalysisItem] = []

    is_game_draw = bool(
        evals
        and (
            positions[-1].is_game_over(claim_draw=False)
            or positions[-1].is_fifty_moves()
            or positions[-1].is_repetition(3)
        )
    )

    for ply_idx, move in enumerate(moves, start=1):
        board_before = positions[ply_idx - 1]
        board_after = positions[ply_idx]
        eval_before = evals[ply_idx - 1]
        eval_after = evals[ply_idx]

        is_white = board_before.turn == chess.WHITE
        move_san = board_before.san(move)

        # Only rewrite the final ply as a procedural draw-claim when:
        #  - the move is genuinely a 50-move or 3-fold claim (NOT an auto-terminal
        #    like 75-move / stalemate / checkmate / locked dead — those are real
        #    moves that LOST the game by blunder, not players taking a draw);
        #  - the move is one of the legal intended-claim moves (a non-resetting,
        #    non-capturing king move for the 50-move rule, or a repetition-completing
        #    move for the threefold rule).
        # Otherwise we score the move as a real play_move so a blunder into an
        # automatic terminal draw (e.g. Qf8+ at halfmove 149) is properly penalized.
        action_type_to_use = "play_move"
        if ply_idx == len(moves) and is_game_draw:
            intended_now = (
                board_after.is_fifty_moves() and not board_after.is_seventyfive_moves()
            ) or board_after.is_repetition(3)
            is_intended_claim = intended_now and (
                not board_after.is_game_over(claim_draw=False) or board_after.can_claim_draw
            )
            if is_intended_claim:
                played_uci = move.uci()
                rule_before = evaluate_rule_status(board_before, history_complete="complete")
                valid_for_intended = (
                    rule_before.can_claim_with_intended_move
                    and played_uci in rule_before.intended_claim_ucis
                )
                if valid_for_intended:
                    action_type_to_use = "claim_draw_with_intended_move"

        score = score_played_move(
            board_before,
            move,
            eval_before,
            eval_after,
            board_after,
            action_type=action_type_to_use,
        )

        mc = score.move_class.value
        cpl = score.centipawn_loss
        win_loss = score.win_loss
        move_acc = max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * win_loss) - 3.1669))
        effective_loss = score.effective_loss

        raw_cpl_val = (
            score.centipawn_loss
            if score.centipawn_loss is not None
            else (score.raw_centipawn_loss if score.raw_centipawn_loss is not None else 0)
        )
        if is_white:
            if raw_cpl_val is not None:
                white_cpls.append(raw_cpl_val)
            if score.raw_centipawn_loss is not None:
                white_raw_cpls.append(score.raw_centipawn_loss)
            elif score.centipawn_loss is not None:
                white_raw_cpls.append(score.centipawn_loss)
            if effective_loss is not None:
                white_eff_losses.append(effective_loss)
            white_accs.append(move_acc)
            if mc == "blunder":
                white_blunders += 1
            elif mc == "mistake":
                white_mistakes += 1
            elif mc == "inaccuracy":
                white_inaccuracies += 1
        else:
            if raw_cpl_val is not None:
                black_cpls.append(raw_cpl_val)
            if score.raw_centipawn_loss is not None:
                black_raw_cpls.append(score.raw_centipawn_loss)
            elif score.centipawn_loss is not None:
                black_raw_cpls.append(score.centipawn_loss)
            if effective_loss is not None:
                black_eff_losses.append(effective_loss)
            black_accs.append(move_acc)
            if mc == "blunder":
                black_blunders += 1
            elif mc == "mistake":
                black_mistakes += 1
            elif mc == "inaccuracy":
                black_inaccuracies += 1

        best_san: str | None = None
        # U-06 (2026-09-01): reconcile `best_san` with the final classification.
        # Without this guard, an analyze_game turning point can report
        # `best_move_san == played_san` while `move_class == "blunder"` —
        # internally contradictory. The bug surfaced at depth=1 where the
        # engine's top line happens to be a losing move (audit U-06
        # promotion-defense reproducer). When the played move equals the
        # engine's reported best but the classifier decided it was a
        # blunder/mistake, suppress the best_move_san to avoid the
        # contradiction. The classify_move path runs a depth+4 verification
        # search to refine; analyze_game doesn't (per-ply cost), so the
        # conservative answer is `best_san = None` here.
        if eval_before.best_move and not (
            score.is_best_engine_move and score.move_class.value in ("blunder", "mistake")
        ):
            try:
                move_obj = chess.Move.from_uci(eval_before.best_move.lower())
                if move_obj in board_before.legal_moves:
                    best_san = board_before.san(move_obj)
            except (
                ValueError,
                chess.IllegalMoveError,
                chess.InvalidMoveError,
                AssertionError,
            ):
                best_san = None

        if (
            (cpl is not None and cpl >= 150)
            or (effective_loss is not None and effective_loss >= 150)
            or mc in ("blunder", "mistake")
        ):
            turning_points.append(
                PlyAnalysisItem(
                    ply=ply_idx,
                    san=move_san,
                    uci=move.uci(),
                    move_class=mc,
                    centipawn_loss=cpl,
                    effective_loss=effective_loss,
                    loss_kind=score.loss_kind,
                    engine_cp_loss=score.engine_cp_loss,
                    mate_distance_penalty=score.mate_distance_penalty,
                    outcome_penalty=score.outcome_penalty,
                    rule_action_penalty=score.rule_action_penalty,
                    best_move_san=best_san,
                    best_action=score.best_action,
                    missed_draw_claim=score.missed_draw_claim,
                    conceded_draw_claim=score.conceded_draw_claim,
                    claim_reason=score.claim_reason,
                    claim_move=score.claim_move,
                )
            )

    white_acc = round(sum(white_accs) / len(white_accs), 1) if white_accs else None
    black_acc = round(sum(black_accs) / len(black_accs), 1) if black_accs else None
    white_raw_acpl = round(sum(white_raw_cpls) / len(white_raw_cpls), 1) if white_raw_cpls else None
    black_raw_acpl = round(sum(black_raw_cpls) / len(black_raw_cpls), 1) if black_raw_cpls else None
    white_avg_eff = (
        round(sum(white_eff_losses) / len(white_eff_losses), 1) if white_eff_losses else None
    )
    black_avg_eff = (
        round(sum(black_eff_losses) / len(black_eff_losses), 1) if black_eff_losses else None
    )
    white_acpl = white_avg_eff
    black_acpl = black_avg_eff

    top_turning_points = sorted(
        sorted(
            turning_points,
            key=lambda x: 1000 if x.effective_loss is None else x.effective_loss,
            reverse=True,
        )[:8],
        key=lambda x: x.ply,
    )

    return (
        white_acc,
        black_acc,
        white_acpl,
        black_acpl,
        white_raw_acpl,
        black_raw_acpl,
        white_avg_eff,
        black_avg_eff,
        (white_blunders, white_mistakes, white_inaccuracies),
        (black_blunders, black_mistakes, black_inaccuracies),
        top_turning_points,
    )


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def analyze_game(  # pyright: ignore[reportGeneralTypeIssues]
    pgn: str,
    depth: int = 14,
    strict: bool = False,
    ctx: Context | None = None,
) -> GameAnalysisResult:
    """Analyze a full game in PGN format with Stockfish, providing accuracy scores, mistake counts, and metadata.

    Supports standard PGN, annotated PGNs (with comments, NAGs, variations), conversational
    preamble/trailer text, markdown-wrapped PGNs, and bare move lists. Side variations in parentheses
    and comments are ignored for the mainline analysis. `white_acpl` / `black_acpl` report the effective
    ACPL across all plies (including 1000cp mate transitions and draw claim forfeitures), while
    `white_raw_acpl` / `black_raw_acpl` report unweighted raw CPL on non-mate plies.

    Args:
        pgn: PGN string, annotated game, or move text.
        depth: Search depth per move (default 14, clamped 1-30).
        strict: When True, reject non-canonical SAN syntax, move number mismatches, or metadata discrepancies (default False).

    Returns:
         GameAnalysisResult with player accuracy %, ACPL, blunder/mistake counts, turning points, and game metadata.
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="analyze_game")
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    try:
        sanitized_pgn, lexical_header_warnings = _sanitize_malformed_pgn_header_lines(
            pgn, strict=strict
        )
        _check_multiple_games(sanitized_pgn)
        if strict:
            _validate_strict_header_syntax(sanitized_pgn)
        canonical_pgn = _extract_canonical_pgn_text(sanitized_pgn)
        game = _extract_game_inner(canonical_pgn, strict=strict)
        if strict:
            _validate_strict_mainline_surface(canonical_pgn, game)

        positions: list[chess.Board] = []
        moves: list[chess.Move] = []
        syntax_warnings: list[str] = []
        curr_board = game.board()
        if not curr_board.is_valid() or curr_board.status() != chess.STATUS_VALID:
            raise ValueError(
                f"INVALID_FEN: Initial position '{curr_board.fen()}' in PGN is not a valid chess position ({format_fen_status_errors(curr_board.status())})."
            )

        positions.append(curr_board.copy(stack=True))
        auto_termination: str | None = None
        reached_terminal = False
        ignored_trailing_plies = 0

        # U-03 (2026-09-01): if the initial FEN is already terminal (75-move
        # draw, checkmate, stalemate, insufficient material, fivefold
        # repetition, dead position), the movetext's first move is bogus —
        # the board has no legal moves. Strict mode raises a
        # STRICT_PGN_ERROR. Non-strict mode records a syntax_warning and
        # treats every following move as a trailing ply so the analysis
        # surfaces 0 executed plies. Without this check the mainline loop
        # silently advanced `ignored_trailing_plies` without ever telling
        # the caller that the starting position was terminal.
        initial_rule = evaluate_rule_status(curr_board, history_complete="complete")
        if initial_rule.terminal is not None:
            auto_termination = initial_rule.terminal
            reached_terminal = True
            if strict:
                raise ValueError(
                    f"STRICT_PGN_ERROR: Initial FEN '{curr_board.fen()}' is already "
                    f"terminal ({initial_rule.terminal}); cannot execute movetext."
                )
            syntax_warnings.append(
                f"Initial FEN is terminal ({initial_rule.terminal}); "
                f"all movetext moves will be ignored."
            )

        # P3/INVESTIGATE (2026-09-02 ultra audit): NAG values outside the
        # PGN 0..255 range were silently dropped in lenient mode (the
        # `_validate_movetext_tokens` helper only flags them in strict
        # mode). Re-scan the movetext here so lenient callers also see
        # the warning. Strict mode still rejects via the helper's
        # `invalid_tokens` branch; this scan catches the lenient case
        # without regressing strict behavior. Comments and variations are
        # already stripped from `cleaned_movetext` below, so the regex
        # scans the same tokens the strict path consumes.
        for nag_match in re.finditer(r"\$([0-9]+)", canonical_pgn):
            nag_val = int(nag_match.group(1))
            if nag_val > 255:
                if strict:
                    # Strict mode: promote to a metadata_warning so the
                    # final pass at the bottom raises STRICT_PGN_ERROR
                    # (mirrors the existing NAG enforcement path).
                    syntax_warnings.append(
                        f"NAG value ${nag_val} outside the PGN-supported range 0..255."
                    )
                else:
                    syntax_warnings.append(
                        f"NAG value ${nag_val} outside the PGN-supported range 0..255."
                    )

        # Extract headers ONLY from contiguous header block at the start of canonical_pgn
        header_end = 0
        first_header = TAG_PAIR_REGEX.search(canonical_pgn)
        first_mv = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", canonical_pgn)
        if first_header and (not first_mv or first_header.start() < first_mv.start()):
            header_end = first_header.end()
            for m in TAG_PAIR_REGEX.finditer(canonical_pgn):
                if m.start() < header_end:
                    continue
                if canonical_pgn[header_end : m.start()].strip() == "":
                    header_end = m.end()
                else:
                    break

        header_section = canonical_pgn[:header_end]
        movetext_section = canonical_pgn[header_end:]

        # Clean movetext for token scanning (strip comments and variations, translate figurines, split attached NAGs)
        movetext_section = re.sub(
            r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)(\$\d+)",
            r"\1 \2",
            movetext_section,
        )
        movetext_section = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)(\$\d+)", r"\1\2 \3", movetext_section)
        cleaned_movetext = _normalize_movetext_figurines(movetext_section)
        if re.search(
            r"(?:^|\s)\(?e\.?p\.?\)?(?=\s|$)",
            cleaned_movetext,
            flags=re.IGNORECASE,
        ):
            syntax_warnings.append("En-passant marker 'e.p.' normalized to canonical SAN.")
        while "{" in cleaned_movetext and "}" in cleaned_movetext:
            prev = cleaned_movetext
            cleaned_movetext = re.sub(r"\{[^{}]*\}", " ", cleaned_movetext, flags=re.DOTALL)
            if cleaned_movetext == prev:
                break
        cleaned_movetext = re.sub(r";[^\r\n]*", " ", cleaned_movetext)
        while "(" in cleaned_movetext and ")" in cleaned_movetext:
            prev = cleaned_movetext
            cleaned_movetext = re.sub(r"\([^()]*\)", " ", cleaned_movetext, flags=re.DOTALL)
            if cleaned_movetext == prev:
                break

        # R4-§C (2026-09-02 ultra audit round 4): PGN §8.1 allows whitespace
        # around the move-number dot (`1 . e4` is equivalent to `1. e4`).
        # The downstream tokenizer expects a single `N.` / `N...` token and
        # emits spurious "Input SAN 'N' normalized" warnings otherwise. Collapse
        # digit-then-dots sequences across whitespace before splitting.
        cleaned_movetext = re.sub(
            r"(?<!\S)(\d+)\s+(\.+)(?=\s|$)",
            lambda m: f" {m.group(1)}{m.group(2)} ",
            cleaned_movetext,
        )

        movetext_tokens = cleaned_movetext.split()
        tok_idx = 0
        expected_fullmove = curr_board.fullmove_number

        # Skip leading non-chess tokens to align with the first actual move (PGN-01)
        for i, tok in enumerate(movetext_tokens):
            clean_tok = tok.strip(".,;:!?")
            num_m = re.match(r"^(\d+)[\.\:]*$", clean_tok)
            if num_m:
                tok_idx = i
                break
            try:
                curr_board.parse_san(clean_tok)
                tok_idx = i
                break
            except Exception:
                continue

        for node in game.mainline():
            if reached_terminal:
                ignored_trailing_plies += 1
                continue

            move = node.move
            if move not in curr_board.legal_moves:
                ignored_trailing_plies += 1
                reached_terminal = True
                continue

            canonical_san = curr_board.san(move)

            # Advance token index through move number tokens or result tokens
            # NOTE: do NOT strip the trailing dots here — the U-15 side
            # marker check needs the original dot count. The previous
            # code stripped dots which made actual_dots empty for any
            # single-dot or triple-dot token, and the side-marker check
            # would then fire as a false positive.
            while tok_idx < len(movetext_tokens):
                raw_tok = movetext_tokens[tok_idx]
                # U-15 (2026-09-01): the previous pattern `(\.|\.\.)*` was
                # a Python regex footgun — alternation inside a `*` group
                # never extends beyond a single match, so group(2) was
                # always "." regardless of how many dots were in the
                # input. That made the wrong-side-marker check a no-op
                # (the actual and expected dots were always the same).
                # `\.+` captures the full dot run in one shot.
                num_m = re.match(r"^(\d+)(\.+)$", raw_tok)
                if num_m:
                    move_num = int(num_m.group(1))
                    if move_num != expected_fullmove:
                        syntax_warnings.append(
                            f"Move number mismatch: found '{movetext_tokens[tok_idx]}' but expected move {expected_fullmove}."
                        )
                    # U-15 (2026-09-01): also flag wrong-dot count. A black
                    # move (board.turn == BLACK at this point in the
                    # mainline) MUST use "..." (triple dot), not ".".
                    # Strict mode promotes the warning to a STRICT_PGN_ERROR;
                    # the final pass at the bottom raises on any
                    # syntax_warnings in strict mode.
                    expected_dots = "..." if curr_board.turn == chess.BLACK else "."
                    actual_dots = num_m.group(2) or ""
                    if actual_dots != expected_dots:
                        syntax_warnings.append(
                            f"Wrong side marker: found '{movetext_tokens[tok_idx]}' "
                            f"but expected '{expected_dots}' for the side to move."
                        )
                    tok_idx += 1
                    continue
                if raw_tok in ("1-0", "0-1", "1/2-1/2", "*") or re.match(r"^\$[0-9]+$", raw_tok):
                    tok_idx += 1
                    continue
                break

            if tok_idx < len(movetext_tokens):
                raw_tok = movetext_tokens[tok_idx].strip(".,;:!?")
                raw_tok_san = raw_tok.rstrip("!?")
                # Round-3 (further super deep): the old check compared the
                # raw SAN to canonical_san verbatim. That rejected the valid
                # PGN §8.1.4 promotion form 'e8Q' (no '=') because
                # canonical_san is 'e8=Q'. Strip the optional '=' from BOTH
                # sides so 'e8=Q' and 'e8Q' both compare equal — same
                # comparison the strict surface validator uses.
                raw_tok_promotionless = _strip_promotion_eq(raw_tok_san)
                canonical_promotionless = _strip_promotion_eq(canonical_san)
                if raw_tok_promotionless != canonical_promotionless and not re.fullmatch(
                    r"[a-h][1-8][a-h][1-8][qrbn]?", raw_tok_san.lower()
                ):
                    syntax_warnings.append(
                        f"Input SAN '{movetext_tokens[tok_idx]}' normalized to '{canonical_san}'"
                    )
                tok_idx += 1

            moves.append(move)
            curr_board.push(move)
            positions.append(curr_board.copy(stack=True))
            if curr_board.turn == chess.WHITE:
                expected_fullmove += 1

            if curr_board.is_repetition(5):
                reached_terminal = True
                auto_termination = "fivefold_repetition"
            else:
                rule_after = evaluate_rule_status(curr_board, history_complete="complete")
                if rule_after.terminal is not None:
                    reached_terminal = True
                    auto_termination = rule_after.terminal

        # Extract headers with TAG_PAIR_REGEX from header_section to handle escaped quotes and robust tag parsing
        # P2 (2026-09-02 ultra audit): the tag name MUST be canonicalized
        # before storage in tags_dict — otherwise [Variant "Standard"] and
        # [variant "Standard"] produced different downstream lookups (Variant
        # returned None from the second form). The metadata pipeline now
        # uses lowercase keys consistently. Downstream code reads both the
        # canonical-key form (lowercase, e.g. "variant") and falls back to
        # python-chess's game.headers for whatever it parsed.
        tags_dict: dict[str, str] = {}
        for tag_m in TAG_PAIR_REGEX.finditer(header_section):
            tag_k = tag_m.group(1).lower()
            tag_v = _unescape_pgn_tag_value(tag_m.group(2))
            if tag_k not in tags_dict and tag_v is not None and tag_v != "?":
                tags_dict[tag_k] = tag_v

        h = game.headers
        white_name = tags_dict.get("white") or (
            _unescape_pgn_tag_value(h.get("White"))
            if h.get("White") and h.get("White") != "?"
            else None
        )
        black_name = tags_dict.get("black") or (
            _unescape_pgn_tag_value(h.get("Black"))
            if h.get("Black") and h.get("Black") != "?"
            else None
        )
        event_name = tags_dict.get("event") or (
            _unescape_pgn_tag_value(h.get("Event"))
            if h.get("Event") and h.get("Event") != "?"
            else None
        )
        site_name = tags_dict.get("site") or (
            _unescape_pgn_tag_value(h.get("Site"))
            if h.get("Site") and h.get("Site") != "?"
            else None
        )
        round_name = tags_dict.get("round") or (
            _unescape_pgn_tag_value(h.get("Round"))
            if h.get("Round") and h.get("Round") != "?"
            else None
        )
        white_elo_val = tags_dict.get("whiteelo") or (
            h.get("WhiteElo") if h.get("WhiteElo") and h.get("WhiteElo") != "?" else None
        )
        black_elo_val = tags_dict.get("blackelo") or (
            h.get("BlackElo") if h.get("BlackElo") and h.get("BlackElo") != "?" else None
        )
        time_control_val = tags_dict.get("timecontrol") or (
            h.get("TimeControl") if h.get("TimeControl") and h.get("TimeControl") != "?" else None
        )
        # P2 (2026-09-02 ultra audit): TimeControl values like "? ",
        # " ?", " ? " were preserved verbatim on the wire — the strict
        # validator allowed them through, but downstream consumers saw
        # a different literal than the canonical sentinel. Strip
        # whitespace and normalize the unknown sentinel to None so the
        # exposed value is consistent across inputs.
        if time_control_val is not None:
            time_control_val = time_control_val.strip()
            if time_control_val == "?":
                time_control_val = None
        variant_val = tags_dict.get("variant") or (
            h.get("Variant") if h.get("Variant") and h.get("Variant") != "?" else None
        )
        date_val = (
            tags_dict.get("date") or tags_dict.get("utcdate") or h.get("Date") or h.get("UTCDate")
        )
        # Round-3 (further super deep): normalize empty / whitespace-only
        # Date tags to None — they all mean "no date" per PGN §7.1. The
        # audit found `[Date ""]` silently accepted (the truthy check above
        # fell through because Python treats "" as falsy) while `[Date " "]`
        # was rejected — inconsistent. Treat them identically.
        if date_val is not None and date_val.strip() in ("", "?", "????.??.??"):
            date_val = None

        result_movetext = _find_movetext_result(canonical_pgn)

        # Extract Result and Termination headers from header_section ONLY
        result_header_raw: str | None = None
        termination_header_val: str | None = None
        for tag_m in TAG_PAIR_REGEX.finditer(header_section):
            tag_k = tag_m.group(1).lower()
            tag_v = _unescape_pgn_tag_value(tag_m.group(2))
            if tag_k == "result" and result_header_raw is None:
                result_header_raw = tag_v
            elif tag_k == "termination" and termination_header_val is None:
                termination_header_val = tag_v

        metadata_warnings: list[str] = list(lexical_header_warnings)

        # R4-§B (2026-09-02 ultra audit round 4): the input was comment-only
        # (no moves after stripping comments + variations). Surface a clear
        # metadata warning in lenient mode so callers see the input was
        # non-empty but contained no moves. Strict mode does NOT raise on
        # this — a comment-only PGN with valid headers is not a metadata
        # inconsistency, so the warning would only confuse the strict
        # validator (which promotes every metadata_warning to a STRICT_PGN_
        # ERROR at the bottom of analyze_game).
        if getattr(game, "comment_only_input", False) and not strict:
            metadata_warnings.append(
                "Input PGN contained only comments (and optionally a result "
                "token) with no moves; returning an empty game."
            )

        # U-14 (2026-09-01): strict mode rejects malformed Date tags.
        # PGN Date is `YYYY.MM.DD`; anything else (e.g. "2026.99.99",
        # "hello", "not.a.date") is a non-canonical value. The legacy
        # behavior silently accepted these and even echoed them back
        # to clients, which is the audit's P2 finding. Strict mode
        # records a metadata_warning; the final strict pass at the
        # bottom of analyze_game raises STRICT_PGN_ERROR on any
        # metadata_warning, so the malformed Date is rejected. The
        # regex is tighter than just `\d{4}\.\d{2}\.\d{2}` — it
        # enforces month 01-12 and day 01-31 so "2026.99.99" is
        # correctly rejected.
        #
        # P2/P3 (2026-09-02 ultra audit): the regex above only catches
        # range issues; it still accepts impossible calendar dates like
        # 2023.02.29, 2026.04.31, 2026.02.31. After the structural
        # check, run the date through Python's `datetime.date`
        # constructor — that raises ValueError for any day that doesn't
        # exist in the given month/year, including the Feb 29 leap-year
        # rule (no Apr 31, no Sep 31, no Feb 30, etc.). In strict mode
        # the impossible date is a metadata_warning that promotes to a
        # STRICT_PGN_ERROR; in lenient mode it is also a warning so
        # downstream callers see that the metadata is suspect, even
        # though parsing continues.
        if date_val is not None:
            # Round-3 (further super deep): the old regex required ALL
            # three components to be concrete digits, which silently
            # rejected the per-component wildcards PGN §7.1 allows
            # (????.09.02, 2026.09.??, 2026.??.02, 2026.??.??). At the
            # same time it accepted ???? and '?' because of the truthy
            # fallback above, which was inconsistent. Validate each
            # component independently:
            #   - YYYY  |  ????   (year)
            #   - MM    |  ??     (month)
            #   - DD    |  ??     (day)
            # Calendar semantics only run when all three components are
            # concrete (Apr 31 / Sep 31 / Feb 29 in non-leap year / etc.).
            _date_err = _validate_pgn_date(date_val)
            if _date_err is not None:
                if strict:
                    metadata_warnings.append(_date_err)
                else:
                    metadata_warnings.append(_date_err)

        CANONICAL_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
        if result_header_raw is not None and result_header_raw != "?":
            if result_header_raw in CANONICAL_RESULTS:
                result_header = result_header_raw
            else:
                metadata_warnings.append(
                    f"Invalid Result header tag '{result_header_raw}'; expected 1-0, 0-1, 1/2-1/2, or *."
                )
                result_header = None
        else:
            result_header = None

        if white_elo_val is not None and white_elo_val != "-":
            if not (white_elo_val.isdigit() and 0 <= int(white_elo_val) <= 4000):
                metadata_warnings.append(
                    f"Invalid WhiteElo header tag '{white_elo_val}'; expected numeric integer rating."
                )
        if black_elo_val is not None and black_elo_val != "-":
            if not (black_elo_val.isdigit() and 0 <= int(black_elo_val) <= 4000):
                metadata_warnings.append(
                    f"Invalid BlackElo header tag '{black_elo_val}'; expected numeric integer rating."
                )
        if time_control_val is not None and not _is_valid_pgn_time_control(time_control_val):
            metadata_warnings.append(f"Invalid TimeControl header tag '{time_control_val}'.")

        eco_header = tags_dict.get("eco") or h.get("ECO")
        opening_header = tags_dict.get("opening") or h.get("Opening")

        # Detect duplicate headers in header block only. P2 (2026-09-02
        # ultra audit): the tag name MUST be canonicalized (lowercased)
        # before duplicate counting — otherwise `[Result "*"]` and
        # `[result "1-0"]` were treated as different tags despite
        # python-chess treating them as the same semantic tag. We also
        # surface value conflicts on the canonical Result tag because
        # the audit flagged that competing values were silently merged.
        tag_counts: dict[str, int] = {}
        tag_values_by_canonical: dict[str, list[str]] = {}
        for tag_m in TAG_PAIR_REGEX.finditer(header_section):
            tag_name_raw = tag_m.group(1)
            tag_value = _unescape_pgn_tag_value(tag_m.group(2))
            tag_name_canonical = tag_name_raw.lower()
            tag_counts[tag_name_canonical] = tag_counts.get(tag_name_canonical, 0) + 1
            if tag_value is not None:
                tag_values_by_canonical.setdefault(tag_name_canonical, []).append(tag_value)
        for tag_name, count in tag_counts.items():
            if count > 1:
                metadata_warnings.append(
                    f"Duplicate PGN tag '[{tag_name}]' detected ({count} occurrences); using canonical tag value."
                )
        # Surface value conflicts on Result / Variant explicitly so the
        # audit's "duplicate detection is not consistently
        # case-insensitive" finding is closed.
        for canonical_name in ("result", "variant"):
            values = tag_values_by_canonical.get(canonical_name) or []
            if len(values) >= 2 and any(v != values[0] for v in values[1:]):
                metadata_warnings.append(
                    f"Conflicting values for PGN tag '{canonical_name}': {values!r}; "
                    f"using the first declared value."
                )

        # Validate SetUp vs FEN tags
        setup_header = h.get("SetUp")
        fen_header = h.get("FEN")
        # P2 (2026-09-02 ultra audit): SetUp tag value domain must be
        # validated. The legacy code special-cased the canonical "1"
        # string and silently accepted every other value (including
        # non-canonical "2", empty string, "true", "false", "01", "-1",
        # and " ") — which the audit showed meant strict mode never
        # rejected malformed SetUp values. Strict mode now rejects any
        # value outside the canonical {"0", "1"} set; lenient mode
        # accepts "1" only (treating everything else as the implicit
        # "SetUp absent" case, with a warning).
        if setup_header is not None:
            if setup_header not in ("0", "1"):
                if strict:
                    metadata_warnings.append(
                        f"Invalid SetUp tag value '{setup_header}': must be exactly '0' or '1'."
                    )
                else:
                    # Lenient: warn but don't reject — preserve
                    # backward compatibility for slightly-malformed
                    # inputs that the caller may not be able to fix.
                    metadata_warnings.append(
                        f"Non-canonical SetUp tag value '{setup_header}': expected '0' or '1'."
                    )
        if setup_header == "1" and not fen_header:
            metadata_warnings.append(
                '[SetUp "1"] tag provided without FEN tag; defaulting to standard starting position.'
            )
        elif fen_header and setup_header != "1":
            metadata_warnings.append(
                'FEN tag provided without [SetUp "1"]; custom position loaded.'
            )

        if game.errors:
            # P2 (2026-09-02 ultra audit): board-detected checkmate path
            # undercounted trailing plies. The legacy code added
            # `len(game.errors)` which is the number of distinct
            # python-chess exceptions raised while parsing — usually 1
            # per movetext that breaks at the first illegal move — rather
            # than the actual number of trailing ply tokens the user
            # wrote. The explicit result-token branch already counted
            # all SAN tokens after the result marker; we now mirror that
            # behavior here, counting remaining movetext tokens after the
            # last successfully executed ply.
            consumed_plies = len(moves)
            tokens_in_movetext = re.findall(
                r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*|O-O-O[\+#\?!]*|O-O[\+#\?!]*)\b",
                cleaned_movetext,
            )
            total_ply_tokens = len(tokens_in_movetext)
            trailing_from_errors = max(0, total_ply_tokens - consumed_plies)
            if trailing_from_errors > 0:
                ignored_trailing_plies += trailing_from_errors
            else:
                # No recoverable move tokens found in the trailing
                # movetext. Fall back to the legacy game.errors
                # count so we never underreport below zero.
                ignored_trailing_plies = max(ignored_trailing_plies, len(game.errors))

        raw_pgn_clean = _strip_pgn_escape_lines(canonical_pgn)
        raw_truncated = _truncate_movetext_at_result(raw_pgn_clean)
        if len(raw_truncated) < len(raw_pgn_clean):
            after_part = raw_pgn_clean[len(raw_truncated) :]
            after_clean = re.sub(r"\{[^{}]*\}", " ", after_part)
            after_clean = re.sub(r";[^\r\n]*", " ", after_clean)
            tokens_after = re.findall(
                r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*|O-O-O[\+#\?!]*|O-O[\+#\?!]*)\b",
                after_clean,
            )
            if tokens_after:
                ignored_trailing_plies += len(tokens_after)

        if ignored_trailing_plies > 0:
            ply_word = "ply" if ignored_trailing_plies == 1 else "plies"
            metadata_warnings.append(
                f"Movetext contained moves after game termination; ignored {ignored_trailing_plies} trailing {ply_word}."
            )

        # Validate game result consistency against final board state
        final_board = positions[-1]
        # positions[] was reconstructed from the complete PGN mainline, so repetition
        # history is authoritative here. Do not downgrade a previously detected fivefold
        # repetition to generic game_over during final result reconciliation.
        rule_final = evaluate_rule_status(final_board, history_complete="complete")
        result_board: str | None = None
        if rule_final.terminal is not None:
            if rule_final.terminal == "checkmate":
                result_board = "1-0" if final_board.turn == chess.BLACK else "0-1"
                auto_termination = "checkmate"
            else:
                result_board = "1/2-1/2"
                auto_termination = rule_final.terminal

        if result_board is not None:
            result_val = result_board
            if result_header and result_header not in ("*", "?") and result_header != result_board:
                metadata_warnings.append(
                    f"Result header '{result_header}' disagrees with board outcome '{result_board}'; using board outcome."
                )
            if (
                result_movetext
                and result_movetext not in ("*", "?")
                and result_movetext != result_board
            ):
                metadata_warnings.append(
                    f"Movetext result '{result_movetext}' disagrees with board outcome '{result_board}'; using board outcome."
                )
        else:
            if result_header_raw and result_movetext and result_header_raw != result_movetext:
                metadata_warnings.append(
                    f"Result header '{result_header_raw}' disagrees with movetext result '{result_movetext}'."
                )

            if result_header and result_header not in ("*", "?"):
                result_val = result_header
            elif result_movetext and result_movetext not in ("*", "?"):
                result_val = result_movetext
            else:
                result_val = result_header or result_movetext or "*"

        # Infer only from explicit winner/loser grammar.
        if result_val == "*" or result_val is None:
            inferred = _infer_result_from_termination(termination_header_val)
            if inferred is not None:
                result_val = inferred

        # Validate Resignation & Time Forfeit & Rules Infraction under FIDE mating possibility rules
        result_val, mate_warnings = validate_mating_possibility(
            final_board, result_val, termination_header_val
        )
        metadata_warnings.extend(mate_warnings)

        # U-14 (2026-09-01): strict-mode Termination validation. The
        # legacy code only flagged Termination when it contradicted
        # the result; arbitrary strings like "foobar" were stored raw
        # without rejection. Strict mode now requires the Termination
        # to either be blank, a known FIDE value, or fall through the
        # normalize_termination mapper; anything else is a metadata
        # warning that strict pass will reject.
        if strict and termination_header_val:
            norm_term = normalize_termination(termination_header_val)
            # If normalize_termination returns None AND the string is
            # not blank/known, it's an unrecognised value.
            if norm_term is None and termination_header_val.strip() not in (
                "",
                "Normal",
                "Time forfeit",
                "Rules infraction",
                "Abandoned",
                "Unterminated",
            ):
                # If normalize_termination returned None but
                # contains a known FIDE term in lowercase, accept.
                lower = termination_header_val.strip().lower()
                if not any(
                    kw in lower
                    for kw in (
                        "resign",
                        "checkmate",
                        "stalemate",
                        "time",
                        "abandon",
                        "rule",
                        "draw",
                        "repetition",
                        "insufficient",
                        "50-move",
                        "75-move",
                    )
                ):
                    metadata_warnings.append(
                        f"Unrecognised Termination tag '{termination_header_val}'."
                    )

        # Check for contradictory metadata
        if termination_header_val:
            norm_term = normalize_termination(termination_header_val)
            if norm_term in (
                "stalemate",
                "insufficient_material",
                "fifty_moves",
                "seventyfive_moves",
                "threefold_repetition",
                "fivefold_repetition",
                "dead_position",
            ) and result_val in ("1-0", "0-1"):
                metadata_warnings.append(
                    f"Contradictory PGN metadata: Termination '{termination_header_val}' contradicts Result '{result_val}'."
                )
            elif norm_term == "checkmate" and result_val in ("1/2-1/2", "*"):
                metadata_warnings.append(
                    f"Contradictory PGN metadata: Termination '{termination_header_val}' contradicts Result '{result_val}'."
                )
            elif norm_term == "unterminated" and result_val in (
                "1-0",
                "0-1",
                "1/2-1/2",
            ):
                metadata_warnings.append(
                    f"Contradictory PGN metadata: Termination '{termination_header_val}' contradicts Result '{result_val}'."
                )

        # Premature draw agreement warning
        if (
            termination_header_val
            and "agreement" in termination_header_val.lower()
            and len(moves) < 2
        ):
            metadata_warnings.append(
                "Draw agreement declared before both players completed at least one move."
            )

        if auto_termination is not None:
            termination_val = auto_termination
            if termination_header_val:
                norm_term_hdr = normalize_termination(termination_header_val)
                if norm_term_hdr == "normal":
                    pass
                else:
                    is_concurrent_draw = norm_term_hdr in (
                        "stalemate",
                        "seventyfive_moves",
                        "fivefold_repetition",
                        "insufficient_material",
                        "fifty_moves",
                        "threefold_repetition",
                        "dead_position",
                    ) and (
                        (norm_term_hdr == "threefold_repetition" and final_board.is_repetition(3))
                        or (
                            norm_term_hdr == "fifty_moves"
                            and (final_board.is_fifty_moves() or final_board.halfmove_clock >= 100)
                        )
                        or (
                            norm_term_hdr == "fivefold_repetition"
                            and final_board.is_fivefold_repetition()
                        )
                        or (
                            norm_term_hdr == "seventyfive_moves"
                            and final_board.is_seventyfive_moves()
                        )
                        or (
                            norm_term_hdr == "insufficient_material"
                            and final_board.is_insufficient_material()
                        )
                        or (
                            norm_term_hdr == "dead_position"
                            and is_locked_dead_position(final_board)
                        )
                        or (norm_term_hdr == "stalemate" and final_board.is_stalemate())
                    )
                    if norm_term_hdr != auto_termination and not is_concurrent_draw:
                        metadata_warnings.append(
                            f"Termination header '{termination_header_val}' disagrees with board outcome '{auto_termination}'; using board outcome."
                        )
        elif termination_header_val:
            norm_term_hdr = normalize_termination(termination_header_val)
            if norm_term_hdr == "normal":
                termination_val = "normal"
            elif norm_term_hdr in ("checkmate", "stalemate"):
                metadata_warnings.append(
                    f"Termination header '{termination_header_val}' contradicts board state (position is not {norm_term_hdr})."
                )
                termination_val = None
            elif norm_term_hdr == "threefold_repetition":
                if not final_board.is_repetition(3):
                    metadata_warnings.append(
                        f"Termination header '{termination_header_val}' contradicts board state (position is not threefold_repetition)."
                    )
                    termination_val = None
                else:
                    termination_val = "threefold_repetition"
            elif norm_term_hdr == "fifty_moves":
                if not final_board.is_fifty_moves() and final_board.halfmove_clock < 100:
                    metadata_warnings.append(
                        f"Termination header '{termination_header_val}' contradicts board state (position is not fifty_moves)."
                    )
                    termination_val = None
                else:
                    termination_val = "fifty_moves"
            elif norm_term_hdr in (
                "insufficient_material",
                "seventyfive_moves",
                "fivefold_repetition",
                "dead_position",
            ):
                metadata_warnings.append(
                    f"Termination header '{termination_header_val}' contradicts board state (position is not {norm_term_hdr})."
                )
                termination_val = None
            else:
                termination_val = norm_term_hdr
        else:
            termination_val = None

        if strict and not moves:
            if syntax_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains syntax normalization or move number mismatch: {syntax_warnings[0]}"
                )
            if metadata_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains metadata inconsistency: {metadata_warnings[0]}"
                )

        is_standard_start = game.board().fen() == chess.STARTING_FEN

        pool = await _get_analyzer_pool(ctx)
        engine_name_str = getattr(pool, "engine_version", getattr(pool, "name", "Stockfish"))

        if not moves:
            detected_opening, detected_eco = (
                lookup_opening([])[:2] if is_standard_start else (None, None)
            )
            return GameAnalysisResult(
                total_plies=0,
                white_accuracy=None,
                black_accuracy=None,
                white_acpl=None,
                black_acpl=None,
                white_raw_acpl=None,
                black_raw_acpl=None,
                white_effective_acpl=None,
                black_effective_acpl=None,
                white_average_effective_loss=None,
                black_average_effective_loss=None,
                white_blunders=0,
                white_mistakes=0,
                white_inaccuracies=0,
                black_blunders=0,
                black_mistakes=0,
                black_inaccuracies=0,
                turning_points=[],
                white=white_name,
                black=black_name,
                event=event_name,
                site=site_name,
                date=date_val,
                round=round_name,
                result=result_val or result_header or "*",
                result_header=result_header,
                result_header_raw=result_header_raw,
                result_movetext=result_movetext,
                result_inferred=result_board,
                white_elo=white_elo_val,
                black_elo=black_elo_val,
                time_control=time_control_val,
                variant=variant_val,
                eco=detected_eco or eco_header,
                opening=detected_opening or opening_header,
                opening_header=opening_header,
                eco_header=eco_header,
                metadata_warnings=metadata_warnings,
                syntax_warnings=syntax_warnings,
                termination=termination_val,
                termination_header=termination_header_val,
                requested_depth=raw_requested_depth,
                searched_depth=0,
                engine="Stockfish",
                engine_version=engine_name_str,
                **_build_identity(pool),
                accuracy_method="win_probability_logistic",
                mate_penalty_policy="1000_cp_mate_transition",
            )

        eval_pairs = await _gather_evaluate_positions_bounded(
            positions,
            depth,
            pool,
            requested_depth=raw_requested_depth,
            history_complete="complete",
        )
        evals: list[MCPEval] = [ep[0] for ep in eval_pairs]
        all_cached = all(ep[1] for ep in eval_pairs)

        (
            white_acc,
            black_acc,
            white_acpl,
            black_acpl,
            white_raw_acpl,
            black_raw_acpl,
            white_avg_eff,
            black_avg_eff,
            (white_blunders, white_mistakes, white_inaccuracies),
            (black_blunders, black_mistakes, black_inaccuracies),
            top_turning_points,
        ) = _compute_game_metrics(positions, moves, evals)

        await metrics.record("analyze_game", (time.time() - t0) * 1000, cache_hit=all_cached)

        uci_moves = [m.uci() for m in moves]
        if is_standard_start:
            detected_opening, detected_eco, _ = lookup_opening(uci_moves)
        else:
            detected_opening, detected_eco = None, None

        final_opening = detected_opening or opening_header
        final_eco = detected_eco or eco_header

        if detected_opening and opening_header:
            det_clean = detected_opening.strip().lower()
            hdr_clean = opening_header.strip().lower()
            det_base = det_clean.split(":")[0].strip()
            hdr_base = hdr_clean.split(":")[0].strip()
            is_parent_child = (
                det_clean.startswith(hdr_clean)
                or hdr_clean.startswith(det_clean)
                or det_base == hdr_base
            )
            if not is_parent_child:
                metadata_warnings.append(
                    f"Opening header '{opening_header}' disagrees with detected opening '{detected_opening}'"
                )
        if (
            detected_eco
            and eco_header
            and detected_eco.strip().upper() != eco_header.strip().upper()
        ):
            metadata_warnings.append(
                f"ECO header '{eco_header}' disagrees with detected ECO '{detected_eco}'"
            )

        if strict:
            if syntax_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains syntax normalization or move number mismatch: {syntax_warnings[0]}"
                )
            if metadata_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains metadata inconsistency: {metadata_warnings[0]}"
                )

        return GameAnalysisResult(
            total_plies=len(moves),
            white_accuracy=white_acc,
            black_accuracy=black_acc,
            white_acpl=white_acpl,
            black_acpl=black_acpl,
            white_raw_acpl=white_raw_acpl,
            black_raw_acpl=black_raw_acpl,
            white_effective_acpl=white_avg_eff,
            black_effective_acpl=black_avg_eff,
            white_average_effective_loss=white_avg_eff,
            black_average_effective_loss=black_avg_eff,
            white_blunders=white_blunders,
            white_mistakes=white_mistakes,
            white_inaccuracies=white_inaccuracies,
            black_blunders=black_blunders,
            black_mistakes=black_mistakes,
            black_inaccuracies=black_inaccuracies,
            turning_points=top_turning_points,
            white=white_name,
            black=black_name,
            event=event_name,
            site=site_name,
            date=date_val,
            round=round_name,
            result=result_val,
            result_header=result_header,
            result_header_raw=result_header_raw,
            result_movetext=result_movetext,
            result_inferred=result_board
            or (
                result_val
                if (result_header_raw in ("*", None) and result_val in ("1-0", "0-1", "1/2-1/2"))
                else None
            ),
            white_elo=white_elo_val,
            black_elo=black_elo_val,
            time_control=time_control_val,
            variant=variant_val,
            eco=final_eco,
            opening=final_opening,
            opening_header=opening_header,
            eco_header=eco_header,
            metadata_warnings=metadata_warnings,
            syntax_warnings=syntax_warnings,
            termination=termination_val,
            termination_header=termination_header_val,
            requested_depth=raw_requested_depth,
            searched_depth=depth,
            engine="Stockfish",
            engine_version=engine_name_str,
            **_build_identity(pool),
            accuracy_method="win_probability_logistic",
            mate_penalty_policy="1000_cp_mate_transition",
        )
    except ToolError:
        await metrics.record("analyze_game", (time.time() - t0) * 1000, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record("analyze_game", (time.time() - t0) * 1000, is_error=True)
        msg = str(exc)
        code = error_code_for(msg)
        raise _tool_error(code=code, message=msg, tool="analyze_game", input=pgn[:100]) from exc
    except Exception as exc:
        await metrics.record("analyze_game", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="analyze_game") from exc


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
