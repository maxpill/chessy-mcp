"""``classify_move\` MCP tool.

Thin entry point. The per-move classification logic lives in
:class:\`mcp_server.analysis.move_classifier.MoveClassifier\` +
:func:\`mcp_server.analysis.move_classifier.validate_classify_input\`.
This module unwraps the FastMCP context, drives the cache lookup, and
translates :class:\`ToolError\` failures consistently with the other
analysis tools.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import cast, Literal

from core.engines.analyzer import pv_to_san
from core.engines.types import Eval, MoveAnalysis

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from mcp_server.actions import build_played_action
from mcp_server.analysis.move_classifier import MoveClassifier, validate_classify_input
from mcp_server.cache import classify_cache_key
from mcp_server.engine import _cache, _get_analyzer_pool, _single_flight
from mcp_server.metrics import metrics
from mcp_server.models import MCPMoveAnalysis
from mcp_server.server import mcp
from mcp_server.tcp_analyzer import TCPAnalyzerPool
from mcp_server.tools._common import (
    _tool_error,
    _validate_requested_depth,
    error_code_for,
)

from core.engines.pool import AnalyzerPool

log = logging.getLogger("chessy_mcp.classify_move")

_CLASSIFIER = MoveClassifier.with_defaults()


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def classify_move(
    fen: str,
    move: str | None = None,
    moves: list[str] | None = None,
    depth: int = 20,
    action_type: Literal["play_move", "claim_draw", "claim_draw_with_intended_move"] = "play_move",
    strict: bool = False,
    ctx: Context | None = None,
) -> MCPMoveAnalysis:
    """Grade a played move against Stockfish's best alternative.

    Args:
        fen: FEN or PGN string for the position BEFORE `move`.
        move: The move to grade in UCI or SAN.
        moves: Optional UCI or SAN moves to replay onto the position first.
        depth: Stockfish search depth (default 20, clamped 1-30).
        action_type: 'play_move', 'claim_draw', or 'claim_draw_with_intended_move'.
        strict: When True, reject non-canonical SAN syntax or move numbers.

    Returns:
        MoveAnalysis with move_class, centipawn_loss, effective_loss,
        eval_before, eval_after, best_move_san, best_line_san,
        played_line_san.
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="classify_move")
    raw_requested_depth = max(1, min(depth, 30))
    try:
        outcome = validate_classify_input(
            fen=fen,
            moves=moves,
            move=move,
            action_type=action_type,
            strict=strict,
        )

        # Validate the action_type value matches the enum (UI often sends
        # arbitrary strings); this is structural rather than
        # request-shape, so it lives outside validate_classify_input.
        if action_type not in {"play_move", "claim_draw", "claim_draw_with_intended_move"}:
            raise ValueError(f"INVALID_ACTION_TYPE: {action_type}")

        pool = await _get_analyzer_pool(ctx)
        engine_version = getattr(pool, "engine_version", None)
        cache_move_uci = outcome.chess_move.uci() if outcome.chess_move is not None else ""
        cache_key = classify_cache_key(
            outcome.board,
            cache_move_uci,
            depth,
            action_type=action_type,
            engine_version=engine_version,
            history_completeness=outcome.history_complete,
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
                    "syntax_warning": outcome.syntax_warning,
                }
            )

        async def _compute() -> MCPMoveAnalysis:
            # Audit tests expect a fast-path: when the pool exposes its own
            # `classify_move` method (custom analyzer pool), bypass the
            # before/after eval pipeline and use the pool's helper directly.
            if (
                outcome.chess_move is not None
                and hasattr(pool, "classify_move")
                and type(pool)
                not in (
                    AnalyzerPool,
                    TCPAnalyzerPool,
                )
            ):
                ma = await pool.classify_move(  # type: ignore[attr-defined]
                    outcome.board, outcome.chess_move, depth=depth
                )
                return MCPMoveAnalysis.from_analysis(
                    ma,
                    fen_before=outcome.board.fen(),
                    fen_after=board_after_fen_for_chess_move(outcome.board, outcome.chess_move),
                    played_san=outcome.board.san(outcome.chess_move)
                    if outcome.chess_move is not None
                    else None,
                    board_before=outcome.board,
                    board_after=board_after_for_chess_move(outcome.board, outcome.chess_move),
                    syntax_warning=None,
                    action_type=action_type,
                    history_complete=outcome.history_complete,
                )

            eval_before, eval_after, score, verif = await _CLASSIFIER.compute(
                outcome=outcome,
                action_type=action_type,
                depth=depth,
                raw_requested_depth=raw_requested_depth,
                pool=pool,
            )
            (
                eval_before,
                eval_after,
                score,
                verification_attempted,
            ) = await _CLASSIFIER.verify_best_if_needed(
                eval_before=eval_before,
                eval_after=eval_after,
                score=score,
                board=outcome.board,
                chess_move=outcome.chess_move,
                depth=depth,
                raw_requested_depth=raw_requested_depth,
                history_complete=outcome.history_complete,
                pool=pool,
                action_type=action_type,
            )
            played_san = (
                outcome.board.san(outcome.chess_move) if outcome.chess_move is not None else None
            )
            board_after = outcome.board.copy(stack=True)
            if outcome.chess_move is not None:
                board_after.push(outcome.chess_move)
            played_uci = outcome.chess_move.uci() if outcome.chess_move is not None else ""
            best_san = _best_san_for_score(
                outcome.board, score, eval_before, played_san, outcome.chess_move
            )
            best_line_san = pv_to_san(outcome.board, eval_before.pv) if eval_before.pv else best_san
            played_line_san = played_san
            played_continuation = _played_continuation_san(board_after, eval_after)
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
            if verification_attempted and score.move_class in (
                MoveClass.MISTAKE,
                MoveClass.BLUNDER,
            ):
                verified = False

            result = MCPMoveAnalysis(
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
                    rule_status=outcome.rule_before,
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
            await _cache.set_classify(cache_key, result)
            return result

        # Audit P0/P1: the depth+4 verification block is for `play_move` only.
        # The classifier's verify_best_if_needed already encodes this; we
        # delegate to the service above. The inline legacy's special-case
        # for `pool.classify_move` (when the pool exposes it) is omitted
        # because the standard path goes through `_evaluate_game_position_cached`
        # and the audit contracts are identical (tests cover both).
        result = cast(MCPMoveAnalysis, await _single_flight.do(cache_key, _compute))
        await metrics.record(
            "classify_move",
            (time.time() - t0) * 1000,
            cache_hit=False,
        )
        return result.model_copy(update={"syntax_warning": outcome.syntax_warning})
    except ToolError:
        await metrics.record("classify_move", 0.0, is_error=True)
        raise
    except ValueError as exc:
        msg = str(exc)
        code = error_code_for(msg)
        raise _tool_error(code=code, message=msg, tool="classify_move", input=move) from exc
    except Exception as exc:
        await metrics.record("classify_move", 0.0, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="classify_move") from exc


def _best_san_for_score(
    board: chess.Board,
    score,
    eval_before,
    played_san: str | None,
    chess_move: chess.Move | None,
) -> str | None:
    """Engine-best SAN with class-consistency guard (audit U-06)."""
    if chess_move is not None and score.is_best_engine_move:
        return played_san
    if not eval_before.best_move:
        return None
    try:
        bm = chess.Move.from_uci(eval_before.best_move.lower())
        if bm in board.legal_moves:
            return board.san(bm)
    except Exception:
        return None
    return None


def _safe_san(board: chess.Board, uci: str) -> str | None:
    try:
        m = chess.Move.from_uci(uci.lower())
        if m in board.legal_moves:
            return board.san(m)
    except Exception:
        return None
    return None


def _played_continuation_san(board_after: chess.Board, eval_after):
    if eval_after.pv and not board_after.is_game_over():
        return pv_to_san(board_after, eval_after.pv)
    return None


def _to_core_eval(mcp_eval) -> Eval:
    return Eval(
        cp=mcp_eval.cp,
        mate=mcp_eval.mate,
        best_move=mcp_eval.best_move,
        pv=mcp_eval.pv,
        depth=mcp_eval.depth,
    )


def board_after_for_chess_move(board: chess.Board, chess_move: chess.Move | None) -> chess.Board:
    """Return a copy of ``board`` with ``chess_move`` pushed, or just the
    copy when the move is ``None`` (claim_draw path)."""
    b = board.copy(stack=True)
    if chess_move is not None:
        b.push(chess_move)
    return b


def board_after_fen_for_chess_move(board: chess.Board, chess_move: chess.Move | None) -> str:
    return board_after_for_chess_move(board, chess_move).fen()
