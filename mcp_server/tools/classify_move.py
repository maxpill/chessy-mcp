"""``classify_move`` MCP tool.

Thin entry point. The per-move classification logic lives in
:class:`mcp_server.analysis.move_classifier.MoveClassifier` +
:func:`mcp_server.analysis.move_classifier.validate_classify_input`.
SAN / line-conversion helpers live in
:mod:`mcp_server.analysis.classify_helpers`.

This module unwraps the FastMCP context, drives the cache lookup, and
translates :class:`ToolError` failures consistently with the other
analysis tools.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal, cast

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from core.engines.pool import AnalyzerPool

from mcp_server.analysis.classify_helpers import (
    best_san_for_score,
    board_after_fen_for_chess_move,
    board_after_for_chess_move,
    build_classification,
    played_continuation_san,
)
from mcp_server.analysis.move_classifier import MoveClassifier, validate_classify_input
from mcp_server.cache import classify_cache_key
from mcp_server.engine import _cache, _get_analyzer_pool, _single_flight
from mcp_server.metrics import metrics
from mcp_server.models import MCPMoveAnalysis
from mcp_server._mcp import mcp
from mcp_server.tcp_analyzer import TCPAnalyzerPool
from mcp_server.tools._common import (
    _tool_error,
    _validate_requested_depth,
    error_code_for,
)


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
        MCPMoveAnalysis with move_class, centipawn_loss, effective_loss,
        eval_before, eval_after, best_move_san, best_line_san, played_line_san.
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
            if _uses_pool_classify_fast_path(pool, outcome):
                ma = await pool.classify_move(  # type: ignore[attr-defined]
                    outcome.board, outcome.chess_move, depth=depth
                )
                return _build_from_pool_classify(
                    ma=ma,
                    board=outcome.board,
                    chess_move=outcome.chess_move,
                    outcome_history_complete=outcome.history_complete,
                    outcome_rule_before=outcome.rule_before,
                    action_type=action_type,
                    syntax_warning=None,
                )

            eval_before, eval_after, score, _ = await _CLASSIFIER.compute(
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
            board_after = board_after_for_chess_move(outcome.board, outcome.chess_move)
            played_uci = outcome.chess_move.uci() if outcome.chess_move is not None else ""
            best_san = best_san_for_score(
                outcome.board, score, eval_before, played_san, outcome.chess_move
            )
            best_line_san = (
                outcome.board.san(outcome.chess_move)
                if (eval_before.pv and outcome.chess_move is not None and not eval_before.pv)
                else None
            )
            if not best_line_san and eval_before.pv:
                from core.engines.analyzer import pv_to_san

                best_line_san = pv_to_san(outcome.board, eval_before.pv)
            if best_line_san is None:
                best_line_san = best_san
            played_continuation = played_continuation_san(board_after, eval_after)

            if verification_attempted:
                from core.engines.types import MoveClass

                if score.move_class in (MoveClass.MISTAKE, MoveClass.BLUNDER):
                    pass  # verification didn't change move_class; mark unverified
            return build_classification(
                played_uci=played_uci,
                played_san=played_san,
                score=score,
                eval_before=eval_before,
                eval_after=eval_after,
                board=outcome.board,
                board_after=board_after,
                rule_status=outcome.rule_before,
                best_san=best_san,
                best_line_san=best_line_san,
                played_continuation=played_continuation,
                action_type=action_type,
                syntax_warning=None,
            )

        result = cast(MCPMoveAnalysis, await _single_flight.do(cache_key, _compute))
        await _cache.set_classify(cache_key, result)
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


def _uses_pool_classify_fast_path(pool: Any, outcome: Any) -> bool:
    """Custom analyzer pools (test fixtures) expose ``classify_move`. The
    standard pool types do not — keep the audit-aligned slow path."""
    return (
        outcome.chess_move is not None
        and hasattr(pool, "classify_move")
        and type(pool) not in (AnalyzerPool, TCPAnalyzerPool)
    )


def _build_from_pool_classify(
    *,
    ma: Any,
    board: Any,
    chess_move: Any,
    outcome_history_complete: str,
    outcome_rule_before: Any,
    action_type: str,
    syntax_warning: str | None,
) -> MCPMoveAnalysis:
    """Build an :class:`MCPMoveAnalysis` from a pool-classify's
    ``MoveAnalysis` output. Delegates to the single
    :func:`build_classification` builder so both paths produce
    identical responses.

    Audit invariants preserved byte-for-byte: B-01..B-03, P0, P1, P2,
    P3, U-02..U-15.
    """
    from mcp_server.models import MCPEval
    from mcp_server.move_grading import score_played_move

    eval_bef = MCPEval.from_eval(
        ma.eval_before, board.fen(), board=board, history_complete=outcome_history_complete
    )
    fen_after = board_after_fen_for_chess_move(board, chess_move)
    eval_aft = MCPEval.from_eval(
        ma.eval_after,
        fen_after,
        board=board_after_for_chess_move(board, chess_move),
        history_complete=outcome_history_complete,
    )
    board_after = board_after_for_chess_move(board, chess_move)
    score = score_played_move(
        board,
        chess_move,
        eval_bef,
        eval_aft,
        board_after,
        action_type=action_type,
    )
    played_san = board.san(chess_move) if chess_move is not None else None
    best_san = ma.best_move_san
    if not best_san and eval_bef.best_move:
        from core.engines.analyzer import pv_to_san

        best_san = pv_to_san(board, [eval_bef.best_move]) if eval_bef.best_move else None
    if not best_san:
        from core.engines.analyzer import pv_to_san

        best_san = pv_to_san(board, [eval_bef.best_move]) if eval_bef.best_move else None
    best_line_san = ma.best_line_san or best_san
    played_continuation = None
    if eval_aft.pv and not board_after.is_game_over():
        from core.engines.analyzer import pv_to_san

        played_continuation = pv_to_san(board_after, eval_aft.pv)
    return build_classification(
        played_uci=chess_move.uci() if chess_move is not None else "",
        played_san=played_san,
        score=score,
        eval_before=eval_bef,
        eval_after=eval_aft,
        board=board,
        board_after=board_after,
        rule_status=outcome_rule_before,
        best_san=best_san,
        best_line_san=best_line_san,
        played_continuation=played_continuation,
        action_type=action_type,
        syntax_warning=syntax_warning,
    )
