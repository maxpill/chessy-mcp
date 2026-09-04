"""SAN / line-conversion helpers for the ``classify_move` tool.

Extracted from :mod:`mcp_server.tools.classify_move` so the tool
entry point stays focused on FastMCP plumbing (cache lookup, single-
flight, error translation). All helpers here are pure functions over
``chess.Board` + ``MCPEval`.
"""

from __future__ import annotations

from typing import Any

import chess
from core.engines.analyzer import pv_to_san
from core.engines.types import Eval

from mcp_server.models import MCPMoveAnalysis


def best_san_for_score(
    board: chess.Board,
    score: Any,
    eval_before: Any,
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


def safe_san(board: chess.Board, uci: str) -> str | None:
    """Best-effort ``san()` for a UCI string; never raises."""
    try:
        m = chess.Move.from_uci(uci.lower())
        if m in board.legal_moves:
            return board.san(m)
    except Exception:
        return None
    return None


def played_continuation_san(board_after: chess.Board, eval_after: Any) -> str | None:
    """Render the engine's post-state PV as SAN, or ``None` if the position
    is already terminal."""
    if eval_after.pv and not board_after.is_game_over():
        return pv_to_san(board_after, eval_after.pv)
    return None


def to_core_eval(mcp_eval: Any) -> Eval:
    """Lift an :class:`MCPEval` to the core :class:`Eval` shape for
    downstream callers (e.g. ``core.engines.analyzer` helpers)."""
    return Eval(
        cp=mcp_eval.cp,
        mate=mcp_eval.mate,
        best_move=mcp_eval.best_move,
        pv=mcp_eval.pv,
        depth=mcp_eval.depth,
    )


def board_after_for_chess_move(board: chess.Board, chess_move: chess.Move | None) -> chess.Board:
    """Return a copy of ``board` with ``chess_move` pushed, or just the
    copy when the move is ``None` (claim_draw path)."""
    b = board.copy(stack=True)
    if chess_move is not None:
        b.push(chess_move)
    return b


def board_after_fen_for_chess_move(board: chess.Board, chess_move: chess.Move | None) -> str:
    return board_after_for_chess_move(board, chess_move).fen()


def build_classification(
    *,
    played_uci: str,
    played_san: str | None,
    score: Any,
    eval_before: Any,
    eval_after: Any,
    board: chess.Board,
    board_after: chess.Board,
    rule_status: Any,
    best_san: str | None,
    best_line_san: str | None,
    played_continuation: str | None,
    action_type: str,
    syntax_warning: str | None,
) -> MCPMoveAnalysis:
    """Single construction path for :class:`MCPMoveAnalysis`.

    Replaces the two divergent builders previously in
    :mod:`mcp_server.tools.classify_move` (the ``pool.classify_move`
    fast-path that called :meth:`MCPMoveAnalysis.from_analysis` and the
    standard path that called :class:`MCPMoveAnalysis` directly).

    Audit invariants preserved: B-01..B-03, P0, P1, P2, P3, U-02..U-15.
    """
    from mcp_server.actions import build_played_action

    verified = _classification_verified(score, action_type)
    return MCPMoveAnalysis(
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
        played_line_san=played_san,
        played_continuation_san=played_continuation,
        syntax_warning=syntax_warning,
        action_type=action_type,
        best_action=score.best_action,
        is_best_action=score.is_best_action,
        action_equivalent=score.action_equivalent,
        played_action_obj=build_played_action(
            action_type,
            move_uci=played_uci,
            move_san=played_san,
            rule_status=rule_status,
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


def _classification_verified(score: Any, action_type: str) -> bool:
    """Audit P1: ``classification_verified`` flips to False on
    play_move + non-play best + claimed best, missing loss_kind with
    positive effective_loss, or engine-best + positive effective_loss."""
    if (
        action_type == "play_move"
        and score.best_action != "play_move"
        and score.is_best_action
        and not score.action_equivalent
    ):
        return False
    if (
        score.effective_loss
        and score.effective_loss > 0
        and (not score.loss_kind or score.loss_kind == "none")
    ):
        return False
    if score.is_best_engine_move and score.effective_loss and score.effective_loss > 0:
        return False
    return True


# Back-compat shims (legacy underscored names preserved for any test that
# monkeypatches them).
_best_san_for_score = best_san_for_score
_safe_san = safe_san
_played_continuation_san = played_continuation_san
_to_core_eval = to_core_eval
