"""SAN / line-conversion helpers for the ``classify_move`` tool.

Extracted from :mod:`mcp_server.tools.classify_move` so the tool
entry point stays focused on FastMCP plumbing (cache lookup, single-
flight, error translation). All helpers here are pure functions over
``chess.Board`` + ``MCPEval``.
"""

from __future__ import annotations

from typing import Any, Literal

import chess
from core.engines.analyzer import pv_to_san
from core.engines.types import Eval

from mcp_server.models import MCPMoveAnalysis
from mcp_server.models.legacy import AnalysisConfidence

ActionType = Literal["play_move", "claim_draw", "claim_draw_with_intended_move"]


def classify_draw_claim_delta(
    *,
    best_action: str,
    played_action: str,
    action_equivalent: bool,
    can_claim_now: bool = False,
    can_claim_with_intended_move: bool = False,
    claim_reason: str | None = None,
    is_best_engine_move: bool = False,
    raw_cpl: int | None = None,
) -> dict[str, Any]:
    """Authoritative draw claim delta evaluator shared across tools."""
    claim_available = bool(can_claim_now or can_claim_with_intended_move)
    kind = "none"
    if can_claim_now:
        kind = "immediate"
    elif can_claim_with_intended_move:
        kind = "intended_move"

    is_intended_claim = bool(
        best_action == "claim_draw_with_intended_move"
        and is_best_engine_move
        and (raw_cpl is None or raw_cpl == 0)
    )

    missed = bool(
        best_action in ("claim_draw", "claim_draw_with_intended_move")
        and not action_equivalent
        and played_action not in ("claim_draw", "claim_draw_with_intended_move")
        and not is_intended_claim
    )

    return {
        "available": claim_available,
        "kind": kind,
        "reason": claim_reason,
        "best_action": best_action,
        "played_action": played_action,
        "equivalent": action_equivalent,
        "missed_draw_claim": missed,
        "missed_draw_claim_kind": kind if missed else "none",
    }


def compute_analysis_confidence(
    eval_before: Any,
    eval_after: Any,
    *,
    requested_depth: int | None,
    searched_depth: int | None,
    board_after: chess.Board | None = None,
) -> AnalysisConfidence:
    """Compute confidence signal based on search depth and exact rule/mate evidence."""
    effective_depth = searched_depth if searched_depth is not None else requested_depth
    is_terminal = bool(
        (board_after and board_after.is_game_over(claim_draw=False))
        or (eval_after and getattr(eval_after, "status", None) in (
            "checkmate", "stalemate", "insufficient_material", "seventyfive_moves", "fivefold_repetition"
        ))
    )
    is_mate = bool(
        (eval_after and getattr(eval_after, "mate", None) is not None)
        or (eval_before and getattr(eval_before, "mate", None) is not None)
    )
    if is_terminal or is_mate:
        return AnalysisConfidence(
            level="high",
            reason_codes=["EXACT_CHESS_RULE_OR_MATE"],
            requested_depth=requested_depth,
            searched_depth=searched_depth,
        )
    if effective_depth is not None and effective_depth <= 6:
        return AnalysisConfidence(
            level="low",
            reason_codes=["SHALLOW_SEARCH_DEPTH"],
            requested_depth=requested_depth,
            searched_depth=searched_depth,
        )
    if effective_depth is not None and effective_depth <= 11:
        return AnalysisConfidence(
            level="medium",
            reason_codes=["MODERATE_SEARCH_DEPTH"],
            requested_depth=requested_depth,
            searched_depth=searched_depth,
        )
    return AnalysisConfidence(
        level="normal",
        reason_codes=[],
        requested_depth=requested_depth,
        searched_depth=searched_depth,
    )


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
    """Best-effort ``san()`` for a UCI string; never raises."""
    try:
        m = chess.Move.from_uci(uci.lower())
        if m in board.legal_moves:
            return board.san(m)
    except Exception:
        return None
    return None


def played_continuation_san(board_after: chess.Board, eval_after: Any) -> str | None:
    """Render the engine's post-state PV as SAN, or ``None`` if the position
    is already terminal."""
    if eval_after.pv and not board_after.is_game_over():
        return pv_to_san(board_after, eval_after.pv)
    return None


def to_core_eval(mcp_eval: Any) -> Eval:
    """Lift an :class:`MCPEval` to the core :class:`Eval` shape for
    downstream callers (e.g. ``core.engines.analyzer`` helpers)."""
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
    action_type: ActionType,
    syntax_warning: str | None,
) -> MCPMoveAnalysis:
    """Single construction path for :class:`MCPMoveAnalysis`.

    Replaces the two divergent builders previously in
    :mod:`mcp_server.tools.classify_move` (the ``pool.classify_move``
    fast-path that called :meth:`MCPMoveAnalysis.from_analysis` and the
    standard path that called :class:`MCPMoveAnalysis` directly).

    Audit invariants preserved: B-01..B-03, P0, P1, P2, P3, U-02..U-15.
    """
    from mcp_server.actions import build_best_action, build_played_action

    verified = _classification_verified(score, action_type)
    played_outcome, played_value = _outcome_from_action(
        action_type, eval_after, board_after, board.turn
    )
    best_outcome, best_value = _outcome_from_action(
        score.best_action, eval_before, board, board.turn
    )

    is_same_move = bool(
        eval_before.best_move and played_uci and eval_before.best_move.lower() == played_uci.lower()
    )
    is_same_action = (
        action_type == score.best_action
        and (
            action_type != "play_move"
            or is_same_move
            or score.is_best_action
            or score.is_best_engine_move
        )
    )
    unify_best_with_played = is_same_action and action_type == "play_move"
    if unify_best_with_played:
        best_post_cp = eval_after.cp
        best_post_mate = eval_after.mate
        best_outcome = played_outcome
        best_value = played_value
    elif is_same_action:
        best_post_cp = eval_before.cp
        best_post_mate = eval_before.mate
        best_outcome = played_outcome
        best_value = played_value
    else:
        best_post_cp = eval_before.cp
        best_post_mate = eval_before.mate

    req_depth = getattr(eval_before, "requested_depth", None) or getattr(eval_before, "depth", None)
    srch_depth = getattr(eval_before, "searched_depth", None) or getattr(eval_before, "depth", None)
    confidence = compute_analysis_confidence(
        eval_before,
        eval_after,
        requested_depth=req_depth,
        searched_depth=srch_depth,
        board_after=board_after,
    )
    final_warning = syntax_warning
    if final_warning is None and confidence.level == "low":
        final_warning = "SHALLOW_DEPTH_CLASSIFICATION_UNSTABLE"

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
        syntax_warning=final_warning,
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
        best_action_obj=(
            build_best_action(
                recommended_action=score.best_action,
                rule_status=rule_status,
                engine_eval=type(
                    "E",
                    (),
                    {
                        "cp": best_post_cp,
                        "mate": best_post_mate,
                        "best_move": eval_before.best_move,
                    },
                )(),
                board=board,
                sign=1 if board.turn == chess.WHITE else -1,
            )
            if unify_best_with_played
            else eval_before.best_action_obj
        ),
        played_outcome=played_outcome.value
        if hasattr(played_outcome, "value")
        else str(played_outcome),
        best_outcome=best_outcome.value if hasattr(best_outcome, "value") else str(best_outcome),
        played_canonical_value=played_value,
        best_canonical_value=best_value,
        missed_draw_claim=score.missed_draw_claim,
        missed_draw_claim_kind=getattr(score, "missed_draw_claim_kind", "none"),
        conceded_draw_claim=score.conceded_draw_claim,
        claim_reason=score.claim_reason,
        claim_move=score.claim_move,
        can_claim_now=score.can_claim_now,
        can_claim_with_intended_move=score.can_claim_with_intended_move,
        claim_moves=score.claim_moves,
        classification_verified=verified,
        action_class=(
            "missed_rule_action"
            if score.missed_draw_claim
            else (
                "conceded_rule_action"
                if score.conceded_draw_claim
                else (
                    "best"
                    if score.is_best_action
                    else (
                        "suboptimal_rule_action"
                        if action_type != score.best_action
                        else score.move_class.value
                    )
                )
            )
        ),
        action_quality_class=(
            "missed_rule_action"
            if score.missed_draw_claim
            else (
                "conceded_rule_action"
                if score.conceded_draw_claim
                else (
                    "best"
                    if score.is_best_action
                    else (
                        "suboptimal_rule_action"
                        if action_type != score.best_action
                        else score.move_class.value
                    )
                )
            )
        ),
        move_quality_class=score.move_class.value,
        requested_depth=req_depth,
        searched_depth=srch_depth,
        analysis_confidence=confidence,
    )



def _outcome_from_action(
    action_type: str,
    eval_obj: Any,
    board: chess.Board,
    mover_color: chess.Color,
) -> tuple[Any, int | None]:
    """Compute the Outcome of an action evaluation from the mover's POV."""
    from mcp_server.domain.types import Outcome

    if action_type in ("claim_draw", "claim_draw_with_intended_move"):
        return Outcome.DRAW, 0
    return _outcome_from_eval(eval_obj, mover_color)


def _outcome_from_eval(eval_obj: Any, mover_color: chess.Color) -> tuple[Any, int | None]:
    """Compute the Outcome of a play_move / game_over eval from the mover's POV."""
    from mcp_server.domain.types import Outcome

    status = getattr(eval_obj, "status", None)
    mate = getattr(eval_obj, "mate", None)
    cp = getattr(eval_obj, "cp", None)

    if status == "checkmate":
        winner = getattr(eval_obj, "winner", None)
        if winner == "white":
            is_win = mover_color == chess.WHITE
            return (Outcome.WIN if is_win else Outcome.LOSS), (100000 if is_win else -100000)
        if winner == "black":
            is_win = mover_color == chess.BLACK
            return (Outcome.WIN if is_win else Outcome.LOSS), (100000 if is_win else -100000)
        return Outcome.ACTIVE, None

    if status in (
        "stalemate",
        "insufficient_material",
        "seventyfive_moves",
        "fivefold_repetition",
        "dead_position",
        "game_over",
    ):
        return Outcome.DRAW, 0

    if mate is not None:
        mover_sign = 1 if mover_color == chess.WHITE else -1
        m = mover_sign * mate
        if m > 0:
            return Outcome.WIN, 100000
        if m < 0:
            return Outcome.LOSS, -100000

    if cp is not None:
        mover_sign = 1 if mover_color == chess.WHITE else -1
        signed_cp = mover_sign * cp
        if signed_cp >= 2000:
            return Outcome.WIN, signed_cp
        if signed_cp <= -2000:
            return Outcome.LOSS, signed_cp
        return Outcome.ACTIVE, signed_cp

    return Outcome.ACTIVE, None


def _classification_verified(score: Any, action_type: str) -> bool:
    """Audit P1: ensure classification metadata is self-consistent."""
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


_best_san_for_score = best_san_for_score
_safe_san = safe_san
_played_continuation_san = played_continuation_san
_to_core_eval = to_core_eval
