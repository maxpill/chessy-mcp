"""Terminal-position strategies: delivered/received checkmate, blundered auto-draw.

Three strategies that fire on board-terminal predicates (checkmate on either
side, automatic terminal-draw transitions from a winning position).
"""

from __future__ import annotations

import chess
from core.engines.types import MoveClass

from mcp_server.analysis.move_grading.helpers import is_before_winning
from mcp_server.analysis.move_grading.types import finalize_score
from mcp_server.domain.rule_status import RuleStatus
from mcp_server.models import MCPEval, PlayedMoveScore

__all__ = [
    "score_blundered_terminal_draw",
    "score_delivered_checkmate",
    "score_received_checkmate",
]


def score_delivered_checkmate(
    *,
    action_type: str,
    board_before: chess.Board,
    board_after: chess.Board,
    is_best_engine_move: bool,
    canonical_best_action: str,
    rule_before: RuleStatus,
) -> PlayedMoveScore | None:
    """Move delivered checkmate — the best outcome for the mover."""
    if not (
        action_type == "play_move"
        and board_after.is_checkmate()
        and board_after.turn != board_before.turn
    ):
        return None
    score = PlayedMoveScore(
        move_class=MoveClass.BEST,
        centipawn_loss=0,
        raw_centipawn_loss=0,
        raw_centipawn_delta=0,
        mate_distance_loss=0,
        effective_loss=0,
        loss_kind="none",
        is_best_engine_move=is_best_engine_move,
        win_loss=0.0,
        best_action=canonical_best_action,
        is_best_action=True,
        action_equivalent=False,
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=None,
        claim_move=None,
    )
    return finalize_score(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


def score_received_checkmate(
    *,
    action_type: str,
    board_before: chess.Board,
    board_after: chess.Board,
    canonical_best_action: str,
    rule_before: RuleStatus,
) -> PlayedMoveScore | None:
    """Move walked into checkmate against the mover — full blunder."""
    if not (
        action_type == "play_move"
        and board_after.is_checkmate()
        and board_after.turn == board_before.turn
    ):
        return None
    score = PlayedMoveScore(
        move_class=MoveClass.BLUNDER,
        centipawn_loss=1000,
        raw_centipawn_loss=1000,
        raw_centipawn_delta=1000,
        mate_distance_loss=None,
        effective_loss=1000,
        loss_kind="mate_transition",
        outcome_penalty=1000,
        is_best_engine_move=False,
        win_loss=100.0,
        best_action=canonical_best_action,
        is_best_action=False,
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=None,
        claim_move=None,
    )
    return finalize_score(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


def score_blundered_terminal_draw(
    *,
    is_auto_terminal_draw: bool,
    is_before_winning: bool,
    board_after: chess.Board,  # noqa: ARG001 — kept for parity with original signature
    is_best_engine_move: bool,
    raw_cpl: int,
    raw_board_delta: int,
    before_mover: int,
    mover_mate_before: int | None,
    canonical_best_action: str,
    rule_before: RuleStatus,
    action_type: str,
) -> PlayedMoveScore | None:
    """Move played into an automatic terminal draw (no rule action)."""
    if not is_auto_terminal_draw:
        return None
    if is_before_winning:
        eff_loss = max(300, min(1000, before_mover if before_mover > 0 else 1000))
        score = PlayedMoveScore(
            move_class=MoveClass.BLUNDER,
            centipawn_loss=None if mover_mate_before is not None else raw_cpl,
            raw_centipawn_loss=None if mover_mate_before is not None else raw_cpl,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=None,
            effective_loss=eff_loss,
            loss_kind="blundered_draw",
            outcome_penalty=eff_loss,
            is_best_engine_move=False,
            win_loss=50.0,
            best_action=canonical_best_action,
            is_best_action=False,
            action_equivalent=False,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=None,
            claim_move=None,
        )
    else:
        score = PlayedMoveScore(
            move_class=MoveClass.BEST if is_best_engine_move else MoveClass.GOOD,
            centipawn_loss=0,
            raw_centipawn_loss=raw_cpl,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=None,
            effective_loss=0,
            loss_kind="none",
            is_best_engine_move=is_best_engine_move,
            win_loss=0.0,
            best_action=canonical_best_action,
            is_best_action=True,
            action_equivalent=canonical_best_action
            in ("claim_draw", "claim_draw_with_intended_move"),
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=None,
            claim_move=None,
        )
    return finalize_score(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )
