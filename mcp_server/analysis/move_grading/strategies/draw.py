"""Draw-claim strategies: claim action grading, conceded-draw detection, optimal-claim-recommended.

Three strategies covering the FIDE 5.2.2 / 9.6.1 / 9.6.2 surface area.
"""

from __future__ import annotations

import chess
from core.engines.types import MoveClass

from mcp_server.analysis.move_grading.types import finalize_score
from mcp_server.domain.rule_status import RuleStatus
from mcp_server.models import MCPEval, PlayedMoveScore
from mcp_server.rules import ChessActionType

__all__ = [
    "score_claim_draw_action",
    "score_conceded_draw",
    "score_optimal_claim_recommended",
]


def score_claim_draw_action(
    *,
    action_type: str,
    move: chess.Move,
    eval_before: MCPEval,
    rule_before: RuleStatus,
    canonical_best_action: str,
    before_mover: int,
    mover_mate_before: int | None,
    is_best_engine_move: bool,
) -> PlayedMoveScore:
    """Grade a procedural draw-claim action.

    Honors the claim only when it's legally available AND the mover wasn't
    in a forced-win / winning-technical-conversion position (audit P0/P1).
    A claim while a forced mate is on the board is a blundered win, not a
    claim.
    """
    is_claim_now_action = action_type in (
        "claim_draw",
        ChessActionType.CLAIM_DRAW_NOW.value,
    )
    played_uci = move.uci().lower()

    claim_legal = (is_claim_now_action and rule_before.can_claim_now) or (
        not is_claim_now_action
        and rule_before.can_claim_with_intended_move
        and played_uci in [u.lower() for u in rule_before.intended_claim_ucis]
    )
    if not claim_legal:
        raise ValueError("ILLEGAL_ACTION: requested draw claim is not legally available")

    if is_claim_now_action:
        reasons = rule_before.claim_reasons_now
    else:
        reason_map = rule_before.intended_claim_reasons_by_uci
        reasons = reason_map.get(move.uci(), [])
    claim_r = reasons[0] if reasons else None

    is_mover_forced_win = (
        mover_mate_before is not None and mover_mate_before > 0
    ) or before_mover >= 200
    is_canonical_play_move = canonical_best_action == "play_move"
    if is_mover_forced_win or is_canonical_play_move:
        # U-02 (2026-09-01): a winning zeroing-capture at halfmove=100 sat at
        # cp=+26 but the rule layer detected a forced technical win; the
        # previous logic honored that as BEST. Now we flag the claim as a
        # smaller blunder — the mover forfeited a winning conversion.
        if is_mover_forced_win:
            eff_loss = 1000
            win_loss = 50.0
            loss_kind = "outcome_penalty"
            outcome_pen = 1000
        else:
            eff_loss = 500
            win_loss = 50.0
            loss_kind = "technical_win_forfeited"
            outcome_pen = None
        score = PlayedMoveScore(
            move_class=MoveClass.BLUNDER,
            centipawn_loss=None,
            raw_centipawn_loss=None,
            raw_centipawn_delta=0,
            mate_distance_loss=None,
            effective_loss=eff_loss,
            loss_kind=loss_kind,
            outcome_penalty=outcome_pen,
            is_best_engine_move=is_best_engine_move,
            win_loss=win_loss,
            best_action=canonical_best_action,
            is_best_action=False,
            action_equivalent=False,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=claim_r,
            claim_move=rule_before.claim_move,
        )
        return finalize_score(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )

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
        is_best_action=canonical_best_action == action_type,
        # Bug fix (chessy-mcp-deep-audit §5): action_equivalent requires the
        # same action_type — only same-kind actions can be equivalent.
        action_equivalent=canonical_best_action == action_type,
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=claim_r,
        claim_move=rule_before.claim_move,
    )
    return finalize_score(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


def score_conceded_draw(
    *,
    opponent_will_claim: bool,
    rule_after: RuleStatus,
    eval_after: MCPEval,
    is_before_winning: bool,
    raw_cpl: int,
    raw_board_delta: int,
    before_mover: int,
    canonical_best_action: str,
    rule_before: RuleStatus,
    action_type: str,
) -> PlayedMoveScore | None:
    """The opponent can claim a draw against us on the next ply (we conceded it)."""
    if not opponent_will_claim:
        return None
    eff_loss = max(500, min(1000, before_mover if before_mover > 0 else 1000))
    claim_r = (
        rule_after.claim_reasons_now[0]
        if rule_after.claim_reasons_now
        else (eval_after.claim_reasons[0] if eval_after.claim_reasons else "threefold_repetition")
    )
    score = PlayedMoveScore(
        move_class=MoveClass.BLUNDER,
        centipawn_loss=raw_cpl,
        raw_centipawn_loss=raw_cpl,
        raw_centipawn_delta=raw_board_delta,
        mate_distance_loss=None,
        effective_loss=eff_loss,
        loss_kind="conceded_draw",
        rule_action_penalty=eff_loss,
        is_best_engine_move=False,
        win_loss=50.0,
        best_action=canonical_best_action,
        is_best_action=False,
        action_equivalent=False,
        missed_draw_claim=False,
        conceded_draw_claim=True,
        claim_reason=claim_r,
        claim_move=None,
    )
    return finalize_score(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


def score_optimal_claim_recommended(
    *,
    optimal_claim_recommended: bool,
    is_auto_terminal_draw: bool,
    eval_before: MCPEval,
    eval_after: MCPEval,
    rule_before: RuleStatus,
    raw_cpl: int,
    raw_board_delta: int,
    is_best_engine_move: bool,
    canonical_best_action: str,
    action_type: str,
    is_after_losing: bool,
    is_after_winning: bool,
    is_down_material: bool,
    is_mover_forced_win: bool,
    before_mover: int,
    after_mover: int,
    opp_mat: int,
    mover_mat: int,
) -> PlayedMoveScore | None:
    """Move played instead of an optimal draw claim — penalize the forfeit."""
    if not (optimal_claim_recommended and not is_auto_terminal_draw):
        return None
    decision_before_draw = bool(
        (eval_before.decision_value and eval_before.decision_value.get("outcome") == "draw")
        or (rule_before.can_claim_draw and before_mover <= 100)
    )
    decision_after_draw = bool(
        eval_after.decision_value and eval_after.decision_value.get("outcome") == "draw"
    )
    draw_preserved = bool(
        decision_before_draw and decision_after_draw and is_best_engine_move and raw_cpl == 0
    )
    if draw_preserved:
        claim_r = (
            rule_before.claim_reasons[0]
            if rule_before.claim_reasons
            else (eval_before.claim_reasons[0] if eval_before.claim_reasons else None)
        )
        score = PlayedMoveScore(
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            raw_centipawn_loss=0,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=None,
            effective_loss=0,
            loss_kind="none",
            is_best_engine_move=True,
            win_loss=0.0,
            best_action=canonical_best_action,
            is_best_action=canonical_best_action == "play_move",
            action_equivalent=True,
            missed_draw_claim=bool(
                is_down_material
                and canonical_best_action in ("claim_draw", "claim_draw_with_intended_move")
                and action_type == "play_move"
            ),
            conceded_draw_claim=False,
            claim_reason=claim_r,
            claim_move=rule_before.claim_move,
        )
        return finalize_score(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )
    if is_after_losing and not is_after_winning:
        loss_val = max(abs(before_mover), abs(after_mover))
        eff_loss = max(
            300,
            min(
                1000,
                loss_val if loss_val > 0 else (opp_mat - mover_mat if is_down_material else 500),
            ),
        )
        final_class = MoveClass.BLUNDER if eff_loss >= 300 else MoveClass.MISTAKE
        claim_r = (
            rule_before.claim_reasons[0]
            if rule_before.claim_reasons
            else (eval_before.claim_reasons[0] if eval_before.claim_reasons else None)
        )
        score = PlayedMoveScore(
            move_class=final_class,
            centipawn_loss=raw_cpl,
            raw_centipawn_loss=raw_cpl,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=None,
            effective_loss=eff_loss,
            loss_kind="draw_claim_forfeit",
            rule_action_penalty=eff_loss,
            is_best_engine_move=is_best_engine_move,
            win_loss=50.0,
            best_action=canonical_best_action,
            is_best_action=False,
            action_equivalent=False,
            missed_draw_claim=True,
            conceded_draw_claim=False,
            claim_reason=claim_r,
            claim_move=rule_before.claim_move,
        )
        return finalize_score(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )
    return None
