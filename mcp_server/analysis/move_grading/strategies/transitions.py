"""Engine-eval-transition strategies: mate→mate, mate→cp, cp→mate.

These three strategies handle the case where the engine's score type changes
between before and after the move — typically because the move itself
delivered mate, allowed mate, or converted a centipawn position into one
with a mate distance.
"""

from __future__ import annotations

import chess
from core.engines.types import MoveClass

from mcp_server.analysis.move_grading.types import finalize_score
from mcp_server.analysis.move_grading.winprob import win_prob_fn as _win_pct
from mcp_server.domain.rule_status import RuleStatus
from mcp_server.models import MCPEval, PlayedMoveScore

__all__ = [
    "score_cp_to_mate",
    "score_mate_to_cp",
    "score_mate_to_mate",
]


def score_mate_to_mate(
    *,
    mover_mate_before: int | None,
    mover_mate_after: int | None,
    board_after: chess.Board,
    is_best_engine_move: bool,
    canonical_best_action: str,
    rule_before: RuleStatus,
    action_type: str,
) -> PlayedMoveScore | None:
    """Both before and after evaluations report a mate distance."""
    if not (mover_mate_before is not None and mover_mate_after is not None):
        return None
    if mover_mate_before > 0:
        if mover_mate_after > 0:
            if board_after.is_checkmate():
                mate_dist_loss = 0
            else:
                mate_dist_loss = max(0, (abs(mover_mate_after) + 1) - abs(mover_mate_before))
            if is_best_engine_move or mate_dist_loss == 0:
                final_class = MoveClass.BEST
                eff_loss = 0
                w_loss = 0.0
            elif mate_dist_loss <= 1:
                final_class = MoveClass.GOOD
                eff_loss = 50
                w_loss = 0.5
            elif mate_dist_loss <= 2:
                final_class = MoveClass.INACCURACY
                eff_loss = 150
                w_loss = 2.0
            else:
                final_class = MoveClass.MISTAKE
                eff_loss = 300
                w_loss = min(20.0, float(mate_dist_loss * 2.0))
            score = PlayedMoveScore(
                move_class=final_class,
                centipawn_loss=0,
                raw_centipawn_loss=0,
                raw_centipawn_delta=0,
                mate_distance_loss=mate_dist_loss,
                effective_loss=eff_loss,
                loss_kind="mate_distance" if mate_dist_loss > 0 else "none",
                mate_distance_penalty=eff_loss if mate_dist_loss > 0 else None,
                is_best_engine_move=is_best_engine_move,
                win_loss=w_loss,
                best_action=canonical_best_action,
                is_best_action=is_best_engine_move
                or (canonical_best_action == "play_move" and mate_dist_loss == 0),
                missed_draw_claim=False,
                conceded_draw_claim=False,
                claim_reason=None,
            )
            return finalize_score(
                score,
                canonical_best_action=canonical_best_action,
                rule_before=rule_before,
                action_type=action_type,
            )
        if mover_mate_after == 0:
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
        score = PlayedMoveScore(
            move_class=MoveClass.BLUNDER,
            centipawn_loss=None,
            raw_centipawn_loss=None,
            raw_centipawn_delta=None,
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
        )
        return finalize_score(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )
    if mover_mate_before < 0 and mover_mate_after < 0:
        # Defender perspective: allowing faster mate against self is a loss
        # of resistance.
        defender_resistance_loss = max(0, abs(mover_mate_before) - abs(mover_mate_after))
        if is_best_engine_move or defender_resistance_loss == 0:
            final_class = MoveClass.BEST
            eff_loss = 0
            w_loss = 0.0
        elif defender_resistance_loss == 1:
            final_class = MoveClass.INACCURACY
            eff_loss = 150
            w_loss = 2.0
        else:
            final_class = MoveClass.BLUNDER if defender_resistance_loss >= 3 else MoveClass.MISTAKE
            eff_loss = 500 if defender_resistance_loss >= 3 else 300
            w_loss = min(20.0, float(defender_resistance_loss * 5.0))
        score = PlayedMoveScore(
            move_class=final_class,
            centipawn_loss=0,
            raw_centipawn_loss=0,
            raw_centipawn_delta=0,
            mate_distance_loss=defender_resistance_loss,
            effective_loss=eff_loss,
            loss_kind="mate_distance" if defender_resistance_loss > 0 else "none",
            mate_distance_penalty=eff_loss if defender_resistance_loss > 0 else None,
            is_best_engine_move=is_best_engine_move,
            win_loss=w_loss,
            best_action=canonical_best_action,
            is_best_action=is_best_engine_move and canonical_best_action == "play_move",
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=None,
            claim_move=rule_before.claim_move,
        )
        return finalize_score(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )
    return None


def score_mate_to_cp(
    *,
    mover_mate_before: int | None,
    eval_move_eval: MCPEval,
    after_mover: int,
    is_best_engine_move: bool,
    canonical_best_action: str,
    rule_before: RuleStatus,
    action_type: str,
    raw_cpl: int,
    raw_board_delta: int,
) -> PlayedMoveScore | None:
    """Mover had mate; after-state is centipawn (e.g. Qh4+ giving perpetual)."""
    if not (
        mover_mate_before is not None and mover_mate_before > 0 and eval_move_eval.mate is None
    ):
        return None
    if after_mover >= 400:
        final_class = MoveClass.BEST if is_best_engine_move else MoveClass.INACCURACY
        eff_loss = 0 if is_best_engine_move else 150
        w_loss = 0.0 if is_best_engine_move else 2.0
        score = PlayedMoveScore(
            move_class=final_class,
            centipawn_loss=0,
            raw_centipawn_loss=0,
            raw_centipawn_delta=0,
            mate_distance_loss=None,
            effective_loss=eff_loss,
            loss_kind="mate_distance" if not is_best_engine_move else "none",
            mate_distance_penalty=eff_loss if not is_best_engine_move else None,
            is_best_engine_move=is_best_engine_move,
            win_loss=w_loss,
            best_action=canonical_best_action,
            is_best_action=is_best_engine_move,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=None,
        )
        return finalize_score(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )
    wp_after = _win_pct(after_mover)
    win_loss = max(0.0, 100.0 - wp_after)
    final_class = MoveClass.BLUNDER if win_loss >= 20.0 else MoveClass.MISTAKE
    eff_loss = 1000 if final_class == MoveClass.BLUNDER else 300
    score = PlayedMoveScore(
        move_class=final_class,
        centipawn_loss=raw_cpl,
        raw_centipawn_loss=raw_cpl,
        raw_centipawn_delta=raw_board_delta,
        mate_distance_loss=None,
        effective_loss=eff_loss,
        loss_kind="outcome_penalty",
        outcome_penalty=eff_loss,
        is_best_engine_move=False,
        win_loss=win_loss,
        best_action=canonical_best_action,
        is_best_action=False,
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=None,
    )
    return finalize_score(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


def score_cp_to_mate(
    *,
    eval_before: MCPEval,
    mover_mate_after: int | None,
    baseline_mover: int,
    optimal_claim_recommended: bool,
    after_mover: int,  # noqa: ARG001
    is_best_engine_move: bool,
    canonical_best_action: str,
    rule_before: RuleStatus,
    action_type: str,
) -> PlayedMoveScore | None:
    """Before was centipawn; after-state now reports mate distance."""
    if not (eval_before.mate is None and mover_mate_after is not None):
        return None
    if mover_mate_after < 0:
        wp_before = _win_pct(baseline_mover)
        win_loss = max(0.0, wp_before - 0.0)
        score = PlayedMoveScore(
            move_class=MoveClass.BLUNDER,
            centipawn_loss=None,
            raw_centipawn_loss=None,
            raw_centipawn_delta=None,
            mate_distance_loss=None,
            effective_loss=1000,
            loss_kind="mate_transition",
            outcome_penalty=1000,
            is_best_engine_move=False,
            win_loss=win_loss,
            best_action=canonical_best_action,
            is_best_action=False,
            missed_draw_claim=optimal_claim_recommended,
            conceded_draw_claim=False,
            claim_reason=rule_before.claim_reasons[0]
            if (optimal_claim_recommended and rule_before.claim_reasons)
            else None,
        )
        return finalize_score(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )
    score = PlayedMoveScore(
        move_class=MoveClass.BEST if is_best_engine_move else MoveClass.GOOD,
        centipawn_loss=0,
        raw_centipawn_loss=0,
        raw_centipawn_delta=0,
        mate_distance_loss=0,
        effective_loss=0,
        loss_kind="none",
        is_best_engine_move=is_best_engine_move,
        win_loss=0.0,
        best_action=canonical_best_action,
        is_best_action=is_best_engine_move,
        action_equivalent=is_best_engine_move
        and canonical_best_action in ("claim_draw", "claim_draw_with_intended_move"),
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=None,
    )
    return finalize_score(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )
