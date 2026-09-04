"""Standard centipawn strategy — the fallback when no terminal/transition fires.

Decisive-position saturation table + raw cp classification. Always runs last
and is responsible for the bulk of the typical-inaccuracies-mistakes-blunders
classifications.
"""

from __future__ import annotations

from core.engines.grading import classify_centipawn_loss
from core.engines.types import MoveClass

from mcp_server.analysis.move_grading.types import finalize_score
from mcp_server.analysis.move_grading.winprob import win_prob_fn as _win_pct
from mcp_server.domain.rule_status import RuleStatus
from mcp_server.models import MCPEval, PlayedMoveScore

__all__ = ["score_standard_cp"]


def score_standard_cp(
    *,
    is_best_engine_move: bool,
    canonical_best_action: str,
    rule_before: RuleStatus,
    eval_before: MCPEval,
    raw_cpl: int,
    raw_board_delta: int,
    baseline_mover: int,
    after_mover: int,
    action_type: str,
) -> PlayedMoveScore:
    """Decisive-position saturation table + the raw cp fallback.

    Saturates engine cp loss at 95%+ winning ranges (cp >= 400, winp >= 95)
    so a tiny position change doesn't inflate the surface; falls back to
    ``classify_centipawn_loss`` for normal positions. Always enforces the
    hard invariant that an is_best_engine_move can never be MISTAKE/BLUNDER.
    """
    if is_best_engine_move:
        is_claim_best = canonical_best_action in ("claim_draw", "claim_draw_with_intended_move")
        claim_r = (
            rule_before.claim_reasons[0]
            if rule_before.claim_reasons
            else (eval_before.claim_reasons[0] if eval_before.claim_reasons else None)
        )
        # Bug fix (chessy-mcp-deep-audit §5): when the engine's best action is
        # a claim but the player instead played the engine's best move (which
        # is also winning), action_equivalent must be FALSE — the outcomes
        # differ (WIN vs DRAW). action_equivalent requires the same action_type.
        score = PlayedMoveScore(
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            raw_centipawn_loss=0,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=None,
            effective_loss=0,
            loss_kind="none",
            is_best_engine_move=is_best_engine_move,
            win_loss=0.0,
            best_action=canonical_best_action,
            is_best_action=not is_claim_best,
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

    eff_delta = baseline_mover - after_mover
    eff_loss = max(0, eff_delta)
    wp_before = _win_pct(baseline_mover)
    wp_after = _win_pct(after_mover)
    win_loss = max(0.0, wp_before - wp_after)

    if (wp_before >= 95.0 and wp_after >= 95.0) or (baseline_mover >= 400 and after_mover >= 400):
        if win_loss < 4.0:
            final_class = MoveClass.BEST if is_best_engine_move else MoveClass.GOOD
            final_cpl = min(eff_loss, 15)
        elif win_loss < 8.0:
            final_class = MoveClass.GOOD
            final_cpl = min(eff_loss, 45)
        else:
            classified = classify_centipawn_loss(eff_loss)
            final_class = (
                MoveClass.GOOD
                if classified == MoveClass.BEST and not is_best_engine_move
                else classified
            )
            final_cpl = min(eff_loss, 1000)
    elif (wp_before >= 90.0 and wp_after >= 90.0) or (baseline_mover >= 300 and after_mover >= 300):
        if win_loss < 4.0:
            final_class = MoveClass.BEST if is_best_engine_move else MoveClass.GOOD
            final_cpl = min(eff_loss, 20)
        elif win_loss < 8.0:
            final_class = MoveClass.GOOD
            final_cpl = min(eff_loss, 50)
        else:
            classified = classify_centipawn_loss(eff_loss)
            final_class = (
                MoveClass.GOOD
                if classified == MoveClass.BEST and not is_best_engine_move
                else classified
            )
            final_cpl = min(eff_loss, 1000)
    else:
        classified = classify_centipawn_loss(eff_loss)
        final_class = (
            MoveClass.GOOD
            if classified == MoveClass.BEST and not is_best_engine_move
            else classified
        )
        final_cpl = min(eff_loss, 1000)

    if is_best_engine_move and final_class in (MoveClass.MISTAKE, MoveClass.BLUNDER):
        final_class = MoveClass.BEST
        final_cpl = 0
        win_loss = 0.0

    score = PlayedMoveScore(
        move_class=final_class,
        centipawn_loss=raw_cpl,
        raw_centipawn_loss=raw_cpl,
        raw_centipawn_delta=raw_board_delta,
        mate_distance_loss=None,
        effective_loss=final_cpl,
        loss_kind="engine_cp" if final_cpl > 0 else "none",
        engine_cp_loss=final_cpl if final_cpl > 0 else None,
        is_best_engine_move=is_best_engine_move,
        win_loss=win_loss,
        best_action=canonical_best_action,
        is_best_action=is_best_engine_move and canonical_best_action == "play_move",
        # Bug fix (chessy-mcp-deep-audit §5): action_equivalent requires the
        # same action_type (a play_move is never equivalent to a draw claim
        # — the outcomes differ). Only set True when both sides recommend
        # the same kind of action AND the played move is the engine's best.
        action_equivalent=is_best_engine_move and canonical_best_action == action_type,
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=rule_before.claim_reasons[0]
        if rule_before.claim_reasons
        else (eval_before.claim_reasons[0] if eval_before.claim_reasons else None),
        claim_move=rule_before.claim_move,
    )
    return finalize_score(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )
