"""Score-finalization helper used by every strategy."""

from __future__ import annotations

from mcp_server.domain.rule_status import RuleStatus
from mcp_server.models.legacy import PlayedMoveScore

__all__ = ["finalize_score"]


def finalize_score(
    score: PlayedMoveScore,
    *,
    canonical_best_action: str,
    rule_before: RuleStatus,
    action_type: str,
) -> PlayedMoveScore:
    """Project the rule-action provenance surface onto ``score``.

    All strategies use this to attach ``best_action``, ``can_claim_*``,
    ``claim_moves``, and ``action_type`` before returning. Keeps the
    per-strategy bodies focused on the move-class verdict and loss numbers.
    """
    score.best_action = canonical_best_action
    score.can_claim_now = rule_before.can_claim_now
    score.can_claim_with_intended_move = rule_before.can_claim_with_intended_move
    score.claim_moves = rule_before.claim_moves
    score.action_type = action_type
    if score.is_best_engine_move and canonical_best_action == action_type:
        score.action_equivalent = True
        score.is_best_action = True

    if canonical_best_action in ("claim_draw", "claim_draw_with_intended_move"):
        if action_type in ("claim_draw", "claim_draw_with_intended_move"):
            score.missed_draw_claim = False
            score.missed_draw_claim_kind = "none"
        else:
            is_intended_claim = bool(
                canonical_best_action == "claim_draw_with_intended_move"
                and score.is_best_engine_move
                and (score.raw_centipawn_loss is None or score.raw_centipawn_loss == 0)
            )
            if is_intended_claim:
                score.missed_draw_claim = False
                score.action_equivalent = True
                score.missed_draw_claim_kind = "none"
            else:
                score.missed_draw_claim = True
                score.is_best_action = False
                score.action_equivalent = False
                score.missed_draw_claim_kind = (
                    "immediate" if rule_before.can_claim_now else "intended_move"
                )
    return score

