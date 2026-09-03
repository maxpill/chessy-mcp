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
    return score
