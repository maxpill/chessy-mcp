"""Draw-claim projection helpers.

Owns :func:`force_draw_outcome` — projects an :class:`MCPEval` onto the
post-claim state so no dummy move or engine view of the pre-claim board
leaks into ``eval_after``.
"""

from __future__ import annotations

from typing import Any

from mcp_server.models import MCPEval

__all__ = ["force_draw_outcome"]


def _build_terminal_state(mcp_eval: MCPEval) -> dict[str, Any]:
    """Build the terminal-state projection dict for a granted draw claim.

    Audit B-02/B-03: classifying a draw claim must not let any dummy move
    leak into ``eval_after``. A granted claim always terminates the game as
    a draw (cp=0, no mate, outcome="draw"), so we force that projection
    here regardless of what the engine reported for the (irrelevant) board
    state.

    U-04 (2026-09-01): the previous projection only zeroed cp/mate/status
    and decision_value, leaving the rest of the eval as Stockfish's view of
    the pre-claim position. That produced self-contradictory fields like
    ``best_move=Qc8#`` alongside ``status=draw, cp=0`` on the same
    ``eval_after`` object. This forces EVERY active-state field to a
    pure-terminal value. The pre-claim engine state is preserved separately
    on ``eval_before`` so callers can still inspect what the engine saw
    before the claim.
    """
    prior_best_action = "claim_draw"
    if mcp_eval.decision_value and isinstance(mcp_eval.decision_value, dict):
        candidate = mcp_eval.decision_value.get("best_action")
        if isinstance(candidate, str):
            prior_best_action = candidate

    return {
        # Score — terminal draw.
        "cp": 0,
        "mate": None,
        "status": "draw",
        "decision_value": {
            "outcome": "draw",
            "cp_equivalent": 0,
            "best_action": prior_best_action,
            "perspective": "white",
        },
        # Engine activity — gone after a granted claim.
        "best_move": None,
        "executable_move": None,
        "pv": [],
        "wdl": None,
        "wdl_pct": None,
        # Rule-action fields — no further claims are possible.
        "can_claim_draw": False,
        "claim_reasons": [],
        "claim_move": None,
        "claim_move_san": None,
        "claim_move_uci": None,
        "can_claim_now": False,
        "claim_reasons_now": [],
        "can_claim_with_intended_move": False,
        "claim_moves": [],
        # Best-action surface — the game is over.
        "recommended_action": "game_over",
        "best_action": "game_over",
        "best_action_type": "game_over",
        "best_action_obj": {
            "type": "game_over",
            "outcome": "draw",
            "reason": "draw_claim",
        },
        "legal_actions": [],
        # Post-state fields — there is no meaningful post-state.
        "post_terminal_status": "draw",
        "post_can_claim_draw": False,
        "post_can_claim_now": False,
        "post_claim_reasons": [],
        "post_claim_moves": [],
        "post_position": {
            "status": "draw",
            "winner": None,
            "can_claim_now": False,
            "can_claim_draw": False,
            "claim_reasons": [],
            "recommended_action": "game_over",
        },
    }


def force_draw_outcome(mcp_eval: MCPEval) -> MCPEval:
    """Return a copy of ``mcp_eval`` projected onto the post-claim terminal state."""
    return mcp_eval.model_copy(update=_build_terminal_state(mcp_eval))


# Underscore-prefixed alias kept for backwards compatibility with the
# pre-split call sites (and the test suite).
_force_draw_outcome = force_draw_outcome
