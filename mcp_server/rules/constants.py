"""Constants for the rules engine.

Nothing but modules-level constants and the ``format_fen_status_errors``
helper. The action-policy metadata constants (``ACTION_POLICY_VERSION`` etc.)
are surfaced via MCPEval.action_policy for observability — they live here so
they sit next to the policy implementation that consumes them.
"""

from __future__ import annotations

from enum import StrEnum

# Audit M-04: explicit, versioned action policy. Single source of truth.
ACTION_POLICY_NAME = "risk_adjusted_draw_claim"
ACTION_POLICY_VERSION = "1.0.0"


class ChessActionType(StrEnum):
    """Legacy vocabulary still imported from mcp_server.rules by move_grading.py
    and historical tests. New code should use ``mcp_server.domain.action.TypeOfAction``.
    """

    PLAY_MOVE = "play_move"
    CLAIM_DRAW_NOW = "claim_draw"
    CLAIM_DRAW_WITH_INTENDED_MOVE = "claim_draw_with_intended_move"
    GAME_OVER = "game_over"


# Claim is preferred when mover-POV cp is at or below this threshold AND the
# mover is materially behind OR the score is unambiguously losing. Mate wins
# always override. See ``choose_recommended_action`` for the live policy.
ACTION_EQUIVALENCE_THRESHOLD_CP = 50
ACTION_MATERIAL_DOWN_THRESHOLD_CP = 200

# Minimum post-state cp (Mover-POV) that counts as a forced win for the mover.
FORCED_WIN_THRESHOLD_CP = 2000

# Terminal statuses that do NOT depend on the move-stack history. The other
# fivefold_repetition is the only audit-visible terminal that requires history.
TERMINAL_VS_HISTORY_INDEPENDENT: frozenset[str] = frozenset(
    {
        "checkmate",
        "stalemate",
        "insufficient_material",
        "seventyfive_moves",
        "dead_position",
    }
)


_STATUS_FLAG_NAMES: list[tuple[int, str]] = [
    (getattr(__import__("chess"), "STATUS_NO_WHITE_KING", 1 << 0), "NO_WHITE_KING"),
    (getattr(__import__("chess"), "STATUS_NO_BLACK_KING", 1 << 1), "NO_BLACK_KING"),
    (getattr(__import__("chess"), "STATUS_TOO_MANY_KINGS", 1 << 2), "TOO_MANY_KINGS"),
    (
        getattr(__import__("chess"), "STATUS_TOO_MANY_WHITE_PAWNS", 1 << 3),
        "TOO_MANY_WHITE_PAWNS",
    ),
    (
        getattr(__import__("chess"), "STATUS_TOO_MANY_BLACK_PAWNS", 1 << 4),
        "TOO_MANY_BLACK_PAWNS",
    ),
    (getattr(__import__("chess"), "STATUS_PAWNS_ON_BACKRANK", 1 << 5), "PAWNS_ON_BACKRANK"),
    (
        getattr(__import__("chess"), "STATUS_TOO_MANY_WHITE_PIECES", 1 << 6),
        "TOO_MANY_WHITE_PIECES",
    ),
    (
        getattr(__import__("chess"), "STATUS_TOO_MANY_BLACK_PIECES", 1 << 7),
        "TOO_MANY_BLACK_PIECES",
    ),
    (
        getattr(__import__("chess"), "STATUS_BAD_CASTLING_RIGHTS", 1 << 8),
        "INVALID_CASTLING_RIGHTS",
    ),
    (getattr(__import__("chess"), "STATUS_INVALID_EP_SQUARE", 1 << 9), "INVALID_EP_SQUARE"),
    (getattr(__import__("chess"), "STATUS_OPPOSITE_CHECK", 1 << 10), "OPPOSITE_CHECK"),
    (getattr(__import__("chess"), "STATUS_EMPTY", 1 << 11), "EMPTY_BOARD"),
    (getattr(__import__("chess"), "STATUS_RACE_CHECK", 1 << 12), "RACE_CHECK"),
    (getattr(__import__("chess"), "STATUS_RACE_OVER", 1 << 13), "RACE_OVER"),
    (
        getattr(__import__("chess"), "STATUS_TOO_MANY_CHECKERS", 1 << 14),
        "TOO_MANY_CHECKERS",
    ),
]


def format_fen_status_errors(status_mask: int) -> str:
    """Format numeric python-chess status bitmask into human-readable error reasons."""
    reasons = [name for flag, name in _STATUS_FLAG_NAMES if flag and (status_mask & flag)]
    if not reasons:
        return f"INVALID_POSITION_STATUS_{status_mask}"
    return ", ".join(reasons)
