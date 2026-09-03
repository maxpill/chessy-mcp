"""Recommend the canonical legal action for an MCP tool to surface.

The pure function ``choose_recommended_action`` decides whether to emit
``play_move`` or one of the two draw-claim actions. It is the ONLY function
in the codebase that takes the raw material / claim flags and produces the
recommended-action string that downstream code stores in
``MCPEval.recommended_action``.

The function is intentionally pure — every input is an argument; no globals,
no I/O, no engine calls.
"""

from __future__ import annotations

import chess

from mcp_server.rules.constants import (
    ACTION_EQUIVALENCE_THRESHOLD_CP,
    ACTION_MATERIAL_DOWN_THRESHOLD_CP,
    FORCED_WIN_THRESHOLD_CP,
)
from mcp_server.domain.position import PIECE_VALUE

__all__ = [
    "FORCED_WIN_THRESHOLD_CP",
    "choose_recommended_action",
]


def choose_recommended_action(
    board: chess.Board,
    *,
    can_claim_now: bool,
    can_claim_with_intended_move: bool,
    mover_score: int | None = None,
    mate_for_mover: int | None = None,
    zeroing_move_best_score: int | None = None,
    zeroing_move_best_mate: int | None = None,
) -> str:
    """Choose one canonical legal root action.

    A draw claim is preferred ONLY when no zeroing move (capture or pawn push)
    can plausibly turn the position into a forced win. The post-state values
    of zeroing moves are the correct comparator: root cp conflates "draw is
    available" with "this is drawn", and a position like K+R vs K after Rxa7
    at halfmove=100 reports a tiny root cp because the draw is on the table
    even though Rxa7 yields a forced win.
    """
    if not (can_claim_now or can_claim_with_intended_move):
        return "play_move"
    if mate_for_mover is not None and mate_for_mover > 0:
        return "play_move"
    if zeroing_move_best_mate is not None and zeroing_move_best_mate > 0:
        return "play_move"
    if zeroing_move_best_score is not None and zeroing_move_best_score >= FORCED_WIN_THRESHOLD_CP:
        return "play_move"

    if _is_mover_in_trouble(board, mover_score):
        if can_claim_now:
            return "claim_draw"
        return "claim_draw_with_intended_move"
    return "play_move"


def _is_mover_in_trouble(board: chess.Board, mover_score: int | None) -> bool:
    """Decide whether the mover should prefer a draw.

    True when:
        - mover is materially down by ``ACTION_MATERIAL_DOWN_THRESHOLD_CP``, OR
        - mover's score is at/below ``ACTION_EQUIVALENCE_THRESHOLD_CP``
          (cp-rough draw-or-loss).
    """
    mover_mat = sum(len(board.pieces(pt, board.turn)) * value for pt, value in PIECE_VALUE.items())
    opp_mat = sum(
        len(board.pieces(pt, not board.turn)) * value for pt, value in PIECE_VALUE.items()
    )
    is_down_material = opp_mat - mover_mat >= ACTION_MATERIAL_DOWN_THRESHOLD_CP

    if mover_score is None:
        claim_preferred = is_down_material
    else:
        claim_preferred = not (
            mover_score > ACTION_EQUIVALENCE_THRESHOLD_CP and not is_down_material
        )
    return claim_preferred
