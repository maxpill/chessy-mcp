"""Predicate helpers used by every move-grading strategy.

Small pure functions — extracted out of ``move_grading.py`` so they can be
unit-tested without dragging the giant strategy dispatcher along.
"""

from __future__ import annotations

import chess

from mcp_server.domain.position import PIECE_VALUE

__all__ = [
    "is_after_losing",
    "is_after_winning",
    "is_before_winning",
    "is_down_material",
    "is_mover_forced_win",
    "material_balance",
]


def is_before_winning(before_mover: int, mover_mate_before: int | None) -> bool:
    """True when the position before the move is materially winning for the mover."""
    return (mover_mate_before is not None and mover_mate_before > 0) or before_mover >= 200


def is_mover_forced_win(before_mover: int, mover_mate_before: int | None) -> bool:
    """True when the mover has a forced win (mate-distance or >=100cp ahead)."""
    return (mover_mate_before is not None and mover_mate_before > 0) or before_mover >= 100


def is_after_winning(mover_mate_after: int | None, after_mover: int) -> bool:
    """True when the post-move position is materially winning for the mover."""
    return (mover_mate_after is not None and mover_mate_after > 0) or (after_mover >= 100)


def is_after_losing(mover_mate_after: int | None, after_mover: int, board: chess.Board) -> bool:
    """True when the post-move position is materially losing for the mover."""
    base = after_mover <= -100 or (mover_mate_after is not None and mover_mate_after < 0)
    if base:
        return True
    own, opp = material_balance(board)
    return opp - own >= 200


def is_down_material(board: chess.Board) -> bool:
    """True when the mover is down at least ``ACTION_MATERIAL_DOWN_THRESHOLD_CP`` material."""
    own, opp = material_balance(board)
    return opp - own >= 200


def material_balance(board: chess.Board) -> tuple[int, int]:
    """Return ``(mover_material, opponent_material)`` using :data:`PIECE_VALUE`."""
    own = sum(len(board.pieces(pt, board.turn)) * value for pt, value in PIECE_VALUE.items())
    opp = sum(len(board.pieces(pt, not board.turn)) * value for pt, value in PIECE_VALUE.items())
    return own, opp
