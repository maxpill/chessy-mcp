"""Move-grading service package.

Public surface:

    - :func:`score_played_move` — free-function dispatcher (legacy entry point).
    - :class:`MoveGrader` — service object with constructor-injectable deps.
    - Strategies are exposed under :mod:`.strategies` for direct access.

Backwards-compatible imports — ``from mcp_server.move_grading import
score_played_move, _score_*`` — keep working through the module shim at
``mcp_server/move_grading.py``.
"""

from __future__ import annotations

import chess

from mcp_server.analysis.move_grading.dispatcher import dispatch_score  # noqa: F401
from mcp_server.analysis.move_grading.grader import MoveGrader
from mcp_server.analysis.move_grading.helpers import (
    is_after_losing,
    is_after_winning,
    is_before_winning,
    is_down_material,
    is_mover_forced_win,
    material_balance,
)
from mcp_server.analysis.move_grading.strategies import (
    score_blundered_terminal_draw,
    score_claim_draw_action,
    score_conceded_draw,
    score_cp_to_mate,
    score_delivered_checkmate,
    score_mate_to_cp,
    score_mate_to_mate,
    score_optimal_claim_recommended,
    score_received_checkmate,
    score_standard_cp,
)
from mcp_server.models import MCPEval, PlayedMoveScore

__all__ = [
    "MoveGrader",
    "is_after_losing",
    "is_after_winning",
    "is_before_winning",
    "is_down_material",
    "is_mover_forced_win",
    "material_balance",
    "score_blundered_terminal_draw",
    "score_claim_draw_action",
    "score_conceded_draw",
    "score_cp_to_mate",
    "score_delivered_checkmate",
    "score_mate_to_cp",
    "score_mate_to_mate",
    "score_optimal_claim_recommended",
    "score_played_move",
    "score_received_checkmate",
    "score_standard_cp",
]


_default_grader = MoveGrader()


def score_played_move(
    board_before: chess.Board,
    move: chess.Move,
    eval_before: MCPEval,
    eval_after: MCPEval,
    board_after: chess.Board | None = None,
    eval_played: MCPEval | None = None,
    action_type: str = "play_move",
) -> PlayedMoveScore:
    """Default-grader wrapper. Backwards-compat shim for ``move_grading.score_played_move``."""
    return _default_grader.score(
        board_before,
        move,
        eval_before,
        eval_after,
        board_after=board_after,
        eval_played=eval_played,
        action_type=action_type,
    )
