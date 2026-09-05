"""``MoveGrader`` service object with constructor-injectable dependencies."""

from __future__ import annotations

from collections.abc import Callable

import chess

from core.engines.types import MoveClass
from mcp_server.analysis.move_grading.dispatcher import dispatch_score
from mcp_server.analysis.move_grading.winprob import win_prob_fn
from mcp_server.models import MCPEval, PlayedMoveScore

__all__ = ["MoveGrader"]


class MoveGrader:
    """Service object that scores a played move against rule-aware policies.

    Construction-time dependencies:

        - ``win_prob_fn``: White-POV win-probability function.
        - ``cp_classifier``: centipawn-loss to move-class function.

    Both are injectable for testing without monkeypatching ``core``.
    """

    def __init__(
        self,
        *,
        win_prob_fn: Callable[[float], float] | None = None,
        cp_classifier: Callable[[int], MoveClass] | None = None,
    ) -> None:
        self._win_prob_fn = win_prob_fn if win_prob_fn is not None else win_prob_fn_default()
        self._cp_classifier = (
            cp_classifier if cp_classifier is not None else default_cp_classifier()
        )

    def score(
        self,
        board_before: chess.Board,
        move: chess.Move,
        eval_before: MCPEval,
        eval_after: MCPEval,
        board_after: chess.Board | None = None,
        eval_played: MCPEval | None = None,
        action_type: str = "play_move",
    ) -> PlayedMoveScore:
        """Single source of truth for move-class + loss + rule-action provenance."""
        return dispatch_score(
            board_before,
            move,
            eval_before,
            eval_after,
            board_after,
            eval_played,
            action_type,
        )


def win_prob_fn_default() -> Callable[[float], float]:
    """Default win-probability function."""
    return win_prob_fn


def default_cp_classifier() -> Callable[[int], MoveClass]:
    """Default centipawn to move-class classifier."""
    from core.engines.grading import classify_centipawn_loss

    return classify_centipawn_loss
