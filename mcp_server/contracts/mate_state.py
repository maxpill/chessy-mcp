"""Canonical mate-state semantics.

Audit Phase 8 (2026-09-14). The previous code computed a mate state from the
raw ``MCPEval.mate`` integer, which has Stockfish's side-to-move perspective.
That produced color-asymmetric results in two scenarios:

1. ``mate == +N`` while Black is to move (meaning "Black mates in N") was
   collapsed to ``"white_mates"``.
2. ``mate == 0`` after a delivered mate was mapped to ``"black_mates"`` when
   the fallback fired before the terminal-winner check.

The only stable invariant is: a mate-state label must be derived from the
side-to-move identity plus the perspective rule, never from the raw numeric
sign alone.

The :func:`semantic_mate_state` helper is the single source of truth used by
``_mate_signature`` (classification_stability, practical_equivalence) and by
any future caller.
"""

from __future__ import annotations

from enum import StrEnum

import chess

from mcp_server.models import MCPEval


class MateState(StrEnum):
    NO_MATE = "no_mate"
    WHITE_MATES = "white_mates"
    BLACK_MATES = "black_mates"
    TERMINAL_WHITE_WON = "terminal_white_won"
    TERMINAL_BLACK_WON = "terminal_black_won"


def semantic_mate_state(
    board: chess.Board,
    eval_after: MCPEval,
    *,
    eval_before: MCPEval | None = None,
) -> MateState:
    """Map an evaluation snapshot to a semantic mate state.

    Rules (in priority order):

    1. If ``eval_after.status == "checkmate"``, return the terminal winner
       state derived from ``eval_after.winner``.
    2. Otherwise, when ``eval_after.mate`` is set, the integer is
       **side-to-move perspective**: ``mate > 0`` means the side to move is
       mating, ``mate < 0`` means the side to move is being mated. Convert
       that into ``WHITE_MATES`` / ``BLACK_MATES`` via ``board.turn``.
    3. Fall back to ``NO_MATE``.
    """
    if eval_after.status == "checkmate":
        winner = eval_after.winner
        if winner == "white":
            return MateState.TERMINAL_WHITE_WON
        if winner == "black":
            return MateState.TERMINAL_BLACK_WON
        # Terminal but winner unknown — fall back to the pre-move
        # perspective (eval_before) if it knew who was mating, otherwise
        # the side-to-move at eval_after.
        if eval_before is not None:
            before_state = semantic_mate_state(board, eval_before)
            if before_state == MateState.WHITE_MATES:
                return MateState.TERMINAL_WHITE_WON
            if before_state == MateState.BLACK_MATES:
                return MateState.TERMINAL_BLACK_WON

    if eval_after.mate is None:
        return MateState.NO_MATE

    if eval_after.mate == 0:
        # mate == 0 is the post-terminal representation detail; if status is
        # not checkmate, treat as no forced mate.
        if eval_after.status == "checkmate":
            if eval_after.winner == "white":
                return MateState.TERMINAL_WHITE_WON
            if eval_after.winner == "black":
                return MateState.TERMINAL_BLACK_WON
        return MateState.NO_MATE

    side_to_mate = board.turn
    if eval_after.mate > 0:
        return MateState.WHITE_MATES if side_to_mate == chess.WHITE else MateState.BLACK_MATES
    if eval_after.mate < 0:
        # Negative means side to move is being mated; the other side mates.
        return MateState.BLACK_MATES if side_to_mate == chess.WHITE else MateState.WHITE_MATES
    return MateState.NO_MATE


def coerce_legacy_signature(value: str | None) -> str:
    """Normalize a legacy ``white_mates|black_mates|no_mate`` signature.

    Old call sites pass ``"white_mates"`` / ``"black_mates"`` /
    ``"no_mate"`` strings. Phase 9 forensics stability keys still use the old
    three-value vocabulary; this helper maps ``MateState`` back to that
    vocabulary for backward compatibility.
    """
    if value in (None, MateState.NO_MATE.value):
        return "no_mate"
    if value in (MateState.WHITE_MATES.value, MateState.TERMINAL_WHITE_WON.value):
        return "white_mates"
    if value in (MateState.BLACK_MATES.value, MateState.TERMINAL_BLACK_WON.value):
        return "black_mates"
    return "no_mate"


__all__ = [
    "MateState",
    "coerce_legacy_signature",
    "semantic_mate_state",
]
