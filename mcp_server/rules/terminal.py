"""Terminal-position predicates.

Single source of truth for "this position cannot continue" checks. Every MCP
tool that short-circuits on game-over MUST call ``is_terminal_position`` so a
position's terminality cannot disagree between evaluate_position, top_moves,
classify_move, etc.

FIDE 5.2.2 dead-position detection lives here too (was a single function in the
old rules.py — kept here as its own file because the implementation is dense
enough to deserve isolation).
"""

from __future__ import annotations


import chess

from mcp_server.rules.constants import format_fen_status_errors  # re-exported
from mcp_server.rules.dead_position import is_locked_dead_position

__all__ = [
    "can_checkmate",
    "format_fen_status_errors",
    "is_locked_dead_position",
    "is_terminal_position",
]


def _can_side_force_checkmate(board: chess.Board, color: chess.Color) -> bool:
    """Conservative FIDE mating-possibility predicate.

    False is returned only when checkmate is impossible by every legal
    continuation. This matters for Laws 5.1.2, 6.9 and 7.5.5 because a
    false negative converts a win on time, resignation, or rules infraction
    into a draw.
    """
    pawns = len(board.pieces(chess.PAWN, color))
    rooks = len(board.pieces(chess.ROOK, color))
    queens = len(board.pieces(chess.QUEEN, color))
    bishops = list(board.pieces(chess.BISHOP, color))
    knights = len(board.pieces(chess.KNIGHT, color))

    if not (pawns or rooks or queens or bishops or knights):
        return False
    if pawns or rooks or queens:
        return True

    opponent_nonking = sum(
        len(board.pieces(pt, not color))
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )

    if opponent_nonking:
        return True

    if knights >= 2:
        return True
    if knights == 1 and not bishops:
        return False

    if bishops:
        if len(bishops) == 1 and knights == 0:
            return False
        if knights:
            return True
        complexes = {(chess.square_rank(sq) + chess.square_file(sq)) & 1 for sq in bishops}
        return len(complexes) >= 2

    return False


def can_checkmate(board: chess.Board, color: chess.Color) -> bool:
    """Return whether `color` can mate by some legal continuation."""
    return _can_side_force_checkmate(board, color)


def is_terminal_position(board: chess.Board) -> bool:
    """Single source of truth for "the position is game over".

    Combines python-chess's terminal checks with FIDE 5.2.2 dead-position
    detection (audit invariant R-AUDIT-TERM-01).
    """
    if board.is_checkmate():
        return True
    if board.is_stalemate():
        return True
    if board.is_insufficient_material():
        return True
    if board.is_seventyfive_moves():
        return True
    if board.is_fivefold_repetition():
        return True
    if is_locked_dead_position(board):
        return True
    return False
