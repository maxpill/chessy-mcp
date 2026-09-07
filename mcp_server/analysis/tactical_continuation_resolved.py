"""Shared tactical-sequence resolution helper.

2026-09-08 ultra-hard test notes §2: ``_principal_line`` (in
``analysis/forensics.py``) and ``_walk_endpoint`` (in
``analysis/candidate_continuation.py``) both compute
``tactical_sequence_resolved`` for the same continuation but previously
disagreed — the candidate comparison said resolved=True for a quiet
final position even when no forcing move had been seen during the walk,
while the nested continuation endpoint required forcing to have been
seen first. The shared helper pins a single semantic so the same
continuation produces the same answer at both levels.

Lives in its own module to break a circular import between forensics
and candidate_continuation (both want to call the helper; neither should
import the other through it).
"""

from __future__ import annotations

import chess


def _forcing(move_board: chess.Board, move: chess.Move) -> bool:
    return move_board.gives_check(move) or move_board.is_capture(move) or move.promotion is not None


def _forcing_available(board: chess.Board) -> bool:
    return any(_forcing(board, move) for move in board.legal_moves)


def tactical_sequence_resolved(forcing_seen: bool, final_board: chess.Board) -> bool:
    """Resolve ``tactical_sequence_resolved`` under a single shared semantic.

    A continuation is resolved iff a forcing move was seen during the walk
    AND the final board is quiet (not in check, and no forcing move is
    still available). This is intentionally more restrictive than
    "final position is quiet alone" — a quiet continuation with no
    forcing action is just an engine continuation in a quiet position, not
    a settled tactic.
    """
    return forcing_seen and not final_board.is_check() and not _forcing_available(final_board)
