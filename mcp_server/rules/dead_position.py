"""FIDE 5.2.2 dead-position detection.

A position is dead iff BOTH colors:
    - have no Q, R, or N on the board;
    - have no pawn that can move (from either color's perspective);
    - have no bishop that can move (from either color's perspective);
    - cannot reach an enemy pawn or cross to the opposite side via king maneuvers.

The function uses ``chess.Board.turn`` to inspect both sides' legal moves
because python-chess only enumerates legal moves for the side-to-move. We
save and restore `turn` to avoid surprising callers.

Implementation notes:
    - pawn-attack masks are precomputed via ``board.attacks_mask(sq)``
    - king reachable-square flood-fills use ``chess.BB_KING_ATTACKS`` for speed
"""

from __future__ import annotations

from collections import deque

import chess


def is_locked_dead_position(board: chess.Board) -> bool:
    """Detect dead positions caused by completely locked pawn structures.

    FIDE 5.2.2: a position where neither player can checkmate by any series of
    legal moves. Returns True when the game cannot continue in a way that
    leads to checkmate — distinct from ``is_insufficient_material`` which
    only checks the static material count.
    """
    if board.is_insufficient_material():
        return True

    # Heavy pieces or knights on either side means the position is not dead.
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.ROOK, chess.KNIGHT):
            if board.pieces(pt, color):
                return False

    # Inspect pawn/bishop mobility for BOTH sides. We must flip turn because
    # ``board.legal_moves`` only enumerates the side-to-move.
    saved_turn = board.turn
    try:
        for color in (chess.WHITE, chess.BLACK):
            board.turn = color
            for m in board.legal_moves:
                if board.piece_type_at(m.from_square) == chess.PAWN:
                    return False
        any_bishop = bool(
            board.pieces(chess.BISHOP, chess.WHITE) or board.pieces(chess.BISHOP, chess.BLACK)
        )
        if any_bishop:
            for color in (chess.WHITE, chess.BLACK):
                board.turn = color
                for m in board.legal_moves:
                    if board.piece_type_at(m.from_square) == chess.BISHOP:
                        return False
    finally:
        board.turn = saved_turn

    # From here, only kings + (locked) pawns are on the board.
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    if white_king_sq is None or black_king_sq is None:
        return False

    # Pawn-attack masks for both sides.
    white_pawn_attacks = chess.SquareSet()
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        white_pawn_attacks |= board.attacks_mask(sq)

    black_pawn_attacks = chess.SquareSet()
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        black_pawn_attacks |= board.attacks_mask(sq)

    # King reachable squares (flood-fill, king-step). A king can step only on
    # empty squares it does not control through pawn attacks.
    occupied_or_attacked_by_black = set(chess.SquareSet(board.occupied)) | set(
        chess.SquareSet(black_pawn_attacks)
    )
    white_reachable: set[int] = {white_king_sq}
    queue: deque[int] = deque([white_king_sq])
    while queue:
        curr = queue.popleft()
        for neighbor in chess.SquareSet(chess.BB_KING_ATTACKS[curr]):
            if neighbor not in white_reachable and neighbor not in occupied_or_attacked_by_black:
                white_reachable.add(neighbor)
                queue.append(neighbor)

    occupied_or_attacked_by_white = set(chess.SquareSet(board.occupied)) | set(
        chess.SquareSet(white_pawn_attacks)
    )
    black_reachable: set[int] = {black_king_sq}
    queue = deque([black_king_sq])
    while queue:
        curr = queue.popleft()
        for neighbor in chess.SquareSet(chess.BB_KING_ATTACKS[curr]):
            if neighbor not in black_reachable and neighbor not in occupied_or_attacked_by_white:
                black_reachable.add(neighbor)
                queue.append(neighbor)

    # Either king attack an enemy pawn?
    black_pawns = set(board.pieces(chess.PAWN, chess.BLACK))
    if any(
        bool(chess.SquareSet(chess.BB_KING_ATTACKS[sq]) & white_reachable) for sq in black_pawns
    ):
        return False

    white_pawns = set(board.pieces(chess.PAWN, chess.WHITE))
    if any(
        bool(chess.SquareSet(chess.BB_KING_ATTACKS[sq]) & black_reachable) for sq in white_pawns
    ):
        return False

    # Neither king can cross to the opposite side → truly dead.
    white_max_rank = max((chess.square_rank(sq) for sq in white_reachable), default=0)
    black_min_rank = min((chess.square_rank(sq) for sq in black_reachable), default=7)
    return white_max_rank < black_min_rank
