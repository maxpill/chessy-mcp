"""Immutable Position wrapper around chess.Board.

`chess.Board` is mutable and ties every consumer to its mutation API. This
wrapper exposes the predicates callers actually use (terminality, material
counts, king-reachable squares, etc.) and forces move application to go
through `with_move(...)`, returning a NEW Position — so functional code can
chain without surprising shared mutation.

The wrapper *holds* a chess.Board but never mutates it externally. Internal
state stays mutating for engine reuse (we don't want to construct a fresh
chess.Board every position check); the Position class treats the board as a
read-only state-snapshot helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

_WHITE = chess.WHITE
_BLACK = chess.BLACK

# Default piece values used by `material_balance()` and downstream material-
# down heuristics. Kept in lockstep with the constant in rules.py.
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


@dataclass(frozen=True)
class Position:
    """Read-only wrapper around a chess.Board snapshot.

    The `board` field is intentionally NOT marked frozen — chess.Board is
    a mutable object — but `Position` instances themselves are frozen. To
    advance the position, call `.with_move(...)` which returns a new
    Position with a copy of the board after the move.
    """

    board: chess.Board

    @classmethod
    def from_fen(cls, fen: str) -> Position:
        return cls(chess.Board(fen))

    @classmethod
    def start(cls) -> Position:
        return cls(chess.Board())

    @classmethod
    def empty(cls) -> Position:
        return cls(chess.Board.empty())

    def with_move(self, move: chess.Move | str) -> Position:
        """Return a new Position after applying `move` to a copy of the board."""
        b = self.board.copy(stack=True)
        m = move if isinstance(move, chess.Move) else chess.Move.from_uci(move)
        b.push(m)
        return Position(b)

    def with_turn(self, color: chess.Color) -> Position:
        b = self.board.copy(stack=False)
        b.turn = color
        return Position(b)

    # ------------------------------------------------------------- canonical ---

    @property
    def fen(self) -> str:
        return self.board.fen()

    @property
    def turn(self) -> chess.Color:
        return self.board.turn

    @property
    def is_game_over(self) -> bool:
        """Single source of truth for "the game cannot continue".

        Combines python-chess's terminal checks with FIDE 5.2.2 dead-position
        detection (see mcp_server.rules.is_locked_dead_position — this method
        delegates to it to preserve the audit invariant R-AUDIT-TERM-01).
        """
        from mcp_server.rules import is_terminal_position  # local import avoids cycle

        return is_terminal_position(self.board)

    # --------------------------------------------------------- sanity flags ---

    @property
    def is_checkmate(self) -> bool:
        return self.board.is_checkmate()

    @property
    def is_stalemate(self) -> bool:
        return self.board.is_stalemate()

    @property
    def is_insufficient_material(self) -> bool:
        return self.board.is_insufficient_material()

    @property
    def is_seventyfive_moves(self) -> bool:
        return self.board.is_seventyfive_moves()

    @property
    def is_fivefold_repetition(self) -> bool:
        return self.board.is_fivefold_repetition()

    @property
    def is_repetition_claimable(self) -> bool:
        return self.board.is_repetition(3)

    @property
    def is_fifty_moves(self) -> bool:
        return self.board.is_fifty_moves()

    @property
    def is_check(self) -> bool:
        return self.board.is_check()

    # ---------------------------------------------------------- side-to-move ---

    def can_side_force_checkmate(self, color: chess.Color) -> bool:
        from mcp_server.rules import can_checkmate  # local — cycle avoidance

        return can_checkmate(self.board, color)

    # ---------------------------------------------------------- material -----

    def material_balance(self, color: chess.Color) -> int:
        """Total material for one color using PIECE_VALUE."""
        return sum(len(self.board.pieces(pt, color)) * value for pt, value in PIECE_VALUE.items())

    def material_down_by(self, threshold: int = 200) -> bool:
        """True iff the mover (side-to-move) is materially down by at least threshold."""
        own = self.material_balance(self.board.turn)
        opp = self.material_balance(not self.board.turn)
        return opp - own >= threshold

    # ---------------------------------------------------------- misc helpers ---

    @property
    def legal_moves_uci(self) -> list[str]:
        return [m.uci() for m in self.board.legal_moves]

    def san(self, move: chess.Move) -> str:
        return self.board.san(move)
