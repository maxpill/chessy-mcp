"""Domain type aliases — the canonical names for chess primitive types.

Used across the analyzer / classifier / pipeline layers so call sites
read as ``Position`` instead of ``chess.Board``, ``PlyIndex`` instead
of ``int``, ``CpLoss`` instead of ``int | None``, etc. Aliases keep the
type surface narrow while staying 100% Pythonic (no runtime cost).
"""

from __future__ import annotations

import chess

# Chess primitives — re-aliased so the analyzer / pipeline modules read as
# chessy vocabulary rather than chess-library vocabulary.
Position = chess.Board
Piece = chess.Piece
Square = chess.Square
Move = chess.Move
PieceType = chess.PieceType

# Analysis-side value types.
PlyIndex = int
"""Zero-based index of a ply (half-move) inside a game."""

FullmoveNumber = int
"""Full-move number (the integer on the right of a FEN)."""

Cp = int
"""Centipawns from the engine's perspective (positive = better for White)."""

MoverCp = int
"""Centipawns from the mover's perspective (always positive = better for mover)."""

CpLoss = int | None
"""Centipawn loss for a played move, or None if not applicable (mate / terminal)."""

EffectiveLoss = int | None
"""Audit M-04 effective centipawn loss (raw + outcome + rule penalties)."""

WinProb = float
"""Win probability in [0.0, 1.0], White-POV (default 0.5 = equal)."""

San = str
"""Standard Algebraic Notation."""

Uci = str
"""Universal Chess Interface move string."""

TimeControl = str | None
"""PGN ``[TimeControl "..."]`` header, or None if not provided / unknown."""


__all__ = [
    "Cp",
    "CpLoss",
    "EffectiveLoss",
    "FullmoveNumber",
    "Move",
    "MoverCp",
    "Piece",
    "PieceType",
    "PlyIndex",
    "Position",
    "San",
    "Square",
    "TimeControl",
    "Uci",
    "WinProb",
]
