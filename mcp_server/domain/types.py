"""Strongly-typed enums and re-exports for chess semantics.

Why this file exists:
    - Single source of truth for the str-enum literals used across MCPEval,
      rules.py, and the audit tests.
    - Imports are all pure re-exports or StrEnum classes — zero logic.
"""

from __future__ import annotations

from enum import StrEnum

import chess

Color = chess.Color  # re-export for clarity at use-sites (mcp_server.domain.Color)
Side = chess.Color

Square = chess.Square  # alias for readability in domain types


class Outcome(StrEnum):
    """Game outcome from White-POV (the codebase convention for decision_value)."""

    ACTIVE = "active"
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


class Perspective(StrEnum):
    """PoV from which cp/mate values are reported."""

    WHITE = "white"
    BLACK = "black"


# Strings that appear in MCPEval.status / winner fields.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "checkmate",
        "stalemate",
        "insufficient_material",
        "seventyfive_moves",
        "fivefold_repetition",
        "dead_position",
        "game_over",
    }
)

# Repetition states surfaced in MCPEval.repetition_status.
REPETITION_NONE = "none"
REPETITION_UNKNOWN = "unknown"
REPETITION_THREEFOLD_CLAIMABLE = "threefold_claimable"
REPETITION_FIVEFOLD = "fivefold"
