"""``mcp_server.core`` — cross-cutting types: domain type aliases and the
unified DI :class:\`ToolContext\`.

Currently exposes:

  * :mod:\`mcp_server.core.types\` — Position / Move / San / CpLoss etc.
  * :mod:\`mcp_server.core.context\` — :class:\`ToolContext\` (frozen DI
    container) + :func:\`make_test_context\` factory.
"""

from __future__ import annotations

from mcp_server.core.context import ToolContext
from mcp_server.core.types import (
    Cp,
    CpLoss,
    EffectiveLoss,
    FullmoveNumber,
    Move,
    MoverCp,
    Piece,
    PieceType,
    PlyIndex,
    Position,
    San,
    Square,
    TimeControl,
    Uci,
    WinProb,
)

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
    "ToolContext",
    "Uci",
    "WinProb",
]
