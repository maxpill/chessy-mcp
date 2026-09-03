"""Typed chess action contract — re-exports from mcp_server.domain.action.

Kept as a top-level module for back-compat with all imports across the
codebase. New code should import from mcp_server.domain instead. See
mcp_server.domain.action for the typed discriminated-union definitions.
"""

from __future__ import annotations

from mcp_server.domain.action import (
    ActionValue,
    ClaimDrawAction,
    ClaimDrawWithIntendedMoveAction,
    GameAction,
    GameOverAction,
    MovePayload,
    PlayMoveAction,
    Outcome,
    TypeOfAction,
    build_best_action,
    build_legal_actions,
    build_played_action,
    parse_action,
)

__all__ = [
    "ActionValue",
    "ClaimDrawAction",
    "ClaimDrawWithIntendedMoveAction",
    "GameAction",
    "GameOverAction",
    "MovePayload",
    "PlayMoveAction",
    "Outcome",
    "TypeOfAction",
    "build_best_action",
    "build_legal_actions",
    "build_played_action",
    "parse_action",
]
