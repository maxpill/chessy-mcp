"""``ActionBlock`` aggregate — action surface of :class:`MCPEval`.

Groups the engine-recommended and legal-action surfaces: best move,
executable move pointer, recommended action type, rule-action claims,
post-position state for top_moves candidates, and the legal-move list
(``legal_move_uci`` plus the legacy ``legal_actions`` alias).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ActionBlock(BaseModel):
    """Action-side fields of a single position evaluation."""

    best_move: str | None = None
    executable_move: str | None = None
    recommended_action: str = "play_move"
    best_action: str = "play_move"
    best_action_type: str = "play_move"
    best_action_obj: dict[str, Any] | None = None
    # U-10 (2026-09-01): ``legal_actions`` is rule-only; ``legal_rule_actions``
    # is the explicit name; ``legal_move_uci`` is the full UCI list.
    legal_actions: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    legal_rule_actions: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    legal_move_uci: list[str] = Field(default_factory=list[str])
    can_claim_draw: bool = False
    claim_reasons: list[str] = Field(default_factory=list[str])
    claim_move: str | None = None
    claim_move_san: str | None = None
    claim_move_uci: str | None = None
    can_claim_now: bool = False
    claim_reasons_now: list[str] = Field(default_factory=list[str])
    can_claim_with_intended_move: bool = False
    claim_moves: list[str] = Field(default_factory=list[str])
    post_terminal_status: str | None = None
    candidate_san: str | None = None
    post_can_claim_draw: bool = False
    post_can_claim_now: bool = False
    post_claim_reasons: list[str] = Field(default_factory=list[str])
    post_claim_moves: list[str] = Field(default_factory=list[str])
    post_position: dict[str, Any] | None = None
