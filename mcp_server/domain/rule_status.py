"""RuleStatus — structured output of `evaluate_rule_status`.

Originally lived in rules.py as a `@dataclass`. Promoted to the domain layer
because it's the contract every MCP tool speaks when talking about rules:
without it, callers re-implement checks that already exist (terminal-status,
draw-claim, repetition bookkeeping).

This module imports from rules.py the action-selection function
(`choose_recommended_action`) and the `RuleStatus` dataclass definition site,
then exposes them in a stable, typed location. rules.py will eventually
re-export these names for backward compatibility — phase 12 — and stop
re-declaring them itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import chess

if TYPE_CHECKING:
    from chess import Move as _Move  # noqa: F401 — type-only import for clarity


class HistoryCompleteness(StrEnum):
    """How much of the move stack the caller supplied with the FEN.

    `complete`  — every move from the start was supplied (repetition-aware).
    `partial`   — partial move stack; some claims may be possible.
    `incomplete` — none / most of the stack missing (default for naked FENs).
    `not_required` — terminal status does not depend on history at all.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    INCOMPLETE = "incomplete"
    NOT_REQUIRED = "not_required"


class RuleAction(StrEnum):
    """MCP tool action vocabulary — what a caller should do at this position."""

    PLAY_MOVE = "play_move"
    CLAIM_DRAW = "claim_draw"
    CLAIM_DRAW_WITH_INTENDED_MOVE = "claim_draw_with_intended_move"
    GAME_OVER = "game_over"


@dataclass
class RuleStatus:
    """Complete rule-evaluation result for one position.

    Field-by-field:
        terminal — None if position is active, otherwise terminal-status string.
        winner — "white" / "black" / None (only set on terminal checkmate).
        can_claim_now / can_claim_with_intended_move — FIDE draw-claim flags.
        intended_claim_* — the moves that, if played, would justify a claim.
        claim_reasons — list of human-readable reasons.
        recommended_action — RuleAction value to take.
        history_dependent_status — true if 75/5-fold deps on the move stack.
        repetition_status — "none" | "unknown" | "threefold_claimable" | "fivefold".
    """

    terminal: str | None = None
    winner: str | None = None
    can_claim_now: bool = False
    claim_reasons_now: list[str] = field(default_factory=list[str])
    can_claim_with_intended_move: bool = False
    intended_claim_moves: list[chess.Move] = field(default_factory=list[chess.Move])
    intended_claim_sans: list[str] = field(default_factory=list[str])
    intended_claim_ucis: list[str] = field(default_factory=list[str])
    intended_claim_reasons_by_uci: dict[str, list[str]] = field(
        default_factory=dict[str, list[str]]
    )
    claim_reasons: list[str] = field(default_factory=list[str])
    can_claim_draw: bool = False
    claim_moves: list[str] = field(default_factory=list[str])
    claim_move: str | None = None
    claim_move_uci: str | None = None
    claim_move_san: str | None = None
    recommended_action: str = "play_move"
    history_dependent_status: bool = False
    requires_move_stack: bool = False
    fen_sufficient_for_status: bool = True
    history_completeness: str = "incomplete"
    repetition_status: str = "unknown"
