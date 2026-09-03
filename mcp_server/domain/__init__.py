"""Domain layer — typed value objects for chess positions, actions, evaluations.

This package contains the pure data types the rest of the codebase is built on.
Modules here MUST be dependency-light: no engine, no cache, no I/O. They are the
vocabulary the engine and tools speak.

Subpackages and modules:
    - action: Typed action discriminated union + builders (moved from actions.py)
    - evaluation: Typed evaluation value object (mover-POV / White-POV)
    - position: Immutable Position wrapper around chess.Board with predicates
    - rule_status: RuleStatus dataclass + RuleAction enum
    - types: Color/Side re-exports + Outcome/Perspective enums
"""

from mcp_server.domain.action import (
    Action,
    ActionValue,
    ClaimDrawAction,
    ClaimDrawWithIntendedMoveAction,
    GameAction,
    GameOverAction,
    MovePayload,
    Outcome,
    PlayMoveAction,
    TypeOfAction,
    build_best_action,
    build_legal_actions,
    build_played_action,
)
from mcp_server.domain.evaluation import Evaluation, ScoreLine
from mcp_server.domain.position import Position
from mcp_server.domain.rule_status import HistoryCompleteness, RuleAction, RuleStatus

__all__ = [
    "Action",
    "ActionValue",
    "ClaimDrawAction",
    "ClaimDrawWithIntendedMoveAction",
    "Evaluation",
    "GameAction",
    "GameOverAction",
    "HistoryCompleteness",
    "MovePayload",
    "Outcome",
    "PlayMoveAction",
    "Position",
    "RuleAction",
    "RuleStatus",
    "ScoreLine",
    "TypeOfAction",
    "build_best_action",
    "build_legal_actions",
    "build_played_action",
]
