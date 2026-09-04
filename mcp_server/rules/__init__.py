"""Pure rule-engine logic for FIDE chess positions.

Modules:

    - ``constants`` — thresholds, action policy metadata, FIDE article refs
    - ``status`` — ``RuleStatus`` dataclass + ``evaluate_rule_status`` (single source of truth)
    - ``terminal`` — terminal-position predicates (is_terminal_position, can_checkmate, is_locked_dead_position)
    - ``action_choice`` — ``choose_recommended_action`` (the play_move vs claim decision)
    - ``pv`` — principal-variation utilities (truncate_pv_at_terminal, validate_mating_possibility)

All modules are FUNCTIONAL — they take chess.Board (and/or RuleStatus inputs) and
return pure data. No I/O, no globals. The dataclass ``RuleStatus`` lives in
``mcp_server.domain.rule_status`` (and is re-exported from ``rules.status``) for
consistency with the rest of the domain layer.
"""

from mcp_server.rules.constants import (
    ACTION_EQUIVALENCE_THRESHOLD_CP,
    ACTION_MATERIAL_DOWN_THRESHOLD_CP,
    ACTION_POLICY_NAME,
    ACTION_POLICY_VERSION,
    FORCED_WIN_THRESHOLD_CP,
    TERMINAL_VS_HISTORY_INDEPENDENT,
)
from mcp_server.rules.pv import truncate_pv_at_terminal, validate_mating_possibility
from mcp_server.rules.status import (
    evaluate_rule_status,
    make_rule_status,
)
from mcp_server.rules.terminal import (
    can_checkmate,
    format_fen_status_errors,
    is_locked_dead_position,
    is_terminal_position,
)
from mcp_server.rules.action_choice import choose_recommended_action
from mcp_server.rules.constants import ChessActionType
from mcp_server.domain.rule_status import RuleStatus

__all__ = [
    "ACTION_EQUIVALENCE_THRESHOLD_CP",
    "ACTION_MATERIAL_DOWN_THRESHOLD_CP",
    "ACTION_POLICY_NAME",
    "ACTION_POLICY_VERSION",
    "FORCED_WIN_THRESHOLD_CP",
    "TERMINAL_VS_HISTORY_INDEPENDENT",
    "ChessActionType",
    "RuleStatus",
    "can_checkmate",
    "choose_recommended_action",
    "evaluate_rule_status",
    "format_fen_status_errors",
    "is_locked_dead_position",
    "is_terminal_position",
    "make_rule_status",
    "truncate_pv_at_terminal",
    "validate_mating_possibility",
]
