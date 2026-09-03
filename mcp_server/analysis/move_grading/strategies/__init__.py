"""Move-grading strategies package — one module per audit invariant family.

Modules:

    - ``terminal`` — delivered/received checkmate, blundered-auto-terminal-draw.
    - ``draw`` — claim-action grading, conceded-draw, optimal-claim-recommended.
    - ``transitions`` — mate→mate, mate→cp, cp→mate.
    - ``standard`` — the fallback cp classifier with decisive-position saturation.

Each module exposes unprefixed ``score_*`` functions. The package ``__init__``
also re-exports them with the legacy underscore prefix used by the old
``mcp_server.move_grading`` module so existing imports keep working.
"""

from mcp_server.analysis.move_grading.strategies.draw import (
    score_claim_draw_action,
    score_conceded_draw,
    score_optimal_claim_recommended,
)
from mcp_server.analysis.move_grading.strategies.standard import score_standard_cp
from mcp_server.analysis.move_grading.strategies.terminal import (
    score_blundered_terminal_draw,
    score_delivered_checkmate,
    score_received_checkmate,
)
from mcp_server.analysis.move_grading.strategies.transitions import (
    score_cp_to_mate,
    score_mate_to_cp,
    score_mate_to_mate,
)

# Back-compat underscored aliases — the original move_grading.py exposed these
# as private helpers; preserving the import surface keeps monkeypatched tests
# working through the migration.
_score_delivered_checkmate = score_delivered_checkmate
_score_received_checkmate = score_received_checkmate
_score_claim_draw_action = score_claim_draw_action
_score_blundered_terminal_draw = score_blundered_terminal_draw
_score_conceded_draw = score_conceded_draw
_score_optimal_claim_recommended = score_optimal_claim_recommended
_score_mate_to_mate = score_mate_to_mate
_score_mate_to_cp = score_mate_to_cp
_score_cp_to_mate = score_cp_to_mate
_score_standard_cp = score_standard_cp

__all__ = [
    "_score_blundered_terminal_draw",
    "_score_claim_draw_action",
    "_score_conceded_draw",
    "_score_cp_to_mate",
    "_score_delivered_checkmate",
    "_score_mate_to_cp",
    "_score_mate_to_mate",
    "_score_optimal_claim_recommended",
    "_score_received_checkmate",
    "_score_standard_cp",
    "score_blundered_terminal_draw",
    "score_claim_draw_action",
    "score_conceded_draw",
    "score_cp_to_mate",
    "score_delivered_checkmate",
    "score_mate_to_cp",
    "score_mate_to_mate",
    "score_optimal_claim_recommended",
    "score_received_checkmate",
    "score_standard_cp",
]
