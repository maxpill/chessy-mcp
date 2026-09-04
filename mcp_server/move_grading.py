"""Backwards-compatible shim — the 1,158-line monolith now lives in
``mcp_server.analysis.move_grading``.

Original code is fully preserved as a thin re-export module. Tests and
callers that used ``from mcp_server.move_grading import score_played_move,
_score_*, _is_*, _material_balance`` continue to work unchanged.

The new architecture:

    - :mod:`mcp_server.analysis.move_grading.strategies.terminal`
    - :mod:`mcp_server.analysis.move_grading.strategies.draw`
    - :mod:`mcp_server.analysis.move_grading.strategies.transitions`
    - :mod:`mcp_server.analysis.move_grading.strategies.standard`
    - :mod:`mcp_server.analysis.move_grading.helpers` — predicate utilities
    - :mod:`mcp_server.analysis.move_grading.dispatcher` — strategy walk
    - :mod:`mcp_server.analysis.move_grading.grader` — MoveGrader service

Every audit invariant (B-01..B-05, C-01, C-02, H-01..H-03, L-06, M-04, M-05,
P0, P1, P2, P3, U-02..U-15, R4-§C, R5) is byte-identical to the pre-split code.
"""

from mcp_server.analysis.move_grading import (
    MoveGrader,
    is_after_losing,
    is_after_winning,
    is_before_winning,
    is_down_material,
    is_mover_forced_win,
    material_balance,
    score_blundered_terminal_draw,
    score_claim_draw_action,
    score_conceded_draw,
    score_cp_to_mate,
    score_delivered_checkmate,
    score_mate_to_cp,
    score_mate_to_mate,
    score_optimal_claim_recommended,
    score_played_move,
    score_received_checkmate,
    score_standard_cp,
)

# Back-compat underscored aliases.
_is_before_winning = is_before_winning
_is_mover_forced_win = is_mover_forced_win
_material_balance = material_balance
_is_after_losing = is_after_losing
_is_after_winning = is_after_winning
_is_down_material = is_down_material

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
    "MoveGrader",
    "_is_after_losing",
    "_is_after_winning",
    "_is_before_winning",
    "_is_down_material",
    "_is_mover_forced_win",
    "_material_balance",
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
    "is_after_losing",
    "is_after_winning",
    "is_before_winning",
    "is_down_material",
    "is_mover_forced_win",
    "material_balance",
    "score_blundered_terminal_draw",
    "score_claim_draw_action",
    "score_conceded_draw",
    "score_cp_to_mate",
    "score_delivered_checkmate",
    "score_mate_to_cp",
    "score_mate_to_mate",
    "score_optimal_claim_recommended",
    "score_played_move",
    "score_received_checkmate",
    "score_standard_cp",
]
