"""MultiPV candidate evaluation helpers — pure functions.

Extracted from ``mcp_server.tools.top_moves``. Two reusable pieces live
here:

  * :func:`select_root_recommended_action` — lifts the recommended action
    for the root position off a list of MultiPV candidates, with audit
    B-04/B-05 zeroing-post-state consideration for draw-pollution
    scenarios.
  * :func:`rank_candidate` — chess-correct ordering key for the
    candidate list (audit U-01, mate-precedence, post-state draw-pollution
    guard).
  * :data:`MATE_RANK_CEILING` / :data:`MATE_RANK_VALUE` — sentinel
    thresholds exposed for the rank key.

These functions are side-effect free; the orchestrator handles the I/O.
"""

from __future__ import annotations

from collections.abc import Iterable

import chess

from mcp_server.models import MCPEval
from mcp_server.rules import choose_recommended_action

# Audit U-01: clamp saturated cp scores so any sentinel like cp=±20000
# can never outrank a forced mate. Stockfish occasionally emits cp=±20000
# for terminal-winning positions whose mate distance hasn't been computed
# at the requested depth.
MATE_RANK_CEILING = 9999.0
MATE_RANK_VALUE = 10000.0


def select_root_recommended_action(
    items: list[MCPEval],
    *,
    board: chess.Board,
    rule_status,
    sign: int,
) -> str:
    """Pick the recommended action for the root position from a ranked
    candidate list.

    Honors audit B-04/B-05 by looking at zeroing-move post-state values
    when they exist (so the policy can prefer play_move over claim_draw
    when a winning zeroing capture exists). Falls back to the engine's
    rule_status.recommended_action when the candidate list is empty or
    every candidate lacks best_move.
    """
    if not items:
        return rule_status.recommended_action
    best = items[0]
    eff_mate = best.post_state_mate if best.post_state_mate is not None else best.mate
    eff_cp = best.post_state_cp if best.post_state_cp is not None else best.cp
    if eff_mate is not None:
        mover_score: int | None = sign * eff_mate * 1000
    elif eff_cp is not None:
        mover_score = sign * eff_cp
    else:
        mover_score = None
    mate_for_mover = sign * eff_mate if eff_mate is not None else None

    zeroing_best_cp: int | None = None
    zeroing_best_mate: int | None = None
    for item in items:
        if not item.best_move:
            continue
        try:
            bm = chess.Move.from_uci(item.best_move)
        except Exception:
            continue
        if not (board.is_capture(bm) or board.piece_type_at(bm.from_square) == chess.PAWN):
            continue
        eff_cp = item.post_state_cp if item.post_state_cp is not None else item.cp
        eff_mate = item.post_state_mate if item.post_state_mate is not None else item.mate
        if eff_mate is not None:
            mover_mate = sign * eff_mate
            if mover_mate > 0 and (zeroing_best_mate is None or mover_mate > zeroing_best_mate):
                zeroing_best_mate = mover_mate
        elif eff_cp is not None:
            mover_cp = sign * eff_cp
            if zeroing_best_cp is None or mover_cp > zeroing_best_cp:
                zeroing_best_cp = mover_cp
    return choose_recommended_action(
        board,
        can_claim_now=rule_status.can_claim_now,
        can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
        mover_score=mover_score,
        mate_for_mover=mate_for_mover,
        zeroing_move_best_score=zeroing_best_cp,
        zeroing_move_best_mate=zeroing_best_mate,
    )


def rank_candidate(eval_item: MCPEval, *, sign: int) -> float:
    """Chess-correct total-ordering key for a MultiPV candidate.

    Order: delivered mate > forced mate for mover > finite-cp win >
    draw > finite-cp loss > forced mate against mover. Saturates cp at
    ``MATE_RANK_CEILING`` so a cp=±20000 sentinel cannot outrank a forced
    mate.
    """
    if eval_item.post_terminal_status == "checkmate":
        return MATE_RANK_VALUE
    if eval_item.post_terminal_status in (
        "stalemate",
        "insufficient_material",
        "seventyfive_moves",
        "fivefold_repetition",
        "dead_position",
    ):
        return 0.0
    eff_mate = (
        eval_item.post_state_mate if eval_item.post_state_mate is not None else eval_item.mate
    )
    if eff_mate is not None:
        mover_mate = sign * eff_mate
        if mover_mate > 0:
            return MATE_RANK_VALUE - abs(mover_mate)
        return -MATE_RANK_VALUE + abs(mover_mate)
    eff_cp = eval_item.post_state_cp if eval_item.post_state_cp is not None else eval_item.cp
    if eff_cp is not None:
        mover_cp = sign * eff_cp
        if mover_cp > MATE_RANK_CEILING:
            return MATE_RANK_CEILING
        if mover_cp < -MATE_RANK_CEILING:
            return -MATE_RANK_CEILING
        return float(mover_cp)
    return 0.0


def sort_candidates(items: Iterable[MCPEval], *, sign: int) -> list[MCPEval]:
    """Stable chess-correct ordering of the candidate list."""
    return sorted(items, key=lambda item: rank_candidate(item, sign=sign), reverse=True)
