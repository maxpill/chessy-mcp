"""Rule-status evaluation.

Two parts:

    - ``make_rule_status`` — pure factory that builds a ``RuleStatus`` for a
      terminal position (no claims, no repetition bookkeeping).
    - ``evaluate_rule_status`` — full evaluator that combines terminal-checks,
      claim-eligibility (50-move / threefold repetition) with explicit history
      provenance, and the ``recommended_action`` policy.

``RuleStatus`` itself is defined in ``mcp_server.domain.rule_status`` (the
domain layer) so non-rules modules can type-hint against it. This module is
the IMPLEMENTATION site for the function and the back-compat re-export.
"""

from __future__ import annotations

import chess

from mcp_server.domain.rule_status import RuleStatus
from mcp_server.rules.constants import TERMINAL_VS_HISTORY_INDEPENDENT
from mcp_server.rules.terminal import is_locked_dead_position

__all__ = ["evaluate_rule_status", "make_rule_status"]


def make_rule_status(
    terminal: str | None,
    winner: str | None = None,
    recommended_action: str = "game_over",
    history_dep: bool = False,
    repetition: str = "none",
    history_state: str = "incomplete",
) -> RuleStatus:
    """Build a ``RuleStatus`` for a terminal position with explicit knobs."""
    return RuleStatus(
        terminal=terminal,
        winner=winner,
        recommended_action=recommended_action,
        history_dependent_status=history_dep,
        requires_move_stack=history_dep,
        fen_sufficient_for_status=not history_dep,
        history_completeness=(
            "not_required" if terminal in TERMINAL_VS_HISTORY_INDEPENDENT else history_state
        ),
        repetition_status=repetition,
    )


def evaluate_rule_status(
    board: chess.Board,
    mover_score: int | None = None,
    mate_for_mover: int | None = None,
    history_complete: str | bool = "incomplete",
    zeroing_move_best_score: int | None = None,
    zeroing_move_best_mate: int | None = None,
) -> RuleStatus:
    """Evaluate terminal rules + optional draw claims with explicit history provenance."""
    history_state = _normalize_history_state(history_complete)
    has_history = history_state in {"complete", "partial"}
    full_history = history_state == "complete"

    # ---- terminal predicates (first one wins) -----------------------------
    if board.is_checkmate():
        return make_rule_status(
            "checkmate",
            winner="black" if board.turn == chess.WHITE else "white",
            history_state=history_state,
        )
    if board.is_stalemate():
        return make_rule_status("stalemate", history_state=history_state)
    if board.is_insufficient_material():
        return make_rule_status("insufficient_material", history_state=history_state)
    if board.is_seventyfive_moves():
        return make_rule_status("seventyfive_moves", history_state=history_state)
    if has_history and board.is_fivefold_repetition():
        return make_rule_status(
            "fivefold_repetition",
            history_dep=True,
            repetition="fivefold",
            history_state=history_state,
        )
    if is_locked_dead_position(board):
        return make_rule_status("dead_position", history_state=history_state)
    if board.is_game_over(claim_draw=False):
        return make_rule_status("game_over", history_state=history_state)

    # ---- claim bookkeeping ------------------------------------------------
    claim_now, reasons_now, intended, reasons_by_uci = _collect_claim_proofs(
        board, has_history, reasons_now_so_far=[]
    )
    intended_ucis = [m.uci() for m in intended]
    intended_sans = [_san_or_uci(board, m) for m in intended]
    intended_reasons_set = _aggregate_intended_reasons(reasons_now, reasons_by_uci)
    intended_reasons_ordered = list(intended_reasons_set)

    can_claim_with_intended_move = bool(intended)
    all_claim_reasons = list(dict.fromkeys(reasons_now + intended_reasons_ordered))
    can_claim_draw = claim_now or can_claim_with_intended_move
    claim_move_san = intended_sans[0] if intended_sans else None
    claim_move_uci = intended_ucis[0] if intended_ucis else None

    # ---- recommended action -----------------------------------------------
    from mcp_server.rules.action_choice import choose_recommended_action  # cycle

    recommended_action = choose_recommended_action(
        board,
        can_claim_now=claim_now,
        can_claim_with_intended_move=can_claim_with_intended_move,
        mover_score=mover_score,
        mate_for_mover=mate_for_mover,
        zeroing_move_best_score=zeroing_move_best_score,
        zeroing_move_best_mate=zeroing_move_best_mate,
    )

    # ---- repetition bookkeeping -------------------------------------------
    repetition_proven = (
        "threefold_repetition" in reasons_now or "threefold_repetition" in intended_reasons_ordered
    )
    if repetition_proven:
        repetition_status = "threefold_claimable"
    elif full_history:
        repetition_status = "none"
    else:
        repetition_status = "unknown"

    requires_stack = repetition_proven or repetition_status == "unknown"
    return RuleStatus(
        terminal=None,
        winner=None,
        can_claim_now=claim_now,
        claim_reasons_now=reasons_now,
        can_claim_with_intended_move=can_claim_with_intended_move,
        intended_claim_moves=intended,
        intended_claim_sans=intended_sans,
        intended_claim_ucis=intended_ucis,
        intended_claim_reasons_by_uci=reasons_by_uci,
        claim_reasons=all_claim_reasons,
        can_claim_draw=can_claim_draw,
        claim_moves=intended_sans,
        claim_move=claim_move_san,
        claim_move_uci=claim_move_uci,
        claim_move_san=claim_move_san,
        recommended_action=recommended_action,
        history_dependent_status=requires_stack,
        requires_move_stack=requires_stack,
        fen_sufficient_for_status=not requires_stack,
        history_completeness=history_state,
        repetition_status=repetition_status,
    )


# ------------------------------------------------------------------------------


def _normalize_history_state(value: str | bool) -> str:
    if isinstance(value, bool):
        return "complete" if value else "incomplete"
    if value not in {"complete", "partial", "incomplete", "not_required"}:
        raise ValueError(f"INVALID_HISTORY_PROVENANCE: {value}")
    return value


def _collect_claim_proofs(
    board: chess.Board,
    has_history: bool,
    *,
    reasons_now_so_far: list[str] | None = None,
) -> tuple[bool, list[str], list[chess.Move], dict[str, list[str]]]:
    """Compute (claim_now, reasons_now, intended_moves, reasons_by_uci)."""
    can_claim_now = False
    reasons_now: list[str] = []
    if board.is_fifty_moves():
        can_claim_now = True
        reasons_now.append("fifty_moves")
    if has_history and board.is_repetition(3):
        can_claim_now = True
        reasons_now.append("threefold_repetition")

    intended: list[chess.Move] = []
    reasons_by_uci: dict[str, list[str]] = {}
    for cand in board.legal_moves:
        cand_uci = cand.uci()
        reasons_for_move: list[str] = []

        if "fifty_moves" not in reasons_now:
            is_pawn = board.piece_type_at(cand.from_square) == chess.PAWN
            is_capture = board.is_capture(cand)
            if not is_pawn and not is_capture and board.halfmove_clock + 1 >= 100:
                reasons_for_move.append("fifty_moves")

        if "threefold_repetition" not in reasons_now and has_history:
            child = board.copy(stack=True)
            child.push(cand)
            if child.is_repetition(3):
                reasons_for_move.append("threefold_repetition")

        if reasons_for_move:
            intended.append(cand)
            reasons_by_uci[cand_uci] = reasons_for_move

    return can_claim_now, reasons_now, intended, reasons_by_uci


def _aggregate_intended_reasons(
    reasons_now: list[str], reasons_by_uci: dict[str, list[str]]
) -> set[str]:
    """Build the deduped set of human-readable reasons across all intended moves."""
    bag: set[str] = set()
    for reasons in reasons_by_uci.values():
        bag.update(reasons)
    bag.update(reasons_now)
    return bag


def _san_or_uci(board: chess.Board, move: chess.Move) -> str:
    try:
        return board.san(move)
    except Exception:
        return move.uci()
