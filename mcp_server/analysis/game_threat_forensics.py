"""Deterministic forcing-threat deltas for selected game critical moments.

The full-game coaching model already has fields for opponent forcing threats,
but those fields need a carefully bounded source of evidence. This module
compares two concrete states around the player's move:

* before the move, what checks/captures/promotions the opponent would have if
  the player hypothetically passed (only when a null move is semantically safe);
* after the played move, what checks/captures/promotions the opponent actually
  has as legal replies.

Exact-UCI comparisons are performed on the complete legal forcing-move sets.
Only the response lists are capped for wire size. This avoids false new/resolved
threat classifications when a position has more forcing moves than the display
limit. The result remains evidence, not proof that a threat is objectively
decisive or that the player failed to notice it.
"""

from __future__ import annotations

from typing import Any

import chess

from mcp_server.analysis.forensics import PIECE_NAMES
from mcp_server.models.forensics import ForcingMoveEvidence

MAX_FORCING_FACTS = 24


def _piece_label(piece: chess.Piece | None, square: chess.Square | None = None) -> str | None:
    if piece is None:
        return None
    color = "white" if piece.color == chess.WHITE else "black"
    base = f"{color}_{PIECE_NAMES[piece.piece_type]}"
    return f"{base}@{chess.square_name(square)}" if square is not None else base


def _captured_piece(board: chess.Board, move: chess.Move) -> tuple[chess.Piece | None, chess.Square | None]:
    if not board.is_capture(move):
        return None, None
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        square = move.to_square + offset
    else:
        square = move.to_square
    return board.piece_at(square), square


def forcing_move_evidence(
    board: chess.Board,
    *,
    limit: int | None = MAX_FORCING_FACTS,
) -> list[ForcingMoveEvidence]:
    """Enumerate legal checks, captures and promotions in deterministic order.

    ``limit=None`` returns the complete legal forcing set and must be used for
    semantic set comparisons. A numeric limit is presentation-only.
    """
    facts: list[ForcingMoveEvidence] = []
    for move in board.legal_moves:
        is_check = board.gives_check(move)
        is_capture = board.is_capture(move)
        if not (is_check or is_capture or move.promotion is not None):
            continue
        captured, captured_square = _captured_piece(board, move)
        is_mate = False
        if is_check:
            child = board.copy(stack=False)
            child.push(move)
            is_mate = child.is_checkmate()
        facts.append(
            ForcingMoveEvidence(
                uci=move.uci(),
                san=board.san(move),
                is_check=is_check,
                is_capture=is_capture,
                captured_piece=_piece_label(captured, captured_square),
                promotion=PIECE_NAMES.get(move.promotion) if move.promotion else None,
                is_mate=is_mate,
            )
        )
    facts.sort(
        key=lambda item: (
            not item.is_mate,
            not item.is_check,
            not item.is_capture,
            item.promotion is None,
            item.san,
            item.uci,
        )
    )
    if limit is None:
        return facts
    return facts[: max(0, limit)]


def _compare_forcing_lists(
    baseline: list[ForcingMoveEvidence],
    after: list[ForcingMoveEvidence],
) -> tuple[list[ForcingMoveEvidence], list[ForcingMoveEvidence], list[ForcingMoveEvidence]]:
    """Return new, resolved and persistent forcing moves using complete UCI sets."""
    baseline_by_uci = {item.uci: item for item in baseline}
    after_by_uci = {item.uci: item for item in after}
    newly = [item for item in after if item.uci not in baseline_by_uci]
    resolved = [item for item in baseline if item.uci not in after_by_uci]
    unresolved = [item for item in after if item.uci in baseline_by_uci]
    return newly, resolved, unresolved


def _forcing_semantic_transitions(
    baseline: list[ForcingMoveEvidence],
    after: list[ForcingMoveEvidence],
    unresolved: list[ForcingMoveEvidence],
) -> tuple[
    list[ForcingMoveEvidence],
    list[ForcingMoveEvidence],
    list[dict[str, Any]],
]:
    """Compute semantic transitions (e.g. check_to_mate) for persistent forcing moves."""
    baseline_by_uci = {item.uci: item for item in baseline}
    strengthened: list[ForcingMoveEvidence] = []
    weakened: list[ForcingMoveEvidence] = []
    transitions: list[dict[str, Any]] = []

    for item in unresolved:
        b_item = baseline_by_uci.get(item.uci)
        if b_item is None:
            continue
        trans_type: str | None = None
        direction: str | None = None

        if not b_item.is_mate and item.is_mate:
            trans_type = "check_to_mate" if b_item.is_check else "move_to_mate"
            direction = "strengthened"
        elif b_item.is_mate and not item.is_mate:
            trans_type = "mate_to_check" if item.is_check else "mate_to_quiet"
            direction = "weakened"
        elif not b_item.is_check and item.is_check:
            trans_type = "quiet_to_check"
            direction = "strengthened"
        elif b_item.is_check and not item.is_check:
            trans_type = "check_to_quiet"
            direction = "weakened"
        elif not b_item.is_capture and item.is_capture:
            trans_type = "quiet_to_capture"
            direction = "strengthened"
        elif b_item.is_capture and not item.is_capture:
            trans_type = "capture_to_quiet"
            direction = "weakened"

        if trans_type and direction:
            transitions.append(
                {
                    "uci": item.uci,
                    "san_before": b_item.san,
                    "san_after": item.san,
                    "transition": trans_type,
                    "direction": direction,
                    "before": {
                        "san": b_item.san,
                        "is_check": b_item.is_check,
                        "is_capture": b_item.is_capture,
                        "is_mate": b_item.is_mate,
                    },
                    "after": {
                        "san": item.san,
                        "is_check": item.is_check,
                        "is_capture": item.is_capture,
                        "is_mate": item.is_mate,
                    },
                }
            )
            if direction == "strengthened":
                strengthened.append(item)
            else:
                weakened.append(item)

    return strengthened, weakened, transitions


def _wire(items: list[ForcingMoveEvidence]) -> list[ForcingMoveEvidence]:
    return items[:MAX_FORCING_FACTS]


def critical_forcing_threat_delta(
    board_before: chess.Board,
    board_after: chess.Board,
) -> dict[str, Any]:
    """Compare opponent forcing threats before a move with actual replies after it.

    ``board_before`` must be the position where the player is to move and
    ``board_after`` the legal position after that move. The baseline requires a
    hypothetical pass, so it is unavailable in check or in terminal positions.
    Actual post-move forcing replies are always enumerated when the resulting
    position is non-terminal.
    """
    after_all = (
        []
        if board_after.is_game_over(claim_draw=False)
        else forcing_move_evidence(board_after, limit=None)
    )

    baseline_available = True
    baseline_reason: str | None = None
    baseline_all: list[ForcingMoveEvidence] = []
    if board_before.is_game_over(claim_draw=False):
        baseline_available = False
        baseline_reason = "terminal_position_before_move"
    elif board_before.is_check():
        baseline_available = False
        baseline_reason = "player_in_check_before_move"
    else:
        passed = board_before.copy(stack=True)
        passed.push(chess.Move.null())
        baseline_all = forcing_move_evidence(passed, limit=None)

    newly_all: list[ForcingMoveEvidence] = []
    resolved_all: list[ForcingMoveEvidence] = []
    unresolved_all: list[ForcingMoveEvidence] = []
    strengthened_all: list[ForcingMoveEvidence] = []
    weakened_all: list[ForcingMoveEvidence] = []
    transitions_all: list[dict[str, Any]] = []
    if baseline_available:
        newly_all, resolved_all, unresolved_all = _compare_forcing_lists(
            baseline_all,
            after_all,
        )
        strengthened_all, weakened_all, transitions_all = _forcing_semantic_transitions(
            baseline_all,
            after_all,
            unresolved_all,
        )

    signatures: list[str] = []
    if baseline_available and baseline_all:
        signatures.append("OPPONENT_FORCING_THREAT_BASELINE_PRESENT")
    if after_all:
        signatures.append("OPPONENT_FORCING_REPLY_AFTER_MOVE")
    if newly_all:
        signatures.append("NEW_OPPONENT_FORCING_REPLY_AFTER_MOVE")
        if any(item.is_check for item in newly_all):
            signatures.append("NEW_OPPONENT_CHECK_AFTER_MOVE")
        if any(item.is_capture for item in newly_all):
            signatures.append("NEW_OPPONENT_CAPTURE_AFTER_MOVE")
        if any(item.promotion is not None for item in newly_all):
            signatures.append("NEW_OPPONENT_PROMOTION_AFTER_MOVE")
    if resolved_all:
        signatures.append("RESOLVED_OPPONENT_FORCING_THREAT_CANDIDATE")
    if baseline_available and baseline_all and unresolved_all:
        signatures.append("FAILED_FORCING_THREAT_UPDATE_CANDIDATE")
    if strengthened_all:
        signatures.append("OPPONENT_FORCING_MOVE_STRENGTHENED")
    if weakened_all:
        signatures.append("OPPONENT_FORCING_MOVE_WEAKENED")

    baseline = _wire(baseline_all)
    after = _wire(after_all)
    newly = _wire(newly_all)
    resolved = _wire(resolved_all)
    unresolved = _wire(unresolved_all)
    strengthened = _wire(strengthened_all)
    weakened = _wire(weakened_all)
    transitions = transitions_all[:MAX_FORCING_FACTS]
    counts = {
        "baseline_total": len(baseline_all),
        "after_total": len(after_all),
        "new_total": len(newly_all),
        "resolved_total": len(resolved_all),
        "persistent_total": len(unresolved_all),
        "strengthened_total": len(strengthened_all),
        "weakened_total": len(weakened_all),
        "transitions_total": len(transitions_all),
    }
    truncated = {
        "baseline": len(baseline_all) > MAX_FORCING_FACTS,
        "after": len(after_all) > MAX_FORCING_FACTS,
        "new": len(newly_all) > MAX_FORCING_FACTS,
        "resolved": len(resolved_all) > MAX_FORCING_FACTS,
        "persistent": len(unresolved_all) > MAX_FORCING_FACTS,
        "strengthened": len(strengthened_all) > MAX_FORCING_FACTS,
        "weakened": len(weakened_all) > MAX_FORCING_FACTS,
        "transitions": len(transitions_all) > MAX_FORCING_FACTS,
    }

    scope = (
        "The pre-move baseline is a hypothetical null-move probe and is unavailable while "
        "the player is in check or the position is terminal. Post-move forcing replies are "
        "real legal checks, captures and promotions. New/resolved/persistent classifications "
        "use the complete legal exact-UCI sets before presentation truncation. Returned lists "
        "are capped for wire size and accompanied by total counts/truncation flags. The delta "
        "does not prove that a threat is objectively decisive or that the player noticed or "
        "missed it."
    )
    return {
        "baseline_available": baseline_available,
        "baseline_reason": baseline_reason,
        "opponent_forcing_threat_candidates_if_pass_before": baseline,
        "opponent_forcing_moves_after_played": after,
        "newly_enabled_opponent_forcing_moves_after_played": newly,
        "resolved_opponent_forcing_threat_candidates": resolved,
        "unresolved_exact_opponent_forcing_threat_candidates": unresolved,
        "strengthened_opponent_forcing_moves": strengthened,
        "weakened_opponent_forcing_moves": weakened,
        "forcing_move_semantic_transitions": transitions,
        "forcing_move_counts": counts,
        "presentation_truncated": truncated,
        "signatures": sorted(set(signatures)),
        "proof_scope": scope,
        "trace": {
            "baseline_available": baseline_available,
            "baseline_reason": baseline_reason,
            "opponent_forcing_threat_candidates_if_pass_before": [
                item.model_dump() for item in baseline
            ],
            "opponent_forcing_moves_after_played": [item.model_dump() for item in after],
            "newly_enabled_opponent_forcing_moves_after_played": [
                item.model_dump() for item in newly
            ],
            "resolved_opponent_forcing_threat_candidates": [
                item.model_dump() for item in resolved
            ],
            "unresolved_exact_opponent_forcing_threat_candidates": [
                item.model_dump() for item in unresolved
            ],
            "strengthened_opponent_forcing_moves": [
                item.model_dump() for item in strengthened
            ],
            "weakened_opponent_forcing_moves": [
                item.model_dump() for item in weakened
            ],
            "forcing_move_semantic_transitions": transitions,
            "forcing_move_counts": counts,
            "presentation_truncated": truncated,
            "proof_scope": scope,
        },
    }
