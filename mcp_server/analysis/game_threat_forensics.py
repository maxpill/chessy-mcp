"""Deterministic forcing-threat deltas for selected game critical moments.

The full-game coaching model already has fields for opponent forcing threats,
but those fields need a carefully bounded source of evidence. This module
compares two concrete states around the player's move:

* before the move, what checks/captures/promotions the opponent would have if
  the player hypothetically passed (only when a null move is semantically safe);
* after the played move, what checks/captures/promotions the opponent actually
  has as legal replies.

The comparison is exact-UCI only. It is useful evidence for position-update
failures, but it is not proof that a threat is objectively decisive or that the
player failed to notice it.
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


def forcing_move_evidence(board: chess.Board) -> list[ForcingMoveEvidence]:
    """Enumerate legal checks, captures and promotions in deterministic order."""
    facts: list[ForcingMoveEvidence] = []
    for move in board.legal_moves:
        is_check = board.gives_check(move)
        is_capture = board.is_capture(move)
        if not (is_check or is_capture or move.promotion is not None):
            continue
        captured, captured_square = _captured_piece(board, move)
        facts.append(
            ForcingMoveEvidence(
                uci=move.uci(),
                san=board.san(move),
                is_check=is_check,
                is_capture=is_capture,
                captured_piece=_piece_label(captured, captured_square),
                promotion=PIECE_NAMES.get(move.promotion) if move.promotion else None,
            )
        )
    facts.sort(
        key=lambda item: (
            not item.is_check,
            not item.is_capture,
            item.promotion is None,
            item.san,
            item.uci,
        )
    )
    return facts[:MAX_FORCING_FACTS]


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
    after = [] if board_after.is_game_over(claim_draw=False) else forcing_move_evidence(board_after)

    baseline_available = True
    baseline_reason: str | None = None
    baseline: list[ForcingMoveEvidence] = []
    if board_before.is_game_over(claim_draw=False):
        baseline_available = False
        baseline_reason = "terminal_position_before_move"
    elif board_before.is_check():
        baseline_available = False
        baseline_reason = "player_in_check_before_move"
    else:
        passed = board_before.copy(stack=True)
        passed.push(chess.Move.null())
        baseline = forcing_move_evidence(passed)

    newly: list[ForcingMoveEvidence] = []
    resolved: list[ForcingMoveEvidence] = []
    unresolved: list[ForcingMoveEvidence] = []
    if baseline_available:
        baseline_by_uci = {item.uci: item for item in baseline}
        after_by_uci = {item.uci: item for item in after}
        newly = [item for item in after if item.uci not in baseline_by_uci]
        resolved = [item for item in baseline if item.uci not in after_by_uci]
        unresolved = [item for item in after if item.uci in baseline_by_uci]

    signatures: list[str] = []
    if baseline_available and baseline:
        signatures.append("OPPONENT_FORCING_THREAT_BASELINE_PRESENT")
    if after:
        signatures.append("OPPONENT_FORCING_REPLY_AFTER_MOVE")
    if newly:
        signatures.append("NEW_OPPONENT_FORCING_REPLY_AFTER_MOVE")
        if any(item.is_check for item in newly):
            signatures.append("NEW_OPPONENT_CHECK_AFTER_MOVE")
        if any(item.is_capture for item in newly):
            signatures.append("NEW_OPPONENT_CAPTURE_AFTER_MOVE")
        if any(item.promotion is not None for item in newly):
            signatures.append("NEW_OPPONENT_PROMOTION_AFTER_MOVE")
    if resolved:
        signatures.append("RESOLVED_OPPONENT_FORCING_THREAT_CANDIDATE")
    if baseline_available and baseline and unresolved:
        signatures.append("FAILED_FORCING_THREAT_UPDATE_CANDIDATE")

    scope = (
        "The pre-move baseline is a hypothetical null-move probe and is unavailable while "
        "the player is in check or the position is terminal. Post-move forcing replies are "
        "real legal checks, captures and promotions. New/resolved/persistent comparisons use "
        "exact UCI only and do not prove that a threat is objectively decisive or that the "
        "player noticed or missed it."
    )
    return {
        "baseline_available": baseline_available,
        "baseline_reason": baseline_reason,
        "opponent_forcing_threat_candidates_if_pass_before": baseline,
        "opponent_forcing_moves_after_played": after,
        "newly_enabled_opponent_forcing_moves_after_played": newly,
        "resolved_opponent_forcing_threat_candidates": resolved,
        "unresolved_exact_opponent_forcing_threat_candidates": unresolved,
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
            "proof_scope": scope,
        },
    }
