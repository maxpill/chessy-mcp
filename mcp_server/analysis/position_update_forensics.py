"""Corrected board-state evidence for updates created by the opponent's last move.

The original proxy detected newly attacked/undefended pieces and new pins, but a
piece that was *already attacked* could become tactically urgent when the
opponent removed its last defender. That defender-removal case is central to
coaching and must not be missed merely because the attacker count was already
non-zero.

This module replaces only the existing ``position_update_after_opponent_move``
mechanism in rich ``classify_move`` evidence. It makes no engine calls and does
not infer the player's mental process.
"""

from __future__ import annotations

from typing import Any

import chess

from mcp_server.analysis.forensics import PIECE_NAMES
from mcp_server.models.forensics import ForensicMoveAnalysis


def _label(piece: chess.Piece, square: chess.Square) -> str:
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color}_{PIECE_NAMES[piece.piece_type]}@{chess.square_name(square)}"


def _state(board: chess.Board, color: chess.Color) -> dict[str, tuple[int, int, bool, chess.Square]]:
    out: dict[str, tuple[int, int, bool, chess.Square]] = {}
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        out[_label(piece, square)] = (
            len(board.attackers(not color, square)),
            len(board.attackers(color, square)),
            board.is_pinned(color, square),
            square,
        )
    return out


def _state_for_original_target_after_move(
    after_user: chess.Board,
    *,
    original_label: str,
    original_square: chess.Square,
    played_move: chess.Move,
    mover: chess.Color,
) -> tuple[int, int, bool] | None:
    square = original_square
    if played_move.from_square == original_square:
        square = played_move.to_square
    piece = after_user.piece_at(square)
    if piece is None or piece.color != mover or piece.piece_type == chess.KING:
        return None
    # If the original piece did not move, guard against a different piece now
    # occupying the square in an unusual transformation/capture sequence.
    if played_move.from_square != original_square and _label(piece, square) != original_label:
        return None
    return (
        len(after_user.attackers(not mover, square)),
        len(after_user.attackers(mover, square)),
        after_user.is_pinned(mover, square),
    )


def build_position_update_evidence(
    board_before: chess.Board,
    played_move: chess.Move | None,
) -> dict[str, Any]:
    """Return deterministic urgent-change evidence from the previous opponent move."""
    if not board_before.move_stack:
        return {
            "mechanism": "position_update_after_opponent_move",
            "history_available": False,
            "inference_boundary": (
                "A naked FEN has no previous-move history, so failed-position-update "
                "evidence cannot be reconstructed."
            ),
        }

    before_opponent = board_before.copy(stack=True)
    opponent_move = before_opponent.pop()
    try:
        opponent_san = before_opponent.san(opponent_move)
    except (ValueError, AssertionError):
        opponent_san = opponent_move.uci()

    mover = board_before.turn
    prior = _state(before_opponent, mover)
    current = _state(board_before, mover)

    newly_attacked: list[str] = []
    newly_exposed: list[str] = []
    newly_pinned: list[str] = []
    defender_losses: list[str] = []
    target_squares: dict[str, chess.Square] = {}

    for label, (attackers_after, defenders_after, pinned_after, square) in current.items():
        previous = prior.get(label)
        if previous is None:
            continue
        attackers_before, defenders_before, pinned_before, _ = previous
        target_squares[label] = square

        if attackers_before == 0 and attackers_after > 0:
            newly_attacked.append(label)
        # The urgent exposure transition is defined by current vulnerability,
        # not only by whether an attacker was newly created. This catches
        # removal of the last defender from an already-attacked piece.
        if (
            attackers_after > 0
            and defenders_after == 0
            and (attackers_before == 0 or defenders_before > 0)
        ):
            newly_exposed.append(label)
        if defenders_after < defenders_before:
            defender_losses.append(f"{label}:{defenders_before}->{defenders_after}")
        if not pinned_before and pinned_after:
            newly_pinned.append(label)

    king_in_check = board_before.is_check()
    urgent_targets = sorted(set(newly_exposed + newly_pinned))
    urgent_change = king_in_check or bool(urgent_targets)

    unresolved: list[str] = []
    addresses_change: bool | None = None
    if played_move is not None and played_move in board_before.legal_moves:
        after_user = board_before.copy(stack=True)
        after_user.push(played_move)
        if after_user.is_checkmate():
            unresolved = []
            addresses_change = True
        else:
            for label in newly_exposed:
                square = target_squares[label]
                state = _state_for_original_target_after_move(
                    after_user,
                    original_label=label,
                    original_square=square,
                    played_move=played_move,
                    mover=mover,
                )
                if state is not None and state[0] > 0 and state[1] == 0:
                    unresolved.append(label)
            for label in newly_pinned:
                square = target_squares[label]
                state = _state_for_original_target_after_move(
                    after_user,
                    original_label=label,
                    original_square=square,
                    played_move=played_move,
                    mover=mover,
                )
                if state is not None and state[2]:
                    unresolved.append(label)
            if king_in_check and after_user.is_check():
                unresolved.append("king_in_check")
            addresses_change = not unresolved

    return {
        "mechanism": "position_update_after_opponent_move",
        "history_available": True,
        "opponent_move_uci": opponent_move.uci(),
        "opponent_move_san": opponent_san,
        "king_in_check_after_opponent_move": king_in_check,
        "newly_attacked_user_pieces": sorted(newly_attacked),
        "newly_exposed_user_pieces": sorted(set(newly_exposed)),
        "newly_pinned_user_pieces": sorted(newly_pinned),
        "defender_count_losses": sorted(defender_losses),
        "opponent_move_created_urgent_change": urgent_change,
        "played_move_addresses_change": addresses_change,
        "unresolved_urgent_targets_after_played_move": sorted(set(unresolved)),
        "proof_scope": (
            "Urgent change is a deterministic board-state proxy: check, a newly pinned piece, "
            "or a piece that becomes attacked with zero defenders, including removal of the "
            "last defender from a piece that was already attacked. Persistence tracks the same "
            "piece through the played move when it moves. This can support but does not prove "
            "a coaching label such as plan persistence or failed position update."
        ),
    }


def apply_position_update_correction(
    result: ForensicMoveAnalysis,
    board_before: chess.Board,
    *,
    played_move: chess.Move | None,
) -> ForensicMoveAnalysis:
    evidence = result.forensics
    if evidence is None:
        return result

    corrected = build_position_update_evidence(board_before, played_move)
    mechanisms = [
        item
        for item in evidence.mechanism_evidence
        if item.get("mechanism") != "position_update_after_opponent_move"
    ]
    mechanisms.append(corrected)

    signatures = [
        item
        for item in evidence.evidence_signatures
        if item not in {"OPPONENT_MOVE_CREATED_URGENT_CHANGE", "FAILED_POSITION_UPDATE_CANDIDATE"}
    ]
    mover_won = (
        result.eval_after.status == "checkmate"
        and (
            (board_before.turn == chess.WHITE and result.eval_after.winner == "white")
            or (board_before.turn == chess.BLACK and result.eval_after.winner == "black")
        )
    )
    if not mover_won and corrected.get("opponent_move_created_urgent_change"):
        signatures.append("OPPONENT_MOVE_CREATED_URGENT_CHANGE")
        if corrected.get("played_move_addresses_change") is False:
            signatures.append("FAILED_POSITION_UPDATE_CANDIDATE")

    updated = evidence.model_copy(
        update={
            "mechanism_evidence": mechanisms,
            "evidence_signatures": sorted(set(signatures)),
        }
    )
    return result.model_copy(update={"forensics": updated})
