"""Materialization timing for the opponent's strongest returned reply.

This module refines existing reply-failure evidence without another engine
search. It walks only the already returned principal variation and records when
a concrete material loss first appears from the played side's perspective.
The result is board/line evidence, not a diagnosis of what the player calculated.
"""

from __future__ import annotations

from typing import Any

import chess

from mcp_server.analysis.forensics import PIECE_NAMES, PIECE_VALUES
from mcp_server.analysis.position_update_forensics import apply_position_update_correction
from mcp_server.models.forensics import ForensicMoveAnalysis


def _material_balance(board: chess.Board, color: chess.Color) -> int:
    own = 0
    opponent = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == color:
            own += value
        else:
            opponent += value
    return own - opponent


def _captured_piece_label(board: chess.Board, move: chess.Move) -> str | None:
    if not board.is_capture(move):
        return None
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        square = move.to_square + offset
    else:
        square = move.to_square
    piece = board.piece_at(square)
    if piece is None:
        return None
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color}_{PIECE_NAMES[piece.piece_type]}@{chess.square_name(square)}"


def build_reply_materialization_trace(
    board_after_played: chess.Board,
    pv_uci: list[str],
    *,
    mover: chess.Color,
) -> dict[str, Any]:
    """Walk the returned PV and locate the first concrete material loss.

    ``mover`` is the side that made the classified move. A material loss is
    recorded when its cumulative material balance has worsened by at least one
    pawn relative to the position immediately after the played move.
    """
    work = board_after_played.copy(stack=True)
    baseline = _material_balance(work, mover)
    first_material_loss_ply: int | None = None
    first_material_loss_cp: int | None = None
    steps: list[dict[str, Any]] = []
    complete = True
    termination_reason = "pv_exhausted" if pv_uci else "no_pv"

    for ply, raw in enumerate(pv_uci, start=1):
        try:
            move = chess.Move.from_uci(str(raw).lower())
        except (ValueError, chess.InvalidMoveError):
            complete = False
            termination_reason = "invalid_pv_move"
            break
        if move not in work.legal_moves:
            complete = False
            termination_reason = "invalid_pv_move"
            break

        moving_side = "white" if work.turn == chess.WHITE else "black"
        before = _material_balance(work, mover)
        san = work.san(move)
        captured_piece = _captured_piece_label(work, move)
        is_check = work.gives_check(move)
        is_capture = work.is_capture(move)
        is_promotion = move.promotion is not None
        work.push(move)
        after = _material_balance(work, mover)
        cumulative = after - baseline
        if first_material_loss_ply is None and cumulative <= -PIECE_VALUES[chess.PAWN]:
            first_material_loss_ply = ply
            first_material_loss_cp = cumulative

        steps.append(
            {
                "ply": ply,
                "side": moving_side,
                "uci": move.uci(),
                "san": san,
                "is_check": is_check,
                "is_capture": is_capture,
                "is_promotion": is_promotion,
                "captured_piece": captured_piece,
                "material_change_for_mover_this_ply_cp": after - before,
                "cumulative_material_change_for_mover_cp": cumulative,
            }
        )
        if work.is_game_over(claim_draw=False):
            termination_reason = "terminal_position"
            break

    final_change = _material_balance(work, mover) - baseline
    return {
        "steps": steps,
        "first_material_loss_ply_for_mover": first_material_loss_ply,
        "loss_realized_within_plies": first_material_loss_ply,
        "first_material_loss_cp": first_material_loss_cp,
        "final_material_change_for_mover_cp": final_change,
        "returned_line_plies_walked": len(steps),
        "returned_line_complete": complete,
        "termination_reason": termination_reason,
        "proof_scope": (
            "Material timing is reconstructed only from the already returned principal "
            "variation. It does not extend engine search, prove that the PV is forced, or "
            "establish whether the player generated the opponent reply but stopped calculating."
        ),
    }


def apply_reply_materialization_evidence(
    result: ForensicMoveAnalysis,
    board_before: chess.Board,
    *,
    played_move: chess.Move | None,
) -> ForensicMoveAnalysis:
    """Attach exact materialization timing to existing reply-failure evidence."""
    result = apply_position_update_correction(
        result,
        board_before,
        played_move=played_move,
    )
    evidence = result.forensics
    if evidence is None or evidence.strongest_reply is None:
        return result

    board_after = board_before.copy(stack=True)
    if played_move is not None:
        if played_move not in board_before.legal_moves:
            return result
        board_after.push(played_move)

    reply = evidence.strongest_reply
    line = list(evidence.forced_line.uci)
    starts_with_reply = bool(line and line[0] == reply.uci)
    if not starts_with_reply:
        # Never splice a strongest reply onto an unrelated PV. We can still
        # report the immediate reply itself, but deeper timing would be invented.
        try:
            reply_move = chess.Move.from_uci(reply.uci)
        except (ValueError, chess.InvalidMoveError):
            return result
        if reply_move not in board_after.legal_moves:
            return result
        line = [reply.uci]

    trace = build_reply_materialization_trace(
        board_after,
        line,
        mover=board_before.turn,
    )
    first_loss = trace["first_material_loss_ply_for_mover"]

    mechanisms = [dict(item) for item in evidence.mechanism_evidence]
    profile_index = next(
        (
            index
            for index, item in enumerate(mechanisms)
            if item.get("mechanism") == "reply_failure_profile"
        ),
        None,
    )
    additions = {
        "returned_pv_starts_with_strongest_reply": starts_with_reply,
        "material_trajectory_for_mover": trace["steps"],
        "first_material_loss_ply_for_mover": first_loss,
        "loss_realized_within_plies": trace["loss_realized_within_plies"],
        "first_material_loss_cp": trace["first_material_loss_cp"],
        "final_material_change_for_mover_cp": trace["final_material_change_for_mover_cp"],
        "returned_line_plies_walked": trace["returned_line_plies_walked"],
        "material_trajectory_complete": trace["returned_line_complete"],
        "material_trajectory_termination_reason": trace["termination_reason"],
        "materialization_proof_scope": trace["proof_scope"],
    }
    if profile_index is None:
        mechanisms.append(
            {
                "mechanism": "reply_failure_profile",
                "reply": reply.san,
                "reply_type": (
                    "check_capture"
                    if reply.is_check and reply.is_capture
                    else "check"
                    if reply.is_check
                    else "capture"
                    if reply.is_capture
                    else "quiet"
                ),
                "opponent_first_pv_move_forcing": reply.is_forcing,
                **additions,
            }
        )
    else:
        mechanisms[profile_index] = {**mechanisms[profile_index], **additions}

    signatures = list(evidence.evidence_signatures)
    if reply.is_forcing and isinstance(first_loss, int):
        if first_loss == 1:
            signatures.append("FORCING_REPLY_MATERIAL_LOSS_IMMEDIATE")
        elif first_loss > 1:
            signatures.append("DELAYED_MATERIALIZATION_AFTER_FORCING_REPLY")
    if not starts_with_reply:
        signatures.append("REPLY_PV_ALIGNMENT_UNAVAILABLE")

    updated = evidence.model_copy(
        update={
            "mechanism_evidence": mechanisms,
            "evidence_signatures": sorted(set(signatures)),
        }
    )
    return result.model_copy(update={"forensics": updated})
