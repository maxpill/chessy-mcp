"""Deterministic immediate-mate evidence for rich ``classify_move`` output.

A common coaching failure mode is not an evaluation nuance but simply missing a
legal mate, allowing one, or failing to answer an immediate mating threat.
Stockfish evaluations can encode those facts, but a coach benefits from an
explicit board-grounded answer.

This module performs only legal move generation, checkmate verification and a
bounded hypothetical null-move probe. It adds no engine search and makes no
claim about why a player missed the move.
"""

from __future__ import annotations

from typing import Any

import chess

from mcp_server.analysis.forensics import PIECE_NAMES
from mcp_server.models.forensics import ForensicMoveAnalysis

MAX_MATE_IN_ONE_MOVES = 16


def _color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def _piece_label(piece: chess.Piece | None, square: chess.Square | None = None) -> str | None:
    if piece is None:
        return None
    base = f"{_color_name(piece.color)}_{PIECE_NAMES[piece.piece_type]}"
    return f"{base}@{chess.square_name(square)}" if square is not None else base


def _back_rank_geometry(
    board: chess.Board,
    move: chess.Move,
    moving_piece: chess.Piece | None,
) -> bool:
    if moving_piece is None or moving_piece.piece_type not in {chess.ROOK, chess.QUEEN}:
        return False
    post = board.copy(stack=True)
    mover = board.turn
    post.push(move)
    king_square = post.king(not mover)
    if king_square is None:
        return False
    king_rank = chess.square_rank(king_square)
    enemy_home_rank = 0 if mover == chess.BLACK else 7
    if king_rank != enemy_home_rank:
        return False
    return chess.square_rank(move.to_square) == king_rank


def mate_in_one_moves(board: chess.Board) -> list[dict[str, Any]]:
    """Return every legal move that checkmates immediately, bounded for wire size."""
    if board.is_game_over(claim_draw=False):
        return []

    out: list[dict[str, Any]] = []
    for move in board.legal_moves:
        if not board.gives_check(move):
            continue
        moving_piece = board.piece_at(move.from_square)
        san = board.san(move)
        post = board.copy(stack=True)
        post.push(move)
        if not post.is_checkmate():
            continue
        out.append(
            {
                "uci": move.uci(),
                "san": san,
                "piece": _piece_label(moving_piece, move.from_square),
                "back_rank_geometry": _back_rank_geometry(board, move, moving_piece),
                "resulting_fen": post.fen(),
            }
        )

    out.sort(key=lambda item: (item["san"], item["uci"]))
    return out[:MAX_MATE_IN_ONE_MOVES]


def mate_in_one_threats_if_pass(
    board: chess.Board,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Return opponent mate-in-one candidates after a hypothetical legal pass.

    Passing is never used while the side to move is in check or after a terminal
    position. The result therefore answers the bounded coaching question "if I
    do nothing, does the opponent have mate in one?" It does not prove that the
    threat cannot be met by a legal move.
    """
    if board.is_game_over(claim_draw=False):
        return [], False, "terminal_position"
    if board.is_check():
        return [], False, "side_to_move_in_check_pass_illegal"
    passed = board.copy(stack=True)
    passed.push(chess.Move.null())
    return mate_in_one_moves(passed), True, None


def apply_mate_forensics(
    result: ForensicMoveAnalysis,
    board_before: chess.Board,
    *,
    played_move: chess.Move | None,
) -> ForensicMoveAnalysis:
    """Attach missed/allowed mate-in-one facts to rich move evidence."""
    evidence = result.forensics
    if evidence is None or result.action_type != "play_move" or played_move is None:
        return result
    if played_move not in board_before.legal_moves:
        return result

    before_mates = mate_in_one_moves(board_before)
    before_mate_ucis = {item["uci"] for item in before_mates}
    played_was_mate = played_move.uci() in before_mate_ucis
    mate_threats, threat_probe_available, threat_probe_reason = mate_in_one_threats_if_pass(
        board_before
    )

    board_after = board_before.copy(stack=True)
    board_after.push(played_move)
    opponent_mates = mate_in_one_moves(board_after)
    opponent_mate_ucis = {item["uci"] for item in opponent_mates}

    strongest_reply_is_mate = bool(
        evidence.strongest_reply is not None
        and evidence.strongest_reply.uci in opponent_mate_ucis
    )
    addresses_mate_threat: bool | None = None
    if mate_threats:
        addresses_mate_threat = not opponent_mates

    mechanism = {
        "mechanism": "mate_in_one_scan",
        "side_to_move_before": _color_name(board_before.turn),
        "mate_in_one_moves_before": before_mates,
        "played_move_was_mate_in_one": played_was_mate,
        "opponent_mate_in_one_threats_if_pass_before": mate_threats,
        "mate_threat_pass_probe_available": threat_probe_available,
        "mate_threat_pass_probe_reason": threat_probe_reason,
        "played_move_addresses_immediate_mate_threat": addresses_mate_threat,
        "opponent_mate_in_one_moves_after_played": opponent_mates,
        "strongest_reply_is_mate_in_one": strongest_reply_is_mate,
        "proof_scope": (
            "The before/after mate-in-one scans are exhaustive legal immediate-mate scans. "
            "The threat list uses a hypothetical null move only when passing is legal, so it "
            "answers what mate-in-one the opponent would have if the mover did nothing. It "
            "does not prove that a legal defense cannot answer the threat and does not infer "
            "why a player did or did not see it."
        ),
    }

    mechanisms = list(evidence.mechanism_evidence)
    mechanisms.append(mechanism)
    signatures = list(evidence.evidence_signatures)

    if before_mates:
        signatures.append("MATE_IN_ONE_AVAILABLE_BEFORE_MOVE")
        if not played_was_mate:
            signatures.append("MISSED_MATE_IN_ONE_CANDIDATE")
    if played_was_mate:
        signatures.append("PLAYED_MATE_IN_ONE")
    if mate_threats:
        signatures.append("OPPONENT_MATE_IN_ONE_THREAT_IF_PASS")
        if addresses_mate_threat is False:
            signatures.append("FAILED_MATE_THREAT_UPDATE_CANDIDATE")
        elif addresses_mate_threat is True:
            signatures.append("IMMEDIATE_MATE_THREAT_ADDRESSED")
    if opponent_mates:
        signatures.append("OPPONENT_MATE_IN_ONE_AFTER_MOVE")
    if strongest_reply_is_mate:
        signatures.append("STRONGEST_REPLY_IS_MATE_IN_ONE")
    if any(
        item["back_rank_geometry"]
        for item in [*before_mates, *mate_threats, *opponent_mates]
    ):
        signatures.append("BACK_RANK_MATE_GEOMETRY_CANDIDATE")

    upgraded = evidence.model_copy(
        update={
            "mechanism_evidence": mechanisms,
            "evidence_signatures": sorted(set(signatures)),
        }
    )
    return result.model_copy(update={"forensics": upgraded})
