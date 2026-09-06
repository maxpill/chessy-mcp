from __future__ import annotations

import chess

from mcp_server.analysis.position_update_forensics import build_position_update_evidence


ROOT = "k3r3/8/8/8/4N3/5P2/6b1/K7 b - - 0 1"


def _position_after_last_defender_removed() -> chess.Board:
    board = chess.Board(ROOT)
    move = chess.Move.from_uci("g2f3")
    assert move in board.legal_moves
    board.push(move)
    return board


def test_already_attacked_piece_becomes_urgent_when_last_defender_is_removed() -> None:
    board = _position_after_last_defender_removed()
    quiet = chess.Move.from_uci("a1b1")
    assert quiet in board.legal_moves

    evidence = build_position_update_evidence(board, quiet)

    assert evidence["history_available"] is True
    assert "white_knight@e4:1->0" in evidence["defender_count_losses"]
    assert "white_knight@e4" in evidence["newly_exposed_user_pieces"]
    assert evidence["opponent_move_created_urgent_change"] is True
    assert evidence["played_move_addresses_change"] is False
    assert "white_knight@e4" in evidence["unresolved_urgent_targets_after_played_move"]


def test_moving_exposed_piece_to_safety_addresses_the_update() -> None:
    board = _position_after_last_defender_removed()
    move = chess.Move.from_uci("e4c3")
    assert move in board.legal_moves

    evidence = build_position_update_evidence(board, move)

    assert evidence["opponent_move_created_urgent_change"] is True
    assert evidence["played_move_addresses_change"] is True
    assert evidence["unresolved_urgent_targets_after_played_move"] == []
