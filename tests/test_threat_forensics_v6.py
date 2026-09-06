from __future__ import annotations

import chess

from mcp_server.analysis.position_integrity import build_rich_tactical_snapshot
from mcp_server.analysis.tactical_snapshot_extensions import extend_tactical_snapshot
from mcp_server.analysis.threat_forensics import (
    opponent_forcing_threat_update,
    relative_pin_candidates,
    zwischenzug_candidate_after_reply,
)


def test_opponent_move_exposes_new_forcing_threat_under_pass_probe() -> None:
    board = chess.Board("6k1/8/8/8/7q/8/8/6K1 b - - 0 1")
    opponent_move = chess.Move.from_uci("h4d8")
    assert opponent_move in board.legal_moves
    board.push(opponent_move)

    evidence = opponent_forcing_threat_update(board, played_move=None)

    assert evidence["history_available"] is True
    assert evidence["pass_probe_available"] is True
    assert evidence["opponent_move_uci"] == "h4d8"
    threats = evidence["newly_enabled_forcing_threats_if_pass"]
    assert any(item["uci"] == "d8d1" and item["is_check"] for item in threats)
    assert evidence["played_move_addresses_exact_new_threats"] is None


def test_naked_fen_does_not_fabricate_previous_move_threat_evidence() -> None:
    board = chess.Board("3q2k1/8/8/8/8/8/8/6K1 w - - 0 1")

    evidence = opponent_forcing_threat_update(board, played_move=None)

    assert evidence["history_available"] is False
    assert evidence["pass_probe_available"] is False
    assert evidence["reason"] == "no_previous_move_history"


def test_rich_snapshot_exposes_opponent_forcing_threats_if_side_passes() -> None:
    board = chess.Board("3q2k1/8/8/8/8/8/8/6K1 w - - 0 1")

    snapshot = extend_tactical_snapshot(board, build_rich_tactical_snapshot(board))

    assert snapshot.threat_probe_available is True
    assert snapshot.threat_probe_reason is None
    assert snapshot.threat_probe_scope is not None
    assert any(
        item.uci == "d8d1" and item.is_check
        for item in snapshot.opponent_forcing_threats_if_pass
    )


def test_rich_snapshot_does_not_use_illegal_pass_while_in_check() -> None:
    board = chess.Board("3q2k1/8/8/8/8/8/8/3K4 w - - 0 1")
    assert board.is_check()

    snapshot = extend_tactical_snapshot(board, build_rich_tactical_snapshot(board))

    assert snapshot.threat_probe_available is False
    assert snapshot.threat_probe_reason == "side_to_move_in_check_pass_illegal"
    assert snapshot.opponent_forcing_threats_if_pass == []


def test_capture_with_recapture_and_intermediate_check_is_zwischenzug_candidate() -> None:
    board_after_played = chess.Board(
        "4k3/4n3/8/3q4/8/8/8/3QR1K1 b - - 0 1"
    )
    reply = chess.Move.from_uci("d5d1")
    assert reply in board_after_played.legal_moves
    assert board_after_played.is_capture(reply)

    evidence = zwischenzug_candidate_after_reply(board_after_played, "d5d1")

    assert evidence is not None
    assert any(item["uci"] == "e1d1" for item in evidence["available_immediate_recaptures"])
    assert any(
        item["uci"] == "e1e7" and item["is_check"]
        for item in evidence["intermediate_forcing_moves"]
    )


def test_relative_pin_geometry_distinguishes_front_and_rear_value() -> None:
    board = chess.Board("7k/6q1/8/8/3n4/8/8/BK6 w - - 0 1")

    candidates = relative_pin_candidates(board)

    assert any(
        item["actor"] == "white_bishop@a1"
        and item["front_target"] == "black_knight@d4"
        and item["rear_target"] == "black_queen@g7"
        for item in candidates
    )
