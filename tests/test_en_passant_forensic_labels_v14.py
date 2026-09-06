from __future__ import annotations

import chess

from mcp_server.analysis.position_integrity import build_rich_tactical_snapshot


def test_checking_en_passant_reports_actual_captured_pawn_square() -> None:
    board = chess.Board("4k3/8/8/3pP3/8/8/8/K3R3 w - d6 0 1")
    move = chess.Move.from_uci("e5d6")
    assert move in board.legal_moves
    assert board.is_en_passant(move)
    assert board.gives_check(move)

    snapshot = build_rich_tactical_snapshot(board)
    candidate = next(
        item
        for item in snapshot.mechanism_candidates
        if item.mechanism == "check_capture" and item.trigger_uci == "e5d6"
    )

    assert candidate.targets == ["black_pawn@d5"]
