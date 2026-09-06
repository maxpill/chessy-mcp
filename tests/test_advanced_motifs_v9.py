from __future__ import annotations

import chess

from mcp_server.analysis.position_integrity import build_rich_tactical_snapshot
from mcp_server.analysis.tactical_snapshot_extensions import extend_tactical_snapshot


def _snapshot(fen: str):
    board = chess.Board(fen)
    return board, extend_tactical_snapshot(board, build_rich_tactical_snapshot(board))


def test_discovered_attack_candidate_records_vacated_slider_ray() -> None:
    board, snapshot = _snapshot("7k/7r/8/8/8/8/2N5/1B4K1 w - - 0 1")

    candidate = next(
        item
        for item in snapshot.mechanism_candidates
        if item.mechanism == "discovered_attack_candidate" and item.trigger_uci == "c2e3"
    )

    assert candidate.trigger_san == "Ne3"
    assert candidate.actor == "white_knight@c2"
    assert candidate.targets == ["black_rook@h7"]
    assert candidate.evidence["discovered_attacker"] == "white_bishop@b1"
    assert candidate.evidence["vacated_square"] == "c2"
    assert "does not prove material" in candidate.proof_scope
    assert chess.Move.from_uci(candidate.trigger_uci) in board.legal_moves


def test_interference_candidate_severs_slider_defense_but_stays_bounded() -> None:
    _board, snapshot = _snapshot("r6k/8/8/n1N5/1B6/8/8/7K w - - 0 1")

    candidate = next(
        item
        for item in snapshot.mechanism_candidates
        if item.mechanism == "interference_candidate" and item.trigger_uci == "c5a6"
    )

    assert candidate.trigger_san == "Na6"
    assert candidate.actor == "white_knight@c5"
    assert candidate.targets == ["black_rook@a8", "black_knight@a5"]
    assert candidate.evidence["interference_square"] == "a6"
    assert candidate.evidence["target_remains_attacked_after_move"] is True
    assert "does not prove the target is lost" in candidate.proof_scope


def test_trapped_piece_candidate_requires_no_geometrically_safe_piece_move() -> None:
    _board, snapshot = _snapshot("n6k/8/2Q5/8/8/8/8/4K3 w - - 0 1")

    candidate = next(
        item
        for item in snapshot.mechanism_candidates
        if item.mechanism == "trapped_piece_candidate" and item.targets == ["black_knight@a8"]
    )

    assert candidate.trigger_uci is None
    assert candidate.evidence["pass_hypothesis_available"] is True
    assert set(candidate.evidence["legal_piece_moves_after_hypothetical_pass"]) == {"Nb6", "Nc7"}
    assert candidate.evidence["geometrically_safe_piece_moves"] == []
    assert set(candidate.evidence["attacked_destination_squares"]) == {"b6", "c7"}
    assert "not proof of material loss" in candidate.proof_scope
