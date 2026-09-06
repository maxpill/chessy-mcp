from __future__ import annotations

import chess

from mcp_server.analysis.forensic_integration import (
    build_candidate_differences,
    enrich_candidate_geometry,
)
from mcp_server.analysis.position_integrity import build_rich_tactical_snapshot
from mcp_server.models.forensics import CandidateEvidence


def _candidate(board: chess.Board, uci: str) -> CandidateEvidence:
    move = chess.Move.from_uci(uci)
    return CandidateEvidence(
        requested=uci,
        uci=uci,
        san=board.san(move),
        resulting_fen=board.fen(),
        tactical_snapshot_after=build_rich_tactical_snapshot(board),
    )


def test_candidate_difference_reports_piece_activity_and_center_control() -> None:
    board = chess.Board()
    e4 = enrich_candidate_geometry(board, _candidate(board, "e2e4"))
    d4 = enrich_candidate_geometry(board, _candidate(board, "d2d4"))

    differences = build_candidate_differences(
        board,
        [e4, d4],
        reference_uci="e2e4",
    )

    assert len(differences) == 1
    diff = differences[0]
    assert diff.reference_san == "e4"
    assert diff.candidate_san == "d4"
    assert diff.only_reference_piece_mobility_changes
    assert diff.only_candidate_piece_mobility_changes
    assert diff.only_reference_strategic_square_control_changes
    assert diff.only_candidate_strategic_square_control_changes
    assert any("white_bishop@f1" in item for item in diff.only_reference_piece_mobility_changes)
    assert any("white_bishop@c1" in item for item in diff.only_candidate_piece_mobility_changes)


def test_candidate_difference_serializes_defender_loss_explicitly() -> None:
    board = chess.Board("6k1/6b1/8/8/3P4/4P3/8/6K1 w - - 0 1")
    e4 = enrich_candidate_geometry(board, _candidate(board, "e3e4"))
    kg2 = enrich_candidate_geometry(board, _candidate(board, "g1g2"))

    differences = build_candidate_differences(
        board,
        [e4, kg2],
        reference_uci="e3e4",
    )

    assert len(differences) == 1
    diff = differences[0]
    assert any(
        "white_pawn@d4:attackers=1->1:defenders=1->0" in item
        for item in diff.only_reference_piece_safety_changes
    )
