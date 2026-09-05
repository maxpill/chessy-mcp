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


def test_candidate_difference_serializes_piece_safety_as_explicit_attacker_defender_counts() -> None:
    board = chess.Board("4k3/8/8/8/4p3/8/3P4/3Q2K1 w - - 0 1")
    d4 = enrich_candidate_geometry(board, _candidate(board, "d2d4"))
    d3 = enrich_candidate_geometry(board, _candidate(board, "d2d3"))

    differences = build_candidate_differences(
        board,
        [d4, d3],
        reference_uci="d2d4",
    )

    assert len(differences) == 1
    diff = differences[0]
    combined = diff.only_reference_piece_safety_changes + diff.only_candidate_piece_safety_changes
    assert all("attackers=" in item and "defenders=" in item for item in combined)
