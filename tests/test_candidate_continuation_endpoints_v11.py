from __future__ import annotations

import chess

from mcp_server.analysis.forensic_integration import (
    build_candidate_differences,
    enrich_candidate_geometry,
)
from mcp_server.analysis.forensics import build_tactical_snapshot
from mcp_server.models.forensics import CandidateEvidence


def _candidate(
    board: chess.Board,
    move_uci: str,
    continuation_uci: list[str],
    *,
    eval_cp: int = 0,
) -> CandidateEvidence:
    move = chess.Move.from_uci(move_uci)
    assert move in board.legal_moves
    post = board.copy(stack=True)
    san = board.san(move)
    post.push(move)
    return CandidateEvidence(
        requested=san,
        uci=move.uci(),
        san=san,
        resulting_fen=post.fen(),
        eval_cp=eval_cp,
        tactical_snapshot_after=build_tactical_snapshot(post),
        continuation_uci=continuation_uci,
        continuation_san=[],
        continuation_termination_reason=("pv_exhausted" if continuation_uci else "no_pv"),
    )


def test_candidate_endpoint_follows_returned_pv_and_exposes_root_to_endpoint_delta() -> None:
    board = chess.Board()
    reference = enrich_candidate_geometry(
        board,
        _candidate(board, "e2e4", ["e7e5", "g1f3"], eval_cp=20),
    )
    alternative = enrich_candidate_geometry(
        board,
        _candidate(board, "d2d4", ["d7d5", "c2c4"], eval_cp=10),
    )

    assert reference.continuation_endpoint is not None
    assert alternative.continuation_endpoint is not None
    assert reference.continuation_endpoint["plies_from_root"] == 3
    assert alternative.continuation_endpoint["plies_from_root"] == 3
    assert reference.continuation_endpoint["termination_reason"] == "pv_exhausted"
    assert alternative.continuation_endpoint["termination_reason"] == "pv_exhausted"
    assert reference.continuation_endpoint["tactical_sequence_resolved"] is False
    assert reference.continuation_endpoint["endpoint_fen"] != alternative.continuation_endpoint["endpoint_fen"]

    ref_position = reference.continuation_endpoint["endpoint_position"]
    alt_position = alternative.continuation_endpoint["endpoint_position"]
    assert "f3" in ref_position["piece_map"]["white"]["knight"]
    assert "c4" in alt_position["piece_map"]["white"]["pawn"]
    assert reference.continuation_endpoint["root_to_endpoint_delta"]["pawn_structure_changes"]
    assert alternative.continuation_endpoint["root_to_endpoint_delta"]["pawn_structure_changes"]

    differences = build_candidate_differences(
        board,
        [reference, alternative],
        reference_uci="e2e4",
    )
    assert len(differences) == 1
    endpoint = differences[0].continuation_endpoint_difference
    assert endpoint["available"] is True
    assert endpoint["reference_plies_from_root"] == 3
    assert endpoint["candidate_plies_from_root"] == 3
    assert endpoint["reference_termination_reason"] == "pv_exhausted"
    assert endpoint["candidate_termination_reason"] == "pv_exhausted"
    assert len(endpoint["reference_irreversible_events"]) == 2
    assert len(endpoint["candidate_irreversible_events"]) == 3


def test_forcing_root_capture_stops_at_material_resolution_before_quiet_pv_padding() -> None:
    board = chess.Board("4k3/8/8/8/8/8/p7/R3K3 w - - 0 1")
    candidate = enrich_candidate_geometry(
        board,
        _candidate(board, "a1a2", ["e8e7"], eval_cp=100),
    )

    assert candidate.continuation_endpoint is not None
    endpoint = candidate.continuation_endpoint
    assert endpoint["termination_reason"] == "material_resolution"
    assert endpoint["tactical_sequence_resolved"] is True
    assert endpoint["plies_from_candidate"] == 0
    assert endpoint["plies_from_root"] == 1
    assert endpoint["returned_pv_plies_available"] == 1
    assert endpoint["endpoint_position"]["piece_map"]["black"]["pawn"] == []
    assert endpoint["endpoint_position"]["piece_map"]["black"]["king"] == ["e8"]


def test_endpoint_difference_reports_material_consequence_after_different_candidates() -> None:
    board = chess.Board("4k3/8/8/8/8/8/p7/R3K3 w - - 0 1")
    reference = enrich_candidate_geometry(
        board,
        _candidate(board, "a1a2", ["e8e7"], eval_cp=100),
    )
    alternative = enrich_candidate_geometry(
        board,
        _candidate(board, "a1b1", ["e8e7"], eval_cp=0),
    )

    difference = build_candidate_differences(
        board,
        [reference, alternative],
        reference_uci="a1a2",
    )[0]
    endpoint = difference.continuation_endpoint_difference

    assert endpoint["available"] is True
    assert endpoint["reference_termination_reason"] == "material_resolution"
    assert endpoint["candidate_termination_reason"] == "pv_exhausted"
    assert endpoint["material_effect_difference_for_mover_cp"] == -100
    assert any("Rxa2" in item for item in endpoint["reference_irreversible_events"])
