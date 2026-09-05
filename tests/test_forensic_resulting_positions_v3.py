"""Regression coverage for resulting-position forensic comparisons."""

from __future__ import annotations

import chess

from mcp_server.analysis.forensic_integration import (
    build_candidate_differences,
    enrich_candidate_geometry,
)
from mcp_server.analysis.position_integrity import build_rich_tactical_snapshot
from mcp_server.models.forensics import CandidateEvidence, StrongestReplyEvidence


def _candidate(
    board: chess.Board,
    uci: str,
    *,
    cp: int,
    reply_uci: str | None = None,
) -> CandidateEvidence:
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    post = board.copy(stack=True)
    san = board.san(move)
    post.push(move)

    reply = None
    if reply_uci is not None:
        reply_move = chess.Move.from_uci(reply_uci)
        assert reply_move in post.legal_moves
        reply_post = post.copy(stack=True)
        reply_san = post.san(reply_move)
        is_check = post.gives_check(reply_move)
        is_capture = post.is_capture(reply_move)
        reply_post.push(reply_move)
        reply = StrongestReplyEvidence(
            uci=reply_move.uci(),
            san=reply_san,
            is_check=is_check,
            is_capture=is_capture,
            is_forcing=is_check or is_capture,
            resulting_fen=reply_post.fen(),
        )

    return CandidateEvidence(
        requested=san,
        uci=move.uci(),
        san=san,
        resulting_fen=post.fen(),
        eval_cp=cp,
        tactical_snapshot_after=build_rich_tactical_snapshot(post),
        opponent_best_reply=reply,
    )


def test_candidate_geometry_exposes_root_and_reply_position_deltas() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    raw = _candidate(board, "e2e4", cp=30, reply_uci="e8e7")

    enriched = enrich_candidate_geometry(board, raw)

    assert enriched.position_after is not None
    assert enriched.position_after.piece_map["white"]["pawn"] == ["e4"]
    assert enriched.position_delta is not None
    assert "white_pawn_removed@e2" in enriched.position_delta.pawn_structure_changes
    assert "white_pawn_added@e4" in enriched.position_delta.pawn_structure_changes

    assert enriched.position_after_reply is not None
    assert enriched.position_after_reply.piece_map["black"]["king"] == ["e7"]
    assert enriched.tactical_after_reply is not None
    assert enriched.reply_delta is not None
    assert "black_king@e8" in enriched.reply_delta.removed_pieces
    assert "black_king@e7" in enriched.reply_delta.added_pieces


def test_candidate_difference_uses_engine_reference_and_compares_resulting_positions() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    best = enrich_candidate_geometry(board, _candidate(board, "e2e4", cp=50))
    alternative = enrich_candidate_geometry(board, _candidate(board, "e2e3", cp=10))

    differences = build_candidate_differences(
        board,
        [alternative, best],
        reference_uci="e2e4",
    )

    assert len(differences) == 1
    diff = differences[0]
    assert diff.reference_uci == "e2e4"
    assert diff.candidate_uci == "e2e3"
    assert diff.eval_gap_candidate_minus_reference_for_mover_cp == -40
    assert "white_pawn_added@e4" in diff.only_reference_pawn_structure_changes
    assert "white_pawn_added@e3" in diff.only_candidate_pawn_structure_changes


def test_candidate_difference_flips_engine_gap_for_black_root_mover() -> None:
    board = chess.Board("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1")
    # Engine evaluations are White POV. For Black, the more negative line is better.
    best = enrich_candidate_geometry(board, _candidate(board, "e7e5", cp=-60))
    alternative = enrich_candidate_geometry(board, _candidate(board, "e7e6", cp=-10))

    differences = build_candidate_differences(
        board,
        [best, alternative],
        reference_uci="e7e5",
    )

    assert len(differences) == 1
    assert differences[0].eval_gap_candidate_minus_reference_for_mover_cp == -50
