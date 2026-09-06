"""Regression coverage for resulting-position forensic comparisons."""

from __future__ import annotations

from types import SimpleNamespace

import chess
import pytest

from mcp_server.analysis.forensic_integration import (
    build_candidate_differences,
    enrich_candidate_geometry,
)
from mcp_server.analysis.forensics import _candidate_evidence
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
    assert diff.first_divergence_ply == 1
    assert diff.first_divergence == {"reference": "e4", "candidate": "e3"}
    assert diff.eval_gap_candidate_minus_reference_for_mover_cp == -40
    assert diff.reference_root_move_irreversible is True
    assert diff.candidate_root_move_irreversible is True
    assert "pawn_move" in diff.reference_root_irreversible_reasons
    assert "pawn_move" in diff.candidate_root_irreversible_reasons
    assert "white_pawn_added@e4" in diff.only_reference_pawn_structure_changes
    assert "white_pawn_added@e3" in diff.only_candidate_pawn_structure_changes


def test_candidate_difference_marks_pawn_move_vs_reversible_king_move() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    pawn = enrich_candidate_geometry(board, _candidate(board, "e2e4", cp=20))
    king = enrich_candidate_geometry(board, _candidate(board, "e1f1", cp=10))

    diff = build_candidate_differences(
        board,
        [pawn, king],
        reference_uci="e2e4",
    )[0]

    assert diff.reference_root_move_irreversible is True
    assert diff.reference_root_irreversible_reasons == ["pawn_move"]
    assert diff.candidate_root_move_irreversible is False
    assert diff.candidate_root_irreversible_reasons == []


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


def test_candidate_difference_separates_reply_forcing_moves_from_root_pass_threats() -> None:
    board = chess.Board("6k1/8/8/8/1b6/2N5/P7/6K1 w - - 0 1")
    reference = enrich_candidate_geometry(board, _candidate(board, "c3b5", cp=20))
    alternative = enrich_candidate_geometry(board, _candidate(board, "a2a3", cp=-40))

    diff = build_candidate_differences(
        board,
        [reference, alternative],
        reference_uci="c3b5",
    )[0]

    # After a3 it is Black to move and ...Bxc3 is immediately legal.
    assert any(
        "b4c3" in item and "capture=1" in item
        for item in diff.only_candidate_immediate_reply_forcing_moves
    )
    # A pass by Black would instead hand White the move, where axb4 is available.
    assert any(
        "a3b4" in item and "capture=1" in item
        for item in diff.only_candidate_root_forcing_threats_if_reply_passes
    )


class _PVPool:
    async def evaluate(self, board: chess.Board, *, depth: int):
        assert board.turn == chess.BLACK
        return SimpleNamespace(
            cp=12,
            mate=None,
            depth=depth,
            best_move="e7e5",
            pv=["e7e5", "g1f3"],
        )


@pytest.mark.asyncio
async def test_candidate_evidence_retains_principal_continuation() -> None:
    candidate = await _candidate_evidence(
        chess.Board(),
        "e4",
        pool=_PVPool(),
        depth=18,
    )

    assert candidate.continuation_uci == ["e7e5", "g1f3"]
    assert candidate.continuation_san == ["e5", "Nf3"]
    assert candidate.continuation_termination_reason == "pv_exhausted"
    assert candidate.continuation_proof_status == "principal_variation_only"
