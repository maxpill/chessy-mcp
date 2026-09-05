"""Regression tests for position-integrity and defender-geometry evidence."""

from __future__ import annotations

import chess

from mcp_server.analysis.position_integrity import (
    build_rich_position_delta,
    build_rich_tactical_snapshot,
    enrich_position_eval,
)
from mcp_server.models.mcpeval import MCPEval


def test_tactical_hanging_candidate_detects_nominal_defender_that_cannot_recapture():
    # The black knight on e7 nominally defends Bf5, but it is absolutely pinned
    # to Ke8 by Re1. After Qxf5 the nominal Nxf5 recapture is illegal.
    board = chess.Board("4k3/4n3/8/5b2/8/8/2Q5/4R1K1 w - - 0 1")

    snapshot = build_rich_tactical_snapshot(board)
    matches = [item for item in snapshot.tactically_hanging_candidates if item.target.square == "f5"]

    assert matches
    assert matches[0].target.piece == "bishop"
    assert matches[0].nominal_defenders == 1
    assert matches[0].capture.uci == "c2f5"
    assert matches[0].legal_immediate_recaptures == []


def test_tactical_hanging_candidate_does_not_flag_working_recapture():
    # Same geometry without the king/rook pin: ...Nxf5 is a legal recapture.
    board = chess.Board("6k1/4n3/8/5b2/8/8/2Q5/6K1 w - - 0 1")

    snapshot = build_rich_tactical_snapshot(board)

    assert not [item for item in snapshot.tactically_hanging_candidates if item.target.square == "f5"]


def test_overloaded_defender_candidate_reports_two_attacked_targets():
    # Nf3 is the shared defender of e5 and h4. Bb8 attacks e5 and Rh8 attacks h4.
    board = chess.Board("1b4kr/8/8/4P3/7P/5N2/8/6K1 w - - 0 1")

    snapshot = build_rich_tactical_snapshot(board)
    knight = [item for item in snapshot.overloaded_defender_candidates if item.square == "f3"]

    assert knight
    assert "white_pawn@e5" in knight[0].attacked_targets
    assert "white_pawn@h4" in knight[0].attacked_targets


def test_position_delta_surfaces_defender_loss_on_piece_that_stays_put():
    before = chess.Board("1b4k1/8/8/4P3/8/5N2/8/6K1 w - - 0 1")
    move = chess.Move.from_uci("f3g5")
    assert move in before.legal_moves
    after = before.copy(stack=True)
    after.push(move)

    delta = build_rich_position_delta(before, after)
    pawn = [item for item in delta.piece_safety_changes if item.target == "white_pawn@e5"]

    assert pawn
    assert pawn[0].attackers_before == 1
    assert pawn[0].attackers_after == 1
    assert pawn[0].defenders_before == 1
    assert pawn[0].defenders_after == 0


def test_forensic_eval_echoes_board_and_best_move_resulting_position():
    board = chess.Board("4k3/4n3/8/5b2/8/8/2Q5/4R1K1 w - - 0 1")
    result = MCPEval(
        cp=120,
        best_move="c2f5",
        pv=["c2f5"],
        depth=20,
        searched_depth=20,
        canonical_fen=board.fen(),
    )

    enriched = enrich_position_eval(result, board, detail="forensic")

    assert enriched.forensics is not None
    assert enriched.forensics.position.canonical_fen == board.fen()
    assert enriched.forensics.position.piece_map["black"]["bishop"] == ["f5"]
    assert enriched.forensics.best_move_uci == "c2f5"
    assert enriched.forensics.position_after_best is not None
    assert enriched.forensics.position_after_best.piece_map["black"]["bishop"] == []
    assert enriched.forensics.best_move_delta is not None
    assert "black_bishop@f5" in enriched.forensics.best_move_delta.removed_pieces
