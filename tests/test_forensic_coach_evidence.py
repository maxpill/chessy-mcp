from __future__ import annotations

import json
from types import SimpleNamespace

import chess
import pytest

from core.engines.types import MoveClass
from mcp_server.analysis.forensics import (
    build_position_delta,
    build_position_fingerprint,
    build_tactical_snapshot,
    enrich_move_analysis,
    parse_candidate_move,
)
from mcp_server.middleware.request_cost import estimate_mcp_request_cost
from mcp_server.models import MCPEval, MCPMoveAnalysis


def test_position_fingerprint_echoes_board_state_deterministically() -> None:
    board = chess.Board()

    first = build_position_fingerprint(board)
    second = build_position_fingerprint(board.copy(stack=True))

    assert first == second
    assert first.canonical_fen == board.fen()
    assert first.side_to_move == "white"
    assert first.castling_rights == "KQkq"
    assert first.in_check is False
    assert first.legal_move_count == 20
    assert first.material == {"white": 4000, "black": 4000}
    assert first.piece_map["white"]["king"] == ["e1"]
    assert first.piece_map["white"]["pawn"] == [
        "a2",
        "b2",
        "c2",
        "d2",
        "e2",
        "f2",
        "g2",
        "h2",
    ]
    assert len(first.position_hash) == 16


def test_tactical_snapshot_lists_checks_captures_en_prise_and_pins() -> None:
    forcing = chess.Board("4k3/4r3/8/8/8/8/4Q3/4K3 w - - 0 1")
    snapshot = build_tactical_snapshot(forcing)

    qxe7 = next(move for move in snapshot.checks if move.uci == "e2e7")
    assert qxe7.san == "Qxe7+"
    assert qxe7.is_check is True
    assert qxe7.is_capture is True
    assert qxe7.captured_piece == "black_rook"
    assert any(piece.square == "e7" for piece in snapshot.en_prise_pieces)

    pinned = chess.Board("4k3/4n3/8/8/8/8/8/4R1K1 w - - 0 1")
    pinned_snapshot = build_tactical_snapshot(pinned)
    assert any(
        piece.color == "black" and piece.piece == "knight" and piece.square == "e7"
        for piece in pinned_snapshot.pinned_pieces
    )


def test_position_delta_reports_capture_and_new_geometry() -> None:
    before = chess.Board("4k3/4r3/8/8/8/8/4Q3/4K3 w - - 0 1")
    after = before.copy(stack=True)
    after.push(chess.Move.from_uci("e2e7"))

    delta = build_position_delta(before, after)

    assert delta.material_delta_white == 0
    assert delta.material_delta_black == -500
    assert "black_rook@e7" in delta.removed_pieces
    assert "white_queen@e2" in delta.removed_pieces
    assert "white_queen@e7" in delta.added_pieces
    assert delta.check_state_changed is True


def test_compare_move_parser_accepts_san_and_uci_and_rejects_illegal() -> None:
    board = chess.Board()
    assert parse_candidate_move(board, "e4").uci() == "e2e4"
    assert parse_candidate_move(board, "e2e4").uci() == "e2e4"
    with pytest.raises(ValueError, match="INVALID_COMPARE_MOVE"):
        parse_candidate_move(board, "Qh8")


def test_forensic_classification_is_charged_more_than_standard() -> None:
    def rpc(arguments: dict[str, object]) -> bytes:
        return json.dumps(
            {
                "params": {
                    "name": "classify_move",
                    "arguments": arguments,
                }
            }
        ).encode()

    standard = estimate_mcp_request_cost(rpc({"fen": "startpos", "move": "e4", "depth": 20}))
    coach = estimate_mcp_request_cost(
        rpc({"fen": "startpos", "move": "e4", "depth": 20, "detail": "coach"})
    )
    forensic = estimate_mcp_request_cost(
        rpc(
            {
                "fen": "startpos",
                "move": "e4",
                "depth": 20,
                "detail": "forensic",
                "compare_moves": ["d4", "Nf3"],
            }
        )
    )

    assert standard < coach < forensic


class _FakePool:
    async def evaluate(self, board: chess.Board, *, depth: int, root_moves=None):
        best_move = None
        if board.turn == chess.BLACK and chess.Move.from_uci("e8e7") in board.legal_moves:
            best_move = "e8e7"
        return SimpleNamespace(cp=0, mate=None, depth=depth, best_move=best_move, pv=[])


@pytest.mark.asyncio
async def test_forensic_enrichment_exposes_strongest_reply_and_candidate_position() -> None:
    board = chess.Board("4k3/4r3/8/8/8/8/4Q3/4K3 w - - 0 1")
    played = chess.Move.from_uci("e2e7")
    result = MCPMoveAnalysis(
        played=played.uci(),
        played_san="Qxe7+",
        move_class=MoveClass.BLUNDER,
        eval_before=MCPEval(cp=0, best_move="e2e7", pv=["e2e7"], depth=8, searched_depth=8),
        eval_after=MCPEval(
            cp=-900,
            best_move="e8e7",
            pv=["e8e7"],
            depth=8,
            searched_depth=8,
        ),
        best_move_san="Qxe7+",
    )

    enriched = await enrich_move_analysis(
        result,
        board_before=board,
        played_move=played,
        pool=_FakePool(),
        depth=8,
        detail="forensic",
        compare_moves=["Qxe7+"],
    )

    assert enriched.forensics is not None
    evidence = enriched.forensics
    assert evidence.position_before.piece_map["black"]["rook"] == ["e7"]
    assert evidence.position_after_played.material["black"] == 0
    assert evidence.strongest_reply is not None
    assert evidence.strongest_reply.uci == "e8e7"
    assert evidence.strongest_reply.is_capture is True
    assert evidence.strongest_reply.captured_piece == "white_queen"
    assert "FORCING_CAPTURE_REPLY" in evidence.evidence_signatures
    assert evidence.candidate_comparisons[0].san == "Qxe7+"
    assert evidence.candidate_comparisons[0].resulting_fen == evidence.position_after_played.canonical_fen
    assert evidence.forced_line.proof_status == "principal_variation_only"
