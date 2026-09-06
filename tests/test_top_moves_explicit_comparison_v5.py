from __future__ import annotations

import chess

from mcp_server.analysis.top_moves_forensics import _comparison_requests
from mcp_server.models import MCPEval
from mcp_server.models.legacy import TopMovesResult


def test_explicit_compare_moves_cannot_be_displaced_by_long_top_n() -> None:
    board = chess.Board()
    result = TopMovesResult(
        result=[
            MCPEval(best_move="g1f3"),
            MCPEval(best_move="b1c3"),
            MCPEval(best_move="e2e4"),
            MCPEval(best_move="d2d4"),
            MCPEval(best_move="c2c4"),
            MCPEval(best_move="g2g3"),
            MCPEval(best_move="b2b3"),
            MCPEval(best_move="f2f4"),
            MCPEval(best_move="a2a4"),
        ]
    )
    explicit = ["a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3"]

    requested = _comparison_requests(
        result,
        board,
        detail="forensic",
        include_moves=explicit,
    )

    assert requested[0] == "Nf3"
    assert len(requested) == 9
    assert requested[1:] == explicit


def test_explicit_uci_and_san_duplicates_are_canonicalized_once() -> None:
    board = chess.Board()
    result = TopMovesResult(result=[MCPEval(best_move="e2e4")])

    requested = _comparison_requests(
        result,
        board,
        detail="forensic",
        include_moves=["e4", "e2e4", "d4", "d2d4"],
    )

    assert requested == ["e4", "d4"]
