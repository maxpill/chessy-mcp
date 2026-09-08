"""2026-09-08 audit regression: ``classify_move`` rich ``_finish_result`` must accept ``strict``.

Bug 2 — ``classify_move(startpos, depth=1, detail="standard",
compare_moves=[8 candidates])`` crashed with::

    [ENGINE_ERROR] name 'strict' is not defined (tool=classify_move)

Root cause: ``_finish_result`` referenced the bare name ``strict`` at line
304 but ``strict`` was not in its keyword-only signature. The
``classify_move`` outer tool has a top-level ``strict`` parameter that was
shadowed by the inner function. The fix threads ``strict`` through the
``_finish_result`` signature from both call sites (cache-hit + compute
paths).
"""

from __future__ import annotations

import asyncio

import chess
import pytest

from core.engines.types import MoveClass

from mcp_server.models.legacy import MCPMoveAnalysis
from mcp_server.models.mcpeval import MCPEval
from mcp_server.tools.classify_move import _finish_result


class _Outcome:
    def __init__(self, board: chess.Board, move: chess.Move | None) -> None:
        self.board = board
        self.chess_move = move
        self.history_complete = "incomplete"
        self.rule_before = None
        self.syntax_warning = None


class _FakePool:
    name = "FakePool"
    engine_version = "FakePool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        return None

    async def top_moves(self, board, n=3, depth=14):
        return []

    async def classify_move(self, board, move, depth=14):
        return None

    async def close(self):
        pass


def _ma(board: chess.Board, move_uci: str = "e2e4") -> MCPMoveAnalysis:
    move = chess.Move.from_uci(move_uci)
    board_copy = board.copy(stack=True)
    if move in board_copy.legal_moves:
        board_copy.push(move)
    eval_before = MCPEval(
        cp=20,
        mate=None,
        depth=1,
        requested_depth=1,
        searched_depth=1,
        best_move=move_uci,
        pv=[move_uci],
        engine_eval={
            "requested_depth": 1,
            "searched_depth": 1,
            "best_move": move_uci,
            "cp": 20,
        },
    )
    eval_after = MCPEval(
        cp=15,
        mate=None,
        depth=1,
        requested_depth=1,
        searched_depth=1,
        best_move="",
        pv=[],
        engine_eval={"requested_depth": 1, "searched_depth": 1, "best_move": "", "cp": 15},
    )
    return MCPMoveAnalysis(
        played=move_uci,
        played_san=board.san(move) if move in board.legal_moves else None,
        move_class=MoveClass.BEST,
        eval_before=eval_before,
        eval_after=eval_after,
    )


@pytest.mark.asyncio
async def test_finish_result_accepts_strict_keyword():
    """Bug 2: pre-fix ``strict=...`` raised ``NameError: name 'strict' is not defined``.

    Passing ``strict=False`` (the audit's exact call site) must succeed.
    """
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    move = chess.Move.from_uci("e2e4")
    ma = _ma(board, "e2e4")
    outcome = _Outcome(board, move)
    pool = _FakePool()

    out = await _finish_result(
        ma,
        outcome=outcome,
        pool=pool,
        depth=1,
        detail="standard",
        compare_moves=None,
        strict=False,
    )
    assert out is not None


@pytest.mark.asyncio
async def test_finish_result_rich_with_compare_moves_accepts_strict():
    """Bug 2: rich ``forensic`` with 8 ``compare_moves`` and ``strict`` kwarg.

    This is the audit's exact failing call shape — supplying
    ``compare_moves`` upgrades standard to forensic, which routes through
    ``enrich_move_analysis(..., strict=strict)``. Pre-fix this crashed
    with ``NameError`` before reaching the enrichment path.
    """
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    move = chess.Move.from_uci("e2e4")
    ma = _ma(board, "e2e4")
    outcome = _Outcome(board, move)
    pool = _FakePool()

    out = await _finish_result(
        ma,
        outcome=outcome,
        pool=pool,
        depth=1,
        detail="forensic",
        compare_moves=["d2d4", "g1f3", "b1c3", "c2c4", "e2e3", "g2g3", "b2b3", "h2h3"],
        strict=False,
    )
    assert out.forensics is not None
    assert len(out.forensics.candidate_comparisons) == 8
