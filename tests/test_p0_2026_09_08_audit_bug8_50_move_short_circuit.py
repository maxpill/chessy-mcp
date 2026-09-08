"""2026-09-08 audit regression: 50-move-rule draw final position is short-circuited.

Bug 8 (audit D.31) — Live ``analyze_game`` with a PGN ending in a
50-move-rule draw returned ``searched_depth=1`` for the final position,
which is wasted work — the player will claim rather than play on.

Fix: a ``short_circuit`` callback is threaded through
``gather_evaluate_positions_bounded``. For a PGN whose final position
has ``is_fifty_moves()`` True AND whose game result is ``1/2-1/2``, the
callback synthesizes a terminal ``MCPEval`` (``status="fifty_moves"``,
``searched_depth=0``, ``can_claim_draw=True``) so the engine is never
called on that position.
"""

from __future__ import annotations

import asyncio

import chess
import pytest

from mcp_server.analysis.game_analyzer import (
    _build_fifty_move_short_circuit,
)
from mcp_server.models import MCPEval


def test_short_circuit_returns_none_for_non_fifty_move() -> None:
    """Non-50-move positions must NOT be short-circuited."""
    board = chess.Board()
    cb = _build_fifty_move_short_circuit(
        game_result="1/2-1/2", requested_depth=14, positions=[board]
    )
    assert cb(board, 0) is None


def test_short_circuit_returns_none_when_result_not_draw() -> None:
    """A 50-move position with a non-draw PGN result must NOT be short-circuited.

    The player didn't claim — they kept playing. The engine eval is
    still useful for accuracy metrics.
    """
    board = chess.Board("7k/8/8/8/8/8/8/4K2k w - - 100 51")
    assert board.is_fifty_moves()
    cb = _build_fifty_move_short_circuit(game_result="1-0", requested_depth=14, positions=[board])
    assert cb(board, 0) is None


def test_short_circuit_returns_terminal_mcpeval_for_fifty_move_draw() -> None:
    """The audit's exact case: 50-move position + draw result -> terminal MCPEval."""
    board = chess.Board("7k/8/8/8/8/8/8/4K2k w - - 100 51")
    assert board.is_fifty_moves()
    cb = _build_fifty_move_short_circuit(
        game_result="1/2-1/2", requested_depth=14, positions=[board]
    )
    mc = cb(board, 0)
    assert mc is not None
    assert isinstance(mc, MCPEval)
    assert mc.status == "fifty_moves"
    assert mc.searched_depth == 0
    assert mc.requested_depth == 14
    assert mc.can_claim_draw is True
    assert mc.can_claim_now is True
    assert "fifty_moves" in mc.claim_reasons
    assert "fifty_moves" in mc.claim_reasons_now
    assert mc.recommended_action == "claim_draw"
    assert mc.best_action == "claim_draw"


def test_short_circuit_does_not_call_engine() -> None:
    """The synthesized MCPEval must NOT trigger an engine call.

    This is the audit's primary concern: the engine was being called
    on a position where the player would claim rather than play on.
    The synthesized MCPEval has ``depth=0`` and ``searched_depth=0``,
    and ``engine_eval`` carries the same zeros so a downstream
    consumer reading just the sub-dict sees the same provenance.
    """
    board = chess.Board("7k/8/8/8/8/8/8/4K2k w - - 100 51")
    cb = _build_fifty_move_short_circuit(
        game_result="1/2-1/2", requested_depth=14, positions=[board]
    )
    mc = cb(board, 0)
    assert mc is not None
    assert mc.depth == 0
    assert mc.searched_depth == 0
    assert mc.engine_eval is not None
    assert mc.engine_eval["depth"] == 0


def test_short_circuit_only_fires_for_last_position() -> None:
    """The 2026-09-08 audit F-04 fix must scope the short-circuit to the LAST position.

    An intermediate position with ``is_fifty_moves()`` True (e.g. the
    pre-Qf8 position at halfmove=149 with a winning continuation that
    walks into 75-move-draw on the next move) still needs a real
    engine eval so blunder detection can compare eval_before vs
    eval_after.
    """
    b_before = chess.Board("7k/5Q2/5K2/8/8/8/8/8 w - - 149 75")
    b_after = chess.Board("7k/5Q2/5K2/8/8/8/8/8 w - - 149 75")
    b_after.push(chess.Move.from_uci("f7f8"))
    # Both positions have is_fifty_moves() True.
    assert b_before.is_fifty_moves()
    assert b_after.is_fifty_moves()
    cb = _build_fifty_move_short_circuit(
        game_result="1/2-1/2", requested_depth=14, positions=[b_before, b_after]
    )
    # First position (idx=0) must NOT be short-circuited — blunder detection
    # needs the real engine eval to compare against the blundered position.
    assert cb(b_before, 0) is None
    # Last position (idx=1) IS short-circuited — game has ended.
    mc = cb(b_after, 1)
    assert mc is not None
    assert mc.status == "fifty_moves"


@pytest.mark.asyncio
async def test_gather_short_circuit_skips_engine_for_fifty_move_position() -> None:
    """End-to-end: ``gather_evaluate_positions_bounded`` short-circuits
    the 50-move position and does NOT invoke the engine on it.

    The other positions are still engine-evaluated.
    """
    eval_calls: list[str] = []

    class _RecordingPool:
        name = "RecordingPool"
        engine_version = "RecordingPool"

        async def evaluate(self, board, *, depth=14, root_moves=None):
            from core.engines.types import Eval

            eval_calls.append(board.fen())
            return Eval(
                cp=20,
                mate=None,
                best_move="e2e4",
                pv=["e2e4"],
                depth=depth,
            )

        async def top_moves(self, board, n=3, depth=14):
            return []

        async def classify_move(self, board, move, depth=14):
            return None

        async def close(self):
            pass

    # Two positions: a normal startpos and a 50-move-claimable FEN.
    b_start = chess.Board()
    b_50 = chess.Board("7k/8/8/8/8/8/8/4K2k w - - 100 51")
    positions = [b_start, b_50]

    cb = _build_fifty_move_short_circuit(
        game_result="1/2-1/2", requested_depth=10, positions=positions
    )
    pool = _RecordingPool()

    from mcp_server.engine.parallel_gather import (
        gather_evaluate_positions_bounded,
    )

    eval_pairs = await gather_evaluate_positions_bounded(
        positions,
        depth=10,
        pool=pool,
        requested_depth=10,
        history_complete="complete",
        short_circuit=cb,
    )
    # Only the startpos was engine-evaluated; the 50-move position was
    # short-circuited.
    assert len(eval_calls) == 1, (
        f"only the non-50-move position should trigger an engine call; got {len(eval_calls)} calls"
    )
    assert eval_calls[0] == b_start.fen()
    # Both positions returned an MCPEval with correct searched_depth.
    assert eval_pairs[0][0].searched_depth == 10
    assert eval_pairs[1][0].searched_depth == 0
    assert eval_pairs[1][0].status == "fifty_moves"
