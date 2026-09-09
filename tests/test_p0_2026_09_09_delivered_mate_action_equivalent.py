"""2026-09-09 master audit F-001: delivered-checkmate must be action_equivalent.

Reproduces the audit's P0 finding: a move that delivers checkmate, is the
engine's exact best move, and is the same action type as the canonical best
must report ``action_equivalent=True``. Previously the ``score_delivered_checkmate``
strategy hardcoded ``action_equivalent=False`` even when every other flag
agreed the move was best, breaking a core semantic invariant.

Two independent mate-in-one fixtures are covered, plus a non-best mating
move control case that must remain ``False``.
"""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module
from mcp_server.engine.retry import reset_breaker


class _MateInOnePool:
    """Pool that reports the played move as engine best with mate=1 then mate=0."""

    name = "MateInOnePool"
    engine_version = "MateInOnePool"

    def __init__(self, played_move: chess.Move, board_before: chess.Board) -> None:
        self._played_move = played_move
        self._board_before = board_before
        self._board_after = board_before.copy(stack=True)
        self._board_after.push(played_move)

    async def evaluate(self, board, *, depth=14, root_moves=None):
        legal = list(board.legal_moves)
        best = legal[0].uci() if legal else None
        if board.is_checkmate():
            return Eval(mate=0, best_move=best, pv=[best] if best else [], depth=depth)
        if any(board.is_capture(m) or board.gives_check(m) for m in board.legal_moves):
            return Eval(
                mate=1, best_move=self._played_move.uci(), pv=[self._played_move.uci()], depth=depth
            )
        return Eval(cp=20, best_move=best, pv=[best] if best else [], depth=depth)

    async def classify_move(self, board, move, depth=14):
        played = move.uci()
        return MoveAnalysis(
            played=played,
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(mate=1, best_move=played),
            eval_after=Eval(mate=0),
            best_move_san=board.san(move),
        )

    async def top_moves(self, board, *, n=3, depth=14):
        return []

    async def close(self) -> None:
        pass


class _NonBestMatePool(_MateInOnePool):
    """Pool that returns a non-best (Good) classification even though a mate exists."""

    async def classify_move(self, board, move, depth=14):  # type: ignore[override]
        played = move.uci()
        return MoveAnalysis(
            played=played,
            move_class=MoveClass.GOOD,
            centipawn_loss=20,
            eval_before=Eval(mate=1, best_move=played),
            eval_after=Eval(mate=0),
            best_move_san=board.san(move),
        )


@pytest.fixture(autouse=True)
async def _cleanup() -> None:
    yield
    reset_breaker()
    await server_module.close_analyzer_pool()


def _qg7_fen() -> str:
    return "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"


def _qxf7_fen() -> str:
    return "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fen, move_san",
    [
        (_qg7_fen(), "Qg7#"),
        (_qxf7_fen(), "Qxf7#"),
    ],
)
async def test_delivered_checkmate_is_action_equivalent(fen: str, move_san: str) -> None:
    """P0 F-001: exact engine-best mating move must report action_equivalent=True."""
    await server_module._cache.clear()
    board = chess.Board(fen)
    move = board.parse_san(move_san)
    server_module._analyzer_pool = _MateInOnePool(move, board)

    res = await server_module.classify_move(fen, move=move_san, depth=10, action_type="play_move")

    assert res.is_engine_best is True, (
        f"is_engine_best must be True for exact mating move; got {res.is_engine_best}"
    )
    assert res.is_best_action is True
    assert res.move_class == MoveClass.BEST
    assert res.action_type == "play_move"
    assert res.best_action == "play_move"
    assert res.action_equivalent is True, (
        f"F-001 regression: exact engine-best mating move must be action_equivalent=True; "
        f"got action_equivalent={res.action_equivalent} "
        f"(is_engine_best={res.is_engine_best}, is_best_action={res.is_best_action})"
    )


@pytest.mark.asyncio
async def test_non_best_mating_move_remains_non_equivalent() -> None:
    """Control case: non-best mating move must NOT be action_equivalent=True."""
    await server_module._cache.clear()
    fen = _qg7_fen()
    board = chess.Board(fen)
    move = board.parse_san("Qg7#")
    server_module._analyzer_pool = _NonBestMatePool(move, board)

    res = await server_module.classify_move(fen, move="Qg7#", depth=10, action_type="play_move")

    assert res.is_engine_best is False
    assert res.action_equivalent is False, (
        "Non-best mating move must NOT be action_equivalent=True. "
        "Only exact engine-best mating play_move is action-equivalent."
    )
