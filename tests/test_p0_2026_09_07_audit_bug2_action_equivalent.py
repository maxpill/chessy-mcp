"""2026-09-07 audit regression: action_equivalent must be True for exact engine-best moves.

Bug §2 — ``classify_move(startpos, e4, depth=1)`` reported::

    is_engine_best = True
    is_best_action = True
    centipawn_loss = 0
    action_equivalent = False   # contradictory

Root cause: the ``is_best_engine_move`` early-return branch in
``score_standard_cp`` hardcoded ``action_equivalent=False``. The non-best
branch already used the correct formula. This test pins the fix.
"""

from __future__ import annotations

import pytest

from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module
from mcp_server.engine.retry import reset_breaker


class _BestMovePool:
    """Minimal pool: classify_move reports the requested move as engine's best."""

    name = "BestMovePool"
    engine_version = "BestMovePool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        legal = list(board.legal_moves)
        best = legal[0].uci() if legal else None
        return Eval(cp=20, best_move=best, pv=[best] if best else [], depth=depth)

    async def classify_move(self, board, move, depth=14):
        played = move.uci() if move else ""
        return MoveAnalysis(
            played=played,
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(cp=20, best_move=played),
            eval_after=Eval(cp=20),
            best_move_san=board.san(move) if move in board.legal_moves else "",
        )

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _cleanup() -> None:
    yield
    reset_breaker()
    await server_module.close_analyzer_pool()


def _startpos() -> str:
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.mark.asyncio
async def test_classify_move_engine_best_move_is_action_equivalent() -> None:
    """§2: engine-best play_move with no rule action must be action_equivalent=True."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _BestMovePool()

    res = await server_module.classify_move(
        _startpos(), move="e2e4", depth=1, action_type="play_move"
    )

    assert res.is_engine_best is True
    assert res.is_best_action is True
    assert res.centipawn_loss == 0
    assert res.action_equivalent is True, (
        f"action_equivalent must be True for the engine's exact best move; "
        f"got action_equivalent={res.action_equivalent}, "
        f"best_move_san={res.best_move_san}, action_type={res.action_type}"
    )


@pytest.mark.asyncio
async def test_classify_move_engine_best_uci_input() -> None:
    """§2: UCI input form must also satisfy the invariant."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _BestMovePool()

    res = await server_module.classify_move(
        _startpos(), move="e2e4", depth=1, action_type="play_move"
    )

    assert res.played == "e2e4"
    assert res.action_equivalent is True


@pytest.mark.asyncio
async def test_classify_move_non_best_move_keeps_action_equivalent_false() -> None:
    """Non-best moves must NOT become action_equivalent=True (control case)."""
    await server_module._cache.clear()

    class _OffBestPool(_BestMovePool):
        async def classify_move(self, board, move, depth=14):  # type: ignore[override]
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.GOOD,
                centipawn_loss=12,
                eval_before=Eval(cp=20, best_move="d2d4"),
                eval_after=Eval(cp=8),
                best_move_san="d4",
            )

    server_module._analyzer_pool = _OffBestPool()
    res = await server_module.classify_move(
        _startpos(), move="e2e4", depth=1, action_type="play_move"
    )

    assert res.is_engine_best is False
    assert res.action_equivalent is False
