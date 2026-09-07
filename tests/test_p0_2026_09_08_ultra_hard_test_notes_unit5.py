"""2026-09-08 ultra-hard test notes (early obs §7): best_move_uci populated in coach mode.

Before the fix, ``evaluate_position(detail="coach")`` reported
``forensics.best_move_uci=null`` even when the top-level ``best_move`` was
populated — only ``detail="forensic"`` populated the post-best evidence.
The fields are cheap (one extra ``board.copy()`` + tactical snapshot), so
the fix widens the population to both coach and forensic modes.
"""

from __future__ import annotations

import pytest

from core.engines.types import Eval
from mcp_server import server as server_module
from mcp_server.tools import evaluate_position as evaluate_position_module


class _BestMovePool:
    name = "BestMovePool"
    engine_version = "BestMovePool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        legal = list(board.legal_moves)
        if not legal:
            return Eval(cp=0, best_move=None, pv=[], depth=depth)
        m = legal[0]
        return Eval(cp=20, best_move=m.uci(), pv=[m.uci()], depth=depth)

    async def top_moves(self, board, n=3, depth=14):
        legal = list(board.legal_moves)[:n]
        return [Eval(cp=20, best_move=m.uci(), pv=[m.uci()], depth=depth) for m in legal]

    async def classify_move(self, board, move, depth=14):
        return None

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _cleanup():
    await server_module._cache.clear()
    server_module._analyzer_pool = _BestMovePool()  # type: ignore[assignment]
    yield
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_evaluate_position_coach_populates_best_move_uci():
    """§7: detail=coach must populate the post-best forensic fields."""
    res = await evaluate_position_module.evaluate_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        depth=1,
        detail="coach",
    )

    assert res.forensics is not None
    assert res.forensics.best_move_uci is not None, (
        f"forensics.best_move_uci must be populated in coach mode; got "
        f"forensics.best_move_uci={res.forensics.best_move_uci!r}, "
        f"top-level best_move={res.best_move!r}"
    )
    assert res.forensics.best_move_san is not None
    assert res.forensics.position_after_best is not None
    assert res.forensics.tactical_after_best is not None
    assert res.forensics.best_move_delta is not None


@pytest.mark.asyncio
async def test_evaluate_position_forensic_still_populates_best_move_uci():
    """Regression guard: detail=forensic must keep populating the post-best fields."""
    res = await evaluate_position_module.evaluate_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        depth=1,
        detail="forensic",
    )

    assert res.forensics is not None
    assert res.forensics.best_move_uci is not None
    assert res.forensics.best_move_san is not None


@pytest.mark.asyncio
async def test_evaluate_position_standard_keeps_forensic_null():
    """Regression guard: detail=standard must NOT add a forensic block."""
    res = await evaluate_position_module.evaluate_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        depth=1,
        detail="standard",
    )

    assert res.forensics is None, (
        f"detail=standard must not attach a forensic block; got forensics={res.forensics!r}"
    )
