"""2026-09-08 ultra-hard test notes Finding #1: depth provenance consistency.

For ``top_moves`` and ``classify_move`` the requested_depth field must
preserve the caller's validated depth (even when 0 or > 30); only the
``depth`` / ``searched_depth`` fields get clamped.

This pins the same contract as ``test_mcp_new_11_requested_depth_semantics_
clamping`` (which already covers ``evaluate_position``).
"""

from __future__ import annotations

import pytest

from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module
from mcp_server.tools import classify_move as classify_move_module
from mcp_server.tools import top_moves as top_moves_module


class _ClampedPool:
    """Minimal analyzer pool that records the clamped depth seen by the engine."""

    name = "ClampedPool"
    engine_version = "ClampedPool"

    def __init__(self) -> None:
        self.last_depth = 0

    async def evaluate(self, board, *, depth=14, root_moves=None):
        self.last_depth = depth
        return Eval(cp=20, best_move="e2e4", pv=["e2e4"], depth=depth)

    async def top_moves(self, board, n=3, depth=14):
        self.last_depth = depth
        return [Eval(cp=20, best_move="e2e4", pv=["e2e4"], depth=depth)]

    async def classify_move(self, board, move, depth=14):
        self.last_depth = depth
        return MoveAnalysis(
            played=move.uci(),
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(cp=20, best_move=move.uci(), pv=[move.uci()], depth=depth),
            eval_after=Eval(cp=20, depth=depth),
            best_move_san=board.san(move),
        )

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _cleanup():
    await server_module._cache.clear()
    server_module._analyzer_pool = _ClampedPool()  # type: ignore[assignment]
    yield
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_top_moves_requested_depth_zero_preserves_caller_intent():
    """§1: top_moves(depth=0) must round-trip requested_depth=0; only depth=1 reaches the engine."""
    pool = server_module._analyzer_pool  # type: ignore[assignment]
    res = await top_moves_module.top_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        n=1,
        depth=0,
    )
    assert res.requested_depth == 0, (
        f"top_moves(depth=0).requested_depth must equal caller's 0; got {res.requested_depth}"
    )
    assert pool.last_depth == 1, (
        f"engine pool must have received the clamped depth 1; got {pool.last_depth}"
    )


@pytest.mark.asyncio
async def test_top_moves_requested_depth_99_preserves_caller_intent():
    """§1: top_moves(depth=99) must round-trip requested_depth=99; only depth=30 reaches the engine."""
    pool = server_module._analyzer_pool  # type: ignore[assignment]
    res = await top_moves_module.top_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        n=1,
        depth=99,
    )
    assert res.requested_depth == 99
    assert pool.last_depth == 30


@pytest.mark.asyncio
async def test_classify_move_requested_depth_zero_preserves_caller_intent():
    """§1: classify_move(depth=0) must round-trip requested_depth=0 on eval_before and eval_after."""
    pool = server_module._analyzer_pool  # type: ignore[assignment]
    res = await classify_move_module.classify_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move="e2e4",
        depth=0,
    )
    assert res.eval_before.requested_depth == 0, (
        f"classify_move(depth=0).eval_before.requested_depth must be 0; got "
        f"{res.eval_before.requested_depth}"
    )
    assert res.eval_after.requested_depth == 0
    assert pool.last_depth == 1


@pytest.mark.asyncio
async def test_classify_move_requested_depth_99_preserves_caller_intent():
    """§1: classify_move(depth=99) must round-trip requested_depth=99 on the evals."""
    pool = server_module._analyzer_pool  # type: ignore[assignment]
    res = await classify_move_module.classify_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move="e2e4",
        depth=99,
    )
    assert res.eval_before.requested_depth == 99
    assert res.eval_after.requested_depth == 99
    assert pool.last_depth == 30


@pytest.mark.asyncio
async def test_top_moves_requested_depth_in_band_unchanged():
    """§1 control case: depth=20 (in band) must NOT be transformed."""
    pool = server_module._analyzer_pool  # type: ignore[assignment]
    res = await top_moves_module.top_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        n=1,
        depth=20,
    )
    assert res.requested_depth == 20
    assert res.searched_depth == 20
    assert pool.last_depth == 20
