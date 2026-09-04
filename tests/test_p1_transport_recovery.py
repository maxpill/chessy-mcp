"""P1 regression tests: TCP transport recovery.

Bug doc §8 — closed transport must auto-reconnect with a single retry,
not surface as a 500. The current code re-raises from pool and tool layer
wraps it as engine_error.
"""

from __future__ import annotations

import pytest

from core.engines.types import Eval
from mcp_server import server as server_module


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


class _DeadThenAlivePool:
    """First call raises ConnectionError, subsequent calls succeed.

    Simulates a TCP transport that died but the next pool member is alive.
    The test verifies the retry mechanism kicks in exactly once.
    """

    name = "DeadThenAlivePool"
    engine_version = "DeadThenAlivePool"

    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, board, *, depth=14, root_moves=None):
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("simulated closed TCPTransport")
        legal = list(board.legal_moves)
        best = legal[0].uci() if legal else None
        return Eval(cp=10, best_move=best, pv=[best] if best else [], depth=depth)

    async def classify_move(self, board, move, depth=14):
        from core.engines.types import MoveAnalysis, MoveClass

        return MoveAnalysis(
            played=move.uci(),
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(cp=10, best_move=move.uci()),
            eval_after=Eval(cp=10),
        )

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_one_retry_after_connection_error_succeeds():
    """A single transient ConnectionError must result in a successful retry."""
    await server_module._cache.clear()
    pool = _DeadThenAlivePool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]

    res = await server_module.evaluate_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", depth=8
    )
    assert res.cp == 10
    assert pool.calls == 2, f"expected 1 retry (2 calls total), got {pool.calls}"


class _PermanentlyDeadPool:
    """Always raises ConnectionError."""

    name = "PermanentlyDeadPool"

    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, board, *, depth=14, root_moves=None):
        self.calls += 1
        raise ConnectionError("permanently dead")

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_permanent_failure_does_not_retry_storm():
    """When the second attempt also fails, must NOT keep retrying."""
    await server_module._cache.clear()
    pool = _PermanentlyDeadPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]

    with pytest.raises(Exception):
        await server_module.evaluate_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", depth=8
        )
    # 2 total: initial + 1 retry. NO storm.
    assert pool.calls <= 3, f"too many attempts ({pool.calls}); retry storm suspected"
