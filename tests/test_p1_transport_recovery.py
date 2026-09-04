"""P1 regression tests: TCP transport recovery.

Bug doc §8 — closed transport must auto-reconnect with a single retry,
not surface as a 500. The current code re-raises from pool and tool layer
wraps it as engine_error.
"""

from __future__ import annotations

import chess
import chess.engine
import pytest

from core.engines.pool import _EnginePool
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


class _DeadFirstAliveSecondPool:
    """Per-instance death toggle. When ``die_on_next=True`` the next evaluate
    call raises chess.engine.EngineError; the instance is then "alive" forever
    (i.e. simulates a one-shot TCPTransport closure that leaves the underlying
    channel working on next connect). The pool must discard the instance after
    the error and refilled via the factory.
    """

    def __init__(self, *, die_on_next: bool) -> None:
        self._die_on_next = die_on_next
        self.success_calls = 0
        self.failed_calls = 0

    async def evaluate(self, board, *, depth=14, root_moves=None):
        if self._die_on_next:
            self._die_on_next = False
            self.failed_calls += 1
            raise chess.engine.EngineError(
                "unable to perform operation on <TCPTransport closed=True reading=False 0xdeadbeef>"
            )
        self.success_calls += 1
        legal = list(board.legal_moves)
        best = legal[0].uci() if legal else None
        return Eval(cp=42, best_move=best, pv=[best] if best else [], depth=depth)

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_pool_replaces_dead_instance_on_engine_error():
    """Pool must discard an instance whose call raises EngineError and try again.

    Without the fix the dead instance is returned to the queue untouched and
    every subsequent call hits the same closed transport.
    """
    pool_size = 2
    spawn_index = 0

    async def make():
        nonlocal spawn_index
        spawn_index += 1
        return _DeadFirstAliveSecondPool(die_on_next=(spawn_index == 1))

    pool = _EnginePool([await make() for _ in range(pool_size)], make, acquire_timeout=5.0)
    assert pool._alive_count == pool_size

    board = chess.Board()
    with pytest.raises(chess.engine.EngineError):
        await pool.run(lambda a: a.evaluate(board, depth=12))
    assert spawn_index == pool_size + 1, (
        f"factory spawn count={spawn_index}, expected {pool_size + 1} after dead-instance discard"
    )
    assert pool._alive_count == pool_size, (
        f"pool did not self-heal: alive={pool._alive_count}/{pool_size}"
    )

    good_call = await pool.run(lambda a: a.evaluate(board, depth=12))
    assert good_call.cp == 42

    await pool.close()
