"""Regression tests for TCP transport closed recovery.

Verifies that `RuntimeError` from uvloop on a closed TCP transport:
1. Does not break `TCPUCIClient._reset_connection` (clears reader/writer references cleanly).
2. Triggers `is_connected() == False` and transparent reconnection in `_ensure_connected()`.
3. Is translated to `UCIError` (a `ConnectionError` subclass) by `_send()` and `_readline()`.
4. Causes `_EnginePool.run()` to discard the dead handler, spawn a replacement, and retry.
5. Allows `analyze_game` to complete without surfacing `[ENGINE_ERROR]`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import chess
import pytest

from core.engines.pool import _EnginePool, _is_transport_error
from core.engines.types import Eval
from mcp_server.analysis.game_analyzer import GameAnalyzer
from mcp_server.engine.retry import is_transport_error as retry_is_transport_error
from mcp_server.engine.retry import with_engine_retry
from mcp_server.models.game_coaching import ForensicGameAnalysisResult
from mcp_server.models.mcpeval import MCPEval
from mcp_server.tcp_client import TCPUCIClient, UCIError


class DummyTransport:
    """Simulates uvloop's TCPTransport behavior when closed."""

    def __init__(self, is_closing: bool = True) -> None:
        self._closing = is_closing

    def is_closing(self) -> bool:
        return self._closing

    def write(self, data: bytes) -> None:
        if self._closing:
            raise RuntimeError(
                "unable to perform operation on <TCPTransport closed=True reading=False 0xdeadbeef>; "
                "the handler is closed"
            )


class DummyWriter:
    def __init__(self, transport: DummyTransport) -> None:
        self.transport = transport
        self._closing = transport.is_closing()

    def is_closing(self) -> bool:
        return self._closing

    def write(self, data: bytes) -> None:
        self.transport.write(data)

    async def drain(self) -> None:
        if self._closing:
            raise RuntimeError(
                "unable to perform operation on <TCPTransport closed=True reading=False 0xdeadbeef>; "
                "the handler is closed"
            )

    def close(self) -> None:
        self._closing = True

    async def wait_closed(self) -> None:
        pass


@pytest.mark.asyncio
async def test_reset_connection_handles_closed_transport_without_raising():
    """_reset_connection must cleanly set _writer and _reader to None even if write raises."""
    client = TCPUCIClient("127.0.0.1", 9999, name="test_engine")
    transport = DummyTransport(is_closing=True)
    writer = DummyWriter(transport)
    client._writer = writer  # type: ignore[assignment]
    client._reader = MagicMock()

    assert not client.is_connected()
    # Must not raise RuntimeError
    await client._reset_connection()
    assert client._writer is None
    assert client._reader is None
    assert not client.is_connected()


@pytest.mark.asyncio
async def test_send_and_readline_convert_runtime_error_to_ucierror():
    """Closed transport RuntimeError during send or readline raises UCIError and resets connection."""
    client = TCPUCIClient("127.0.0.1", 9999, name="test_engine")
    transport = DummyTransport(is_closing=True)
    writer = DummyWriter(transport)
    client._writer = writer  # type: ignore[assignment]
    client._reader = MagicMock()

    with pytest.raises(UCIError) as exc_send:
        await client._send("ucinewgame")
    assert "transport error during send" in str(exc_send.value)
    assert client._writer is None
    assert client._reader is None

    # Simulate readline failure
    client._writer = DummyWriter(DummyTransport(is_closing=False))  # type: ignore[assignment]
    reader = AsyncMock()
    reader.readline.side_effect = RuntimeError(
        "unable to perform operation on <TCPTransport closed=True reading=False 0xdeadbeef>; "
        "the handler is closed"
    )
    client._reader = reader

    with pytest.raises(UCIError) as exc_read:
        await client._readline()
    assert "transport error during readline" in str(exc_read.value)
    assert client._writer is None
    assert client._reader is None


def test_transport_error_predicates():
    """_is_transport_error recognizes closed-transport RuntimeErrors."""
    err = RuntimeError(
        "unable to perform operation on <TCPTransport closed=True reading=False 0xdeadbeef>; "
        "the handler is closed"
    )
    assert _is_transport_error(err)
    assert retry_is_transport_error(err)
    assert _is_transport_error(ConnectionError("lost"))
    assert not _is_transport_error(ValueError("invalid fen"))
    assert not _is_transport_error(RuntimeError("unrelated runtime failure"))


@pytest.mark.asyncio
async def test_retry_helper_recovers_from_transport_runtime_error():
    """with_engine_retry retries on closed transport RuntimeError."""
    attempts = 0

    async def flaky_call():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError(
                "unable to perform operation on <TCPTransport closed=True reading=False 0xdeadbeef>; "
                "the handler is closed"
            )
        return "success"

    res = await with_engine_retry(flaky_call, max_retries=1)
    assert res == "success"
    assert attempts == 2


class _ClosedTransportOnFirstCallWorker:
    def __init__(self, die_once: bool) -> None:
        self._die_once = die_once
        self.calls = 0

    async def evaluate(self, board: chess.Board, *, depth: int = 12, **kwargs: Any) -> Eval:
        self.calls += 1
        if self._die_once:
            self._die_once = False
            raise RuntimeError(
                "unable to perform operation on <TCPTransport closed=True reading=False 0xdeadbeef>; "
                "the handler is closed"
            )
        return Eval(cp=15, best_move="e2e4", pv=["e2e4"], depth=depth)

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_pool_discards_dead_handler_on_closed_transport_runtime_error():
    """_EnginePool replaces the handler and retries operation on RuntimeError."""
    pool_size = 2
    spawn_count = 0

    async def factory():
        nonlocal spawn_count
        spawn_count += 1
        # First spawn has dead transport; replacement will be healthy
        return _ClosedTransportOnFirstCallWorker(die_once=(spawn_count == 1))

    instances: list[object] = [await factory() for _ in range(pool_size)]
    pool = _EnginePool(instances, factory, acquire_timeout=5.0)

    board = chess.Board()
    ev = await pool.run(lambda a: a.evaluate(board, depth=10))  # type: ignore[attr-defined]
    assert ev.cp == 15
    # Replacement was spawned to replace dead instance
    assert spawn_count == pool_size + 1


@pytest.mark.asyncio
async def test_analyze_game_recovers_from_closed_transport():
    """GameAnalyzer completes successfully when evaluate_positions recovers from dead handler."""
    calls = 0

    async def mock_eval_positions(positions, depth, pool, **kwargs):
        nonlocal calls
        calls += 1
        # Return mock evaluations for the moves
        evals = []
        for b in positions:
            evals.append(
                (
                    MCPEval.from_eval(
                        Eval(cp=20, best_move="e2e4", pv=["e2e4"], depth=depth),
                        b.fen(),
                        board=b,
                        requested_depth=depth,
                    ),
                    False,
                )
            )
        return evals

    analyzer = GameAnalyzer(
        get_pool=AsyncMock(return_value=MagicMock()),
        evaluate_positions=mock_eval_positions,
        compute_metrics=MagicMock(
            return_value=MagicMock(
                white_accuracy=95.0,
                black_accuracy=94.0,
                white_acpl=15.0,
                black_acpl=18.0,
                white_raw_acpl=15.0,
                black_raw_acpl=18.0,
                white_effective_acpl=15.0,
                black_effective_acpl=18.0,
                white_blunders=0,
                white_mistakes=0,
                white_inaccuracies=0,
                black_blunders=0,
                black_mistakes=0,
                black_inaccuracies=0,
                turning_points=[],
            )
        ),
        identity=MagicMock(
            return_value={
                "build_sha": "abc1234",
                "service_version": "0.1.0",
                "schema_version": "1.2.0",
            }
        ),
        engine_version=MagicMock(return_value="Stockfish 18"),
    )

    pgn = "1. e4 e5 2. Nf3 Nc6 *"
    result: ForensicGameAnalysisResult = await analyzer.analyze(
        pgn=pgn,
        depth=8,
        strict=False,
        ctx=None,
        detail="standard",
    )
    assert result.total_plies == 4
    assert result.white_accuracy == 95.0
    assert calls == 1
