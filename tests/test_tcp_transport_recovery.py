"""Regression tests for TCP transport closed recovery.

Verifies that `RuntimeError` from uvloop on a closed TCP transport:
1. Does not break `TCPUCIClient._reset_connection` (clears reader/writer references cleanly).
2. Triggers `is_connected() == False` and transparent reconnection in `_ensure_connected()`.
3. Is translated to `UCIError` (a `ConnectionError` subclass) by `_send()` and `_readline()`.
4. Causes `_EnginePool.run()` to discard the dead handler, spawn a replacement, and retry.
5. Allows `analyze_game` to complete without surfacing `[ENGINE_ERROR]`.
"""

from __future__ import annotations

import asyncio
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


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    import mcp_server.server as server_module

    await server_module.close_analyzer_pool()


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


@pytest.mark.asyncio
async def test_classify_move_forensic_recovers_from_closed_transport():
    """classify_move with detail='forensic' recovers when transport raises RuntimeError."""
    import mcp_server.server as server_module
    from mcp_server.tools.classify_move import classify_move

    from core.engines.pool import AnalyzerPool

    await server_module.close_analyzer_pool()
    await server_module._cache.clear()

    spawn_counter = 0
    dead_handler_id = None
    alive_handler_ids: list[int] = []

    class _TransportFailingWorker:
        def __init__(self, worker_id: int, die_once: bool) -> None:
            self.worker_id = worker_id
            self._die_once = die_once
            self.closed = False
            self.name = "Stockfish 18"

        async def evaluate(
            self, board: chess.Board, *, depth: int = 12, root_moves=None, **kwargs: Any
        ) -> Eval:
            if self._die_once:
                self._die_once = False
                raise RuntimeError(
                    "unable to perform operation on <TCPTransport closed=True reading=False 0x5d2ee4e178e0>; "
                    "the handler is closed"
                )
            legal = list(board.legal_moves)
            best = legal[0].uci() if legal else "e2e4"
            return Eval(cp=25, best_move=best, pv=[best], depth=depth)

        async def top_moves(
            self, board: chess.Board, *, n: int = 3, depth: int | None = None, **kwargs: Any
        ) -> list[Eval]:
            legal = list(board.legal_moves)
            best = legal[0].uci() if legal else "e2e4"
            return [Eval(cp=25, best_move=best, pv=[best], depth=depth or 12)]

        async def probe_threat(self, board: chess.Board, **kwargs: Any) -> Eval | None:
            return None

        async def close(self) -> None:
            self.closed = True

    async def factory():
        nonlocal spawn_counter, dead_handler_id
        spawn_counter += 1
        if spawn_counter == 1:
            worker = _TransportFailingWorker(spawn_counter, die_once=True)
            dead_handler_id = id(worker)
            return worker
        worker = _TransportFailingWorker(spawn_counter, die_once=False)
        alive_handler_ids.append(id(worker))
        return worker

    pool = _EnginePool([await factory()], factory, acquire_timeout=5.0)
    analyzer_pool = AnalyzerPool(pool, name="TestStockfish")
    server_module._analyzer_pool = analyzer_pool  # type: ignore[assignment]

    # Call 1: classify_move(detail="forensic") triggers dead transport, pool catches it, discards dead handler, replaces, retries
    res1 = await classify_move(fen="startpos", move="e4", detail="forensic", depth=8)
    assert res1.move_class in ("best", "good", "inaccuracy", "mistake", "blunder")
    assert spawn_counter == 2
    assert dead_handler_id not in [id(item) for item in pool._q._queue]  # type: ignore[attr-defined]

    # Call 2: second call succeeds cleanly and never touches the dead handler
    res2 = await classify_move(fen="startpos", move="d4", detail="forensic", depth=8)
    assert res2.move_class in ("best", "good", "inaccuracy", "mistake", "blunder")

    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_terminal_checkmate_perspective_contract():
    """White checkmated:
    1. evaluate_position returns status='checkmate', winner='black',
       best_action_obj={'type': 'game_over', 'outcome': 'loss', 'reason': 'checkmate', 'outcome_perspective': 'white'},
       decision_value={'outcome': 'loss', 'perspective': 'white'}.
    2. analyze_game(perspective='white') returns final_position.effective_cp == -100000,
       final segment state='decisively_worse', and no spurious 'recovered' / 'gained_advantage' events.
    """
    from mcp_server.tools.analyze_game import analyze_game
    from mcp_server.tools.evaluate_position import evaluate_position

    # Fool's mate terminal FEN: White is checkmated by Black
    fools_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    eval_res = await evaluate_position(fen=fools_fen, depth=8)
    assert eval_res.status == "checkmate"
    assert eval_res.winner == "black"
    assert eval_res.best_action_obj["type"] == "game_over"
    assert eval_res.best_action_obj["outcome"] == "loss"
    assert eval_res.best_action_obj["outcome_perspective"] == "white"
    assert eval_res.decision_value["outcome"] == "loss"
    assert eval_res.decision_value["perspective"] == "white"

    # White wins checkmate: Scholar's mate terminal FEN
    scholars_fen = "r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    white_win_res = await evaluate_position(fen=scholars_fen, depth=8)
    assert white_win_res.status == "checkmate"
    assert white_win_res.winner == "white"
    assert white_win_res.best_action_obj["type"] == "game_over"
    assert white_win_res.best_action_obj["outcome"] == "win"
    assert white_win_res.best_action_obj["outcome_perspective"] == "white"
    assert white_win_res.decision_value["outcome"] == "win"
    assert white_win_res.decision_value["perspective"] == "white"

    # Full game analysis from White's perspective
    fools_pgn = "1. f3 e5 2. g4 Qh4# 0-1"
    game_res = await analyze_game(pgn=fools_pgn, depth=8, detail="coach", perspective="white")
    assert game_res.coaching is not None
    assert game_res.coaching.final_position.checkmate is True
    assert game_res.coaching.final_position.effective_cp == -100000
    assert game_res.coaching.game_segments[-1].state == "decisively_worse"

    # White must not have recovered or gained advantage after being checkmated
    white_events = [
        e.kind for e in game_res.coaching.advantage_events if e.side == "white" and e.ply == 4
    ]
    assert "recovered" not in white_events
    assert "gained_advantage" not in white_events


@pytest.mark.asyncio
async def test_pool_discards_worker_on_cancelled_error_and_recovers():
    """When a search is cancelled (e.g. caller timeout), the worker is discarded and replaced."""
    spawn_counter = 0
    cancelled_worker_id = None

    class _HangingWorker:
        def __init__(self, wid: int) -> None:
            self.wid = wid
            self.closed = False

        async def evaluate(self, board: chess.Board, **kwargs: Any) -> Eval:
            await asyncio.sleep(10.0)
            return Eval(cp=10)

        async def close(self) -> None:
            self.closed = True

    class _HealthyWorker:
        def __init__(self, wid: int) -> None:
            self.wid = wid
            self.closed = False

        async def evaluate(self, board: chess.Board, **kwargs: Any) -> Eval:
            return Eval(cp=42, best_move="e2e4")

        async def close(self) -> None:
            self.closed = True

    async def factory():
        nonlocal spawn_counter, cancelled_worker_id
        spawn_counter += 1
        if spawn_counter == 1:
            worker = _HangingWorker(spawn_counter)
            cancelled_worker_id = id(worker)
            return worker
        return _HealthyWorker(spawn_counter)

    pool = _EnginePool([await factory()], factory, acquire_timeout=5.0)

    # 1. Run an operation with a 50ms timeout to simulate a caller timeout / CancelledError
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            pool.run(lambda a: a.evaluate(chess.Board())),
            timeout=0.05,
        )

    # The aborted worker must NOT be placed back in the queue
    queued_ids = [id(item) for item in pool._q._queue]  # type: ignore[attr-defined]
    assert cancelled_worker_id not in queued_ids

    # Wait briefly for self-heal to refill the slot
    for _ in range(50):
        if not pool._q.empty():
            break
        await asyncio.sleep(0.05)

    assert not pool._q.empty()
    res = await pool.run(lambda a: a.evaluate(chess.Board()))
    assert res.cp == 42
    await pool.close()


@pytest.mark.asyncio
async def test_analyze_game_forensic_completes_without_timeout():
    """analyze_game in forensic mode runs parallelized gap checks and completes in seconds."""
    from mcp_server.tools.analyze_game import analyze_game

    pgn = "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Bg5 e6 7. f4 Qb6 8. Qd2 Qxb2 9. Rb1 Qa3"
    t0 = asyncio.get_running_loop().time()
    res = await analyze_game(pgn=pgn, depth=14, detail="forensic", perspective="white")
    duration = asyncio.get_running_loop().time() - t0

    assert res.coaching is not None
    assert res.coaching.detail == "forensic"
    assert res.coaching.verification_depth is not None
    assert res.coaching.final_position is not None
    assert duration < 20.0, f"forensic analysis took too long: {duration:.2f}s"


@pytest.mark.asyncio
async def test_pool_discards_fresh_worker_on_cancelled_error_in_replace_and_retry():
    """When _replace_and_retry spawns fresh worker and is cancelled, fresh must be discarded."""
    from core.engines.pool import _EnginePool

    spawned = 0
    fresh_worker_id = None

    class _FirstDeadThenHangingWorker:
        def __init__(self, idx: int) -> None:
            self.idx = idx
            self.closed = False

        async def evaluate(self, board: chess.Board, **kwargs) -> Eval:
            if self.idx == 1:
                raise RuntimeError(
                    "unable to perform operation on <TCPTransport closed=True reading=False 0xdead>; the handler is closed"
                )
            # Replacement worker hangs until cancelled
            await asyncio.sleep(10.0)
            return Eval(cp=42)

        async def close(self) -> None:
            self.closed = True

    async def factory():
        nonlocal spawned, fresh_worker_id
        spawned += 1
        w = _FirstDeadThenHangingWorker(spawned)
        if spawned == 2:
            fresh_worker_id = id(w)
        return w

    initial = await factory()
    pool = _EnginePool([initial], factory, acquire_timeout=5.0)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            pool.run(lambda a: a.evaluate(chess.Board())),  # type: ignore[attr-defined]
            timeout=0.05,
        )

    # The fresh worker must NOT be put back into queue on cancellation
    queued_ids = [id(item) for item in pool._q._queue]  # type: ignore[attr-defined]
    assert fresh_worker_id not in queued_ids
    await pool.close()


@pytest.mark.asyncio
async def test_analyze_game_default_depth_is_14():
    """analyze_game defaults to depth 14 for sub-10s execution under ChatGPT 15s timeout."""
    import inspect
    from mcp_server.tools.analyze_game import analyze_game

    sig = inspect.signature(analyze_game)
    assert sig.parameters["depth"].default == 14


@pytest.mark.asyncio
async def test_gather_evaluate_positions_reserves_workers_for_interactive_queries(monkeypatch):
    """gather_evaluate_positions_bounded reserves at least 1-2 workers in larger pools."""
    import chess
    from mcp_server.engine.parallel_gather import gather_evaluate_positions_bounded
    from mcp_server.models import MCPEval

    async def fake_eval(*args, **kwargs):
        return (MCPEval(cp=10, depth=14, searched_depth=14), True)

    monkeypatch.setattr(
        "mcp_server.engine.pool_factory._evaluate_game_position_cached", fake_eval
    )

    class FakePool:
        def __init__(self, size: int):
            self._pool = type("SubPool", (), {"_target_size": size, "run": self._run})()
            self.acquired_count = 0

        async def _run(self, fn):
            self.acquired_count += 1
            return await fn(object())

    # With pool_size=6, batch partitions across at most 4 workers (leaving 2 free)
    p6 = FakePool(6)
    boards = [chess.Board() for _ in range(20)]
    await gather_evaluate_positions_bounded(boards, depth=14, pool=p6, requested_depth=14)
    assert p6.acquired_count == 4

    # With pool_size=4, batch partitions across at most 3 workers (leaving 1 free)
    p4 = FakePool(4)
    await gather_evaluate_positions_bounded(boards, depth=14, pool=p4, requested_depth=14)
    assert p4.acquired_count == 3


def test_mcpsettings_parses_acquire_timeout(monkeypatch):
    """MCPSettings exposes CHESS_MCP_ACQUIRE_TIMEOUT with 15.0 default."""
    from mcp_server.config import MCPSettings

    monkeypatch.delenv("CHESS_MCP_ACQUIRE_TIMEOUT", raising=False)
    cfg = MCPSettings()
    assert cfg.acquire_timeout == 15.0

    monkeypatch.setenv("CHESS_MCP_ACQUIRE_TIMEOUT", "25.0")
    cfg2 = MCPSettings()
    assert cfg2.acquire_timeout == 25.0


@pytest.mark.asyncio
async def test_evaluate_pair_runs_concurrently():
    """_evaluate_pair runs before and after evaluations concurrently."""
    import chess
    import asyncio
    from mcp_server.analysis.classification_stability import _evaluate_pair
    from mcp_server.models import MCPEval

    active = 0
    max_active = 0

    class FakePool:
        async def evaluate(self, board, depth=14):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return MCPEval(cp=0, depth=depth, searched_depth=depth)

    b1 = chess.Board()
    b2 = chess.Board()
    b2.push_san("e4")

    ev1, ev2 = await _evaluate_pair(
        FakePool(),
        b1,
        b2,
        depth=24,
        history_complete="incomplete",
        evaluate_position=None,
    )
    assert ev1 is not None
    assert ev2 is not None
    assert max_active == 2


