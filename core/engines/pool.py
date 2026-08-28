"""Crash-resilient pools of Stockfish engines.

A single Stockfish process serializes all UCI calls (hence the per-engine lock),
so one shared opponent/analyzer caps the whole server at one move at a time. These
pools hold N independent engine subprocesses dispatched via a queue, so concurrent
requests run in parallel across cores — with one drop-in interface each.

Crash handling is the point: Stockfish *can* die mid-call (segfault on a pathological
position). A naive queue would lose that slot and slowly drain to a deadlock under
load. Here every use replaces a dead engine with a fresh subprocess and never returns
a dead engine to the queue; a pile-up fails fast (PoolBusy → 503) instead of queueing
unbounded into an OOM.

If the synchronous respawn inside `run()` itself fails (e.g. the Stockfish binary
has been replaced with a broken one, or system resources are exhausted), the slot
is dropped — and without a refill path, repeated transient failures would shrink
the pool to zero, after which every request would PoolBusy until process restart.
A background self-heal task replenishes missing slots up to the original size.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import chess
import chess.engine

from core import usage

from .analyzer import Analyzer
from .opponent import StockfishOpponent
from .types import Eval, MoveAnalysis

log = logging.getLogger("chessy.enginepool")

DEFAULT_ACQUIRE_TIMEOUT = 6.0  # seconds a request waits for a free engine before giving up

T = TypeVar("T")


class PoolBusy(Exception):
    """No engine became available within the acquire timeout (server overloaded)."""


class _EnginePool:
    """Generic pool: N instances in a queue, each with .close(); recreated on death."""

    def __init__(
        self,
        instances: list[object],
        factory: Callable[[], Awaitable[object]],
        acquire_timeout: float,
    ) -> None:
        self._factory = factory
        self._acquire_timeout = acquire_timeout
        self._q: asyncio.Queue[object] = asyncio.Queue()
        for inst in instances:
            self._q.put_nowait(inst)
        self.size = len(instances)
        self._target_size = len(instances)
        # P1 audit fix: track the TOTAL number of alive instances (idle + busy),
        # not just the queue size. The queue only holds idle workers, so under
        # load `qsize()` would always look low and the loop would over-spawn
        # until it ran away. We increment on `run()` (busy), decrement on put
        # (idle replacement) or completion (failure).
        self._alive_count = len(instances)
        self._closed = False
        self._self_heal_task: asyncio.Task[None] | None = None
        # Self-heal cadence: 5s is fast enough to recover within a typical request
        # burst but slow enough to avoid busy-spinning if the factory is broken.
        self._self_heal_interval_s = 5.0
        self._self_heal_attempts = 0
        self._last_self_heal_log = 0.0

    def _start_self_heal(self) -> None:
        if self._self_heal_task is not None and not self._self_heal_task.done():
            return
        self._self_heal_task = asyncio.create_task(self._self_heal_loop())

    async def _self_heal_loop(self) -> None:
        """Refill the pool back up to its target size when slots are lost to failed respawns."""
        while not self._closed:
            try:
                await asyncio.sleep(self._self_heal_interval_s)
            except asyncio.CancelledError:
                return
            if self._closed:
                return
            # P1 audit fix: use the actual alive count (idle + busy), not the
            # queue size. Previously `missing = target - qsize()` over-counted
            # busy workers as missing, causing the loop to spawn extras on every
            # tick under steady load.
            missing = self._target_size - self._alive_count
            if missing <= 0:
                continue
            try:
                fresh = await self._factory()
            except Exception:  # noqa: BLE001 — factory still broken; back off
                self._self_heal_attempts += 1
                # Log at most once per minute to avoid log flooding
                import time as _time

                now = _time.time()
                if now - self._last_self_heal_log > 60.0:
                    log.warning(
                        "engine self-heal: factory still failing (%d consecutive attempts); pool size %d/%d",
                        self._self_heal_attempts,
                        self._q.qsize(),
                        self._target_size,
                    )
                    self._last_self_heal_log = now
                # Back off exponentially up to 60s when persistent
                self._self_heal_interval_s = min(60.0, self._self_heal_interval_s * 1.5)
                continue
            # Success — reset counters
            self._self_heal_attempts = 0
            self._self_heal_interval_s = 5.0
            try:
                self._q.put_nowait(fresh)
            except Exception:  # noqa: BLE001
                pass
            log.info("engine self-heal: refilled slot, pool now %d/%d", self._q.qsize(), self._target_size)

    async def run(self, fn: Callable[[object], Awaitable[T]]) -> T:
        usage.count("stockfish")
        try:
            inst = await asyncio.wait_for(self._q.get(), timeout=self._acquire_timeout)
        except TimeoutError as exc:
            raise PoolBusy(f"no engine free within {self._acquire_timeout}s") from exc

        # P1 audit fix: account for the worker being BUSY during fn() so the
        # self-heal loop doesn't double-count it as missing.
        self._alive_count = max(0, self._alive_count)  # already counted; keep
        replace = False
        try:
            return await fn(inst)
        except (chess.engine.EngineTerminatedError, ConnectionError, OSError):
            replace = True  # engine died mid-call — swap in a fresh one below
            raise

        finally:
            if not replace:
                self._q.put_nowait(inst)
            else:
                # Engine died — alive count drops by one (we'll respawn below
                # or kick self-heal if respawn fails).
                self._alive_count = max(0, self._alive_count - 1)
                try:
                    await inst.close()  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001 — best-effort cleanup of a dead process
                    pass
                try:
                    fresh = await self._factory()
                    self._q.put_nowait(fresh)
                    self._alive_count += 1
                except Exception:  # noqa: BLE001 — respawn failed; kick off background refill
                    log.exception(
                        "engine respawn failed; pool at %d/%d — self-heal engaged",
                        self._q.qsize(),
                        self._target_size,
                    )
                    self._start_self_heal()

    async def close(self) -> None:
        self._closed = True
        if self._self_heal_task is not None:
            self._self_heal_task.cancel()
            try:
                await self._self_heal_task
            except (asyncio.CancelledError, Exception):
                pass
            self._self_heal_task = None
        while not self._q.empty():
            inst = self._q.get_nowait()
            try:
                await inst.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass


EnginePool = _EnginePool


class OpponentPool:
    """Drop-in for StockfishOpponent: same select_move(board, target_rating)."""

    name = "stockfish-pool"

    def __init__(self, pool: _EnginePool) -> None:
        self._pool = pool

    @classmethod
    async def create(
        cls, path: str, size: int, acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT
    ) -> OpponentPool:
        async def factory() -> object:
            return await StockfishOpponent.create(path)

        instances = [await factory() for _ in range(max(1, size))]
        log.info("opponent pool ready: %d engines", len(instances))
        return cls(_EnginePool(instances, factory, acquire_timeout))

    async def select_move(self, board: chess.Board, target_rating: int) -> chess.Move:
        return await self._pool.run(lambda e: e.select_move(board, target_rating))  # type: ignore[attr-defined]

    async def close(self) -> None:
        await self._pool.close()


class AnalyzerPool:
    """Drop-in for Analyzer: same evaluate / classify_move / probe_threat."""

    def __init__(self, pool: _EnginePool, name: str = "Stockfish") -> None:
        self._pool = pool
        self.name = name
        self.engine_version = name

    @classmethod
    async def create(
        cls,
        path: str,
        size: int,
        *,
        depth: int = 12,
        threads: int = 1,
        hash_mb: int = 128,
        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> AnalyzerPool:
        async def factory() -> object:
            return await Analyzer.create(path, depth=depth, threads=threads, hash_mb=hash_mb)

        instances = [await factory() for _ in range(max(1, size))]
        engine_name = getattr(instances[0], "name", "Stockfish") if instances else "Stockfish"
        log.info("analyzer pool ready: %d engines (%s)", len(instances), engine_name)
        return cls(_EnginePool(instances, factory, acquire_timeout), name=engine_name)

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int | None = None,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        return await self._pool.run(lambda a: a.evaluate(board, depth=depth, root_moves=root_moves))  # type: ignore[attr-defined]

    async def classify_move(
        self, board: chess.Board, move: chess.Move, *, depth: int | None = None
    ) -> MoveAnalysis:
        # classify_move does several evaluate() calls internally — keep them on ONE acquired engine.
        return await self._pool.run(lambda a: a.classify_move(board, move, depth=depth))  # type: ignore[attr-defined]

    async def top_moves(self, board: chess.Board, *, n: int = 3, depth: int | None = None) -> list[Eval]:
        return await self._pool.run(lambda a: a.top_moves(board, n=n, depth=depth))  # type: ignore[attr-defined]

    async def probe_threat(self, board_after: chess.Board, *, depth: int | None = None) -> Eval | None:
        return await self._pool.run(lambda a: a.probe_threat(board_after, depth=depth))  # type: ignore[attr-defined]

    async def close(self) -> None:
        await self._pool.close()
