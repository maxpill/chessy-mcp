"""Crash-resilient pools of chess engines."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import chess
import chess.engine

from core import usage

from .analyzer import Analyzer
from .types import Eval, MoveAnalysis

log = logging.getLogger("chessy.enginepool")

DEFAULT_ACQUIRE_TIMEOUT = 15.0
T = TypeVar("T")
_TRANSPORT_ERRORS = (
    chess.engine.EngineError,
    chess.engine.EngineTerminatedError,
    ConnectionError,
    OSError,
)


class PoolBusy(Exception):
    """No engine became available within the acquire timeout."""


class _EnginePool:
    """Generic fixed-size engine pool with crash replacement and self-heal."""

    def __init__(
        self,
        instances: list[object],
        factory: Callable[[], Awaitable[object]],
        acquire_timeout: float,
    ) -> None:
        self._factory = factory
        self._acquire_timeout = acquire_timeout
        self._q: asyncio.Queue[object] = asyncio.Queue(maxsize=len(instances))
        for inst in instances:
            self._q.put_nowait(inst)
        self.size = len(instances)
        self._target_size = len(instances)
        self._alive_count = len(instances)
        self._closed = False
        self._self_heal_task: asyncio.Task[None] | None = None
        self._self_heal_interval_s = 5.0
        self._self_heal_attempts = 0
        self._last_self_heal_log = 0.0
        self._cardinality_lock = asyncio.Lock()

    def _start_self_heal(self) -> None:
        if self._closed:
            return
        if self._self_heal_task is not None and not self._self_heal_task.done():
            return
        self._self_heal_task = asyncio.create_task(self._self_heal_loop())

    async def _discard(self, inst: object) -> None:
        try:
            await inst.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    async def _accept_fresh(self, fresh: object) -> bool:
        """Return a healthy replacement to the pool without exceeding target size."""
        accepted = False
        async with self._cardinality_lock:
            if not self._closed and self._alive_count < self._target_size:
                try:
                    self._q.put_nowait(fresh)
                except asyncio.QueueFull:
                    accepted = False
                else:
                    self._alive_count += 1
                    accepted = True
        if not accepted:
            await self._discard(fresh)
        return accepted

    async def _self_heal_loop(self) -> None:
        """Refill lost slots without ever exceeding the original target size."""
        while not self._closed:
            try:
                await asyncio.sleep(self._self_heal_interval_s)
            except asyncio.CancelledError:
                return
            if self._closed:
                return

            async with self._cardinality_lock:
                missing = self._target_size - self._alive_count
            if missing <= 0:
                continue

            try:
                fresh = await self._factory()
            except Exception:
                self._self_heal_attempts += 1
                now = time.time()
                if now - self._last_self_heal_log > 60.0:
                    log.warning(
                        "engine self-heal: factory failing; alive=%d target=%d attempts=%d",
                        self._alive_count,
                        self._target_size,
                        self._self_heal_attempts,
                    )
                    self._last_self_heal_log = now
                self._self_heal_interval_s = min(
                    60.0,
                    self._self_heal_interval_s * 1.5,
                )
                continue

            if not await self._accept_fresh(fresh):
                continue

            self._self_heal_attempts = 0
            self._self_heal_interval_s = 5.0
            log.info(
                "engine self-heal: refilled slot; alive=%d target=%d",
                self._alive_count,
                self._target_size,
            )

    async def _replace_and_retry(
        self,
        dead: object,
        fn: Callable[[object], Awaitable[T]],
        first_exc: BaseException,
    ) -> T:
        """Replace a dead handler and retry the same operation once on the fresh one.

        Retrying on the newly spawned handler, instead of returning it to the queue first,
        is important when every queued TCP handler went stale at the same time. In that
        state a caller-level single retry can simply pick a second dead handler and still
        surface ``TCPTransport closed=True`` even though replacement is working.
        """
        async with self._cardinality_lock:
            self._alive_count = max(0, self._alive_count - 1)
        await self._discard(dead)

        try:
            fresh = await self._factory()
        except Exception:
            log.exception(
                "engine respawn failed; alive=%d target=%d",
                self._alive_count,
                self._target_size,
            )
            self._start_self_heal()
            raise first_exc

        try:
            result = await fn(fresh)
        except _TRANSPORT_ERRORS:
            await self._discard(fresh)
            self._start_self_heal()
            raise
        except BaseException:
            await self._accept_fresh(fresh)
            raise

        await self._accept_fresh(fresh)
        return result

    async def run(self, fn: Callable[[object], Awaitable[T]]) -> T:
        usage.count("stockfish")
        try:
            inst = await asyncio.wait_for(
                self._q.get(),
                timeout=self._acquire_timeout,
            )
        except TimeoutError as exc:
            raise PoolBusy(f"no engine free within {self._acquire_timeout}s") from exc

        try:
            result = await fn(inst)
        except _TRANSPORT_ERRORS as exc:
            return await self._replace_and_retry(inst, fn, exc)
        except BaseException:
            self._q.put_nowait(inst)
            raise
        else:
            self._q.put_nowait(inst)
            return result

    async def close(self) -> None:
        self._closed = True
        if self._self_heal_task is not None:
            self._self_heal_task.cancel()
            try:
                await self._self_heal_task
            except asyncio.CancelledError:
                pass
            self._self_heal_task = None
        while not self._q.empty():
            inst = self._q.get_nowait()
            await self._discard(inst)
        async with self._cardinality_lock:
            self._alive_count = 0


EnginePool = _EnginePool


class AnalyzerPool:
    """Drop-in pool for Analyzer."""

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
        show_wdl: bool = False,
        syzygy_path: str | None = None,
        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> AnalyzerPool:
        async def factory() -> object:
            return await Analyzer.create(
                path,
                depth=depth,
                threads=threads,
                hash_mb=hash_mb,
                show_wdl=show_wdl,
                syzygy_path=syzygy_path,
            )

        instances = [await factory() for _ in range(max(1, size))]
        engine_name = getattr(instances[0], "name", "Stockfish") if instances else "Stockfish"
        log.info(
            "analyzer pool ready: %d engines (%s)",
            len(instances),
            engine_name,
        )
        return cls(
            _EnginePool(instances, factory, acquire_timeout),
            name=engine_name,
        )

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int | None = None,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        return await self._pool.run(
            lambda a: a.evaluate(  # type: ignore[attr-defined]
                board,
                depth=depth,
                root_moves=root_moves,
            )
        )

    async def classify_move(
        self,
        board: chess.Board,
        move: chess.Move,
        *,
        depth: int | None = None,
    ) -> MoveAnalysis:
        return await self._pool.run(
            lambda a: a.classify_move(  # type: ignore[attr-defined]
                board,
                move,
                depth=depth,
            )
        )

    async def top_moves(
        self,
        board: chess.Board,
        *,
        n: int = 3,
        depth: int | None = None,
    ) -> list[Eval]:
        return await self._pool.run(
            lambda a: a.top_moves(  # type: ignore[attr-defined]
                board,
                n=n,
                depth=depth,
            )
        )

    async def probe_threat(
        self,
        board_after: chess.Board,
        *,
        depth: int | None = None,
    ) -> Eval | None:
        return await self._pool.run(
            lambda a: a.probe_threat(  # type: ignore[attr-defined]
                board_after,
                depth=depth,
            )
        )

    async def close(self) -> None:
        await self._pool.close()
