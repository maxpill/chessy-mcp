"""SingleFlight request coalescer.

Ensures that for a given key, only one in-flight async operation is executed
at a time. All concurrent callers for the same key await the same task
result.

Cancellation safety: every waiter awaits the shared future through
``asyncio.shield(...)`` so one disconnected caller cannot cancel shared work.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


def _consume_unobserved_exception[T](future: asyncio.Future[T]) -> None:
    """Mark producer-only failures as observed without changing waiter semantics."""
    if future.cancelled():
        return
    try:
        future.exception()
    except BaseException:
        pass


class SingleFlight[T]:
    """Request coalescer (SingleFlight pattern)."""

    def __init__(self) -> None:
        self._in_flight: dict[str, asyncio.Future[T]] = {}
        self._lock = asyncio.Lock()

    async def do(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            if key in self._in_flight:
                future = self._in_flight[key]
                do_wait = True
            else:
                future = asyncio.get_running_loop().create_future()
                future.add_done_callback(_consume_unobserved_exception)
                self._in_flight[key] = future
                do_wait = False

        if do_wait:
            return await asyncio.shield(future)

        try:
            result = await fn()
            if not future.done():
                future.set_result(result)
            return result
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            raise
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._in_flight.pop(key, None)
