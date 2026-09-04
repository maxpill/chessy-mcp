"""Engine-call retry helper (bug fix for §8).

Wraps an idempotent engine call with a single-shot retry on
``ConnectionError`` / ``OSError`` / ``EngineTerminatedError``. The pool
already replaces the dead instance internally, so the retry gets a
fresh handler.

Bounded: never more than one retry, never exponential backoff beyond
50ms. The endpoints are idempotent (engine search on a board state),
so retrying is always safe.

Circuit breaker: after 3 consecutive failures within 30s, refuse new
requests for 5s to avoid a retry storm when the engine process is
broken.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from chess.engine import EngineError, EngineTerminatedError


log = logging.getLogger("chessy_mcp.engine.retry")


_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    OSError,
    EngineTerminatedError,
    EngineError,
)


@dataclass
class _BreakerState:
    failures: int = 0
    opened_at: float = 0.0


_breaker = _BreakerState()
_BREAKER_THRESHOLD = 3
_BREAKER_WINDOW_S = 30.0
_BREAKER_COOLDOWN_S = 5.0


class EngineUnavailable(Exception):
    """Raised when the circuit breaker is open."""


async def with_engine_retry(fn, *, max_retries: int = 1) -> object:
    """Run an idempotent engine call with at most ``max_retries`` retries.

    Raises ``EngineUnavailable`` if the breaker is open. Re-raises the
    transport exception if all retries fail.
    """
    _check_breaker()
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            result = await fn()
            _record_success()
            return result
        except _TRANSPORT_ERRORS as exc:
            last_exc = exc
            _record_failure()
            if attempt < max_retries:
                log.warning(
                    "engine transport error on attempt %d/%d, retrying: %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                await asyncio.sleep(0.05)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("with_engine_retry: no attempts ran")


def _check_breaker() -> None:
    now = time.time()
    if _breaker.opened_at == 0.0:
        return
    # Only refuse requests when the breaker is *tripped* (failures >= threshold).
    # A few transient failures below the threshold must still be retried.
    if _breaker.failures < _BREAKER_THRESHOLD:
        return
    if now - _breaker.opened_at < _BREAKER_COOLDOWN_S:
        raise EngineUnavailable(
            f"engine circuit breaker open "
            f"({_breaker.failures} failures in last {_BREAKER_WINDOW_S:.0f}s)"
        )
    # Cool-down elapsed — half-open. Allow one trial.
    _breaker.opened_at = 0.0
    _breaker.failures = 0


def _record_success() -> None:
    _breaker.failures = 0
    _breaker.opened_at = 0.0


def _record_failure() -> None:
    now = time.time()
    if _breaker.opened_at == 0.0 or (now - _breaker.opened_at) > _BREAKER_WINDOW_S:
        _breaker.opened_at = now
        _breaker.failures = 1
    else:
        _breaker.failures += 1
    if _breaker.failures >= _BREAKER_THRESHOLD:
        log.error("engine circuit breaker tripped after %d failures", _breaker.failures)


def reset_breaker() -> None:
    """Test-only: reset the global circuit-breaker state."""
    global _breaker
    _breaker = _BreakerState()
