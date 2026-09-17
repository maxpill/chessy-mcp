"""Cache + single-flight wrapper for Lichess Explorer responses.

Two tiers, both best-effort:
    L1: in-process ``AsyncLRUCache`` (4096 entries).
    L2: shared ``SQLiteDiskCache`` keyed with a ``lichess_explorer:`` prefix so we
        never collide with Stockfish eval entries in the same SQLite WAL.

TTL = 8 min (matches ``lila.core.opening.OpeningApi.defaultCache``).

SingleFlight coalesces N concurrent identical requests into one HTTP call.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, Final

from mcp_server.cache.disk import SQLiteDiskCache
from mcp_server.cache.memory import AsyncLRUCache
from mcp_server.cache.single_flight import SingleFlight
from mcp_server.config import get_mcp_settings

__all__ = [
    "EXPLORER_CACHE_KEY_PREFIX",
    "EXPLORER_CACHE_L1_SIZE",
    "EXPLORER_CACHE_TTL_S",
    "LichessExplorerCache",
    "explorer_cache_key",
]


EXPLORER_CACHE_KEY_PREFIX: Final[str] = "lichess_explorer:"
EXPLORER_CACHE_TTL_S: Final[int] = 8 * 60
EXPLORER_CACHE_L1_SIZE: Final[int] = 4096


def explorer_cache_key(
    db: str,
    variant: str,
    fen: str,
    play: tuple[str, ...],
    speeds: frozenset[str],
    ratings: frozenset[int],
    modes: frozenset[str],
    since: str,
    until: str,
    player: str,
    color: str,
    top_games: int,
    recent_games: int,
) -> str:
    """Deterministic cache key. Stable across processes; tests assert on it."""
    play_key = ",".join(play) if play else "-"
    speeds_key = ",".join(sorted(speeds)) if speeds else "-"
    ratings_key = ",".join(str(r) for r in sorted(ratings)) if ratings else "-"
    modes_key = ",".join(sorted(modes)) if modes else "-"
    return (
        f"{EXPLORER_CACHE_KEY_PREFIX}v1:"
        f"db={db}:var={variant}:fen={fen}:play={play_key}:"
        f"sp={speeds_key}:rt={ratings_key}:md={modes_key}:"
        f"since={since or '-'}:until={until or '-'}:"
        f"player={player or '-'}:color={color or '-'}:"
        f"top={top_games}:rec={recent_games}"
    )


class _CacheEntry:
    __slots__ = ("expires_at", "value")

    def __init__(self, value: dict[str, Any], expires_at: float) -> None:
        self.value = value
        self.expires_at = expires_at

    def is_alive(self, now: float) -> bool:
        return now < self.expires_at


class LichessExplorerCache:
    """L1 + L2 + single-flight wrapper around ``LichessExplorerClient.query``."""

    def __init__(
        self, l1_size: int = EXPLORER_CACHE_L1_SIZE, ttl_s: int = EXPLORER_CACHE_TTL_S
    ) -> None:
        self._l1: AsyncLRUCache[_CacheEntry] = AsyncLRUCache(maxsize=l1_size)
        self._l2: SQLiteDiskCache = SQLiteDiskCache(get_mcp_settings().cache_db)
        self._ttl_s: int = ttl_s
        self._flight: SingleFlight[dict[str, Any]] = SingleFlight()

    async def get_or_fetch(
        self,
        key: str,
        fetcher: Callable[[], Awaitable[dict[str, Any]]],
    ) -> tuple[dict[str, Any], bool]:
        """Return ``(value, cache_hit)`` for ``key``; call ``fetcher`` on miss.

        ``fetcher`` is awaited through ``SingleFlight`` so concurrent identical
        requests coalesce into one HTTP call.
        """
        import os

        print(
            f"DEBUG GOF: cache_id={id(self)} l2_id={id(self._l2)} "
            f"db={self._l2.db_path} file_size={os.path.getsize(self._l2.db_path)}"
        )
        now = time.time()
        entry = await self._l1.get(key)
        if entry is not None and entry.is_alive(now):
            return entry.value, True

        raw = await self._l2.get(key)
        print(
            f"DEBUG GOF: L2 raw len={len(raw) if raw else 0} raw_repr={raw[:100] if raw else 'None'}"
        )
        if raw is not None:
            try:
                payload = json.loads(raw)
                expires_at = float(payload["_expires_at"])
                if now < expires_at:
                    value = payload["value"]
                    await self._l1.set(key, _CacheEntry(value, expires_at))
                    return value, True
            except Exception:
                pass

        async def _call() -> dict[str, Any]:
            return await fetcher()

        value = await self._flight.do(key, _call)
        expires_at = now + self._ttl_s
        await self._l1.set(key, _CacheEntry(value, expires_at))
        try:
            await self._l2.set(
                key,
                json.dumps({"_expires_at": expires_at, "value": value}, separators=(",", ":")),
            )
        except Exception:
            pass
        return value, False
