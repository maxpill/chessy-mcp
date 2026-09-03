"""L1 in-memory + L2 SQLite multi-tier cache for typed models.

The ``MultiTierCache`` composes an ``AsyncLRUCache[Any]`` with a
``SQLiteDiskCache``. Reads check L1, fall back to L2 (and rehydrate L1),
returns deserialized Pydantic objects. Writes go to L1 unconditionally and to
L2 only when L1 was cold before the write — under bursty ``SingleFlight``
coalescing or ``analyze_game`` fan-out, the first writer persists to L2 and
every subsequent redundant ``set`` is L1-only. That saves the tmpfs WAL write
+ ``asyncio.to_thread`` spawn per redundant call.
"""

from __future__ import annotations

import json
from typing import Any, cast

from mcp_server.cache.disk import SQLiteDiskCache
from mcp_server.cache.memory import AsyncLRUCache
from mcp_server.models import MCPEval, MCPMoveAnalysis

__all__ = ["MultiTierCache"]


class MultiTierCache:
    """Two-tier cache: L1 in-memory LRU + L2 SQLite WAL disk cache."""

    def __init__(self, l1_size: int = 50_000, db_path: str | None = None) -> None:
        self._l1: AsyncLRUCache[Any] = AsyncLRUCache[Any](maxsize=l1_size)
        self._l2: SQLiteDiskCache = SQLiteDiskCache(db_path)

    async def get_eval(self, key: str) -> MCPEval | None:
        v = await self._l1.get(key)
        if isinstance(v, MCPEval):
            return v
        raw = await self._l2.get(key)
        if raw is not None:
            try:
                val = MCPEval.model_validate_json(raw)
                await self._l1.set(key, val)
                return val
            except Exception:
                pass
        return None

    async def set_eval(self, key: str, val: MCPEval) -> None:
        was_cold = await self._l1.get(key) is None
        await self._l1.set(key, val)
        if was_cold:
            await self._l2.set(key, val.model_dump_json())

    async def get_top_moves(self, key: str) -> list[MCPEval] | None:
        v = await self._l1.get(key)
        if isinstance(v, list):
            return cast(list[MCPEval], v)
        raw = await self._l2.get(key)
        if raw is not None:
            try:
                data = json.loads(raw)
                vals = [MCPEval.model_validate(x) for x in data]
                await self._l1.set(key, vals)
                return vals
            except Exception:
                pass
        return None

    async def set_top_moves(self, key: str, vals: list[MCPEval]) -> None:
        was_cold = await self._l1.get(key) is None
        await self._l1.set(key, vals)
        if was_cold:
            raw = json.dumps([x.model_dump() for x in vals])
            await self._l2.set(key, raw)

    async def get_classify(self, key: str) -> MCPMoveAnalysis | None:
        v = await self._l1.get(key)
        if isinstance(v, MCPMoveAnalysis):
            return v
        raw = await self._l2.get(key)
        if raw is not None:
            try:
                val = MCPMoveAnalysis.model_validate_json(raw)
                await self._l1.set(key, val)
                return val
            except Exception:
                pass
        return None

    async def set_classify(self, key: str, val: MCPMoveAnalysis) -> None:
        was_cold = await self._l1.get(key) is None
        await self._l1.set(key, val)
        if was_cold:
            await self._l2.set(key, val.model_dump_json())

    async def clear(self) -> None:
        await self._l1.clear()
        await self._l2.clear()
