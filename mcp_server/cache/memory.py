"""Async L1 in-memory LRU cache.

Thin wrapper around ``OrderedDict`` with an ``asyncio.Lock`` guarding both
mutation and ordered iteration. Suitable as the fast tier in a multi-tier
cache — values store whatever the L2 returns (typically a JSON-decoded
Pydantic model, but the cache is generic over ``T``).
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Generic, TypeVar

T = TypeVar("T")


class AsyncLRUCache(Generic[T]):
    """Thread-safe / async in-memory LRU cache."""

    def __init__(self, maxsize: int = 50_000) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> T | None:
        async with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    async def set(self, key: str, value: T) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()

    async def size(self) -> int:
        async with self._lock:
            return len(self._cache)

    @property
    def maxsize(self) -> int:
        return self._maxsize
