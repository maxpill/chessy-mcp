from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mcp_server.cache.multi_tier import MultiTierCache
from mcp_server.models import MCPEval


@pytest.mark.asyncio
async def test_eval_l2_miss_rechecks_l1_after_concurrent_writer(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale L2 miss must not escape after another request filled L1.

    This is the deterministic form of the burst race that could otherwise let a
    late caller reach SingleFlight after the original producer had already
    completed, causing a second engine evaluation for one concurrent request
    wave.
    """
    cache = MultiTierCache(db_path=str(tmp_path / "cache.sqlite3"))
    entered_l2 = asyncio.Event()
    release_l2 = asyncio.Event()

    async def delayed_miss(_key: str) -> None:
        entered_l2.set()
        await release_l2.wait()
        return None

    monkeypatch.setattr(cache._l2, "get", delayed_miss)

    reader = asyncio.create_task(cache.get_eval("same-position"))
    await entered_l2.wait()

    expected = MCPEval(cp=42, best_move="e2e4", depth=3)
    await cache._l1.set("same-position", expected)
    release_l2.set()

    result = await reader
    assert result is expected
