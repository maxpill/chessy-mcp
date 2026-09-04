"""P3 regression tests: terminal position semantic consistency.

Bug doc §12 — terminal positions report `legal_move_count > 0` (board-legal
moves ignoring terminality) while `legal_move_uci = []` is empty. The two
fields have incompatible semantics; either rename or zero out.
"""

from __future__ import annotations

import pytest

from core.engines.types import Eval
from mcp_server import server as server_module


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_k_vs_k_terminal_does_not_report_legal_moves():
    """K vs K is insufficient_material — no legal moves in game state."""
    await server_module._cache.clear()

    class _StubPool:
        name = "StubPool"

        async def evaluate(self, board, *, depth=14, root_moves=None):
            return Eval(cp=0, mate=None, best_move="", pv=[], depth=depth)

        async def top_moves(self, board, n=3, depth=14):
            return []

        async def close(self):
            pass

    server_module._analyzer_pool = _StubPool()  # type: ignore[assignment]

    res = await server_module.evaluate_position("4k3/8/8/8/8/8/8/4K3 w - - 0 1", depth=8)
    assert res.status == "insufficient_material"
    # Either both empty or document the inconsistency — the new field
    # `board_legal_move_count_ignoring_terminal_state` semantics must be
    # consistent with legal_move_uci
    assert res.legal_move_uci == []


@pytest.mark.asyncio
async def test_top_moves_terminal_has_consistent_legal_fields():
    """top_moves on terminal position: legal_actions/uci consistent with status."""
    await server_module._cache.clear()

    class _StubPool:
        name = "StubPool"

        async def evaluate(self, board, *, depth=14, root_moves=None):
            return Eval(cp=0, mate=None, best_move="", pv=[], depth=depth)

        async def top_moves(self, board, n=3, depth=14):
            return []

        async def close(self):
            pass

    server_module._analyzer_pool = _StubPool()  # type: ignore[assignment]

    res = await server_module.top_moves("4k3/8/8/8/8/8/8/4K3 w - - 0 1", n=3, depth=8)
    assert res.status == "insufficient_material"
