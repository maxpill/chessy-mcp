"""P1 regression tests: same move must have one canonical value in classify_move.

Bug doc §6.1 — for FEN 7k/8/8/8/8/8/4K2P/R7 w - - 100 51, move h3:
best_action_obj reports h3 with cp=+26 (root MultiPV), played_action_obj
reports h3 with cp=+20000 (post-position). The same LegalAction has two
different values in one response. This is the documented §6 bug.
"""

from __future__ import annotations

import pytest

from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module


class _RootVsPostStatePool:
    """Engine returns cp=+26 at root but cp=+20000 on post-state eval."""

    name = "RootVsPostStatePool"
    engine_version = "RootVsPostStatePool"

    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, board, *, depth=14, root_moves=None):
        self.calls += 1
        # Post-state eval (called with root_moves): huge cp
        if root_moves:
            m = root_moves[0]
            return Eval(cp=20000, best_move=m.uci(), pv=[m.uci()], depth=depth)
        # Root eval: small cp with h2h3 as best
        return Eval(cp=26, best_move="h2h3", pv=["h2h3"], depth=depth)

    async def classify_move(self, board, move, depth=14):
        return MoveAnalysis(
            played=move.uci(),
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(cp=26, best_move="h2h3"),
            eval_after=Eval(cp=20000, best_move="h2h3"),
            best_move_san=board.san(move),
        )

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_same_move_same_value_in_classify_move():
    """When played == best == h3, both action_obj.value fields must match."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _RootVsPostStatePool()  # type: ignore[assignment]

    fen = "7k/8/8/8/8/8/4K2P/R7 w - - 100 51"
    analysis = await server_module.classify_move(fen, "h2h3", depth=8)

    # Confirm both action_obj point at h3
    pa = analysis.played_action_obj or {}
    ba = analysis.best_action_obj or {}
    pa_uci = pa.get("move", {}).get("uci") if isinstance(pa.get("move"), dict) else pa.get("move")
    ba_uci = ba.get("move", {}).get("uci") if isinstance(ba.get("move"), dict) else ba.get("move")
    assert pa_uci == "h2h3"
    assert ba_uci == "h2h3"

    # The bug: pa.value.cp=20000 vs ba.value.cp=26
    pa_val = pa.get("value") or {}
    ba_val = ba.get("value") or {}
    pa_cp = pa_val.get("cp")
    ba_cp = ba_val.get("cp")

    assert pa_cp == ba_cp, (
        f"played_action_obj.value.cp ({pa_cp}) must equal best_action_obj.value.cp ({ba_cp}) "
        f"when played == best == h2h3"
    )
