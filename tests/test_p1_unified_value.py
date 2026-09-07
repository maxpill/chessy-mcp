"""P1 regression tests: best_action_obj must keep its semantic type identity.

2026-09-07 audit §3 update: the previous invariant
``best_action_obj.move.uci == played_uci`` was encoding the audit's §6 bug.
With the fix, when policy selects a claim action (``canonical_best_action ==
"claim_draw"``) and the player plays the engine's exact best move (which
happens to coincide with the engine's MultiPV best), the response MUST keep
``best_action_obj.type == "claim_draw"`` — the old code rewrote it to
``play_move`` and silently lied about what the policy selected.

This test now pins the corrected invariant: ``best_action_obj.type`` mirrors
``best_action`` (the policy-selected action), and ``played_action_obj.type``
mirrors ``action_type`` (the caller's stated action).
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
        if root_moves:
            m = root_moves[0]
            return Eval(cp=20000, best_move=m.uci(), pv=[m.uci()], depth=depth)
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
async def test_best_action_obj_type_matches_best_action_string_under_claim_recommendation():
    """2026-09-07 audit §3: best_action_obj.type must equal best_action — not whatever
    action happened to coincide with eval_before.best_move."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _RootVsPostStatePool()  # type: ignore[assignment]

    fen = "7k/8/8/8/8/8/4K2P/R7 w - - 100 51"
    analysis = await server_module.classify_move(fen, "h2h3", depth=8)

    pa = analysis.played_action_obj or {}
    ba = analysis.best_action_obj or {}

    # 1) The caller's action_type round-trips on played_action_obj.
    assert pa.get("type") == analysis.action_type == "play_move", (
        f"played_action_obj.type must mirror action_type; got pa={pa!r}, "
        f"action_type={analysis.action_type!r}"
    )
    # The caller did play h2h3; that UCI must be on played_action_obj.
    pa_uci = pa.get("move", {}).get("uci") if isinstance(pa.get("move"), dict) else pa.get("move")
    assert pa_uci == "h2h3"

    # 2) The policy's recommended best_action must round-trip on best_action_obj.
    # At halfmove=100 with claim_available, policy selects claim_draw; even though
    # the engine's MultiPV top happens to be h2h3 (which the player also played),
    # best_action_obj MUST keep its semantic type.
    assert analysis.best_action == ba.get("type"), (
        f"semantic invariant: best_action={analysis.best_action!r} must equal "
        f"best_action_obj.type={ba.get('type')!r}; "
        f"best_action_obj={ba!r}"
    )

    # 3) Same-type evaluation primitives: when best_action is a claim, the
    # best_action_obj has no "move" field (claim dicts don't carry one). When
    # best_action is play_move, the move field tracks eval_before.best_move.
    if analysis.best_action in ("claim_draw", "claim_draw_with_intended_move"):
        assert "move" not in ba, f"claim dict must not carry a 'move' field; got {ba!r}"
    else:
        ba_uci = (
            ba.get("move", {}).get("uci") if isinstance(ba.get("move"), dict) else ba.get("move")
        )
        assert ba_uci == "h2h3"
