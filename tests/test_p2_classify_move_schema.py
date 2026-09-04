"""P2 regression tests: claim_draw schema must not require a placeholder move.

Bug doc §10 — current classify_move requires `move: str` and rejects
non-empty move for claim_draw (strict) or warns (lenient). The fix is
to use a discriminated union where `move` is absent for plain claim_draw.
"""

from __future__ import annotations

import pytest

from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module


class _SimplePool:
    name = "SimplePool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        legal = list(board.legal_moves)
        best = legal[0].uci() if legal else None
        return Eval(cp=10, best_move=best, pv=[best] if best else [], depth=depth)

    async def classify_move(self, board, move, depth=14):
        return MoveAnalysis(
            played=move.uci() if move else "",
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(cp=10),
            eval_after=Eval(cp=10),
        )

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    from mcp_server.engine.retry import reset_breaker

    reset_breaker()
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_classify_move_claim_draw_legal_no_move_required():
    """§32 + §10: claim_draw must be invokable WITHOUT supplying a move arg."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _SimplePool()  # type: ignore[assignment]

    fen = "7k/8/8/8/8/8/P3K3/6R1 w - - 100 51"

    # Should not raise. Current API requires move=""; the new schema should
    # accept no move argument for claim_draw.
    try:
        res = await server_module.classify_move(fen, move=None, depth=8, action_type="claim_draw")
    except Exception as exc:
        # Old API throws on missing move for claim_draw. After fix this should not.
        if "MISSING_MOVE" in str(exc) or "move" in str(exc).lower():
            pytest.fail(f"claim_draw must not require a move argument; got {exc!r}")
        raise

    # The recommendation for a winning pawn-push position must be play_move,
    # but the caller asked to classify claim_draw → action_type must round-trip
    assert res.action_type == "claim_draw"
