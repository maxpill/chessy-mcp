"""P2 regression tests: top_moves.legal_rule_actions must be populated.

Bug doc §9 — top_moves returns legal_rule_actions=[] even when claim_draw
is legal. evaluate_position returns it correctly. The root cause is that
build_top_moves_response only sets legal_actions=, not legal_rule_actions=.
"""

from __future__ import annotations

import pytest

from core.engines.types import Eval
from mcp_server import server as server_module


class _NormalPool:
    name = "NormalPool"
    engine_version = "NormalPool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        if root_moves:
            m = root_moves[0]
            b2 = board.copy(stack=True)
            b2.push(m)
            if b2.is_checkmate():
                return Eval(cp=None, mate=1, best_move=m.uci(), pv=[m.uci()], depth=depth)
        legal = list(board.legal_moves)
        best = legal[0].uci() if legal else None
        return Eval(cp=26, best_move=best, pv=[best] if best else [], depth=depth)

    async def top_moves(self, board, n=3, depth=14):
        legal = list(board.legal_moves)
        return [Eval(cp=26, best_move=m.uci(), pv=[m.uci()], depth=depth) for m in legal[:n]]

    async def classify_move(self, board, move, depth=14):
        from core.engines.types import MoveAnalysis, MoveClass

        return MoveAnalysis(
            played=move.uci(),
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(cp=26, best_move=move.uci()),
            eval_after=Eval(cp=26),
        )

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_top_moves_legal_rule_actions_populated_when_claim_legal():
    """§9.1: when claim_draw is legal, legal_rule_actions must not be empty."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NormalPool()  # type: ignore[assignment]

    fen = "7k/8/8/8/8/8/P3K3/6R1 w - - 100 51"
    res = await server_module.top_moves(fen, n=1, depth=8)

    # claim_draw is legal at halfmove=100
    assert res.can_claim_draw is True
    # legal_rule_actions must contain the claim entry
    assert isinstance(res.legal_rule_actions, list)
    assert len(res.legal_rule_actions) >= 1, (
        f"expected legal_rule_actions to be populated when claim is legal; got {res.legal_rule_actions!r}"
    )
    types = {a.get("type") for a in res.legal_rule_actions}
    assert "claim_draw" in types or "claim_draw_with_intended_move" in types, (
        f"expected a claim action in legal_rule_actions; got types={types}"
    )
