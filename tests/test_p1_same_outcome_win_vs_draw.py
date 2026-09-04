"""P1 regression tests: same_outcome must reflect actual outcome, not is_best_action.

Bug doc §5.1 — `same_outcome = is_best_action` is wrong. When best_action
recommends claim_draw and played_action is a4 (a winning move), the outcomes
are DRAW vs WIN. They are NOT the same. The current code returns
same_outcome=True which is a semantic violation.
"""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module


class _ClaimVsWinPool:
    """Engine recommends claim_draw at root but played move a4 wins."""

    name = "ClaimVsWinPool"
    engine_version = "ClaimVsWinPool"

    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, board, *, depth=14, root_moves=None):
        self.calls += 1
        if board.is_checkmate():
            return Eval(cp=0, mate=0, best_move="", pv=[], depth=depth)
        return Eval(cp=26, best_move="a2a4", pv=["a2a4"], depth=depth)

    async def top_moves(self, board, n=3, depth=14):
        return [
            Eval(cp=26, best_move="a2a4", pv=["a2a4"], depth=depth),
            Eval(cp=20, best_move="g1g2", pv=["g1g2"], depth=depth),
        ][:n]

    async def classify_move(self, board, move, depth=14):
        return MoveAnalysis(
            played=move.uci(),
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(cp=26, best_move="a2a4"),
            eval_after=Eval(cp=None, mate=5, best_move=move.uci()),
            best_move_san=board.san(move) if move in board.legal_moves else "",
        )

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_play_winning_move_vs_claim_is_not_same_outcome():
    """§5.1 reproducer: played a4 (WIN), best claim (DRAW) → same_outcome must be False."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _ClaimVsWinPool()  # type: ignore[assignment]

    fen = "7k/8/8/8/8/8/P3K3/6R1 w - - 100 51"
    analysis = await server_module.classify_move(fen, "a2a4", depth=8)

    # Sanity: a4 wins, eval_after has mate=5
    assert analysis.eval_after.mate is not None and analysis.eval_after.mate > 0

    # The bug: same_outcome is currently is_best_action (True) but outcomes differ
    # WIN (a4) vs DRAW (claim) → same_outcome must be False after fix
    assert analysis.same_outcome is False, (
        f"played WIN vs best DRAW must NOT be same_outcome; got {analysis.same_outcome}"
    )

    # action_equivalent must also be False — WIN is never equivalent to DRAW
    assert analysis.action_equivalent is False, (
        f"WIN vs DRAW must NOT be action_equivalent; got {analysis.action_equivalent}"
    )


@pytest.mark.asyncio
async def test_two_winning_moves_are_same_outcome():
    """Sanity check: two winning actions must have same_outcome=True."""
    fen = "7k/P7/8/8/8/8/4K3/8 w - - 100 51"

    class _AllWinPool:
        name = "AllWinPool"

        async def evaluate(self, board, *, depth=14, root_moves=None):
            if root_moves:
                m = root_moves[0]
                b2 = board.copy(stack=True)
                b2.push(m)
                if b2.is_checkmate():
                    return Eval(cp=None, mate=1, best_move=m.uci(), pv=[m.uci()], depth=depth)
            legal = list(board.legal_moves)
            best = legal[0].uci() if legal else None
            return Eval(cp=20000, best_move=best, pv=[best] if best else [], depth=depth)

        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=20000, best_move=move.uci()),
                eval_after=Eval(cp=None, mate=1, best_move=move.uci()),
                best_move_san=board.san(move),
            )

        async def close(self):
            pass

    await server_module._cache.clear()
    server_module._analyzer_pool = _AllWinPool()  # type: ignore[assignment]

    analysis = await server_module.classify_move(fen, "a7a8q", depth=8)
    assert analysis.eval_after.mate is not None and analysis.eval_after.mate > 0
    assert analysis.same_outcome is True, "two WIN outcomes must be same_outcome"
