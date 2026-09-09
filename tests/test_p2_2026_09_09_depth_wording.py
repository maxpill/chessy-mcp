"""2026-09-09 master audit F-007: depth error wording must describe actual clamp semantics.

The audit confirmed the error message said "positive integer" while the runtime
behavior accepted zero and negative integers and clamped searched depth to
1..30. This test pins the corrected wording and behavior.
"""

from __future__ import annotations

import pytest

from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module
from mcp_server.engine.retry import reset_breaker
from mcp_server.tools._common import _validate_requested_depth


class _StaticPool:
    name = "StaticPool"
    engine_version = "StaticPool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        legal = list(board.legal_moves)
        best = legal[0].uci() if legal else None
        return Eval(cp=20, best_move=best, pv=[best] if best else [], depth=max(depth, 1))

    async def classify_move(self, board, move, depth=14):
        played = move.uci() if move else ""
        return MoveAnalysis(
            played=played,
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(cp=20, best_move=played),
            eval_after=Eval(cp=20),
            best_move_san=board.san(move) if move in board.legal_moves else "",
        )

    async def top_moves(self, board, *, n=3, depth=14):
        return []

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _cleanup() -> None:
    yield
    reset_breaker()
    await server_module.close_analyzer_pool()


def test_depth_error_wording_mentions_clamp_not_positive() -> None:
    """The validation error must describe clamp semantics, not 'positive integer'."""
    try:
        _validate_requested_depth("not-an-int", tool="evaluate_position")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
    else:
        pytest.fail("Expected a validation error for non-integer depth")

    assert "clamp" in msg.lower() or "1..30" in msg, (
        f"Depth error must mention clamp/1..30; got {msg!r}"
    )
    assert "positive" not in msg.lower(), (
        f"Depth error must NOT say 'positive' (clamp semantics is the real contract); got {msg!r}"
    )


@pytest.mark.asyncio
async def test_zero_and_negative_depth_are_accepted_with_clamp_searched() -> None:
    """Zero/negative depths must be accepted; searched_depth clamped to 1."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _StaticPool()

    res_zero = await server_module.evaluate_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        depth=0,
    )
    res_neg = await server_module.evaluate_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        depth=-999,
    )

    assert res_zero.requested_depth == 0
    assert res_zero.searched_depth == 1
    assert res_neg.requested_depth == -999
    assert res_neg.searched_depth == 1
