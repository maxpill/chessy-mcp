"""Regression tests for the 2026-08-31 extreme audit B-04/B-05 fixes.

The original audit reported that at halfmove=100 the action policy was
preferring `claim_draw` over a winning zeroing move because the root
MultiPV cp was "draw-polluted" (the engine sees the draw on the table and
reports a tiny cp even when the winning move is right there). B-05 was
the surface contract: the candidate's reported cp must NOT be silently
rewritten to a re-evaluated value — back-compat callers depend on the
multipv ranking.
"""

from __future__ import annotations

from typing import Any

import chess
import pytest

from core.engines.types import Eval
from mcp_server import server as server_module


class _PositionAwarePool:
    """Returns cp<=0 for multipv root (draw-polluted), cp=20000 for any
    re-evaluation of a post-state (halfmove==0, forced win)."""

    def __init__(self, *, multipv_cp: int = 0, post_cp: int = 20000) -> None:
        self.multipv_cp = multipv_cp
        self.post_cp = post_cp
        self.name = "PositionAwarePool"
        self.engine_version = "PositionAwarePool"

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int = 14,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        legal = list(root_moves) if root_moves else list(board.legal_moves)
        best = legal[0].uci() if legal else None
        cp = self.post_cp if board.halfmove_clock == 0 else self.multipv_cp
        return Eval(cp=cp, mate=None, best_move=best, pv=[best], depth=depth)

    async def top_moves(self, board: chess.Board, *, n: int = 3, depth: int = 14) -> list[Eval]:
        return [
            Eval(
                cp=self.multipv_cp,
                mate=None,
                best_move=move.uci(),
                pv=[move.uci()],
                depth=depth,
            )
            for move in list(board.legal_moves)[:n]
        ]

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _clean_global_state() -> Any:
    await server_module._cache.clear()
    await server_module.close_analyzer_pool()
    yield
    await server_module._cache.clear()
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_b04_top_moves_prefers_play_move_when_zeroing_wins() -> None:
    """B-04: at halfmove=100 with a winning zeroing capture, top_moves must
    recommend play_move, not claim_draw, even though the multipv cp looks
    draw-polluted."""
    pool = _PositionAwarePool(multipv_cp=0, post_cp=20000)
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    fen = "7k/n7/8/8/8/8/R7/4K3 w - - 100 51"
    res = await server_module.top_moves(fen, n=3, depth=2)
    assert res.recommended_action == "play_move", (
        f"Audit B-04: claim_draw must not be recommended when a winning zeroing "
        f"move exists, got {res.recommended_action}"
    )


@pytest.mark.asyncio
async def test_b05_post_state_cp_does_not_overwrite_multipv_cp() -> None:
    """B-05: candidate's reported cp stays at multipv value (back-compat);
    action policy reflects the post-state re-evaluation."""
    pool = _PositionAwarePool(multipv_cp=0, post_cp=20000)
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    fen = "7k/n7/8/8/8/8/R7/4K3 w - - 100 51"
    res = await server_module.top_moves(fen, n=3, depth=2)
    cand = res.result[0]
    assert cand.cp == 0  # back-compat: cp stays at multipv value
    assert cand.post_state_cp == 20000  # post-state tracked separately
    assert res.recommended_action == "play_move"


@pytest.mark.asyncio
async def test_b05_post_state_persists_to_cache_and_survives_warm_path() -> None:
    """B-05: post_state_cp/mate must persist on cache entries so a subsequent
    cache hit still surfaces the winning zeroing move to the action policy."""
    pool = _PositionAwarePool(multipv_cp=0, post_cp=20000)
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    fen = "7k/n7/8/8/8/8/R7/4K3 w - - 100 51"
    first = await server_module.top_moves(fen, n=3, depth=2)
    assert first.recommended_action == "play_move"
    second = await server_module.top_moves(fen, n=3, depth=2)
    assert second.recommended_action == "play_move"
    # Second call goes through the cache; the post_state_cp must survive.
    assert second.result[0].post_state_cp == 20000
