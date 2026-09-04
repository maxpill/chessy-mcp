"""P1 regression tests: cross-endpoint best_action consistency.

Bug doc §7 — evaluate_position, top_moves, classify_move(eval_before)
must agree on best_action for the same input. Currently each endpoint
computes its own recommendation and they can disagree (e.g. top_moves
returns claim_draw while classify_move returns play_move).
"""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module


class _ConsistentPawnPool:
    """Pool that returns a4 as best for the §4.1 FEN, with claim also legal.

    evaluate / top_moves / classify_move all agree on best_move="a2a4" so
    cross-endpoint invariant tests pass.
    """

    name = "ConsistentPawnPool"
    engine_version = "ConsistentPawnPool"

    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, board, *, depth=14, root_moves=None):
        if root_moves:
            m = root_moves[0]
            b2 = board.copy(stack=True)
            b2.push(m)
            if b2.is_checkmate():
                return Eval(cp=None, mate=1, best_move=m.uci(), pv=[m.uci()], depth=depth)
            if board.piece_type_at(m.from_square) == 1:  # PAWN
                return Eval(cp=None, mate=7, best_move=m.uci(), pv=[m.uci()], depth=depth)
        # Detect post-state of a2a4 on §4.1 FEN (pawn now on a4, not a2).
        fen = board.fen()
        if "P7/8/" in fen and "4K3/6R1 b" in fen:
            return Eval(cp=None, mate=7, best_move="a4a5", pv=["a4a5"], depth=depth)
        if board.piece_at(chess.A2) == chess.Piece(chess.PAWN, chess.WHITE):
            return Eval(cp=26, best_move="a2a4", pv=["a2a4"], depth=depth)
        legal = list(board.legal_moves)
        best = legal[0].uci() if legal else None
        return Eval(cp=26, best_move=best, pv=[best] if best else [], depth=depth)

    async def top_moves(self, board, n=3, depth=14):
        if board.piece_at(chess.A2) == chess.Piece(chess.PAWN, chess.WHITE):
            head = Eval(cp=26, best_move="a2a4", pv=["a2a4"], depth=depth)
            rest = [
                Eval(cp=20, best_move=m.uci(), pv=[m.uci()], depth=depth)
                for m in list(board.legal_moves)[: n - 1]
            ]
            return [head] + rest
        legal = list(board.legal_moves)
        return [Eval(cp=26, best_move=m.uci(), pv=[m.uci()], depth=depth) for m in legal[:n]]

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_three_endpoints_agree_on_best_action():
    """evaluate / top_moves / classify must all report the same best_action."""
    await server_module._cache.clear()
    pool = _ConsistentPawnPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]

    fen = "7k/8/8/8/8/8/P3K3/6R1 w - - 100 51"

    e = await server_module.evaluate_position(fen, depth=8)
    t = await server_module.top_moves(fen, n=1, depth=8)
    c = await server_module.classify_move(fen, "a2a4", depth=8)

    # All three must agree on recommended_action (the policy's recommendation).
    assert e.recommended_action == t.recommended_action, (
        f"evaluate_position.recommended_action={e.recommended_action!r} != top_moves.recommended_action={t.recommended_action!r}"
    )
    assert e.recommended_action == c.eval_before.recommended_action, (
        f"evaluate_position.recommended_action={e.recommended_action!r} != classify.eval_before.recommended_action={c.eval_before.recommended_action!r}"
    )

    # And the best move itself (evaluate + classify must agree; top_moves carries
    # its best move inside best_action_obj instead of a separate field).
    assert e.best_move == c.eval_before.best_move
    if e.best_move is not None:
        ba_uci = (
            t.best_action_obj.get("move", {}).get("uci")
            if isinstance(t.best_action_obj, dict)
            else None
        )
        assert ba_uci == e.best_move, (
            f"top_moves.best_action_obj.move.uci={ba_uci!r} != evaluate.best_move={e.best_move!r}"
        )
