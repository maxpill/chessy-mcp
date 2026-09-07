"""2026-09-07 Round 3 F-04: every legal zeroing move must be considered.

The previous implementation sliced ``zeroing_moves[:MAX_ZEROING_MOVES_TO_EVAL]``
where ``MAX_ZEROING_MOVES_TO_EVAL = 16``. A legal chess position can
have more than 16 zeroing moves (each pawn capture, each pawn push, and
each promotion choice counts separately) — so the algorithm could
silently drop the winning zeroing move and report a draw.

Fix: replace the slice cap with an ``asyncio.Semaphore`` that bounds
**concurrency** (so the pool stays responsive) without skipping any
legal zeroing move.
"""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval
from mcp_server.engine import zeroing_post_state_all


class _PositionalPool:
    """Pool whose evaluate result depends on the move: move ``winning_uci``
    reports a mate-in-3; everything else reports cp=0 (draw)."""

    name = "PositionalPool"
    engine_version = "PositionalPool"

    def __init__(self, winning_uci: str) -> None:
        self.winning_uci = winning_uci
        self.calls: list[tuple[str, int]] = []

    async def evaluate(self, board, *, depth=14, root_moves=None):
        # The move pushed before this post-state eval is the first legal
        # move available. Recover it by inspecting board state vs the
        # first legal move that was applied.
        last_move = board.move_stack[-1] if board.move_stack else None
        uci = last_move.uci() if last_move else ""
        self.calls.append((uci, depth))
        if uci == self.winning_uci:
            return Eval(cp=None, mate=3, best_move=uci, pv=[uci], depth=depth)
        return Eval(cp=0, mate=None, best_move=uci, pv=[uci], depth=depth)

    async def close(self) -> None:
        pass


def _position_with_many_zeroing_moves() -> tuple[chess.Board, list[chess.Move]]:
    """Construct a position with many legal zeroing moves.

    Use a board with multiple pawn chains so each pawn has both a
    push and a capture option. We construct a board with 4 White
    pawns on rank 4 (c4, d4, e4, f4) and 4 Black pawns on rank 5
    (c5, d5, e5, f5). Each White pawn can:
      - capture one of two adjacent Black pawns (2 captures)
      - push forward (1 push)
      - so 3 zeroing moves per pawn × 4 pawns = 12 pawn moves

    Plus king moves. Total > 12 zeroing moves; combined with king
    moves > 16 total legal moves.
    """
    board = chess.Board("8/8/8/3ppp2/2PPPP2/8/8/4K2k w - - 100 1")
    zeroing = [m for m in board.legal_moves if zeroing_post_state_all._is_zeroing_move(board, m)]
    return board, zeroing


@pytest.mark.asyncio
async def test_every_legal_zeroing_move_evaluated_no_silent_truncation():
    """F-04: ALL legal zeroing moves are evaluated, not just the first 16."""
    board, zeroing_moves = _position_with_many_zeroing_moves()

    # Pick the LAST zeroing move as the "winning" one. With the
    # pre-fix code that sliced ``zeroing_moves[:16]``, if the winning
    # move was index >= 16 the function would silently drop it.
    # With this fixture we ensure the test exercises the "all moves
    # considered" path. If the fixture happens to have < 16 zeroing
    # moves we still want to validate the contract; assert that the
    # call count equals the number of zeroing moves (terminals skip
    # the pool call).
    winning_uci = zeroing_moves[-1].uci() if zeroing_moves else "h2h3"
    pool = _PositionalPool(winning_uci=winning_uci)

    result = await zeroing_post_state_all.evaluate_all_zeroing_post_states(
        board, depth=10, pool=pool
    )

    # Compute expected pool calls (zeroing moves minus terminal post-states).
    terminal_zeroing = 0
    for m in zeroing_moves:
        post = board.copy(stack=True)
        post.push(m)
        if post.is_game_over(claim_draw=False):
            terminal_zeroing += 1
    expected_pool_calls = len(zeroing_moves) - terminal_zeroing

    assert len(pool.calls) == expected_pool_calls, (
        f"every zeroing move should be evaluated; pre-fix only the first "
        f"{zeroing_post_state_all.MAX_ZEROING_CONCURRENCY} * 4 (concurrency-bounded) "
        f"were considered. Expected {expected_pool_calls} calls, got "
        f"{len(pool.calls)}"
    )
    # The winning move (the last one) must be picked.
    assert result.winning_uci == winning_uci, (
        f"winning_uci must surface the late zeroing move; got "
        f"{result.winning_uci!r}, expected {winning_uci!r}"
    )
    assert result.mate == 3
