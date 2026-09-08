"""2026-09-07 Round 3 F-04: every legal zeroing move must be considered.

The previous implementation sliced ``zeroing_moves[:MAX_ZEROING_MOVES_TO_EVAL]``
where ``MAX_ZEROING_MOVES_TO_EVAL = 16``. A legal chess position can
have more than 16 zeroing moves (each pawn capture, each pawn push, and
each promotion choice counts separately) — so the algorithm could
silently drop the winning zeroing move and report a draw.

Fix: replace the slice cap with an ``asyncio.Semaphore`` that bounds
**concurrency** (so the pool stays responsive) without skipping any
legal zeroing move.

2026-09-08 audit fix: replace the 4×3 fixture (12 zeroing moves,
weakens the regression contract) with a fixture that GUARANTEES
``len(zeroing_moves) > 16`` and uses an index-16+ winning move.
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
        last_move = board.move_stack[-1] if board.move_stack else None
        uci = last_move.uci() if last_move else ""
        self.calls.append((uci, depth))
        if uci == self.winning_uci:
            return Eval(cp=None, mate=3, best_move=uci, pv=[uci], depth=depth)
        return Eval(cp=0, mate=None, best_move=uci, pv=[uci], depth=depth)

    async def close(self) -> None:
        pass


def _position_with_many_zeroing_moves() -> tuple[chess.Board, list[chess.Move]]:
    """Construct a legal position with strictly more than 16 zeroing moves.

    The previous fixture (4 White pawns on rank 4 vs 4 Black pawns on
    rank 5) produced only 12 zeroing moves — fewer than the 16 the
    regression test claimed to protect against. The audit explicitly
    flagged this gap.

    The replacement uses 8 White pawns on the 7th rank
    (``8/PPPPPPPP/8/8/8/8/8/4K2k w - - 100 1``). Each pawn has four
    promotion options (Q/R/B/N) and all promotions target empty rank-8
    squares, yielding **32 zeroing moves** — well above the 16-move
    truncation threshold.

    The 50-move claim is also valid for this position (halfmove=100).
    """
    board = chess.Board("8/PPPPPPPP/8/8/8/8/8/4K2k w - - 100 1")
    assert board.is_valid(), "fixture must be a legal chess position"
    zeroing = [m for m in board.legal_moves if zeroing_post_state_all._is_zeroing_move(board, m)]
    return board, zeroing


@pytest.mark.asyncio
async def test_every_legal_zeroing_move_evaluated_no_silent_truncation():
    """F-04: ALL legal zeroing moves are evaluated, not just the first 16.

    2026-09-08 audit: the fixture is now guaranteed to have
    ``len(zeroing_moves) > 16`` and the winning move is taken from
    index 16+ so the test exercises the "all moves considered" path
    without weakening the contract.
    """
    board, zeroing_moves = _position_with_many_zeroing_moves()

    # Pin the fixture's contract: this test MUST use a position with
    # more than 16 zeroing moves, otherwise it cannot detect a
    # 16-move truncation regression.
    assert len(zeroing_moves) > 16, (
        f"fixture must have strictly more than 16 zeroing moves to exercise "
        f"the regression contract; got {len(zeroing_moves)}"
    )

    # Pick the winning move at index 16 (or later). With the pre-fix
    # code that sliced ``zeroing_moves[:16]``, the winning move would
    # be silently dropped.
    winning_uci = zeroing_moves[16].uci()
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
    # The late winning move must be picked — the one that would have been
    # sliced off by the pre-fix 16-move cap.
    assert result.winning_uci == winning_uci, (
        f"winning_uci must surface the late zeroing move at index 16; got "
        f"{result.winning_uci!r}, expected {winning_uci!r}"
    )
    assert result.mate == 3
