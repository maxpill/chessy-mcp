"""All-zeroing-post-state evaluator (bug fix for §4.1, §4.3, §4.4).

The original `evaluate_zeroing_post_state` only re-evaluated the engine's
MultiPV best move. At halfmove >= 100, Stockfish's root cp can be polluted
by the draw being on the table — a quiet non-zeroing move may report
cp=+26 even though a pawn push would lead to forced mate, because the
pawn push isn't in MultiPV top-N.

This module generalizes: it iterates every legal zeroing move
(captures and pawn pushes — the moves that reset the halfmove clock
or land in a clearly winning post-state), evaluates the post-state of
each at a capped depth, and returns the best winning cp/mate.

For positions with no winning zeroing moves it returns the noop result.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess

if TYPE_CHECKING:
    from core.engines.pool import AnalyzerPool

    from mcp_server.tcp_analyzer import TCPAnalyzerPool


log = logging.getLogger("chessy_mcp.engine.zeroing_post_state_all")


@dataclass
class ZeroingOverrideResult:
    """Best winning post-state among all legal zeroing moves."""

    cp: int | None
    mate: int | None
    winning_uci: str | None = None


_NOOP_RESULT = ZeroingOverrideResult(cp=None, mate=None, winning_uci=None)

# Cap post-state depth so this doesn't dominate latency.
MAX_POST_STATE_DEPTH = 6
# 2026-09-07 Round 3 F-04: concurrency cap, NOT a candidate cap. A
# position can have more than 16 legal zeroing moves (each pawn capture
# and pawn push counts, plus promotions). Limiting to the first 16
# silently truncated correctness — the winning zeroing move could be
# the 17th. Use a semaphore to bound simultaneous engine work without
# skipping any legal zeroing move.
MAX_ZEROING_CONCURRENCY = 4


def _is_zeroing_move(board: chess.Board, move: chess.Move) -> bool:
    """A zeroing move: capture or pawn push. Promotion is also a pawn push."""
    if board.is_capture(move):
        return True
    return board.piece_type_at(move.from_square) == chess.PAWN


def _post_state_winning(
    post: chess.Board,
    *,
    mover_color: chess.Color,
    mate: int | None,
    cp: int | None,
) -> tuple[int | None, int | None]:
    """Return (cp, mate) for the mover, or (None, None) if not winning."""
    mover_sign = 1 if mover_color == chess.WHITE else -1
    if mate is not None:
        m = mover_sign * mate
        if m > 0:
            return None, m
        return None, None
    if cp is not None:
        c = mover_sign * cp
        if c > 0:
            return c, None
    return None, None


async def _eval_one_post_state(
    board_after: chess.Board,
    *,
    mover_color: chess.Color,
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
) -> tuple[int | None, int | None]:
    """Single post-state eval, returning mover-POV (cp, mate) or (None, None)."""
    try:
        ev = await pool.evaluate(board_after, depth=depth)
    except Exception:
        return None, None
    return _post_state_winning(board_after, mover_color=mover_color, mate=ev.mate, cp=ev.cp)


async def evaluate_all_zeroing_post_states(
    b: chess.Board,
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
) -> ZeroingOverrideResult:
    """Re-evaluate the post-state of every legal zeroing move and return
    the best winning cp/mate.

    Returns noop when:
      - the position is already terminal,
      - halfmove < 100 (no draw pollution to worry about),
      - the position has no legal zeroing moves,
      - all post-state evaluations fail or yield no winning signal.
    """
    if b.is_game_over() or b.halfmove_clock < 100:
        return _NOOP_RESULT

    zeroing_moves: list[chess.Move] = []
    for m in b.legal_moves:
        if _is_zeroing_move(b, m):
            zeroing_moves.append(m)
    if not zeroing_moves:
        return _NOOP_RESULT

    eval_depth = min(depth, MAX_POST_STATE_DEPTH)
    mover_color = b.turn

    # 2026-09-07 Round 3 F-04: every legal zeroing move is considered.
    # Concurrency is bounded by a semaphore so the pool stays responsive
    # even when a position has many captures (e.g. late middlegame with
    # both sides able to capture).
    sem = asyncio.Semaphore(MAX_ZEROING_CONCURRENCY)

    async def _bounded_eval(move: chess.Move) -> tuple[str, tuple[int | None, int | None]]:
        async with sem:
            post = b.copy(stack=True)
            post.push(move)
            if post.is_game_over(claim_draw=False):
                if post.is_checkmate():
                    return move.uci(), (None, 1)
                return move.uci(), (0, None)
            res = await _eval_one_post_state(
                post, mover_color=mover_color, depth=eval_depth, pool=pool
            )
            return move.uci(), res

    results = await asyncio.gather(
        *(_bounded_eval(m) for m in zeroing_moves),
        return_exceptions=True,
    )

    best_cp: int | None = None
    best_mate: int | None = None
    best_uci: str | None = None

    for entry in results:
        if isinstance(entry, BaseException):
            continue
        if not isinstance(entry, tuple) or len(entry) != 2:
            continue
        uci, res = entry
        if not isinstance(res, tuple):
            continue
        cp, mate = res
        if cp is None and mate is None:
            continue

        # Mate-first ranking: smaller mate distance is better
        if mate is not None:
            if mate <= 0:
                continue
            if best_mate is None or mate < best_mate:
                best_mate = mate
                best_cp = None
                best_uci = uci
            continue

        if cp is None or cp <= 0:
            continue
        if best_mate is not None:
            continue
        if best_cp is None or cp > best_cp:
            best_cp = cp
            best_uci = uci

    return ZeroingOverrideResult(cp=best_cp, mate=best_mate, winning_uci=best_uci)
