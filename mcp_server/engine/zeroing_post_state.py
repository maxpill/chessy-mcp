"""Zeroing-post-state evaluator (audit U-02 helper).

Extracted from :mod:\`mcp_server.engine.eval_pipeline\`. The original
inline block (was inside :func:\`_evaluate_game_position_cached\`) was
lifting the engine's best zeroing move, re-evaluating the post-state,
and reporting a winning cp/mate when draw pollution was suppressing
the root score. Lifts that pure decision into a focused helper.

Returns a small namespace object with ``cp`` / ``mate`` attributes set
to ``None`` when the override does not apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess

if TYPE_CHECKING:
    from core.engines.pool import AnalyzerPool

    from mcp_server.tcp_analyzer import TCPAnalyzerPool


@dataclass
class ZeroingPostState:
    cp: int | None
    mate: int | None


_NOOP_RESULT = ZeroingPostState(cp=None, mate=None)


async def evaluate_zeroing_post_state(
    b: chess.Board,
    best_move_uci: str,
    depth: int,
    pool: "AnalyzerPool | TCPAnalyzerPool",
) -> ZeroingPostState:
    """If the engine's best move is zeroing and ``b.halfmove_clock >= 100``
    and the position isn't already terminal, re-eval the post-state from the
    mover's perspective and return the winning cp/mate (if any). Otherwise
    return a no-op result.

    Audit U-02 (2026-09-01).
    """
    try:
        bm_obj = chess.Move.from_uci(best_move_uci.lower())
        if bm_obj not in b.legal_moves:
            return _NOOP_RESULT
        is_zeroing = b.is_capture(bm_obj) or (b.piece_type_at(bm_obj.from_square) == chess.PAWN)
        if not is_zeroing:
            return _NOOP_RESULT
        b_after = b.copy(stack=True)
        b_after.push(bm_obj)
        if b_after.is_game_over(claim_draw=False):
            return _NOOP_RESULT
        try:
            post_ev = await pool.evaluate(b_after, depth=depth)
        except Exception:
            return _NOOP_RESULT
        mover_sign = 1 if b.turn == chess.WHITE else -1
        if post_ev.mate is not None:
            mover_mate = mover_sign * post_ev.mate
            if mover_mate > 0:
                return ZeroingPostState(cp=None, mate=mover_mate)
        elif post_ev.cp is not None:
            mover_cp = mover_sign * post_ev.cp
            if mover_cp > 0:
                return ZeroingPostState(cp=mover_cp, mate=None)
    except Exception:
        return _NOOP_RESULT
    return _NOOP_RESULT
