"""Transport-independent chess analysis algorithms.

The classical analyzer and the TCP analyzer both implement the same chess
analysis behavior. Only the transport differs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import chess

from .grading import classify_centipawn_loss
from .types import Eval, MoveAnalysis


def pv_to_san(board: chess.Board, pv_uci: list[str], limit: int = 6) -> str | None:
    """Render an engine principal variation as SAN from the given position."""
    replay = board.copy(stack=False)
    sans: list[str] = []
    for uci in pv_uci[:limit]:
        try:
            move = chess.Move.from_uci(uci.lower())
            sans.append(replay.san(move))
            replay.push(move)
        except (ValueError, AssertionError, chess.IllegalMoveError, chess.InvalidMoveError):
            break
    return " ".join(sans) or None


@runtime_checkable
class AnalysisBackend(Protocol):
    """Transport-independent engine operations."""

    name: str

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int | None = None,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval: ...

    async def top_moves(
        self,
        board: chess.Board,
        *,
        n: int = 3,
        depth: int | None = None,
    ) -> list[Eval]: ...

    async def close(self) -> None: ...


def _played_eval_after(
    *,
    move: chess.Move,
    board_after: chess.Board,
    eval_before: Eval,
    eval_after_lookup: Eval | None,
    sign: int,
    actual_depth: int,
) -> tuple[Eval, list[str], int]:
    """Compute post-move evaluation and CP loss for one played move.

    A claimable draw is deliberately not treated as game termination. FIDE draw
    claims are optional procedural actions and are handled by the MCP action
    layer, not by this transport-level classifier.
    """
    move_uci = move.uci()

    is_engine_best = move_uci.lower() == (eval_before.best_move or "").lower()

    if board_after.is_checkmate():
        return Eval(cp=None, mate=0, depth=0), [move_uci], 0

    if board_after.is_game_over(claim_draw=False):
        before_mover = sign * (eval_before.cp if eval_before.cp is not None else 0)
        return Eval(cp=0, depth=0), [move_uci], max(0, before_mover)

    if eval_after_lookup is None:
        raise ValueError(
            "classify_move requires a post-move eval for non-terminal non-best moves"
        )

    eval_after = eval_after_lookup
    played_pv = [move_uci] + (eval_after.pv or [])

    if is_engine_best:
        return eval_after, played_pv, 0

    if eval_before.mate is not None and eval_after.mate is not None:
        if sign * eval_before.mate > 0 and sign * eval_after.mate > 0:
            mate_dist = max(0, eval_after.mate - eval_before.mate)
            loss = 0 if mate_dist == 0 else (50 if mate_dist == 1 else 300)
        elif sign * eval_before.mate > 0 and sign * eval_after.mate <= 0:
            loss = 1000
        elif sign * eval_before.mate < 0 and sign * eval_after.mate < 0:
            mate_dist = max(0, abs(eval_before.mate) - abs(eval_after.mate))
            loss = 0 if mate_dist == 0 else 50
        else:
            loss = 0
    elif eval_before.mate is not None and eval_after.mate is None:
        loss = 1000 if sign * eval_before.mate > 0 else 0
    elif eval_before.mate is None and eval_after.mate is not None:
        loss = 0 if sign * eval_after.mate > 0 else 1000
    else:
        before_mover = sign * (eval_before.cp if eval_before.cp is not None else 0)
        after_mover = sign * (eval_after.cp if eval_after.cp is not None else 0)
        loss = max(0, before_mover - after_mover)

    return eval_after, played_pv, loss


async def classify_move(
    backend: AnalysisBackend,
    board: chess.Board,
    move: chess.Move,
    *,
    depth: int | None = None,
) -> MoveAnalysis:
    """Grade a played move against the engine's best at depth."""
    actual_depth = depth if depth is not None else getattr(backend, "_depth", 12)
    sign = 1 if board.turn == chess.WHITE else -1
    eval_before = await backend.evaluate(board, depth=actual_depth)

    board_after = board.copy(stack=True)
    board_after.push(move)

    if board_after.is_checkmate() or board_after.is_game_over(claim_draw=False):
        eval_after, played_pv, loss = _played_eval_after(
            move=move,
            board_after=board_after,
            eval_before=eval_before,
            eval_after_lookup=None,
            sign=sign,
            actual_depth=actual_depth,
        )
    else:
        post_eval = await backend.evaluate(board_after, depth=actual_depth)
        eval_after, played_pv, loss = _played_eval_after(
            move=move,
            board_after=board_after,
            eval_before=eval_before,
            eval_after_lookup=post_eval,
            sign=sign,
            actual_depth=actual_depth,
        )

    best_san: str | None = None
    if eval_before.best_move:
        try:
            best_san = board.san(chess.Move.from_uci(eval_before.best_move))
        except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError):
            best_san = None

    if played_pv and played_pv[0] == move.uci():
        full_played_pv = played_pv
    else:
        full_played_pv = [move.uci()] + (eval_after.pv or [])

    return MoveAnalysis(
        played=move.uci(),
        move_class=classify_centipawn_loss(loss),
        centipawn_loss=loss,
        eval_before=eval_before,
        eval_after=eval_after,
        best_move_san=best_san,
        best_line_san=pv_to_san(board, eval_before.pv),
        played_line_san=pv_to_san(board, full_played_pv),
    )


async def probe_threat(
    backend: AnalysisBackend,
    board_after: chess.Board,
    *,
    depth: int | None = None,
) -> Eval | None:
    """Null-move probe for the side that just moved."""
    if board_after.is_check() or board_after.is_game_over(claim_draw=False):
        return None
    probe = board_after.copy(stack=False)
    probe.push(chess.Move.null())
    return await backend.evaluate(probe, depth=depth)
