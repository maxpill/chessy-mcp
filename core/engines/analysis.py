"""Transport-independent chess analysis algorithms.

The classical analyzer and the TCP analyzer both implement the same chess
analysis behavior — evaluation, top-moves, move classification, threat probing.
Only the transport (subprocess vs network) differs. The algorithms live here,
in one place, so a new transport just adapts the wire protocol.

Public API:
    AnalysisBackend    — Protocol implemented by every transport
    classify_move      — single algorithm shared by every backend
    probe_threat       — null-move threat probe shared by every backend
    pv_to_san          — render an engine principal variation as SAN

Each transport adapter (Analyzer, TCPAnalyzer) implements evaluate/top_moves
and delegates classify_move/probe_threat to the helpers in this module.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import chess

from .grading import classify_centipawn_loss
from .types import Eval, MoveAnalysis


def pv_to_san(board: chess.Board, pv_uci: list[str], limit: int = 6) -> str | None:
    """Render an engine principal-variation (UCI) as a SAN line from the given position."""
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
    """Transport-independent engine operations.

    Every analysis adapter (local UCI subprocess, TCP network client, …) implements
    these four operations. Consumers depend on the Protocol; tests can swap a fake.
    """

    name: str

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int | None = None,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval: ...

    async def top_moves(self, board: chess.Board, *, n: int = 3, depth: int | None = None) -> list[Eval]: ...

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
    """Compute the post-move eval, played PV head, and cp-loss for a single played move.

    Centralizes every edge case:
        - played-move-equals-best → reuse the before-eval tail
        - checkmate on the move
        - draw / game-over after the move
        - mate-distance transitions in either direction
        - regular cp loss

    Returns (eval_after, played_pv_head, centipawn_loss).
    """
    move_uci = move.uci()

    if move_uci.lower() == (eval_before.best_move or "").lower():
        loss = 0
        played_pv = eval_before.pv
        if board_after.is_checkmate():
            eval_after = Eval(cp=None, mate=0, depth=0)
        elif board_after.is_game_over() or board_after.can_claim_draw():
            eval_after = Eval(cp=0, depth=0)
        else:
            tail = played_pv[1:] if len(played_pv) > 1 else []
            eval_after = Eval(
                cp=eval_before.cp,
                mate=eval_before.mate,
                best_move=tail[0] if tail else None,
                pv=tail,
                depth=actual_depth,
            )
        return eval_after, played_pv, loss

    if board_after.is_checkmate():
        return Eval(cp=None, mate=0, depth=0), [move_uci], 0

    if board_after.is_game_over() or board_after.can_claim_draw():
        before_mover = sign * (eval_before.cp if eval_before.cp is not None else 0)
        return Eval(cp=0, depth=0), [move_uci], max(0, before_mover)

    if eval_after_lookup is None:
        # Caller must supply a post-move eval for any non-trivial case. Falling back
        # to the before-eval is a programming error: every non-terminal non-best line
        # needs a real search.
        raise ValueError("classify_move requires a post-move eval for non-terminal non-best moves")

    eval_after = eval_after_lookup
    played_pv = [move_uci] + (eval_after.pv or [])

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
        loss = 1000 if (sign * eval_before.mate > 0) else 0
    elif eval_before.mate is None and eval_after.mate is not None:
        loss = 0 if (sign * eval_after.mate > 0) else 1000
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
    """Grade a played move against the engine's best at `depth`.

    This is the single source of truth for move classification. Both the local
    UCI subprocess and the TCP analyzer delegate here so they share the same
    cp-loss / mate-distance semantics.
    """
    actual_depth = depth if depth is not None else getattr(backend, "_depth", 12)
    sign = 1 if board.turn == chess.WHITE else -1

    eval_before = await backend.evaluate(board, depth=actual_depth)

    board_after = board.copy(stack=True)
    board_after.push(move)

    eval_after: Eval
    played_pv: list[str]
    loss: int

    if move.uci().lower() == (eval_before.best_move or "").lower():
        eval_after, played_pv, loss = _played_eval_after(
            move=move,
            board_after=board_after,
            eval_before=eval_before,
            eval_after_lookup=None,
            sign=sign,
            actual_depth=actual_depth,
        )
    elif board_after.is_checkmate():
        eval_after, played_pv, loss = _played_eval_after(
            move=move,
            board_after=board_after,
            eval_before=eval_before,
            eval_after_lookup=None,
            sign=sign,
            actual_depth=actual_depth,
        )
    elif board_after.is_game_over() or board_after.can_claim_draw():
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

    full_played_pv = (
        played_pv if (played_pv and played_pv[0] == move.uci()) else [move.uci()] + (eval_after.pv or [])
    )

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
    """Null-move probe: what does the side that JUST moved threaten?

    Returns the eval after the side to move passes (so the mover's threat
    materializes), or None when the probe is illegal (in check) or the game
    is already over.
    """
    if board_after.is_check() or board_after.is_game_over():
        return None
    probe = board_after.copy(stack=False)
    probe.push(chess.Move.null())
    return await backend.evaluate(probe, depth=depth)
