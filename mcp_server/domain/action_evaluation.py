"""ActionEvaluation: canonical value object for chess action values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import chess

from mcp_server.domain.action import GameAction
from mcp_server.domain.types import Outcome

if TYPE_CHECKING:
    from mcp_server.engine.protocol import EnginePoolLike

__all__ = [
    "MATE_CP",
    "ActionEvaluation",
    "ActionSource",
    "choose_recommended_action",
    "evaluate_action",
    "terminal_outcome_from_status",
]

MATE_CP = 100_000


class ActionSource(StrEnum):
    ROOT_MULTIPV = "root_multipv"
    POST_POSITION = "post_position"
    RULE_DEFAULT = "rule_default"
    INHERITED = "inherited"


@dataclass(frozen=True)
class ActionEvaluation:
    action: GameAction
    outcome: Outcome
    canonical_value: int | None
    mate_distance: int | None
    source: ActionSource
    rule_value: Outcome | None = None

    @property
    def is_winning(self) -> bool:
        return self.outcome == Outcome.WIN

    @property
    def is_draw(self) -> bool:
        return self.outcome == Outcome.DRAW

    @property
    def is_terminal(self) -> bool:
        return self.outcome in (Outcome.WIN, Outcome.LOSS, Outcome.DRAW)


def terminal_outcome_from_status(rule_status: Any) -> Outcome:
    if rule_status is None or getattr(rule_status, "terminal", None) is None:
        return Outcome.ACTIVE
    term = rule_status.terminal
    winner = getattr(rule_status, "winner", None)
    if term == "checkmate":
        if winner == "white":
            return Outcome.WIN if _mover_is(rule_status, chess.WHITE) else Outcome.LOSS
        if winner == "black":
            return Outcome.WIN if _mover_is(rule_status, chess.BLACK) else Outcome.LOSS
        return Outcome.ACTIVE
    return Outcome.DRAW


def _mover_is(rule_status: Any, color: chess.Color) -> bool:
    winner = getattr(rule_status, "winner", None)
    if winner == "white":
        return color == chess.WHITE
    if winner == "black":
        return color == chess.BLACK
    return False


async def evaluate_action(
    board: chess.Board,
    action: GameAction,
    *,
    pool: EnginePoolLike | None = None,
    depth: int = 8,
    rule_status: Any | None = None,
) -> ActionEvaluation:
    from mcp_server.domain.action import (
        ClaimDrawAction,
        ClaimDrawWithIntendedMoveAction,
        GameOverAction,
        PlayMoveAction,
    )

    if isinstance(action, GameOverAction):
        outcome = _outcome_from_game_over(action)
        return ActionEvaluation(
            action=action,
            outcome=outcome,
            canonical_value=None,
            mate_distance=None,
            source=ActionSource.RULE_DEFAULT,
            rule_value=outcome,
        )

    if isinstance(action, (ClaimDrawAction, ClaimDrawWithIntendedMoveAction)):
        return ActionEvaluation(
            action=action,
            outcome=Outcome.DRAW,
            canonical_value=0,
            mate_distance=None,
            source=ActionSource.RULE_DEFAULT,
            rule_value=Outcome.DRAW,
        )

    if isinstance(action, PlayMoveAction):
        return await _evaluate_play_move(
            board, action, pool=pool, depth=depth, rule_status=rule_status
        )

    raise TypeError(f"unsupported action type: {type(action).__name__}")


def _outcome_from_game_over(action: Any) -> Outcome:
    raw = getattr(action, "outcome", "draw")
    if raw == "win":
        return Outcome.WIN
    if raw == "loss":
        return Outcome.LOSS
    return Outcome.DRAW


async def _evaluate_play_move(
    board: chess.Board,
    action: Any,
    *,
    pool: EnginePoolLike | None,
    depth: int,
    rule_status: Any | None,
) -> ActionEvaluation:
    from mcp_server.domain.action import PlayMoveAction

    if not isinstance(action, PlayMoveAction):
        raise TypeError(f"expected PlayMoveAction, got {type(action).__name__}")
    try:
        move = chess.Move.from_uci(action.move.uci.lower())
    except Exception:
        move = None
    if move is None or move not in board.legal_moves:
        return ActionEvaluation(
            action=action,
            outcome=Outcome.ACTIVE,
            canonical_value=None,
            mate_distance=None,
            source=ActionSource.RULE_DEFAULT,
        )
    post = board.copy(stack=True)
    post.push(move)

    post_rule_status = rule_status
    if post_rule_status is None or getattr(post_rule_status, "_evaluated_on", None) is not post:
        post_rule_status = _quick_post_rule_status(post)

    outcome = _post_outcome(post, post_rule_status, mover_color=board.turn)
    if outcome in (Outcome.WIN, Outcome.LOSS, Outcome.DRAW):
        mate_distance = _post_mate_distance(post, mover_color=board.turn)
        cp: int | None
        if outcome == Outcome.WIN:
            cp = MATE_CP
        elif outcome == Outcome.LOSS:
            cp = -MATE_CP
        else:
            cp = 0
        return ActionEvaluation(
            action=action,
            outcome=outcome,
            canonical_value=cp,
            mate_distance=mate_distance,
            source=ActionSource.POST_POSITION,
            rule_value=outcome,
        )

    canonical_value, mate_distance = await _post_state_engine_eval(
        post, mover_color=board.turn, pool=pool, depth=depth
    )
    return ActionEvaluation(
        action=action,
        outcome=outcome,
        canonical_value=canonical_value,
        mate_distance=mate_distance,
        source=ActionSource.POST_POSITION,
    )


def _quick_post_rule_status(post: chess.Board) -> Any:
    @dataclass
    class _Stub:
        terminal: str | None = None
        winner: str | None = None

    if post.is_checkmate():
        winner = "black" if post.turn == chess.WHITE else "white"
        return _Stub(terminal="checkmate", winner=winner)
    if post.is_stalemate():
        return _Stub(terminal="stalemate")
    if post.is_insufficient_material():
        return _Stub(terminal="insufficient_material")
    if post.is_seventyfive_moves():
        return _Stub(terminal="seventyfive_moves")
    if post.is_fivefold_repetition():
        return _Stub(terminal="fivefold_repetition")
    return _Stub(terminal=None)


def _post_outcome(post: chess.Board, rule_status: Any, *, mover_color: chess.Color) -> Outcome:
    term = getattr(rule_status, "terminal", None)
    if term is None:
        return Outcome.ACTIVE
    if term == "checkmate":
        winner = getattr(rule_status, "winner", None)
        if winner == "white":
            return Outcome.WIN if mover_color == chess.WHITE else Outcome.LOSS
        if winner == "black":
            return Outcome.WIN if mover_color == chess.BLACK else Outcome.LOSS
        return Outcome.ACTIVE
    return Outcome.DRAW


def _post_mate_distance(post: chess.Board, *, mover_color: chess.Color) -> int | None:
    if not post.is_checkmate():
        return None
    return 1


async def _post_state_engine_eval(
    post: chess.Board,
    *,
    mover_color: chess.Color,
    pool: EnginePoolLike | None,
    depth: int,
) -> tuple[int | None, int | None]:
    if pool is None:
        return None, None
    try:
        ev = await pool.evaluate(post, depth=depth)
    except Exception:
        return None, None
    mover_sign = 1 if mover_color == chess.WHITE else -1
    if getattr(ev, "mate", None) is not None:
        m = mover_sign * ev.mate
        if m > 0:
            return MATE_CP, m
        if m < 0:
            return -MATE_CP, m
        return 0, 0
    cp = getattr(ev, "cp", None)
    if cp is not None:
        return mover_sign * cp, None
    return None, None


def choose_recommended_action(
    board: chess.Board,
    evaluations: list[ActionEvaluation],
    *,
    can_claim_now: bool = False,
    can_claim_with_intended_move: bool = False,
) -> ActionEvaluation:
    winning = [
        e
        for e in evaluations
        if e.outcome == Outcome.WIN and getattr(e.action, "type", None) == "play_move"
    ]
    if winning:
        winning.sort(key=lambda e: (e.mate_distance or MATE_CP, -(e.canonical_value or 0)))
        return winning[0]

    if can_claim_now or can_claim_with_intended_move:
        claim_evals = [
            e
            for e in evaluations
            if e.outcome == Outcome.DRAW and getattr(e.action, "type", None) != "play_move"
        ]
        if claim_evals:
            return claim_evals[0]

    active = [
        e
        for e in evaluations
        if e.outcome == Outcome.ACTIVE and getattr(e.action, "type", None) == "play_move"
    ]
    if active:
        active.sort(
            key=lambda e: (-(e.canonical_value if e.canonical_value is not None else -MATE_CP),)
        )
        return active[0]

    losing = [
        e
        for e in evaluations
        if e.outcome == Outcome.LOSS and getattr(e.action, "type", None) == "play_move"
    ]
    if losing:
        losing.sort(key=lambda e: -(e.canonical_value or 0))
        return losing[0]

    return evaluations[0]
