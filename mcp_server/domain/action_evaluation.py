"""ActionEvaluation — the canonical value-object for chess action values.

Bug doc §3.2, §4.6, §22 — the codebase had four endpoints computing
their own notion of "what's this action worth?" using three different
signals (root cp, post-state cp, post-state mate). Each endpoint reached
its own conclusion and they disagreed.

This module fixes that by introducing a single immutable value object:

    ActionEvaluation(
        action: LegalAction,
        outcome: Outcome,            # WIN / LOSS / DRAW / ACTIVE_UNKNOWN
        canonical_value: int | None,  # signed-mover-POV cp; MATE_CP for mate
        mate_distance: int | None,    # positive=WIN, negative=LOSS, None=non-mate
        source: ActionSource,         # ROOT_MULTIPV / POST_POSITION / RULE_DEFAULT
        rule_value: Outcome | None,   # the rule-mandated outcome (claim/game-over)
    )

Every tool that emits a "best_action_obj" / "played_action_obj" / "value"
goes through this contract. Same LegalAction → one canonical value,
guaranteed.

    evaluate_action(board, action, *, pool, depth) -> ActionEvaluation
        Pure-ish: applies the action, computes terminal outcome, optionally
        calls the engine for an active post-state evaluation.

    choose_recommended_action(board, evaluations: list[ActionEvaluation]) -> LegalAction
        Pure: given a list of evaluations, picks the one LegalAction.
        Rule: any WIN play_move beats any DRAW claim beats any active play_move.

This is the only allowed entry-point for action-value reasoning. The
pre-existing helpers in rules/action_choice.py are kept as a thin
back-compat shim that delegates here.
"""

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
    """Where the value signal came from."""

    ROOT_MULTIPV = "root_multipv"
    POST_POSITION = "post_position"
    RULE_DEFAULT = "rule_default"
    INHERITED = "inherited"


@dataclass(frozen=True)
class ActionEvaluation:
    """Canonical value of one LegalAction from this position.

    Frozen — once evaluated, an action has exactly one value. Equality
    compares action + outcome + canonical_value + mate_distance.
    """

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
    """Map a RuleStatus.terminal string to an Outcome (mover-POV)."""
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
    """Heuristic: we don't know the board from the rule_status, but the
    status knows the side that just moved (the loser of checkmate). For
    the WIN determination we need the mover's color — which is the side
    NOT in check. Use the stalemate/checkmate convention: the winner.
    """
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
    """Compute the canonical ActionEvaluation for one LegalAction.

    For play_move: apply, look up post-state terminal status, optionally
    call engine on post-state for active positions.
    For claim_draw / claim_draw_with_intended_move: rule_default → DRAW.
    For game_over: rule_status → WIN/LOSS/DRAW.
    """
    from mcp_server.domain.action import (
        ClaimDrawAction,
        ClaimDrawWithIntendedMoveAction,
        GameOverAction,
        PlayMoveAction,
    )

    if isinstance(action, GameOverAction):
        return ActionEvaluation(
            action=action,
            outcome=_outcome_from_game_over(action),
            canonical_value=None,
            mate_distance=None,
            source=ActionSource.RULE_DEFAULT,
            rule_value=_outcome_from_game_over(action),
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
    if outcome.is_terminal:
        mate_distance = _post_mate_distance(post, mover_color=board.turn)
        cp: int | None = None
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
    """Cheap post-state rule status: only the predicates that don't need
    move-stack history (terminal-in-board + 50-move)."""

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
    """Determine outcome of the post-state from the mover's POV."""
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
    """Mate distance in plies. 1 = mate-in-1 for mover. Negative = mover
    is mated. Returns None if not checkmate."""
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
    """Run a short engine call on the post-state, return (canonical_cp, mate_distance)
    from mover-POV. Returns (None, None) when pool is unavailable or the call fails."""
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
    """Pure: pick the canonical ActionEvaluation.

    Rule (in order):
      1. Any play_move with Outcome.WIN wins (mate-distance ascending).
      2. Otherwise any active play_move with the highest canonical_value.
      3. Otherwise the DRAW claim (if legal).
      4. Otherwise the best active play_move (lowest canonical_value / longest mate loss).
    """
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
