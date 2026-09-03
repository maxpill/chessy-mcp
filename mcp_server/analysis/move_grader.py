"""Move-grading service: rule-aware classification of a played chess action.

Wraps the 14-branch dispatcher in :mod:`mcp_server.move_grading` as a
:class:`MoveGrader` so callers can substitute classification strategies and
inject dependencies (win-probability function, centipawn classifier, etc.) at
construction time. The legacy free-function :func:`score_played_move` is kept
as a thin wrapper that uses the default grader — preserves the existing
import contract (``from mcp_server.move_grading import score_played_move``).

Strategy dispatch lives here, NOT in the helpers. Every strategy still
returns ``PlayedMoveScore | None`` from a hand-coded helper and the
:class:`MoveGrader` walks them in audit order.

Behavior-preserving: every audit invariant (B-01..B-05, C-01, C-02, H-01..H-03,
L-06, M-04, M-05, P0, P1, P2, P3, U-02..U-15, R4-§C, R5) is identical.
"""

from __future__ import annotations

from typing import Literal

import chess

from core.engines.grading import classify_centipawn_loss
from core.winprob import win_prob as _default_win_prob
from mcp_server.models import MCPEval, PlayedMoveScore
from mcp_server.move_grading import (
    _is_before_winning,
    _is_mover_forced_win,
    _material_balance,
    _score_blundered_terminal_draw,
    _score_claim_draw_action,
    _score_conceded_draw,
    _score_cp_to_mate,
    _score_delivered_checkmate,
    _score_mate_to_cp,
    _score_mate_to_mate,
    _score_optimal_claim_recommended,
    _score_received_checkmate,
    _score_standard_cp,
)
from mcp_server.rules import evaluate_rule_status

__all__ = ["MoveGrader", "score_played_move"]


class MoveGrader:
    """Service object that scores a played move against rule-aware policies.

    Construction-time dependencies:
        - ``win_prob_fn`` — White-POV win-probability function (default
          :func:`core.winprob.win_prob`).
        - ``cp_classifier`` — centipawn-loss → move-class function (default
          :func:`core.engines.grading.classify_centipawn_loss`).

    Both are injectable for testing without monkeypatching ``core``.
    """

    def __init__(
        self,
        *,
        win_prob_fn=None,
        cp_classifier=None,
    ) -> None:
        # Resolution order: injected → fall back to defaults. Keeping ``None``
        # bound lazily lets tests pass substitutes that aren't yet importable
        # at grader construction time (e.g. during module reload).
        self._win_prob_fn = win_prob_fn if win_prob_fn is not None else _default_win_prob
        self._cp_classifier = (
            cp_classifier if cp_classifier is not None else classify_centipawn_loss
        )

    def score(
        self,
        board_before: chess.Board,
        move: chess.Move,
        eval_before: MCPEval,
        eval_after: MCPEval,
        board_after: chess.Board | None = None,
        eval_played: MCPEval | None = None,
        action_type: Literal[
            "play_move", "claim_draw", "claim_draw_with_intended_move"
        ] = "play_move",
    ) -> PlayedMoveScore:
        """Unified, rule-aware single source of truth for move grading and loss across all tools."""
        # Bodies live in :mod:`mcp_server.move_grading`; this method is the
        # composition entry-point and keeps the dispatcher in sync with the
        # strategy helpers when new branches are added.
        return _dispatch_score(
            self._win_prob_fn,
            self._cp_classifier,
            board_before,
            move,
            eval_before,
            eval_after,
            board_after,
            eval_played,
            action_type,
        )


def _dispatch_score(
    win_prob_fn,
    cp_classifier,
    board_before: chess.Board,
    move: chess.Move,
    eval_before: MCPEval,
    eval_after: MCPEval,
    board_after: chess.Board | None,
    eval_played: MCPEval | None,
    action_type: str,
) -> PlayedMoveScore:
    """Internal dispatch body — kept separate so :class:`MoveGrader.score`
    stays one-line and the strategy helpers stay unit-testable.
    """
    if action_type not in {"play_move", "claim_draw", "claim_draw_with_intended_move"}:
        raise ValueError(f"INVALID_ACTION_TYPE: {action_type}")
    if board_after is None:
        board_after = board_before.copy(stack=True)
        board_after.push(move)

    is_white = board_before.turn == chess.WHITE
    sign = 1 if is_white else -1

    # Audit P0/P1: draw-claim actions are procedural — the supplied `move`
    # is not the engine's play_move reference. Force False so callers never
    # see a claim masquerading as the engine's best play.
    is_best_engine_move = bool(
        action_type == "play_move"
        and eval_before.best_move
        and move.uci().lower() == eval_before.best_move.lower()
    )

    eval_move_eval = eval_played if eval_played is not None else eval_after

    before_mover = sign * (eval_before.cp if eval_before.cp is not None else 0)
    if eval_before.cp is None and eval_before.mate is not None:
        before_mover = 10000 if (sign * eval_before.mate > 0) else -10000

    after_mover = sign * (eval_move_eval.cp if eval_move_eval.cp is not None else 0)
    if eval_move_eval.cp is None and eval_move_eval.mate is not None:
        after_mover = 10000 if (sign * eval_move_eval.mate > 0) else -10000

    mover_mate_before = sign * eval_before.mate if eval_before.mate is not None else None
    mover_mate_after = sign * eval_move_eval.mate if eval_move_eval.mate is not None else None

    before_mover_score = (
        before_mover
        if eval_before.mate is None
        else (mover_mate_before * 1000 if mover_mate_before is not None else 0)
    )
    history_state = eval_before.history_completeness
    rule_before = evaluate_rule_status(
        board_before,
        mover_score=before_mover_score,
        mate_for_mover=mover_mate_before,
        history_complete=history_state,
    )
    rule_after = evaluate_rule_status(board_after, history_complete=history_state)
    canonical_best_action = eval_before.best_action or rule_before.recommended_action

    # First two strategies run unconditionally (every play_move + every claim).
    for dispatch in (
        lambda: _score_delivered_checkmate(
            action_type,
            board_before,
            board_after,
            is_best_engine_move,
            canonical_best_action,
            rule_before,
        ),
        lambda: _score_received_checkmate(
            action_type,
            board_before,
            board_after,
            canonical_best_action,
            rule_before,
        ),
    ):
        score = dispatch()
        if score is not None:
            return score

    # Draw-claim branch (audit P0/P1).
    if action_type in ("claim_draw", "claim_draw_with_intended_move"):
        return _score_claim_draw_action(
            action_type=action_type,
            move=move,
            eval_before=eval_before,
            rule_before=rule_before,
            canonical_best_action=canonical_best_action,
            before_mover=before_mover,
            mover_mate_before=mover_mate_before,
            is_best_engine_move=is_best_engine_move,
        )

    is_auto_terminal_draw = bool(
        rule_after.terminal
        in (
            "stalemate",
            "insufficient_material",
            "seventyfive_moves",
            "fivefold_repetition",
            "dead_position",
        )
    )
    can_claim_before = bool(rule_before.can_claim_draw or eval_before.can_claim_draw)
    baseline_mover = max(before_mover, 0) if can_claim_before else before_mover
    raw_board_delta = before_mover - after_mover
    raw_cpl = 0 if is_best_engine_move else max(0, raw_board_delta)

    is_before_winning = _is_before_winning(before_mover, mover_mate_before)

    # Remaining strategies in audit priority order.
    for dispatch in (
        lambda: _score_blundered_terminal_draw(
            is_auto_terminal_draw=is_auto_terminal_draw,
            is_before_winning=is_before_winning,
            board_after=board_after,
            is_best_engine_move=is_best_engine_move,
            raw_cpl=raw_cpl,
            raw_board_delta=raw_board_delta,
            before_mover=before_mover,
            mover_mate_before=mover_mate_before,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        ),
        lambda: _score_conceded_draw(
            opponent_will_claim=bool(
                (rule_after.can_claim_now or eval_after.can_claim_now or eval_after.can_claim_draw)
                and (is_before_winning or before_mover >= 200)
            ),
            rule_after=rule_after,
            eval_after=eval_after,
            is_before_winning=is_before_winning,
            raw_cpl=raw_cpl,
            raw_board_delta=raw_board_delta,
            before_mover=before_mover,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        ),
        lambda: _score_optimal_claim_recommended(
            optimal_claim_recommended=_compute_optimal_claim_recommended(
                is_mover_forced_win=_is_mover_forced_win(before_mover, mover_mate_before),
                is_after_winning=_is_after_winning(mover_mate_after, after_mover),
                is_auto_terminal_draw=is_auto_terminal_draw,
                canonical_best_action=canonical_best_action,
                can_claim_before=can_claim_before,
                before_mover=before_mover,
                is_down_material=_is_down_material(board_before),
                mover_mate_before=mover_mate_before,
            ),
            is_auto_terminal_draw=is_auto_terminal_draw,
            eval_before=eval_before,
            eval_after=eval_after,
            rule_before=rule_before,
            raw_cpl=raw_cpl,
            raw_board_delta=raw_board_delta,
            is_best_engine_move=is_best_engine_move,
            canonical_best_action=canonical_best_action,
            action_type=action_type,
            is_after_losing=_is_after_losing(mover_mate_after, after_mover, board_before),
            is_after_winning=_is_after_winning(mover_mate_after, after_mover),
            is_down_material=_is_down_material(board_before),
            is_mover_forced_win=_is_mover_forced_win(before_mover, mover_mate_before),
            before_mover=before_mover,
            after_mover=after_mover,
            opp_mat=_material_balance(board_before)[1],
            mover_mat=_material_balance(board_before)[0],
        ),
        lambda: _score_mate_to_mate(
            mover_mate_before=mover_mate_before,
            mover_mate_after=mover_mate_after,
            board_after=board_after,
            is_best_engine_move=is_best_engine_move,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        ),
        lambda: _score_mate_to_cp(
            mover_mate_before=mover_mate_before,
            eval_move_eval=eval_move_eval,
            after_mover=after_mover,
            is_best_engine_move=is_best_engine_move,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
            raw_cpl=raw_cpl,
            raw_board_delta=raw_board_delta,
        ),
        lambda: _score_cp_to_mate(
            eval_before=eval_before,
            mover_mate_after=mover_mate_after,
            baseline_mover=baseline_mover,
            optimal_claim_recommended=_compute_optimal_claim_recommended(
                is_mover_forced_win=_is_mover_forced_win(before_mover, mover_mate_before),
                is_after_winning=_is_after_winning(mover_mate_after, after_mover),
                is_auto_terminal_draw=is_auto_terminal_draw,
                canonical_best_action=canonical_best_action,
                can_claim_before=can_claim_before,
                before_mover=before_mover,
                is_down_material=_is_down_material(board_before),
                mover_mate_before=mover_mate_before,
            ),
            after_mover=after_mover,
            is_best_engine_move=is_best_engine_move,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        ),
    ):
        score = dispatch()
        if score is not None:
            return score

    return _score_standard_cp(
        is_best_engine_move=is_best_engine_move,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        eval_before=eval_before,
        raw_cpl=raw_cpl,
        raw_board_delta=raw_board_delta,
        baseline_mover=baseline_mover,
        after_mover=after_mover,
        action_type=action_type,
    )


def _is_after_winning(mover_mate_after: int | None, after_mover: int) -> bool:
    return (mover_mate_after is not None and mover_mate_after > 0) or (after_mover >= 100)


def _is_after_losing(mover_mate_after: int | None, after_mover: int, board: chess.Board) -> bool:
    base = after_mover <= -100 or (mover_mate_after is not None and mover_mate_after < 0)
    if base:
        return True
    mover_mat, opp_mat = _material_balance(board)
    return opp_mat - mover_mat >= 200


def _is_down_material(board: chess.Board) -> bool:
    mover_mat, opp_mat = _material_balance(board)
    return opp_mat - mover_mat >= 200


def _compute_optimal_claim_recommended(
    *,
    is_mover_forced_win: bool,
    is_after_winning: bool,
    is_auto_terminal_draw: bool,
    canonical_best_action: str,
    can_claim_before: bool,
    before_mover: int,
    is_down_material: bool,
    mover_mate_before: int | None,
) -> bool:
    return (
        not is_mover_forced_win
        and not is_after_winning
        and not is_auto_terminal_draw
        and (
            canonical_best_action in ("claim_draw", "claim_draw_with_intended_move")
            or (
                can_claim_before
                and (
                    before_mover <= -100
                    or is_down_material
                    or (mover_mate_before is not None and mover_mate_before < 0)
                )
            )
        )
    )


# Backwards-compat shim — kept so existing import surface still works.
_default_grader = MoveGrader()


def score_played_move(
    board_before: chess.Board,
    move: chess.Move,
    eval_before: MCPEval,
    eval_after: MCPEval,
    board_after: chess.Board | None = None,
    eval_played: MCPEval | None = None,
    action_type: Literal["play_move", "claim_draw", "claim_draw_with_intended_move"] = "play_move",
) -> PlayedMoveScore:
    """Default-grader wrapper for ``score_played_move``. See :class:`MoveGrader` for the typed service object."""
    return _default_grader.score(
        board_before,
        move,
        eval_before,
        eval_after,
        board_after=board_after,
        eval_played=eval_played,
        action_type=action_type,
    )
