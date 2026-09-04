"""Strategy-dispatcher body used by ``MoveGrader.score`` and the public shim.

The dispatcher walks the strategies in audit-priority order and returns the
first one whose predicate fires. :func:`dispatch_score` is a pure function
of its inputs — no module globals, no I/O — so it's straightforward to
unit-test and to substitute test doubles for the win-probability fn.
"""

from __future__ import annotations

from typing import Any

import chess

from mcp_server.analysis.move_grading.helpers import (
    is_after_losing,
    is_after_winning,
    is_before_winning,
    is_down_material,
    is_mover_forced_win,
    material_balance,
)
from mcp_server.analysis.move_grading.strategies import (
    score_blundered_terminal_draw,
    score_claim_draw_action,
    score_conceded_draw,
    score_cp_to_mate,
    score_delivered_checkmate,
    score_mate_to_cp,
    score_mate_to_mate,
    score_optimal_claim_recommended,
    score_received_checkmate,
    score_standard_cp,
)
from mcp_server.models import MCPEval, PlayedMoveScore
from mcp_server.rules import evaluate_rule_status

__all__ = ["dispatch_score"]


_VALID_ACTIONS: frozenset[str] = frozenset(
    {"play_move", "claim_draw", "claim_draw_with_intended_move"}
)


def dispatch_score(
    board_before: chess.Board,
    move: chess.Move,
    eval_before: MCPEval,
    eval_after: MCPEval,
    board_after: chess.Board | None,
    eval_played: MCPEval | None,
    action_type: str,
) -> PlayedMoveScore:
    """Walk strategies in audit order; return the first one whose predicate fires."""
    if action_type not in _VALID_ACTIONS:
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

    # win_before = is_before_winning(before_mover, mover_mate_before)
    # print(
    #     f"DISPATCHER: rule_after.cn={rule_after.can_claim_now} eval_after.cn={eval_after.can_claim_now} eval_after.cd={eval_after.can_claim_draw} win_before={win_before} before_mover={before_mover} canonical_best_action={canonical_best_action} board_after.fen={board_after.fen()} board_after.hm={board_after.halfmove_clock} board_after.is_repetition(3)={board_after.is_repetition(3)}",
    #     flush=True,
    # )
    for dispatched in (
        lambda: score_delivered_checkmate(
            action_type=action_type,
            board_before=board_before,
            board_after=board_after,
            is_best_engine_move=is_best_engine_move,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
        ),
        lambda: score_received_checkmate(
            action_type=action_type,
            board_before=board_before,
            board_after=board_after,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
        ),
    ):
        score = dispatched()
        if score is not None:
            return score

    # 2. Draw-claim action (audit P0/P1).
    if action_type in ("claim_draw", "claim_draw_with_intended_move"):
        return score_claim_draw_action(
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

    win_before = is_before_winning(before_mover, mover_mate_before)
    mover_mat, opp_mat = material_balance(board_before)
    down_material = is_down_material(board_before)
    win_after = is_after_winning(mover_mate_after, after_mover)
    losing_after = is_after_losing(mover_mate_after, after_mover, board_before)
    forced_win = is_mover_forced_win(before_mover, mover_mate_before)

    optimal_claim_recommended = (
        not forced_win
        and not win_after
        and not is_auto_terminal_draw
        and (
            canonical_best_action in ("claim_draw", "claim_draw_with_intended_move")
            or (
                can_claim_before
                and (
                    before_mover <= -100
                    or down_material
                    or (mover_mate_before is not None and mover_mate_before < 0)
                )
            )
        )
    )

    # 3. Order: blundered-auto-draw → conceded-draw → optimal-claim →
    #    mate→mate → mate→cp → cp→mate → standard cp.
    for dispatched in (
        lambda: score_blundered_terminal_draw(
            is_auto_terminal_draw=is_auto_terminal_draw,
            is_before_winning=win_before,
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
        lambda: score_conceded_draw(
            opponent_will_claim=bool(
                (rule_after.can_claim_now or eval_after.can_claim_now or eval_after.can_claim_draw)
                and (win_before or before_mover >= 200)
            ),
            rule_after=rule_after,
            eval_after=eval_after,
            is_before_winning=win_before,
            raw_cpl=raw_cpl,
            raw_board_delta=raw_board_delta,
            before_mover=before_mover,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        ),
        lambda: score_optimal_claim_recommended(
            optimal_claim_recommended=optimal_claim_recommended,
            is_auto_terminal_draw=is_auto_terminal_draw,
            eval_before=eval_before,
            eval_after=eval_after,
            rule_before=rule_before,
            raw_cpl=raw_cpl,
            raw_board_delta=raw_board_delta,
            is_best_engine_move=is_best_engine_move,
            canonical_best_action=canonical_best_action,
            action_type=action_type,
            is_after_losing=losing_after,
            is_after_winning=win_after,
            is_down_material=down_material,
            is_mover_forced_win=forced_win,
            before_mover=before_mover,
            after_mover=after_mover,
            opp_mat=opp_mat,
            mover_mat=mover_mat,
        ),
        lambda: score_mate_to_mate(
            mover_mate_before=mover_mate_before,
            mover_mate_after=mover_mate_after,
            board_after=board_after,
            is_best_engine_move=is_best_engine_move,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        ),
        lambda: score_mate_to_cp(
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
        lambda: score_cp_to_mate(
            eval_before=eval_before,
            mover_mate_after=mover_mate_after,
            baseline_mover=baseline_mover,
            optimal_claim_recommended=optimal_claim_recommended,
            after_mover=after_mover,
            is_best_engine_move=is_best_engine_move,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        ),
    ):
        score = dispatched()
        if score is not None:
            return score

    return score_standard_cp(
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
