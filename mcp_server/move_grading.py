"""Move grading: rule-aware classification of a played chess action.

Extracted from ``mcp_server.models`` for navigability. The single source of
truth for the move-class + effective-loss surface across every tool.

Strategy dispatch
-----------------

``score_played_move`` is a 14-branch decision tree that classifies a played
move into ``MoveClass`` (best/good/inaccuracy/mistake/blunder) and reports
the rule-aware loss surface (centipawn loss, mate distance loss, draw-claim
forfeit penalty, …). Each branch is encoded as a small named helper that
returns either a :class:`PlayedMoveScore` (match) or ``None`` (no match);
``score_played_move`` is a 60-line dispatcher that walks them in order.

Every audit invariant (B-01..B-05, C-01, C-02, H-01..H-03, L-06, M-04, M-05,
P0, P1, P2, P3, U-02..U-15, R4-§C, R5) is preserved byte-identical to the
pre-split ``models.score_played_move``.
"""

from __future__ import annotations

from typing import Literal

import chess

from core.engines.grading import classify_centipawn_loss
from core.winprob import win_prob as _win_pct
from mcp_server.models import MCPEval, PlayedMoveScore
from mcp_server.rules import ChessActionType, evaluate_rule_status

__all__ = ["score_played_move"]


_VALID_ACTIONS: Final[frozenset[str]] = frozenset(
    {"play_move", "claim_draw", "claim_draw_with_intended_move"}
)


def _is_before_winning(before_mover: int, mover_mate_before: int | None) -> bool:
    return (mover_mate_before is not None and mover_mate_before > 0) or before_mover >= 200


def _is_mover_forced_win(before_mover: int, mover_mate_before: int | None) -> bool:
    return (mover_mate_before is not None and mover_mate_before > 0) or before_mover >= 100


def _common_score_kwargs(
    score: PlayedMoveScore,
    *,
    canonical_best_action: str,
    rule_before,
    action_type: str,
) -> dict[str, Any]:
    """Project the rule-action provenance surface onto ``score``."""
    score.best_action = canonical_best_action
    score.can_claim_now = rule_before.can_claim_now
    score.can_claim_with_intended_move = rule_before.can_claim_with_intended_move
    score.claim_moves = rule_before.claim_moves
    score.action_type = action_type
    return score


def _score_delivered_checkmate(
    action_type: str,
    board_before: chess.Board,
    board_after: chess.Board,
    is_best_engine_move: bool,
    canonical_best_action: str,
    rule_before,
) -> PlayedMoveScore | None:
    """Move delivered checkmate — the best outcome for the mover."""
    if not (
        action_type == "play_move"
        and board_after.is_checkmate()
        and board_after.turn != board_before.turn
    ):
        return None
    score = PlayedMoveScore(
        move_class=MoveClass.BEST,
        centipawn_loss=0,
        raw_centipawn_loss=0,
        raw_centipawn_delta=0,
        mate_distance_loss=0,
        effective_loss=0,
        loss_kind="none",
        is_best_engine_move=is_best_engine_move,
        win_loss=0.0,
        best_action=canonical_best_action,
        is_best_action=True,
        action_equivalent=False,
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=None,
        claim_move=None,
    )
    return _common_score_kwargs(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


def _score_received_checkmate(
    action_type: str,
    board_before: chess.Board,
    board_after: chess.Board,
    canonical_best_action: str,
    rule_before,
) -> PlayedMoveScore | None:
    """Move walked into checkmate against the mover — full blunder."""
    if not (
        action_type == "play_move"
        and board_after.is_checkmate()
        and board_after.turn == board_before.turn
    ):
        return None
    score = PlayedMoveScore(
        move_class=MoveClass.BLUNDER,
        centipawn_loss=1000,
        raw_centipawn_loss=1000,
        raw_centipawn_delta=1000,
        mate_distance_loss=None,
        effective_loss=1000,
        loss_kind="mate_transition",
        outcome_penalty=1000,
        is_best_engine_move=False,
        win_loss=100.0,
        best_action=canonical_best_action,
        is_best_action=False,
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=None,
        claim_move=None,
    )
    return _common_score_kwargs(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


def _score_claim_draw_action(
    *,
    action_type: str,
    move: chess.Move,
    eval_before: MCPEval,
    rule_before,
    canonical_best_action: str,
    before_mover: int,
    mover_mate_before: int | None,
    is_best_engine_move: bool,
) -> PlayedMoveScore:
    """Grade a procedural draw-claim action.

    Honors the claim only when it's legally available AND the mover wasn't
    in a forced-win / winning-technical-conversion position (audit P0/P1).
    A claim while a forced mate is on the board is a blundered win, not a
    claim.
    """
    is_claim_now_action = action_type in (
        "claim_draw",
        ChessActionType.CLAIM_DRAW_NOW.value,
    )
    played_uci = move.uci().lower()

    claim_legal = (is_claim_now_action and rule_before.can_claim_now) or (
        not is_claim_now_action
        and rule_before.can_claim_with_intended_move
        and played_uci in [u.lower() for u in rule_before.intended_claim_ucis]
    )
    if not claim_legal:
        raise ValueError("ILLEGAL_ACTION: requested draw claim is not legally available")

    if is_claim_now_action:
        reasons = rule_before.claim_reasons_now
    else:
        reason_map = rule_before.intended_claim_reasons_by_uci
        reasons = reason_map.get(move.uci(), [])
    claim_r = reasons[0] if reasons else None

    is_mover_forced_win = (
        mover_mate_before is not None and mover_mate_before > 0
    ) or before_mover >= 200
    is_canonical_play_move = canonical_best_action == "play_move"
    if is_mover_forced_win or is_canonical_play_move:
        # U-02 (2026-09-01): a winning zeroing-capture at halfmove=100 sat at
        # cp=+26 but the rule layer detected a forced technical win; the
        # previous logic honored that as BEST. Now we flag the claim as a
        # smaller blunder — the mover forfeited a winning conversion.
        if is_mover_forced_win:
            eff_loss = 1000
            win_loss = 50.0
            loss_kind = "outcome_penalty"
            outcome_pen = 1000
        else:
            eff_loss = 500
            win_loss = 50.0
            loss_kind = "technical_win_forfeited"
            outcome_pen = None
        score = PlayedMoveScore(
            move_class=MoveClass.BLUNDER,
            centipawn_loss=None,
            raw_centipawn_loss=None,
            raw_centipawn_delta=0,
            mate_distance_loss=None,
            effective_loss=eff_loss,
            loss_kind=loss_kind,
            outcome_penalty=outcome_pen,
            is_best_engine_move=is_best_engine_move,
            win_loss=win_loss,
            best_action=canonical_best_action,
            is_best_action=False,
            action_equivalent=False,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=claim_r,
            claim_move=rule_before.claim_move,
        )
        return _common_score_kwargs(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )

    score = PlayedMoveScore(
        move_class=MoveClass.BEST,
        centipawn_loss=0,
        raw_centipawn_loss=0,
        raw_centipawn_delta=0,
        mate_distance_loss=0,
        effective_loss=0,
        loss_kind="none",
        is_best_engine_move=is_best_engine_move,
        win_loss=0.0,
        best_action=canonical_best_action,
        is_best_action=canonical_best_action == action_type,
        action_equivalent=canonical_best_action != action_type,
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=claim_r,
        claim_move=rule_before.claim_move,
    )
    return _common_score_kwargs(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


def _score_blundered_terminal_draw(
    *,
    is_auto_terminal_draw: bool,
    is_before_winning: bool,
    board_after: chess.Board,
    is_best_engine_move: bool,
    raw_cpl: int,
    raw_board_delta: int,
    before_mover: int,
    mover_mate_before: int | None,
    canonical_best_action: str,
    rule_before,
    action_type: str,
) -> PlayedMoveScore | None:
    if not is_auto_terminal_draw:
        return None
    if is_before_winning:
        eff_loss = max(300, min(1000, before_mover if before_mover > 0 else 1000))
        score = PlayedMoveScore(
            move_class=MoveClass.BLUNDER,
            centipawn_loss=None if mover_mate_before is not None else raw_cpl,
            raw_centipawn_loss=None if mover_mate_before is not None else raw_cpl,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=None,
            effective_loss=eff_loss,
            loss_kind="blundered_draw",
            outcome_penalty=eff_loss,
            is_best_engine_move=False,
            win_loss=50.0,
            best_action=canonical_best_action,
            is_best_action=False,
            action_equivalent=False,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=None,
            claim_move=None,
        )
    else:
        score = PlayedMoveScore(
            move_class=MoveClass.BEST if is_best_engine_move else MoveClass.GOOD,
            centipawn_loss=0,
            raw_centipawn_loss=raw_cpl,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=None,
            effective_loss=0,
            loss_kind="none",
            is_best_engine_move=is_best_engine_move,
            win_loss=0.0,
            best_action=canonical_best_action,
            is_best_action=True,
            action_equivalent=canonical_best_action
            in ("claim_draw", "claim_draw_with_intended_move"),
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=None,
            claim_move=None,
        )
    return _common_score_kwargs(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


def _score_conceded_draw(
    *,
    opponent_will_claim: bool,
    rule_after,
    eval_after: MCPEval,
    is_before_winning: bool,
    raw_cpl: int,
    raw_board_delta: int,
    before_mover: int,
    canonical_best_action: str,
    rule_before,
    action_type: str,
) -> PlayedMoveScore | None:
    if not opponent_will_claim:
        return None
    eff_loss = max(500, min(1000, before_mover if before_mover > 0 else 1000))
    claim_r = (
        rule_after.claim_reasons_now[0]
        if rule_after.claim_reasons_now
        else (eval_after.claim_reasons[0] if eval_after.claim_reasons else "threefold_repetition")
    )
    score = PlayedMoveScore(
        move_class=MoveClass.BLUNDER,
        centipawn_loss=raw_cpl,
        raw_centipawn_loss=raw_cpl,
        raw_centipawn_delta=raw_board_delta,
        mate_distance_loss=None,
        effective_loss=eff_loss,
        loss_kind="conceded_draw",
        rule_action_penalty=eff_loss,
        is_best_engine_move=False,
        win_loss=50.0,
        best_action=canonical_best_action,
        is_best_action=False,
        action_equivalent=False,
        missed_draw_claim=False,
        conceded_draw_claim=True,
        claim_reason=claim_r,
        claim_move=None,
    )
    return _common_score_kwargs(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


def _score_optimal_claim_recommended(
    *,
    optimal_claim_recommended: bool,
    is_auto_terminal_draw: bool,
    eval_before: MCPEval,
    eval_after: MCPEval,
    rule_before,
    raw_cpl: int,
    raw_board_delta: int,
    is_best_engine_move: bool,
    canonical_best_action: str,
    action_type: str,
    is_after_losing: bool,
    is_after_winning: bool,
    is_down_material: bool,
    is_mover_forced_win: bool,
    before_mover: int,
    after_mover: int,
    opp_mat: int,
    mover_mat: int,
) -> PlayedMoveScore | None:
    if not (optimal_claim_recommended and not is_auto_terminal_draw):
        return None
    decision_before_draw = bool(
        (eval_before.decision_value and eval_before.decision_value.get("outcome") == "draw")
        or (rule_before.can_claim_draw and before_mover <= 100)
    )
    decision_after_draw = bool(
        eval_after.decision_value and eval_after.decision_value.get("outcome") == "draw"
    )
    draw_preserved = bool(
        decision_before_draw and decision_after_draw and is_best_engine_move and raw_cpl == 0
    )
    if draw_preserved:
        claim_r = (
            rule_before.claim_reasons[0]
            if rule_before.claim_reasons
            else (eval_before.claim_reasons[0] if eval_before.claim_reasons else None)
        )
        score = PlayedMoveScore(
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            raw_centipawn_loss=0,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=None,
            effective_loss=0,
            loss_kind="none",
            is_best_engine_move=True,
            win_loss=0.0,
            best_action=canonical_best_action,
            is_best_action=canonical_best_action == "play_move",
            action_equivalent=True,
            missed_draw_claim=bool(
                is_down_material
                and canonical_best_action in ("claim_draw", "claim_draw_with_intended_move")
                and action_type == "play_move"
            ),
            conceded_draw_claim=False,
            claim_reason=claim_r,
            claim_move=rule_before.claim_move,
        )
        return _common_score_kwargs(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )
    if is_after_losing and not is_after_winning:
        loss_val = max(abs(before_mover), abs(after_mover))
        eff_loss = max(
            300,
            min(
                1000,
                loss_val if loss_val > 0 else (opp_mat - mover_mat if is_down_material else 500),
            ),
        )
        final_class = MoveClass.BLUNDER if eff_loss >= 300 else MoveClass.MISTAKE
        claim_r = (
            rule_before.claim_reasons[0]
            if rule_before.claim_reasons
            else (eval_before.claim_reasons[0] if eval_before.claim_reasons else None)
        )
        score = PlayedMoveScore(
            move_class=final_class,
            centipawn_loss=raw_cpl,
            raw_centipawn_loss=raw_cpl,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=None,
            effective_loss=eff_loss,
            loss_kind="draw_claim_forfeit",
            rule_action_penalty=eff_loss,
            is_best_engine_move=is_best_engine_move,
            win_loss=50.0,
            best_action=canonical_best_action,
            is_best_action=False,
            action_equivalent=False,
            missed_draw_claim=True,
            conceded_draw_claim=False,
            claim_reason=claim_r,
            claim_move=rule_before.claim_move,
        )
        return _common_score_kwargs(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )
    return None


def _score_mate_to_mate(
    *,
    mover_mate_before: int | None,
    mover_mate_after: int | None,
    board_after: chess.Board,
    is_best_engine_move: bool,
    canonical_best_action: str,
    rule_before,
    action_type: str,
) -> PlayedMoveScore | None:
    """Both before and after evaluations report a mate distance."""
    if not (mover_mate_before is not None and mover_mate_after is not None):
        return None
    if mover_mate_before > 0:
        if mover_mate_after > 0:
            if board_after.is_checkmate():
                mate_dist_loss = 0
            else:
                mate_dist_loss = max(0, (abs(mover_mate_after) + 1) - abs(mover_mate_before))
            if is_best_engine_move or mate_dist_loss == 0:
                final_class = MoveClass.BEST
                eff_loss = 0
                w_loss = 0.0
            elif mate_dist_loss <= 1:
                final_class = MoveClass.GOOD
                eff_loss = 50
                w_loss = 0.5
            elif mate_dist_loss <= 2:
                final_class = MoveClass.INACCURACY
                eff_loss = 150
                w_loss = 2.0
            else:
                final_class = MoveClass.MISTAKE
                eff_loss = 300
                w_loss = min(20.0, float(mate_dist_loss * 2.0))
            score = PlayedMoveScore(
                move_class=final_class,
                centipawn_loss=0,
                raw_centipawn_loss=0,
                raw_centipawn_delta=0,
                mate_distance_loss=mate_dist_loss,
                effective_loss=eff_loss,
                loss_kind="mate_distance" if mate_dist_loss > 0 else "none",
                mate_distance_penalty=eff_loss if mate_dist_loss > 0 else None,
                is_best_engine_move=is_best_engine_move,
                win_loss=w_loss,
                best_action=canonical_best_action,
                is_best_action=is_best_engine_move
                or (canonical_best_action == "play_move" and mate_dist_loss == 0),
                missed_draw_claim=False,
                conceded_draw_claim=False,
                claim_reason=None,
            )
            return _common_score_kwargs(
                score,
                canonical_best_action=canonical_best_action,
                rule_before=rule_before,
                action_type=action_type,
            )
        if mover_mate_after == 0:
            score = PlayedMoveScore(
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                raw_centipawn_loss=0,
                raw_centipawn_delta=0,
                mate_distance_loss=0,
                effective_loss=0,
                loss_kind="none",
                is_best_engine_move=is_best_engine_move,
                win_loss=0.0,
                best_action=canonical_best_action,
                is_best_action=True,
                missed_draw_claim=False,
                conceded_draw_claim=False,
                claim_reason=None,
                claim_move=None,
            )
            return _common_score_kwargs(
                score,
                canonical_best_action=canonical_best_action,
                rule_before=rule_before,
                action_type=action_type,
            )
        score = PlayedMoveScore(
            move_class=MoveClass.BLUNDER,
            centipawn_loss=None,
            raw_centipawn_loss=None,
            raw_centipawn_delta=None,
            mate_distance_loss=None,
            effective_loss=1000,
            loss_kind="mate_transition",
            outcome_penalty=1000,
            is_best_engine_move=False,
            win_loss=100.0,
            best_action=canonical_best_action,
            is_best_action=False,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=None,
        )
        return _common_score_kwargs(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )
    if mover_mate_before < 0 and mover_mate_after < 0:
        # Defender perspective: allowing faster mate against self is a loss
        # of resistance.
        defender_resistance_loss = max(0, abs(mover_mate_before) - abs(mover_mate_after))
        if is_best_engine_move or defender_resistance_loss == 0:
            final_class = MoveClass.BEST
            eff_loss = 0
            w_loss = 0.0
        elif defender_resistance_loss == 1:
            final_class = MoveClass.INACCURACY
            eff_loss = 150
            w_loss = 2.0
        else:
            final_class = MoveClass.BLUNDER if defender_resistance_loss >= 3 else MoveClass.MISTAKE
            eff_loss = 500 if defender_resistance_loss >= 3 else 300
            w_loss = min(20.0, float(defender_resistance_loss * 5.0))
        score = PlayedMoveScore(
            move_class=final_class,
            centipawn_loss=0,
            raw_centipawn_loss=0,
            raw_centipawn_delta=0,
            mate_distance_loss=defender_resistance_loss,
            effective_loss=eff_loss,
            loss_kind="mate_distance" if defender_resistance_loss > 0 else "none",
            mate_distance_penalty=eff_loss if defender_resistance_loss > 0 else None,
            is_best_engine_move=is_best_engine_move,
            win_loss=w_loss,
            best_action=canonical_best_action,
            is_best_action=is_best_engine_move and canonical_best_action == "play_move",
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=None,
            claim_move=rule_before.claim_move,
        )
        return _common_score_kwargs(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )
    return None


def _score_mate_to_cp(
    *,
    mover_mate_before: int | None,
    eval_move_eval: MCPEval,
    after_mover: int,
    is_best_engine_move: bool,
    canonical_best_action: str,
    rule_before,
    action_type: str,
    raw_cpl: int,
    raw_board_delta: int,
) -> PlayedMoveScore | None:
    """Mover had mate; after-state is centipawn (e.g. Qh4+ giving perpetual)."""
    if not (
        mover_mate_before is not None and mover_mate_before > 0 and eval_move_eval.mate is None
    ):
        return None
    if after_mover >= 400:
        final_class = MoveClass.BEST if is_best_engine_move else MoveClass.INACCURACY
        eff_loss = 0 if is_best_engine_move else 150
        w_loss = 0.0 if is_best_engine_move else 2.0
        score = PlayedMoveScore(
            move_class=final_class,
            centipawn_loss=0,
            raw_centipawn_loss=0,
            raw_centipawn_delta=0,
            mate_distance_loss=None,
            effective_loss=eff_loss,
            loss_kind="mate_distance" if not is_best_engine_move else "none",
            mate_distance_penalty=eff_loss if not is_best_engine_move else None,
            is_best_engine_move=is_best_engine_move,
            win_loss=w_loss,
            best_action=canonical_best_action,
            is_best_action=is_best_engine_move,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=None,
        )
        return _common_score_kwargs(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )
    wp_after = _win_pct(after_mover)
    win_loss = max(0.0, 100.0 - wp_after)
    final_class = MoveClass.BLUNDER if win_loss >= 20.0 else MoveClass.MISTAKE
    eff_loss = 1000 if final_class == MoveClass.BLUNDER else 300
    score = PlayedMoveScore(
        move_class=final_class,
        centipawn_loss=raw_cpl,
        raw_centipawn_loss=raw_cpl,
        raw_centipawn_delta=raw_board_delta,
        mate_distance_loss=None,
        effective_loss=eff_loss,
        loss_kind="outcome_penalty",
        outcome_penalty=eff_loss,
        is_best_engine_move=False,
        win_loss=win_loss,
        best_action=canonical_best_action,
        is_best_action=False,
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=None,
    )
    return _common_score_kwargs(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


def _score_cp_to_mate(
    *,
    eval_before: MCPEval,
    mover_mate_after: int | None,
    baseline_mover: int,
    optimal_claim_recommended: bool,
    after_mover: int,
    is_best_engine_move: bool,
    canonical_best_action: str,
    rule_before,
    action_type: str,
) -> PlayedMoveScore | None:
    """Before was centipawn; after-state now reports mate distance."""
    if not (eval_before.mate is None and mover_mate_after is not None):
        return None
    if mover_mate_after < 0:
        wp_before = _win_pct(baseline_mover)
        win_loss = max(0.0, wp_before - 0.0)
        score = PlayedMoveScore(
            move_class=MoveClass.BLUNDER,
            centipawn_loss=None,
            raw_centipawn_loss=None,
            raw_centipawn_delta=None,
            mate_distance_loss=None,
            effective_loss=1000,
            loss_kind="mate_transition",
            outcome_penalty=1000,
            is_best_engine_move=False,
            win_loss=win_loss,
            best_action=canonical_best_action,
            is_best_action=False,
            missed_draw_claim=optimal_claim_recommended,
            conceded_draw_claim=False,
            claim_reason=rule_before.claim_reasons[0]
            if (optimal_claim_recommended and rule_before.claim_reasons)
            else None,
        )
        return _common_score_kwargs(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )
    score = PlayedMoveScore(
        move_class=MoveClass.BEST if is_best_engine_move else MoveClass.GOOD,
        centipawn_loss=0,
        raw_centipawn_loss=0,
        raw_centipawn_delta=0,
        mate_distance_loss=0,
        effective_loss=0,
        loss_kind="none",
        is_best_engine_move=is_best_engine_move,
        win_loss=0.0,
        best_action=canonical_best_action,
        is_best_action=is_best_engine_move,
        action_equivalent=is_best_engine_move
        and canonical_best_action in ("claim_draw", "claim_draw_with_intended_move"),
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=None,
    )
    return _common_score_kwargs(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


def _score_standard_cp(
    *,
    is_best_engine_move: bool,
    canonical_best_action: str,
    rule_before,
    eval_before: MCPEval,
    raw_cpl: int,
    raw_board_delta: int,
    baseline_mover: int,
    after_mover: int,
    action_type: str,
) -> PlayedMoveScore:
    """Decisive-position saturation table + the raw cp fallback.

    Saturates engine cp loss at 95%+ winning ranges (cp >= 400, winp >= 95)
    so a tiny position change doesn't inflate the surface; falls back to
    ``classify_centipawn_loss`` for normal positions. Always enforces the
    hard invariant that an is_best_engine_move can never be MISTAKE/BLUNDER.
    """
    if is_best_engine_move:
        is_claim_best = canonical_best_action in ("claim_draw", "claim_draw_with_intended_move")
        claim_r = (
            rule_before.claim_reasons[0]
            if rule_before.claim_reasons
            else (eval_before.claim_reasons[0] if eval_before.claim_reasons else None)
        )
        score = PlayedMoveScore(
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            raw_centipawn_loss=0,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=None,
            effective_loss=0,
            loss_kind="none",
            is_best_engine_move=is_best_engine_move,
            win_loss=0.0,
            best_action=canonical_best_action,
            is_best_action=not is_claim_best,
            action_equivalent=is_claim_best,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=claim_r,
            claim_move=rule_before.claim_move,
        )
        return _common_score_kwargs(
            score,
            canonical_best_action=canonical_best_action,
            rule_before=rule_before,
            action_type=action_type,
        )

    eff_delta = baseline_mover - after_mover
    eff_loss = max(0, eff_delta)
    wp_before = _win_pct(baseline_mover)
    wp_after = _win_pct(after_mover)
    win_loss = max(0.0, wp_before - wp_after)

    if (wp_before >= 95.0 and wp_after >= 95.0) or (baseline_mover >= 400 and after_mover >= 400):
        if win_loss < 4.0:
            final_class = MoveClass.BEST if is_best_engine_move else MoveClass.GOOD
            final_cpl = min(eff_loss, 15)
        elif win_loss < 8.0:
            final_class = MoveClass.GOOD
            final_cpl = min(eff_loss, 45)
        else:
            classified = classify_centipawn_loss(eff_loss)
            final_class = (
                MoveClass.GOOD
                if classified == MoveClass.BEST and not is_best_engine_move
                else classified
            )
            final_cpl = min(eff_loss, 1000)
    elif (wp_before >= 90.0 and wp_after >= 90.0) or (baseline_mover >= 300 and after_mover >= 300):
        if win_loss < 4.0:
            final_class = MoveClass.BEST if is_best_engine_move else MoveClass.GOOD
            final_cpl = min(eff_loss, 20)
        elif win_loss < 8.0:
            final_class = MoveClass.GOOD
            final_cpl = min(eff_loss, 50)
        else:
            classified = classify_centipawn_loss(eff_loss)
            final_class = (
                MoveClass.GOOD
                if classified == MoveClass.BEST and not is_best_engine_move
                else classified
            )
            final_cpl = min(eff_loss, 1000)
    else:
        classified = classify_centipawn_loss(eff_loss)
        final_class = (
            MoveClass.GOOD
            if classified == MoveClass.BEST and not is_best_engine_move
            else classified
        )
        final_cpl = min(eff_loss, 1000)

    if is_best_engine_move and final_class in (MoveClass.MISTAKE, MoveClass.BLUNDER):
        final_class = MoveClass.BEST
        final_cpl = 0
        win_loss = 0.0

    score = PlayedMoveScore(
        move_class=final_class,
        centipawn_loss=raw_cpl,
        raw_centipawn_loss=raw_cpl,
        raw_centipawn_delta=raw_board_delta,
        mate_distance_loss=None,
        effective_loss=final_cpl,
        loss_kind="engine_cp" if final_cpl > 0 else "none",
        engine_cp_loss=final_cpl if final_cpl > 0 else None,
        is_best_engine_move=is_best_engine_move,
        win_loss=win_loss,
        best_action=canonical_best_action,
        is_best_action=is_best_engine_move and canonical_best_action == "play_move",
        action_equivalent=is_best_engine_move
        and canonical_best_action in ("claim_draw", "claim_draw_with_intended_move"),
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=rule_before.claim_reasons[0]
        if rule_before.claim_reasons
        else (eval_before.claim_reasons[0] if eval_before.claim_reasons else None),
        claim_move=rule_before.claim_move,
    )
    return _common_score_kwargs(
        score,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )


_PIECE_VALUES: Final[dict[chess.PieceType, int]] = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def _material_balance(board: chess.Board) -> tuple[int, int]:
    """Return (mover_material, opponent_material) in centipawns."""
    mover_color = board.turn
    mover_mat = sum(len(board.pieces(pt, mover_color)) * val for pt, val in _PIECE_VALUES.items())
    opp_mat = sum(len(board.pieces(pt, not mover_color)) * val for pt, val in _PIECE_VALUES.items())
    return mover_mat, opp_mat


def score_played_move(
    board_before: chess.Board,
    move: chess.Move,
    eval_before: MCPEval,
    eval_after: MCPEval,
    board_after: chess.Board | None = None,
    eval_played: MCPEval | None = None,
    action_type: Literal["play_move", "claim_draw", "claim_draw_with_intended_move"] = "play_move",
) -> PlayedMoveScore:
    """Unified, rule-aware single source of truth for move grading and loss across all tools."""
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

    is_before_winning = _is_before_winning(before_mover, mover_mate_before)
    score = _score_blundered_terminal_draw(
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
    )
    if score is not None:
        return score

    opponent_will_claim = bool(
        (rule_after.can_claim_now or eval_after.can_claim_now or eval_after.can_claim_draw)
        and (is_before_winning or before_mover >= 200)
    )
    score = _score_conceded_draw(
        opponent_will_claim=opponent_will_claim,
        rule_after=rule_after,
        eval_after=eval_after,
        is_before_winning=is_before_winning,
        raw_cpl=raw_cpl,
        raw_board_delta=raw_board_delta,
        before_mover=before_mover,
        canonical_best_action=canonical_best_action,
        rule_before=rule_before,
        action_type=action_type,
    )
    if score is not None:
        return score

    mover_mat, opp_mat = _material_balance(board_before)
    is_down_material = opp_mat - mover_mat >= 200
    is_after_winning = (mover_mate_after is not None and mover_mate_after > 0) or (
        after_mover >= 100
    )
    is_after_losing = (
        after_mover <= -100 or (mover_mate_after is not None and mover_mate_after < 0)
    ) or is_down_material
    is_mover_forced_win = _is_mover_forced_win(before_mover, mover_mate_before)
    optimal_claim_recommended = (
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

    score = _score_optimal_claim_recommended(
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
        is_after_losing=is_after_losing,
        is_after_winning=is_after_winning,
        is_down_material=is_down_material,
        is_mover_forced_win=is_mover_forced_win,
        before_mover=before_mover,
        after_mover=after_mover,
        opp_mat=opp_mat,
        mover_mat=mover_mat,
    )
    if score is not None:
        return score

    for dispatch in (
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
            optimal_claim_recommended=optimal_claim_recommended,
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


from typing import Any, Final  # noqa: E402  (Final needs to be available to module-level constants above)

from core.engines.types import MoveClass  # noqa: E402  (used by the helpers above)
