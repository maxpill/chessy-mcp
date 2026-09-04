"""Aggregate ACPL / accuracy / turning-point metrics for ``analyze_game``.

Extracted from ``mcp_server.server``. Computes per-side accuracy, average
effective loss, error counts, and the top turning points from a sequence
of position evaluations.
"""

from __future__ import annotations

import math

import chess

from mcp_server.models import MCPEval, PlyAnalysisItem
from mcp_server.move_grading import score_played_move
from mcp_server.rules import evaluate_rule_status


def _compute_game_metrics(
    positions: list[chess.Board],
    moves: list[chess.Move],
    evals: list[MCPEval],
) -> tuple[
    float | None,  # white_acc
    float | None,  # black_acc
    float | None,  # white_acpl
    float | None,  # black_acpl
    float | None,  # white_raw_acpl
    float | None,  # black_raw_acpl
    float | None,  # white_average_effective_loss
    float | None,  # black_average_effective_loss
    tuple[int, int, int],  # white blunders, mistakes, inaccuracies
    tuple[int, int, int],  # black blunders, mistakes, inaccuracies
    list[PlyAnalysisItem],  # turning_points
]:
    """Calculate ACPL, accuracy %, mistakes, and turning points from position evaluations."""
    white_cpls: list[int] = []
    black_cpls: list[int] = []
    white_raw_cpls: list[int] = []
    black_raw_cpls: list[int] = []
    white_eff_losses: list[int] = []
    black_eff_losses: list[int] = []
    white_accs: list[float] = []
    black_accs: list[float] = []
    white_blunders = white_mistakes = white_inaccuracies = 0
    black_blunders = black_mistakes = black_inaccuracies = 0
    turning_points: list[PlyAnalysisItem] = []

    is_game_draw = bool(
        evals
        and (
            positions[-1].is_game_over(claim_draw=False)
            or positions[-1].is_fifty_moves()
            or positions[-1].is_repetition(3)
        )
    )

    for ply_idx, move in enumerate(moves, start=1):
        board_before = positions[ply_idx - 1]
        board_after = positions[ply_idx]
        eval_before = evals[ply_idx - 1]
        eval_after = evals[ply_idx]

        is_white = board_before.turn == chess.WHITE
        move_san = board_before.san(move)

        # Only rewrite the final ply as a procedural draw-claim when:
        #  - the move is genuinely a 50-move or 3-fold claim (NOT an auto-terminal
        #    like 75-move / stalemate / checkmate / locked dead — those are real
        #    moves that LOST the game by blunder, not players taking a draw);
        #  - the move is one of the legal intended-claim moves (a non-resetting,
        #    non-capturing king move for the 50-move rule, or a repetition-completing
        #    move for the threefold rule).
        # Otherwise we score the move as a real play_move so a blunder into an
        # automatic terminal draw (e.g. Qf8+ at halfmove 149) is properly penalized.
        action_type_to_use = "play_move"
        if ply_idx == len(moves) and is_game_draw:
            intended_now = (
                board_after.is_fifty_moves() and not board_after.is_seventyfive_moves()
            ) or board_after.is_repetition(3)
            is_intended_claim = intended_now and (
                not board_after.is_game_over(claim_draw=False) or board_after.can_claim_draw
            )
            if is_intended_claim:
                played_uci = move.uci()
                rule_before = evaluate_rule_status(board_before, history_complete="complete")
                valid_for_intended = (
                    rule_before.can_claim_with_intended_move
                    and played_uci in rule_before.intended_claim_ucis
                )
                if valid_for_intended:
                    action_type_to_use = "claim_draw_with_intended_move"

        score = score_played_move(
            board_before,
            move,
            eval_before,
            eval_after,
            board_after,
            action_type=action_type_to_use,
        )

        mc = score.move_class.value
        cpl = score.centipawn_loss
        win_loss = score.win_loss
        move_acc = max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * win_loss) - 3.1669))
        effective_loss = score.effective_loss

        raw_cpl_val = (
            score.centipawn_loss
            if score.centipawn_loss is not None
            else (score.raw_centipawn_loss if score.raw_centipawn_loss is not None else 0)
        )
        if is_white:
            if raw_cpl_val is not None:
                white_cpls.append(raw_cpl_val)
            if score.raw_centipawn_loss is not None:
                white_raw_cpls.append(score.raw_centipawn_loss)
            elif score.centipawn_loss is not None:
                white_raw_cpls.append(score.centipawn_loss)
            if effective_loss is not None:
                white_eff_losses.append(effective_loss)
            white_accs.append(move_acc)
            if mc == "blunder":
                white_blunders += 1
            elif mc == "mistake":
                white_mistakes += 1
            elif mc == "inaccuracy":
                white_inaccuracies += 1
        else:
            if raw_cpl_val is not None:
                black_cpls.append(raw_cpl_val)
            if score.raw_centipawn_loss is not None:
                black_raw_cpls.append(score.raw_centipawn_loss)
            elif score.centipawn_loss is not None:
                black_raw_cpls.append(score.centipawn_loss)
            if effective_loss is not None:
                black_eff_losses.append(effective_loss)
            black_accs.append(move_acc)
            if mc == "blunder":
                black_blunders += 1
            elif mc == "mistake":
                black_mistakes += 1
            elif mc == "inaccuracy":
                black_inaccuracies += 1

        best_san: str | None = None
        # U-06 (2026-09-01): reconcile `best_san` with the final classification.
        # Without this guard, an analyze_game turning point can report
        # `best_move_san == played_san` while `move_class == "blunder"` —
        # internally contradictory. The bug surfaced at depth=1 where the
        # engine's top line happens to be a losing move (audit U-06
        # promotion-defense reproducer). When the played move equals the
        # engine's reported best but the classifier decided it was a
        # blunder/mistake, suppress the best_move_san to avoid the
        # contradiction. The classify_move path runs a depth+4 verification
        # search to refine; analyze_game doesn't (per-ply cost), so the
        # conservative answer is `best_san = None` here.
        if eval_before.best_move and not (
            score.is_best_engine_move and score.move_class.value in ("blunder", "mistake")
        ):
            try:
                move_obj = chess.Move.from_uci(eval_before.best_move.lower())
                if move_obj in board_before.legal_moves:
                    best_san = board_before.san(move_obj)
            except (
                ValueError,
                chess.IllegalMoveError,
                chess.InvalidMoveError,
                AssertionError,
            ):
                best_san = None

        if (
            (cpl is not None and cpl >= 150)
            or (effective_loss is not None and effective_loss >= 150)
            or mc in ("blunder", "mistake")
        ):
            turning_points.append(
                PlyAnalysisItem(
                    ply=ply_idx,
                    san=move_san,
                    uci=move.uci(),
                    move_class=mc,
                    centipawn_loss=cpl,
                    effective_loss=effective_loss,
                    loss_kind=score.loss_kind,
                    engine_cp_loss=score.engine_cp_loss,
                    mate_distance_penalty=score.mate_distance_penalty,
                    outcome_penalty=score.outcome_penalty,
                    rule_action_penalty=score.rule_action_penalty,
                    best_move_san=best_san,
                    best_action=score.best_action,
                    missed_draw_claim=score.missed_draw_claim,
                    conceded_draw_claim=score.conceded_draw_claim,
                    claim_reason=score.claim_reason,
                    claim_move=score.claim_move,
                )
            )

    white_acc = round(sum(white_accs) / len(white_accs), 1) if white_accs else None
    black_acc = round(sum(black_accs) / len(black_accs), 1) if black_accs else None
    white_raw_acpl = round(sum(white_raw_cpls) / len(white_raw_cpls), 1) if white_raw_cpls else None
    black_raw_acpl = round(sum(black_raw_cpls) / len(black_raw_cpls), 1) if black_raw_cpls else None
    white_avg_eff = (
        round(sum(white_eff_losses) / len(white_eff_losses), 1) if white_eff_losses else None
    )
    black_avg_eff = (
        round(sum(black_eff_losses) / len(black_eff_losses), 1) if black_eff_losses else None
    )
    white_acpl = white_avg_eff
    black_acpl = black_avg_eff

    top_turning_points = sorted(
        sorted(
            turning_points,
            key=lambda x: 1000 if x.effective_loss is None else x.effective_loss,
            reverse=True,
        )[:8],
        key=lambda x: x.ply,
    )

    return (
        white_acc,
        black_acc,
        white_acpl,
        black_acpl,
        white_raw_acpl,
        black_raw_acpl,
        white_avg_eff,
        black_avg_eff,
        (white_blunders, white_mistakes, white_inaccuracies),
        (black_blunders, black_mistakes, black_inaccuracies),
        top_turning_points,
    )
