"""Selective multi-depth verification for rich ``classify_move`` evidence.

The ordinary move classifier already has a narrow consistency check for the
special case where the played move is reported as engine-best but still grades
as a large error. Coaching needs a different question: does the pedagogically
important classification itself survive a deeper search?

This module performs that verification only for ``forensic`` move analysis and
only for inaccuracy/mistake/blunder results. It re-searches the position before
and after the played move at a higher depth, recomputes the normal move grade,
and escalates once more when the first verification is unstable. Instability is
not limited to a changed class: a material WDL/loss-magnitude shift or a changed
short PV prefix in an already tactical position can also justify one final
verification search. Standard and coach modes keep their existing cost.

Production callers can inject the repository's cached/rule-aware evaluator, so
these extra searches retain the same history, terminal and SingleFlight/cache
semantics as the rest of the MCP. Tests and isolated callers may fall back to a
minimal raw-pool adapter.

The output is evidence, not a new grading policy. The original wire-level
classification is left untouched; the richer ``forensics.stability`` block says
whether deeper searches agree, why escalation happened and what changed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import chess

from mcp_server.models import MCPEval
from mcp_server.models.forensics import ForensicMoveAnalysis
from mcp_server.move_grading import score_played_move

_REVERIFY_CLASSES = {"inaccuracy", "mistake", "blunder"}
_WDL_SHIFT_ESCALATE_PP = 3.0
_EFFECTIVE_LOSS_SHIFT_ESCALATE_CP = 75
_PV_PREFIX_PLIES = 3
_TACTICAL_STABILITY_SIGNATURES = {
    "FORCING_REPLY_MATERIAL_LOSS_IMMEDIATE",
    "DELAYED_MATERIALIZATION_AFTER_FORCING_REPLY",
    "OPPONENT_MATE_IN_ONE_AFTER_MOVE",
    "STRONGEST_REPLY_IS_MATE_IN_ONE",
    "FAILED_MATE_THREAT_UPDATE_CANDIDATE",
    "FORCING_REPLY_AVAILABLE",
}
CachedEvaluator = Callable[..., Awaitable[tuple[MCPEval, bool]]]


def _class_name(value: Any) -> str:
    return str(getattr(value, "value", value))


def _mate_signature(ev: MCPEval) -> Literal["white_mates", "black_mates", "no_mate"]:
    if ev.status == "checkmate" or ev.mate == 0:
        if ev.winner == "white":
            return "white_mates"
        if ev.winner == "black":
            return "black_mates"
        if ev.cp is not None and ev.cp != 0:
            return "white_mates" if ev.cp > 0 else "black_mates"
    if ev.mate is None:
        return "no_mate"
    return "white_mates" if ev.mate > 0 else "black_mates"


def _wdl_expectation(
    wdl: tuple[int, int, int] | None,
    color: chess.Color,
) -> float | None:
    if wdl is None:
        return None
    wins, draws, losses = wdl
    total = wins + draws + losses
    if total <= 0:
        return None
    white = (wins + 0.5 * draws) / total
    return white if color == chess.WHITE else 1.0 - white


def _wdl_loss_percentage_points(before: MCPEval, after: MCPEval, mover: chess.Color) -> float | None:
    before_expectation = _wdl_expectation(before.wdl, mover)
    after_expectation = _wdl_expectation(after.wdl, mover)
    if before_expectation is None or after_expectation is None:
        return None
    return round(max(0.0, (before_expectation - after_expectation) * 100.0), 3)


def _pv_prefix(ev: MCPEval) -> tuple[str, ...]:
    return tuple(str(move).lower() for move in ev.pv[:_PV_PREFIX_PLIES])


def _pv_prefix_stable(left: MCPEval, right: MCPEval) -> bool | None:
    if not left.pv or not right.pv:
        return None
    common = min(_PV_PREFIX_PLIES, len(left.pv), len(right.pv))
    if common <= 0:
        return None
    return tuple(str(move).lower() for move in left.pv[:common]) == tuple(
        str(move).lower() for move in right.pv[:common]
    )


def _tactical_context(result: ForensicMoveAnalysis) -> bool:
    evidence = result.forensics
    if evidence is None:
        return False
    if evidence.strongest_reply is not None and evidence.strongest_reply.is_forcing:
        return True
    return bool(set(evidence.evidence_signatures) & _TACTICAL_STABILITY_SIGNATURES)


def _evaluation_magnitude_stability(
    *,
    initial_wdl_loss: float | None,
    verified_wdl_loss: float | None,
    initial_effective_loss: int | None,
    verified_effective_loss: int | None,
) -> dict[str, Any]:
    if initial_wdl_loss is not None and verified_wdl_loss is not None:
        wdl_delta = round(abs(verified_wdl_loss - initial_wdl_loss), 3)
        return {
            "evaluation_magnitude_stable": wdl_delta <= _WDL_SHIFT_ESCALATE_PP,
            "evaluation_stability_basis": "wdl_loss_percentage_points",
            "wdl_loss_shift_percentage_points": wdl_delta,
            "effective_loss_shift_cp": None,
        }
    if initial_effective_loss is not None and verified_effective_loss is not None:
        effective_delta = abs(verified_effective_loss - initial_effective_loss)
        return {
            "evaluation_magnitude_stable": effective_delta <= _EFFECTIVE_LOSS_SHIFT_ESCALATE_CP,
            "evaluation_stability_basis": "effective_loss_cp",
            "wdl_loss_shift_percentage_points": None,
            "effective_loss_shift_cp": effective_delta,
        }
    return {
        "evaluation_magnitude_stable": None,
        "evaluation_stability_basis": "insufficient_numeric_evidence",
        "wdl_loss_shift_percentage_points": None,
        "effective_loss_shift_cp": None,
    }


def _as_mcpeval(
    raw: Any,
    board: chess.Board,
    *,
    depth: int,
    history_complete: str,
) -> MCPEval:
    if isinstance(raw, MCPEval):
        return raw.model_copy(update={"requested_depth": depth})
    return MCPEval.from_eval(
        raw,
        board.fen(),
        board=board,
        requested_depth=depth,
        history_complete=history_complete,
    )


async def _evaluate_one(
    pool: Any,
    board: chess.Board,
    *,
    depth: int,
    history_complete: str,
    evaluate_position: CachedEvaluator | None,
) -> MCPEval:
    if evaluate_position is not None:
        evaluated, _cache_hit = await evaluate_position(
            board,
            depth,
            pool,
            requested_depth=depth,
            history_complete=history_complete,
        )
        return evaluated
    raw = await pool.evaluate(board, depth=depth)
    return _as_mcpeval(raw, board, depth=depth, history_complete=history_complete)


async def _evaluate_pair(
    pool: Any,
    board_before: chess.Board,
    board_after: chess.Board,
    *,
    depth: int,
    history_complete: str,
    evaluate_position: CachedEvaluator | None,
) -> tuple[MCPEval, MCPEval]:
    ev_before, ev_after = await asyncio.gather(
        _evaluate_one(
            pool,
            board_before,
            depth=depth,
            history_complete=history_complete,
            evaluate_position=evaluate_position,
        ),
        _evaluate_one(
            pool,
            board_after,
            depth=depth,
            history_complete=history_complete,
            evaluate_position=evaluate_position,
        ),
    )
    return ev_before, ev_after


def _score_at_depth(
    result: ForensicMoveAnalysis,
    board_before: chess.Board,
    board_after: chess.Board,
    played_move: chess.Move,
    eval_before: MCPEval,
    eval_after: MCPEval,
) -> Any:
    return score_played_move(
        board_before,
        played_move,
        eval_before,
        eval_after,
        board_after,
        action_type=result.action_type,
    )


def _comparison(
    result: ForensicMoveAnalysis,
    verified_before: MCPEval,
    verified_after: MCPEval,
    verified_score: Any,
    *,
    mover: chess.Color,
) -> dict[str, Any]:
    initial_class = _class_name(result.move_class)
    verified_class = _class_name(verified_score.move_class)
    initial_best = (result.eval_before.best_move or "").lower() or None
    verified_best = (verified_before.best_move or "").lower() or None
    initial_mate_before = _mate_signature(result.eval_before)
    initial_mate_after = _mate_signature(result.eval_after)
    verified_mate_before = _mate_signature(verified_before)
    verified_mate_after = _mate_signature(verified_after)
    initial_wdl_loss = _wdl_loss_percentage_points(result.eval_before, result.eval_after, mover)
    verified_wdl_loss = _wdl_loss_percentage_points(verified_before, verified_after, mover)
    numeric = _evaluation_magnitude_stability(
        initial_wdl_loss=initial_wdl_loss,
        verified_wdl_loss=verified_wdl_loss,
        initial_effective_loss=result.effective_loss,
        verified_effective_loss=verified_score.effective_loss,
    )
    tactical_context = _tactical_context(result)
    before_pv_stable = _pv_prefix_stable(result.eval_before, verified_before)
    after_pv_stable = _pv_prefix_stable(result.eval_after, verified_after)
    tactical_pv_stable: bool | None = None
    if tactical_context:
        available = [item for item in (before_pv_stable, after_pv_stable) if item is not None]
        tactical_pv_stable = all(available) if available else None

    return {
        "verified_class": verified_class,
        "verified_effective_loss": verified_score.effective_loss,
        "verified_centipawn_loss": verified_score.centipawn_loss,
        "verified_best_move_uci": verified_best,
        "verified_wdl_loss_percentage_points": verified_wdl_loss,
        "verified_mate_before": verified_mate_before,
        "verified_mate_after": verified_mate_after,
        "verified_pv_before_prefix": list(_pv_prefix(verified_before)),
        "verified_pv_after_prefix": list(_pv_prefix(verified_after)),
        "class_stable": verified_class == initial_class,
        "best_move_stable": verified_best == initial_best,
        "mate_status_stable": (
            verified_mate_before == initial_mate_before
            and verified_mate_after == initial_mate_after
        ),
        "tactical_context_for_pv_stability": tactical_context,
        "pv_before_prefix_stable": before_pv_stable,
        "pv_after_prefix_stable": after_pv_stable,
        "tactical_pv_stable": tactical_pv_stable,
        **numeric,
    }


def _disagreement_reasons(comparison: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if comparison.get("class_stable") is False:
        reasons.append("classification_changed")
    if comparison.get("best_move_stable") is False:
        reasons.append("engine_best_move_changed")
    if comparison.get("mate_status_stable") is False:
        reasons.append("mate_status_changed")
    if comparison.get("evaluation_magnitude_stable") is False:
        basis = comparison.get("evaluation_stability_basis")
        if basis == "wdl_loss_percentage_points":
            reasons.append("wdl_loss_shift")
        else:
            reasons.append("effective_loss_shift")
    if comparison.get("tactical_pv_stable") is False:
        reasons.append("tactical_pv_prefix_changed")
    return reasons


def _verification_convergence_evidence(
    first_before: MCPEval,
    first_after: MCPEval,
    first_score: Any,
    second_before: MCPEval,
    second_after: MCPEval,
    second_score: Any,
    *,
    mover: chess.Color,
    tactical_context: bool,
) -> dict[str, Any]:
    first_wdl = _wdl_loss_percentage_points(first_before, first_after, mover)
    second_wdl = _wdl_loss_percentage_points(second_before, second_after, mover)
    numeric = _evaluation_magnitude_stability(
        initial_wdl_loss=first_wdl,
        verified_wdl_loss=second_wdl,
        initial_effective_loss=first_score.effective_loss,
        verified_effective_loss=second_score.effective_loss,
    )
    pv_before = _pv_prefix_stable(first_before, second_before)
    pv_after = _pv_prefix_stable(first_after, second_after)
    tactical_pv: bool | None = None
    if tactical_context:
        available = [item for item in (pv_before, pv_after) if item is not None]
        tactical_pv = all(available) if available else None

    class_converged = _class_name(second_score.move_class) == _class_name(first_score.move_class)
    best_converged = (second_before.best_move or "").lower() == (
        first_before.best_move or ""
    ).lower()
    mate_converged = (
        _mate_signature(second_before) == _mate_signature(first_before)
        and _mate_signature(second_after) == _mate_signature(first_after)
    )
    converged = bool(
        class_converged
        and best_converged
        and mate_converged
        and numeric["evaluation_magnitude_stable"] is not False
        and tactical_pv is not False
    )
    return {
        "converged": converged,
        "class_converged": class_converged,
        "best_move_converged": best_converged,
        "mate_status_converged": mate_converged,
        "evaluation_magnitude_converged": numeric["evaluation_magnitude_stable"],
        "evaluation_convergence_basis": numeric["evaluation_stability_basis"],
        "wdl_loss_shift_between_verifications": numeric["wdl_loss_shift_percentage_points"],
        "effective_loss_shift_between_verifications_cp": numeric["effective_loss_shift_cp"],
        "pv_before_prefix_converged": pv_before,
        "pv_after_prefix_converged": pv_after,
        "tactical_pv_converged": tactical_pv,
    }


async def verify_forensic_classification_stability(
    result: ForensicMoveAnalysis,
    board_before: chess.Board,
    *,
    played_move: chess.Move | None,
    pool: Any,
    depth: int,
    history_complete: str,
    evaluate_position: CachedEvaluator | None = None,
) -> ForensicMoveAnalysis:
    """Attach real higher-depth classification stability to forensic evidence.

    The helper is deliberately best-effort. If the extra verification search
    fails, the already-computed move analysis is returned with a status marker
    rather than converting an optional coaching check into a tool failure.
    """
    evidence = result.forensics
    if evidence is None or evidence.detail != "forensic":
        return result

    stability = dict(evidence.stability)
    initial_class = _class_name(result.move_class)
    stability.update(
        {
            "initial_depth": depth,
            "initial_class": initial_class,
            "initial_effective_loss": result.effective_loss,
            "initial_centipawn_loss": result.centipawn_loss,
            "initial_best_move_uci": (result.eval_before.best_move or "").lower() or None,
            "initial_wdl_loss_percentage_points": _wdl_loss_percentage_points(
                result.eval_before,
                result.eval_after,
                board_before.turn,
            ),
            "initial_mate_before": _mate_signature(result.eval_before),
            "initial_mate_after": _mate_signature(result.eval_after),
            "initial_pv_before_prefix": list(_pv_prefix(result.eval_before)),
            "initial_pv_after_prefix": list(_pv_prefix(result.eval_after)),
            "verification_performed": False,
            "verification_status": "not_needed",
            "verification_depth": None,
            "escalation_depth": None,
            "verification_converged": None,
            "verification_convergence": None,
            "first_verification_disagreement_reasons": [],
            "stable": None,
            "verification_uses_cached_rule_aware_evaluator": evaluate_position is not None,
            "stability_thresholds": {
                "wdl_loss_shift_escalate_percentage_points": _WDL_SHIFT_ESCALATE_PP,
                "effective_loss_shift_escalate_cp": _EFFECTIVE_LOSS_SHIFT_ESCALATE_CP,
                "tactical_pv_prefix_plies": _PV_PREFIX_PLIES,
            },
            "proof_scope": (
                "Selective multi-depth engine re-search of the same before/after positions. "
                "It tests class, best move, mate state, evaluation magnitude and, only in "
                "an already tactical context, short principal-variation stability. It does "
                "not prove the engine has solved the position or that a human could find the "
                "verified move OTB."
            ),
        }
    )

    if result.action_type != "play_move" or played_move is None:
        stability["verification_status"] = "non_move_action"
        return result.model_copy(
            update={"forensics": evidence.model_copy(update={"stability": stability})}
        )
    if initial_class not in _REVERIFY_CLASSES:
        stability["verification_status"] = "low_coaching_priority_class"
        return result.model_copy(
            update={"forensics": evidence.model_copy(update={"stability": stability})}
        )
    if depth >= 30:
        stability["verification_status"] = "depth_cap_reached"
        return result.model_copy(
            update={"forensics": evidence.model_copy(update={"stability": stability})}
        )

    board_after = board_before.copy(stack=True)
    if played_move not in board_after.legal_moves:
        stability["verification_status"] = "played_move_not_legal_on_reconstructed_board"
        return result.model_copy(
            update={"forensics": evidence.model_copy(update={"stability": stability})}
        )
    board_after.push(played_move)

    verification_depth = min(max(depth + 4, 24), 30)
    try:
        verified_before, verified_after = await _evaluate_pair(
            pool,
            board_before,
            board_after,
            depth=verification_depth,
            history_complete=history_complete,
            evaluate_position=evaluate_position,
        )
        verified_score = _score_at_depth(
            result,
            board_before,
            board_after,
            played_move,
            verified_before,
            verified_after,
        )
        first = _comparison(
            result,
            verified_before,
            verified_after,
            verified_score,
            mover=board_before.turn,
        )
        stability.update(first)
        stability["verification_performed"] = True
        stability["verification_status"] = "verified"
        stability["verification_depth"] = verification_depth

        disagreement_reasons = _disagreement_reasons(first)
        stability["first_verification_disagreement_reasons"] = disagreement_reasons
        disagreement = bool(disagreement_reasons)
        final_score = verified_score
        final_before = verified_before
        final_after = verified_after
        verification_converged: bool | None = None

        if disagreement and verification_depth < 30:
            escalation_depth = min(verification_depth + 2, 30)
            escalated_before, escalated_after = await _evaluate_pair(
                pool,
                board_before,
                board_after,
                depth=escalation_depth,
                history_complete=history_complete,
                evaluate_position=evaluate_position,
            )
            escalated_score = _score_at_depth(
                result,
                board_before,
                board_after,
                played_move,
                escalated_before,
                escalated_after,
            )
            convergence = _verification_convergence_evidence(
                verified_before,
                verified_after,
                verified_score,
                escalated_before,
                escalated_after,
                escalated_score,
                mover=board_before.turn,
                tactical_context=bool(first["tactical_context_for_pv_stability"]),
            )
            verification_converged = bool(convergence["converged"])
            stability["escalation_depth"] = escalation_depth
            stability["verification_converged"] = verification_converged
            stability["verification_convergence"] = convergence
            final_score = escalated_score
            final_before = escalated_before
            final_after = escalated_after
            stability["verification_depth"] = escalation_depth
            stability["verification_status"] = "escalated"

        final_comparison = _comparison(
            result,
            final_before,
            final_after,
            final_score,
            mover=board_before.turn,
        )
        stability.update(final_comparison)
        stability["stable"] = bool(
            final_comparison["class_stable"]
            and final_comparison["best_move_stable"]
            and final_comparison["mate_status_stable"]
            and final_comparison["evaluation_magnitude_stable"] is not False
            and final_comparison["tactical_pv_stable"] is not False
            and (verification_converged is not False)
        )
    except Exception:
        stability["verification_performed"] = False
        stability["verification_status"] = "engine_verification_failed"

    return result.model_copy(
        update={"forensics": evidence.model_copy(update={"stability": stability})}
    )
