"""Evidence-bounded practical-equivalence evidence for rich move analysis.

Engine-best and practically-equivalent are deliberately different questions.
This module does not change the move grade. It adds a coaching-priority view
using already available WDL, rule-outcome, mate and tactical-punishment
signals so a consumer can avoid turning tiny engine preferences into fake
lessons.

The policy is intentionally conservative and explicit. WDL dominates when it
is available. Centipawns are only a fallback when WDL is unavailable. Any
mate deterioration, rule-outcome change or concrete forcing-punishment signal
blocks an ``equivalent`` label.
"""

from __future__ import annotations

from typing import Any, Literal

import chess

from mcp_server.models.forensics import ForensicMoveAnalysis
from mcp_server.models.mcpeval import MCPEval

_EQUIVALENT_WDL_LOSS_PP = 2.0
_NON_EQUIVALENT_WDL_LOSS_PP = 5.0
_FALLBACK_EQUIVALENT_CP = 30
_FALLBACK_NON_EQUIVALENT_CP = 100

_HARD_TACTICAL_PUNISHMENT_SIGNATURES = {
    "OPPONENT_MATE_IN_ONE_AFTER_MOVE",
    "STRONGEST_REPLY_IS_MATE_IN_ONE",
    "FAILED_MATE_THREAT_UPDATE_CANDIDATE",
    "FORCING_REPLY_MATERIAL_LOSS_IMMEDIATE",
    "DELAYED_MATERIALIZATION_AFTER_FORCING_REPLY",
}


def _class_name(value: Any) -> str:
    return str(getattr(value, "value", value))


def _mover_wdl_expectation(
    wdl: tuple[int, int, int] | None,
    mover: chess.Color,
) -> float | None:
    if wdl is None:
        return None
    wins, draws, losses = wdl
    total = wins + draws + losses
    if total <= 0:
        return None
    white_expectation = (wins + 0.5 * draws) / total
    return white_expectation if mover == chess.WHITE else 1.0 - white_expectation


def _wdl_loss_percentage_points(result: ForensicMoveAnalysis, mover: chess.Color) -> float | None:
    before = _mover_wdl_expectation(result.eval_before.wdl, mover)
    after = _mover_wdl_expectation(result.eval_after.wdl, mover)
    if before is None or after is None:
        return None
    return round(max(0.0, (before - after) * 100.0), 3)


def _opponent_mate_signature(mover: chess.Color) -> str:
    return "black_mates" if mover == chess.WHITE else "white_mates"


def _mate_signature_from_eval(ev: MCPEval) -> Literal["white_mates", "black_mates", "no_mate"]:
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


def _selected_evidence(
    result: ForensicMoveAnalysis,
    mover: chess.Color,
) -> dict[str, Any]:
    evidence = result.forensics
    stability = dict(evidence.stability) if evidence is not None else {}
    verified = bool(stability.get("verification_performed"))

    wdl_loss = (
        stability.get("verified_wdl_loss_percentage_points")
        if verified
        else stability.get("initial_wdl_loss_percentage_points")
    )
    if not isinstance(wdl_loss, (int, float)):
        wdl_loss = _wdl_loss_percentage_points(result, mover)
    normalized_wdl_loss = (
        round(float(wdl_loss), 3) if isinstance(wdl_loss, (int, float)) else None
    )

    effective_loss = (
        stability.get("verified_effective_loss")
        if verified
        else stability.get("initial_effective_loss")
    )
    if not isinstance(effective_loss, int):
        effective_loss = result.effective_loss
    if effective_loss is None:
        effective_loss = result.centipawn_loss

    move_class = stability.get("verified_class") if verified else stability.get("initial_class")
    if not isinstance(move_class, str):
        move_class = _class_name(result.move_class)

    mate_before = stability.get("verified_mate_before") if verified else stability.get("initial_mate_before")
    mate_after = stability.get("verified_mate_after") if verified else stability.get("initial_mate_after")
    if not isinstance(mate_before, str):
        mate_before = _mate_signature_from_eval(result.eval_before)
    if not isinstance(mate_after, str):
        mate_after = _mate_signature_from_eval(result.eval_after)

    return {
        "basis": "verified" if verified else "initial",
        "wdl_loss_percentage_points": normalized_wdl_loss,
        "effective_loss_cp": effective_loss,
        "move_class": move_class,
        "mate_before": mate_before,
        "mate_after": mate_after,
    }


def build_practical_equivalence_evidence(
    result: ForensicMoveAnalysis,
    *,
    mover: chess.Color,
) -> dict[str, Any]:
    """Return a conservative practical-equivalence / coach-priority assessment.

    ``practical_equivalent`` is not a statement that two moves are objectively
    equal. It means the available WDL/outcome/mate/tactical evidence does not
    justify spending coaching attention on the engine preference under this
    bounded policy.
    """
    evidence = result.forensics
    selected = _selected_evidence(result, mover)
    wdl_loss = selected["wdl_loss_percentage_points"]
    effective_loss = selected["effective_loss_cp"]
    signatures = set(evidence.evidence_signatures if evidence is not None else [])
    hard_tactical = sorted(signatures & _HARD_TACTICAL_PUNISHMENT_SIGNATURES)

    strongest_reply_forcing = bool(
        evidence is not None
        and evidence.strongest_reply is not None
        and evidence.strongest_reply.is_forcing
    )
    forcing_large_loss = bool(
        strongest_reply_forcing
        and isinstance(effective_loss, int)
        and effective_loss >= _FALLBACK_NON_EQUIVALENT_CP
    )
    tactical_punishment = bool(hard_tactical or forcing_large_loss)

    mover_won_by_mate = (
        result.eval_after.status == "checkmate"
        and (
            (mover == chess.WHITE and result.eval_after.winner == "white")
            or (mover == chess.BLACK and result.eval_after.winner == "black")
        )
    )
    opponent_mate = _opponent_mate_signature(mover)
    mate_deterioration = (
        not mover_won_by_mate
        and selected["mate_after"] == opponent_mate
        and selected["mate_before"] != opponent_mate
    )
    same_rule_outcome = bool(result.same_outcome)

    status: Literal["equivalent", "not_equivalent", "indeterminate"]
    practical_equivalent: bool | None
    reason_codes: list[str] = []

    if result.action_type != "play_move":
        status = "indeterminate"
        practical_equivalent = None
        reason_codes.append("NON_MOVE_ACTION")
    elif result.is_best_engine_move or getattr(result, "is_engine_best", False):
        status = "equivalent"
        practical_equivalent = True
        if mover_won_by_mate:
            reason_codes.append("ENGINE_BEST_WINNING_MOVE")
        else:
            reason_codes.append("ENGINE_BEST_MOVE")

    elif mate_deterioration:
        status = "not_equivalent"
        practical_equivalent = False
        reason_codes.append("MATE_STATUS_DETERIORATED")
    elif not same_rule_outcome:
        status = "not_equivalent"
        practical_equivalent = False
        reason_codes.append("RULE_OUTCOME_CHANGED")
    elif tactical_punishment:
        status = "not_equivalent"
        practical_equivalent = False
        reason_codes.append("CONCRETE_FORCING_PUNISHMENT_EVIDENCE")
    elif isinstance(wdl_loss, float):
        if wdl_loss <= _EQUIVALENT_WDL_LOSS_PP:
            status = "equivalent"
            practical_equivalent = True
            reason_codes.append("WDL_LOSS_WITHIN_EQUIVALENCE_BAND")
        elif wdl_loss >= _NON_EQUIVALENT_WDL_LOSS_PP:
            status = "not_equivalent"
            practical_equivalent = False
            reason_codes.append("WDL_LOSS_EXCEEDS_COACHING_BAND")
        else:
            status = "indeterminate"
            practical_equivalent = None
            reason_codes.append("WDL_LOSS_IN_GRAY_ZONE")
    elif isinstance(effective_loss, int):
        if effective_loss <= _FALLBACK_EQUIVALENT_CP:
            status = "equivalent"
            practical_equivalent = True
            reason_codes.append("CP_FALLBACK_WITHIN_EQUIVALENCE_BAND")
        elif effective_loss >= _FALLBACK_NON_EQUIVALENT_CP:
            status = "not_equivalent"
            practical_equivalent = False
            reason_codes.append("CP_FALLBACK_EXCEEDS_COACHING_BAND")
        else:
            status = "indeterminate"
            practical_equivalent = None
            reason_codes.append("CP_FALLBACK_IN_GRAY_ZONE")
    else:
        status = "indeterminate"
        practical_equivalent = None
        reason_codes.append("INSUFFICIENT_WDL_OR_LOSS_EVIDENCE")

    coach_priority: Literal["negligible", "low", "medium", "high"]
    if practical_equivalent is True:
        coach_priority = "negligible"
    elif mate_deterioration or not same_rule_outcome or tactical_punishment:
        coach_priority = "high"
    elif (
        (isinstance(wdl_loss, float) and wdl_loss >= 15.0)
        or (isinstance(effective_loss, int) and effective_loss >= 200)
    ):
        coach_priority = "high"
    elif practical_equivalent is False:
        coach_priority = "medium"
    else:
        coach_priority = "low"

    return {
        "status": status,
        "practical_equivalent": practical_equivalent,
        "coach_priority": coach_priority,
        "evidence_basis": selected["basis"],
        "move_class_used": selected["move_class"],
        "is_engine_best": bool(result.is_best_engine_move),
        "same_rule_outcome": same_rule_outcome,
        "wdl_loss_percentage_points": wdl_loss,
        "effective_loss_cp": effective_loss,
        "mate_before": selected["mate_before"],
        "mate_after": selected["mate_after"],
        "mate_deterioration_for_mover": mate_deterioration,
        "strongest_reply_is_forcing": strongest_reply_forcing,
        "tactical_punishment_evidence": tactical_punishment,
        "tactical_punishment_signatures": hard_tactical,
        "reason_codes": reason_codes,
        "thresholds": {
            "equivalent_wdl_loss_percentage_points_max": _EQUIVALENT_WDL_LOSS_PP,
            "not_equivalent_wdl_loss_percentage_points_min": _NON_EQUIVALENT_WDL_LOSS_PP,
            "cp_fallback_equivalent_max": _FALLBACK_EQUIVALENT_CP,
            "cp_fallback_not_equivalent_min": _FALLBACK_NON_EQUIVALENT_CP,
        },
        "proof_scope": (
            "Coaching-priority heuristic over already available engine/board evidence. "
            "WDL is preferred when present; centipawns are fallback only. Any mate "
            "deterioration, rule-outcome change or concrete forcing-punishment evidence "
            "blocks an equivalent label. This does not prove two moves are objectively "
            "equal or equally easy for a human OTB."
        ),
    }


def apply_practical_equivalence(
    result: ForensicMoveAnalysis,
    *,
    mover: chess.Color,
) -> ForensicMoveAnalysis:
    """Attach practical equivalence inside ``forensics.stability`` without search."""
    evidence = result.forensics
    if evidence is None:
        return result

    stability = dict(evidence.stability)
    practical = build_practical_equivalence_evidence(result, mover=mover)
    stability.update(
        {
            "practical_equivalence": practical,
            "practical_equivalent": practical["practical_equivalent"],
            "practical_equivalence_status": practical["status"],
            "coach_priority": practical["coach_priority"],
        }
    )
    return result.model_copy(
        update={"forensics": evidence.model_copy(update={"stability": stability})}
    )
