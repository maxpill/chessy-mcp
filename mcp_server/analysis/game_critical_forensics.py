"""Board-grounded forensic enrichment for selected game critical moments.

``build_game_coaching_evidence`` already selects the pedagogically important
plies and optionally verifies them more deeply. This module adds deterministic
facts that are especially useful for explaining those moments without another
engine search:

* exhaustive mate-in-one availability before and after the played move;
* bounded immediate mate-threat evidence under a hypothetical legal pass;
* concrete opponent forcing-reply deltas around the played move;
* a bounded per-ply position-delta trace over the already returned post-move PV;
* transition anchors for when material, defender, pin, line, square-control or
  king-pressure changes first appear;
* strongest-reply materialization timing over the already returned PV;
* actual-game materialization links kept separate from principal-variation evidence;
* normalized evidence categories that can be aggregated across games without
  turning one position into a psychological diagnosis.

The result remains evidence. It does not diagnose the player's thought process
and the principal-variation trace is not a proof that the line is forced.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import chess

from mcp_server.analysis.causal_trace import build_causal_position_trace
from mcp_server.analysis.forensic_extensions import build_adaptive_forcing_resolution
from mcp_server.analysis.game_threat_forensics import critical_forcing_threat_delta
from mcp_server.analysis.mate_forensics import (
    mate_in_one_moves,
    mate_in_one_threats_if_pass,
)
from mcp_server.analysis.reply_materialization import build_reply_materialization_trace
from mcp_server.models import MCPEval
from mcp_server.models.game_coaching import (
    FailureCorpusBucket,
    FailureEvidenceCategory,
    GameCoachingEvidence,
    GameFailureCorpusSummary,
)


SIGNATURE_CATEGORY_MAP: dict[str, FailureEvidenceCategory] = {
    "FAILED_FORCING_THREAT_UPDATE_CANDIDATE": "failed_forcing_threat_update_candidate",
    "FAILED_POSITION_UPDATE_CANDIDATE": "failed_position_update_candidate",
    "FAILED_MATE_THREAT_UPDATE_CANDIDATE": "immediate_mate_threat_update_failure_candidate",
    "MISSED_MATE_IN_ONE_CANDIDATE": "mate_in_one_miss_candidate",
    "MISSED_FORCING_REPLY_CANDIDATE": "missed_forcing_reply_candidate",
    "FORCING_REPLY_MATERIAL_LOSS_IMMEDIATE": "missed_forcing_reply_candidate",
    "DELAYED_MATERIALIZATION_AFTER_FORCING_REPLY": "missed_forcing_reply_candidate",
    "NEW_EN_PRISE_PIECE_AFTER_MOVE": "new_en_prise_piece_after_move",
    "NEW_TACTICALLY_HANGING_CANDIDATE_AFTER_MOVE": (
        "new_tactically_hanging_candidate_after_move"
    ),
    "ONLY_MOVE_MISSED_CANDIDATE": "only_move_missed_candidate",
    "PAWN_MOVE_FORCING_PUNISHMENT": "pawn_move_forcing_punishment",
}


def _adaptive_limit(
    board_after: chess.Board,
    pv: list[str],
    *,
    mover: chess.Color,
) -> tuple[int, dict[str, Any]]:
    adaptive = build_adaptive_forcing_resolution(board_after, pv, mover=mover)
    consumed = adaptive.get("plies_consumed")
    if isinstance(consumed, int) and consumed >= 0:
        return consumed, adaptive
    return len(pv), adaptive


def _trace_signatures(trace: dict[str, Any]) -> list[str]:
    signatures: list[str] = []
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        return signatures
    signatures.append("CRITICAL_CAUSAL_POSITION_DELTA_TRACE_AVAILABLE")
    flags = {
        flag
        for step in steps
        if isinstance(step, dict)
        for flag in step.get("causal_flags", [])
        if isinstance(flag, str)
    }
    mapping = {
        "material_changed": "CRITICAL_LINE_MATERIAL_CHANGE",
        "defender_count_dropped": "CRITICAL_LINE_DEFENDER_COUNT_DROPPED",
        "new_en_prise_piece": "CRITICAL_LINE_NEW_EN_PRISE_PIECE",
        "new_pin": "CRITICAL_LINE_NEW_PIN",
        "slider_line_opened": "CRITICAL_LINE_SLIDER_LINE_OPENED",
        "strategic_control_lost": "CRITICAL_LINE_STRATEGIC_CONTROL_LOST",
        "king_ring_pressure_changed": "CRITICAL_LINE_KING_PRESSURE_CHANGED",
    }
    signatures.extend(value for key, value in mapping.items() if key in flags)
    return signatures


def _reply_materialization_evidence(
    board_after: chess.Board,
    pv: list[str],
    *,
    mover: chess.Color,
    strongest_reply_uci: str | None,
    strongest_reply_forcing: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not strongest_reply_uci:
        return None, []
    try:
        reply = chess.Move.from_uci(str(strongest_reply_uci).lower())
    except (ValueError, chess.InvalidMoveError):
        return None, []
    if reply not in board_after.legal_moves:
        return None, []

    aligned = bool(pv and str(pv[0]).lower() == reply.uci())
    line = list(pv) if aligned else [reply.uci()]
    materialization = build_reply_materialization_trace(
        board_after,
        line,
        mover=mover,
    )
    materialization = {
        **materialization,
        "returned_pv_starts_with_strongest_reply": aligned,
    }

    signatures: list[str] = []
    first_loss = materialization.get("first_material_loss_ply_for_mover")
    if strongest_reply_forcing and isinstance(first_loss, int):
        if first_loss == 1:
            signatures.append("FORCING_REPLY_MATERIAL_LOSS_IMMEDIATE")
        elif first_loss > 1 and aligned:
            signatures.append("DELAYED_MATERIALIZATION_AFTER_FORCING_REPLY")
    if not aligned:
        signatures.append("REPLY_PV_ALIGNMENT_UNAVAILABLE")
    return materialization, signatures


def _actual_game_materialization_link(
    coaching: GameCoachingEvidence,
    *,
    root_ply: int,
) -> dict[str, Any] | None:
    link = next(
        (item for item in coaching.root_cause_links if item.root_cause_ply == root_ply),
        None,
    )
    if link is None:
        return None
    return {
        "root_cause_ply": link.root_cause_ply,
        "root_cause_san": link.root_cause_san,
        "materialization_ply": link.materialization_ply,
        "materialization_san": link.materialization_san,
        "material_swing_cp": link.material_swing_cp,
        "plies_later": link.plies_later,
        "basis": link.basis,
        "source": "actual_game_mainline_material_balance",
        "proof_scope": (
            "Temporal actual-game evidence only: this is the first adverse material-balance "
            "change found within the bounded six-ply window after the selected critical move. "
            "It does not prove that the earlier move uniquely caused the later loss, and it "
            "must not be merged with the engine principal variation as if both were one line."
        ),
    }


def _failure_categories(signatures: list[str]) -> list[FailureEvidenceCategory]:
    return sorted(
        {SIGNATURE_CATEGORY_MAP[item] for item in signatures if item in SIGNATURE_CATEGORY_MAP}
    )


def _is_major_error(moment: Any) -> bool:
    loss = moment.verified_effective_loss
    if loss is None:
        loss = moment.effective_loss
    if loss is None:
        loss = moment.centipawn_loss
    grade = moment.verified_move_class or moment.move_class
    return bool((loss or 0) >= 100 or grade in {"mistake", "blunder"})


def _failure_corpus(moments: list[Any]) -> GameFailureCorpusSummary:
    plies_by_category: dict[FailureEvidenceCategory, list[int]] = defaultdict(list)
    self_report_by_category: dict[FailureEvidenceCategory, list[int]] = defaultdict(list)
    signatures_by_category: dict[FailureEvidenceCategory, set[str]] = defaultdict(set)
    major_error_plies: list[int] = []
    categorized_critical_plies: set[int] = set()

    for moment in moments:
        categories = list(moment.failure_evidence_categories)
        if _is_major_error(moment):
            major_error_plies.append(moment.ply)
        if categories:
            categorized_critical_plies.add(moment.ply)
        for category in categories:
            plies_by_category[category].append(moment.ply)
            if moment.user_comment_raw:
                self_report_by_category[category].append(moment.ply)
            signatures_by_category[category].update(
                signature
                for signature in moment.evidence_signatures
                if SIGNATURE_CATEGORY_MAP.get(signature) == category
            )

    buckets = [
        FailureCorpusBucket(
            category=category,
            count=len(plies_by_category[category]),
            plies=sorted(plies_by_category[category]),
            self_report_overlap_count=len(self_report_by_category[category]),
            self_reported_plies=sorted(self_report_by_category[category]),
            supporting_signatures=sorted(signatures_by_category[category]),
        )
        for category in sorted(plies_by_category)
    ]
    uncategorized = sorted(set(major_error_plies) - categorized_critical_plies)
    return GameFailureCorpusSummary(
        major_error_critical_moments=len(set(major_error_plies)),
        categorized_critical_moments=len(categorized_critical_plies),
        buckets=buckets,
        uncategorized_major_error_plies=uncategorized,
    )


def enrich_game_critical_forensics(
    coaching: GameCoachingEvidence,
    *,
    positions: list[chess.Board],
    evals: list[MCPEval],
) -> GameCoachingEvidence:
    """Enrich selected critical moments using existing boards and scan PVs only."""
    if not coaching.critical_moments:
        return coaching.model_copy(update={"failure_corpus": GameFailureCorpusSummary()})

    enriched = []
    for moment in coaching.critical_moments:
        if moment.ply <= 0 or moment.ply >= len(positions) or moment.ply >= len(evals):
            categories = _failure_categories(list(moment.evidence_signatures))
            enriched.append(moment.model_copy(update={"failure_evidence_categories": categories}))
            continue

        board_before = positions[moment.ply - 1]
        board_after = positions[moment.ply]
        forcing_delta = critical_forcing_threat_delta(board_before, board_after)
        mate_before = mate_in_one_moves(board_before)
        mate_after = mate_in_one_moves(board_after)
        mate_threats, threat_probe_available, threat_probe_reason = mate_in_one_threats_if_pass(
            board_before
        )
        before_mate_ucis = {item["uci"] for item in mate_before}
        after_mate_ucis = {item["uci"] for item in mate_after}
        played_was_mate = moment.uci in before_mate_ucis
        strongest_reply_is_mate = bool(
            moment.strongest_reply_uci
            and moment.strongest_reply_uci in after_mate_ucis
        )
        addresses_mate_threat: bool | None = None
        if mate_threats:
            addresses_mate_threat = not mate_after

        pv = list(evals[moment.ply].pv)
        trace: dict[str, Any] | None = None
        if pv:
            limit, adaptive = _adaptive_limit(
                board_after,
                pv,
                mover=board_before.turn,
            )
            trace = build_causal_position_trace(board_after, pv, max_plies=limit)
            trace = {
                **trace,
                "adaptive_forcing_resolution": adaptive,
                "source": "scan_or_cached_post_move_principal_variation",
            }

        reply_materialization, reply_materialization_signatures = _reply_materialization_evidence(
            board_after,
            pv,
            mover=board_before.turn,
            strongest_reply_uci=moment.strongest_reply_uci,
            strongest_reply_forcing=bool(
                moment.strongest_reply_is_check or moment.strongest_reply_is_capture
            ),
        )
        if trace is not None and reply_materialization is not None:
            trace = {**trace, "strongest_reply_materialization": reply_materialization}

        actual_materialization = _actual_game_materialization_link(
            coaching,
            root_ply=moment.ply,
        )
        if actual_materialization is not None:
            if trace is None:
                trace = {
                    "steps": [],
                    "source": "actual_game_materialization_only",
                    "actual_game_materialization_link": actual_materialization,
                }
            else:
                trace = {
                    **trace,
                    "actual_game_materialization_link": actual_materialization,
                }

        if trace is None:
            trace = {
                "steps": [],
                "source": "critical_forcing_threat_delta_only",
                "opponent_forcing_threat_delta": forcing_delta["trace"],
            }
        else:
            trace = {
                **trace,
                "opponent_forcing_threat_delta": forcing_delta["trace"],
            }

        signatures = list(moment.evidence_signatures)
        signatures.extend(reply_materialization_signatures)
        signatures.extend(forcing_delta["signatures"])
        if actual_materialization is not None:
            signatures.append("ACTUAL_GAME_MATERIALIZATION_LINK_AVAILABLE")
            if actual_materialization["plies_later"] == 1:
                signatures.append("ACTUAL_GAME_MATERIALIZATION_NEXT_PLY")
        if mate_before:
            signatures.append("MATE_IN_ONE_AVAILABLE_BEFORE_MOVE")
            if not played_was_mate:
                signatures.append("MISSED_MATE_IN_ONE_CANDIDATE")
        if played_was_mate:
            signatures.append("PLAYED_MATE_IN_ONE")
        if mate_threats:
            signatures.append("OPPONENT_MATE_IN_ONE_THREAT_IF_PASS")
            if addresses_mate_threat is False:
                signatures.append("FAILED_MATE_THREAT_UPDATE_CANDIDATE")
            elif addresses_mate_threat is True:
                signatures.append("IMMEDIATE_MATE_THREAT_ADDRESSED")
        if mate_after:
            signatures.append("OPPONENT_MATE_IN_ONE_AFTER_MOVE")
        if strongest_reply_is_mate:
            signatures.append("STRONGEST_REPLY_IS_MATE_IN_ONE")
        if any(item["back_rank_geometry"] for item in [*mate_before, *mate_threats, *mate_after]):
            signatures.append("BACK_RANK_MATE_GEOMETRY_CANDIDATE")
        signatures.extend(_trace_signatures(trace))

        signatures = sorted(set(signatures))
        categories = _failure_categories(signatures)
        enriched.append(
            moment.model_copy(
                update={
                    "opponent_forcing_threat_baseline_available": forcing_delta[
                        "baseline_available"
                    ],
                    "opponent_forcing_moves_after_played": forcing_delta[
                        "opponent_forcing_moves_after_played"
                    ],
                    "newly_enabled_opponent_forcing_moves_after_played": forcing_delta[
                        "newly_enabled_opponent_forcing_moves_after_played"
                    ],
                    "resolved_opponent_forcing_threat_candidates": forcing_delta[
                        "resolved_opponent_forcing_threat_candidates"
                    ],
                    "mate_in_one_moves_before": mate_before,
                    "played_move_was_mate_in_one": played_was_mate,
                    "opponent_mate_in_one_threats_if_pass_before": mate_threats,
                    "mate_threat_pass_probe_available": threat_probe_available,
                    "mate_threat_pass_probe_reason": threat_probe_reason,
                    "played_move_addresses_immediate_mate_threat": addresses_mate_threat,
                    "opponent_mate_in_one_moves_after_played": mate_after,
                    "strongest_reply_is_mate_in_one": strongest_reply_is_mate,
                    "causal_trace": trace,
                    "evidence_signatures": signatures,
                    "failure_evidence_categories": categories,
                }
            )
        )

    corpus = _failure_corpus(enriched)
    return coaching.model_copy(
        update={
            "critical_moments": enriched,
            "failure_corpus": corpus,
        }
    )
