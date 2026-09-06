"""Board-grounded forensic enrichment for selected game critical moments.

``build_game_coaching_evidence`` already selects the pedagogically important
plies and optionally verifies them more deeply. This module adds deterministic
facts that are especially useful for explaining those moments without another
engine search:

* exhaustive mate-in-one availability before and after the played move;
* bounded immediate mate-threat evidence under a hypothetical legal pass;
* a bounded per-ply position-delta trace over the already returned post-move PV;
* transition anchors for when material, defender, pin, line, square-control or
  king-pressure changes first appear.

The result remains evidence. It does not diagnose the player's thought process
and the principal-variation trace is not a proof that the line is forced.
"""

from __future__ import annotations

from typing import Any

import chess

from mcp_server.analysis.causal_trace import build_causal_position_trace
from mcp_server.analysis.forensic_extensions import build_adaptive_forcing_resolution
from mcp_server.analysis.mate_forensics import (
    mate_in_one_moves,
    mate_in_one_threats_if_pass,
)
from mcp_server.models import MCPEval
from mcp_server.models.game_coaching import GameCoachingEvidence


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


def enrich_game_critical_forensics(
    coaching: GameCoachingEvidence,
    *,
    positions: list[chess.Board],
    evals: list[MCPEval],
) -> GameCoachingEvidence:
    """Enrich selected critical moments using existing boards and scan PVs only."""
    if not coaching.critical_moments:
        return coaching

    enriched = []
    for moment in coaching.critical_moments:
        if moment.ply <= 0 or moment.ply >= len(positions) or moment.ply >= len(evals):
            enriched.append(moment)
            continue

        board_before = positions[moment.ply - 1]
        board_after = positions[moment.ply]
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

        signatures = list(moment.evidence_signatures)
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
        if trace is not None:
            signatures.extend(_trace_signatures(trace))

        enriched.append(
            moment.model_copy(
                update={
                    "mate_in_one_moves_before": mate_before,
                    "played_move_was_mate_in_one": played_was_mate,
                    "opponent_mate_in_one_threats_if_pass_before": mate_threats,
                    "mate_threat_pass_probe_available": threat_probe_available,
                    "mate_threat_pass_probe_reason": threat_probe_reason,
                    "played_move_addresses_immediate_mate_threat": addresses_mate_threat,
                    "opponent_mate_in_one_moves_after_played": mate_after,
                    "strongest_reply_is_mate_in_one": strongest_reply_is_mate,
                    "causal_trace": trace,
                    "evidence_signatures": sorted(set(signatures)),
                }
            )
        )

    return coaching.model_copy(update={"critical_moments": enriched})
