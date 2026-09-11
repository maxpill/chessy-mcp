"""Structured, evidence-first chess coaching models.

These models deliberately contain chess facts rather than natural-language
coaching conclusions. The MCP server establishes what is on the board,
which forcing moves exist, what the engine's strongest reply is, and how
candidate positions differ. A coach/LLM can then infer the human process
error without the engine service pretending to read the player's mind.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from mcp_server.models.legacy import MCPMoveAnalysis, TopMovesResult
from mcp_server.models.mcpeval import MCPEval


class PositionFingerprint(BaseModel):
    canonical_fen: str
    side_to_move: Literal["white", "black"]
    piece_map: dict[str, dict[str, list[str]]]
    material: dict[str, int]
    castling_rights: str
    en_passant: str | None = None
    in_check: bool
    legal_move_count: int
    position_hash: str


class ForcingMoveEvidence(BaseModel):
    uci: str
    san: str
    is_check: bool = False
    is_capture: bool = False
    captured_piece: str | None = None
    promotion: str | None = None
    is_mate: bool = False


class PieceEvidence(BaseModel):
    color: Literal["white", "black"]
    piece: str
    square: str
    attackers: int = 0
    defenders: int = 0


class DefenderLoadEvidence(BaseModel):
    color: Literal["white", "black"]
    piece: str
    square: str
    attacked_by: int = 0
    defended_targets: list[str] = Field(default_factory=list)
    attacked_targets: list[str] = Field(default_factory=list)
    sole_defense_targets: list[str] = Field(default_factory=list)


class TacticalHangingEvidence(BaseModel):
    target: PieceEvidence
    capture: ForcingMoveEvidence
    nominal_defenders: int
    legal_immediate_recaptures: list[str] = Field(default_factory=list)
    reason: Literal[
        "defended_but_no_legal_immediate_recapture",
        "local_capture_exchange_profitable",
    ] = "defended_but_no_legal_immediate_recapture"
    local_exchange_gain_cp: int | None = None
    local_exchange_line_uci: list[str] = Field(default_factory=list)
    local_exchange_line_san: list[str] = Field(default_factory=list)
    local_exchange_tree_complete: bool | None = None
    proof_scope: str = (
        "Immediate recapture legality only. This is a tactical-hanging candidate, "
        "not a full exchange-sequence or SEE proof."
    )


class MechanismCandidateEvidence(BaseModel):
    """Evidence-bounded tactical motif candidate.

    `mechanism` names the board geometry that was detected. `proof_scope`
    states exactly what was and was not established so a coaching layer does
    not promote a geometric candidate into a forced tactical claim.
    """

    mechanism: Literal[
        "absolute_pin",
        "check_capture",
        "discovered_attack_candidate",
        "discovered_check",
        "fork_candidate",
        "interference_candidate",
        "overloaded_defender_candidate",
        "promotion_tactic",
        "relative_pin_candidate",
        "removal_of_defender_candidate",
        "skewer_candidate",
        "trapped_piece_candidate",
    ]
    trigger_uci: str | None = None
    trigger_san: str | None = None
    actor: str | None = None
    targets: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    proof_scope: str
    # F-010 fix (2026-09-09): presentation priority bucket. Computed from
    # the candidate's category + whether it is supported by engine PV /
    # concrete material consequence. Used to rank presentation_mechanisms
    # without dropping raw evidence.
    presentation_priority: Literal[
        "immediate_mate",
        "checking_move",
        "capturing_move",
        "promotion",
        "forced_reply",
        "new_en_prise",
        "pinned_defender",
        "engine_pv_supported",
        "pure_geometry",
    ] = "pure_geometry"
    relevance: Literal[
        "tactically_actionable",
        "engine_supported",
        "pure_geometry",
    ] = "pure_geometry"
    evidence_level: Literal[
        "pure_geometry",
        "legally_actionable",
        "exchange_validated",
        "engine_supported",
        "forced_or_exhaustively_proven",
    ] = "pure_geometry"

    @model_validator(mode="before")
    @classmethod
    def _populate_relevance(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "relevance" not in data or data["relevance"] is None:
                priority = data.get("presentation_priority", "pure_geometry")
                if priority in (
                    "immediate_mate",
                    "checking_move",
                    "capturing_move",
                    "promotion",
                    "forced_reply",
                    "new_en_prise",
                    "pinned_defender",
                ):
                    data["relevance"] = "tactically_actionable"
                elif priority == "engine_pv_supported":
                    data["relevance"] = "engine_supported"
                else:
                    data["relevance"] = "pure_geometry"
            if "evidence_level" not in data or data["evidence_level"] is None:
                priority = data.get("presentation_priority", "pure_geometry")
                if priority == "immediate_mate":
                    data["evidence_level"] = "forced_or_exhaustively_proven"
                elif priority == "engine_pv_supported":
                    data["evidence_level"] = "engine_supported"
                elif priority in ("checking_move", "capturing_move", "forced_reply", "promotion"):
                    data["evidence_level"] = "legally_actionable"
                elif priority in ("new_en_prise", "pinned_defender"):
                    data["evidence_level"] = "exchange_validated"
                else:
                    data["evidence_level"] = "pure_geometry"
        return data


class TacticalSnapshot(BaseModel):
    side_to_move: Literal["white", "black"]
    checks: list[ForcingMoveEvidence] = Field(default_factory=list)
    captures: list[ForcingMoveEvidence] = Field(default_factory=list)
    loose_pieces: list[PieceEvidence] = Field(default_factory=list)
    undefended_pieces: list[PieceEvidence] = Field(default_factory=list)
    attacked_undefended_pieces: list[PieceEvidence] = Field(default_factory=list)
    tactically_loose_pieces: list[PieceEvidence] = Field(default_factory=list)
    attacked_pieces: list[PieceEvidence] = Field(default_factory=list)
    en_prise_pieces: list[PieceEvidence] = Field(default_factory=list)
    pinned_pieces: list[PieceEvidence] = Field(default_factory=list)
    tactically_hanging_candidates: list[TacticalHangingEvidence] = Field(default_factory=list)
    attacked_defenders: list[DefenderLoadEvidence] = Field(default_factory=list)
    overloaded_defender_candidates: list[DefenderLoadEvidence] = Field(default_factory=list)
    mechanism_candidates: list[MechanismCandidateEvidence] = Field(default_factory=list)
    # F-010 fix (2026-09-09): the geometry-only mechanism list can dominate
    # payload while contributing little coaching value. ``presentation_mechanisms``
    # is the same data re-ranked by concrete consequence priority (mate >
    # check > capture > promotion > forced reply > en prise > PV-supported >
    # pure geometry). The full ``mechanism_candidates`` list is still emitted
    # for tooling that wants the exhaustive evidence.
    presentation_mechanisms: list[MechanismCandidateEvidence] = Field(default_factory=list)
    opponent_forcing_threats_if_pass: list[ForcingMoveEvidence] = Field(default_factory=list)
    threat_probe_available: bool = False
    threat_probe_reason: str | None = None
    threat_probe_scope: str | None = None


class StrongestReplyEvidence(BaseModel):
    uci: str
    san: str
    is_check: bool = False
    is_capture: bool = False
    is_forcing: bool = False
    captured_piece: str | None = None
    resulting_fen: str
    eval_after_reply_cp: int | None = None
    eval_after_reply_mate: int | None = None
    searched_depth: int | None = None


class PieceSafetyDelta(BaseModel):
    target: str
    attackers_before: int
    attackers_after: int
    defenders_before: int
    defenders_after: int


class PieceMobilityDelta(BaseModel):
    target: str
    mobility_before: int
    mobility_after: int
    gained_squares: list[str] = Field(default_factory=list)
    lost_squares: list[str] = Field(default_factory=list)


class SquareControlDelta(BaseModel):
    square: str
    white_attackers_before: int
    white_attackers_after: int
    black_attackers_before: int
    black_attackers_after: int


class PositionDelta(BaseModel):
    material_delta_white: int = 0
    material_delta_black: int = 0
    removed_pieces: list[str] = Field(default_factory=list)
    added_pieces: list[str] = Field(default_factory=list)
    newly_loose_pieces: list[str] = Field(default_factory=list)
    newly_en_prise_pieces: list[str] = Field(default_factory=list)
    resolved_en_prise_pieces: list[str] = Field(default_factory=list)
    newly_pinned_pieces: list[str] = Field(default_factory=list)
    removed_pins: list[str] = Field(default_factory=list)
    piece_safety_changes: list[PieceSafetyDelta] = Field(default_factory=list)
    piece_mobility_changes: list[PieceMobilityDelta] = Field(default_factory=list)
    strategic_square_control_changes: list[SquareControlDelta] = Field(default_factory=list)
    opened_files: list[str] = Field(default_factory=list)
    closed_files: list[str] = Field(default_factory=list)
    pawn_structure_changes: list[str] = Field(default_factory=list)
    king_ring_attack_delta_white: int = 0
    king_ring_attack_delta_black: int = 0
    check_state_changed: bool = False


CandidateContinuationTermination = Literal[
    "invalid_candidate_move",
    "forced_mate_in_returned_pv",
    "terminal_position",
    "material_resolution",
    "quiet_position",
    "invalid_pv_move",
    "pv_exhausted",
    "no_pv",
]


class CandidateContinuationEndpointEvidence(BaseModel):
    """Board state reached by walking one candidate's already returned PV."""

    plies_from_candidate: int = 0
    plies_from_root: int = 0
    termination_reason: CandidateContinuationTermination
    tactical_sequence_resolved: bool = False
    returned_pv_plies_available: int = 0
    endpoint_fen: str | None = None
    endpoint_position: PositionFingerprint | None = None
    endpoint_tactical_snapshot: TacticalSnapshot | None = None
    root_to_endpoint_delta: PositionDelta | None = None
    irreversible_events: list[dict[str, Any]] = Field(default_factory=list)
    proof_scope: str


class CandidateContinuationEndpointDifference(BaseModel):
    """Difference between two evidence-bounded candidate PV endpoints."""

    available: bool
    reason: str | None = None
    reference_endpoint_fen: str | None = None
    candidate_endpoint_fen: str | None = None
    reference_plies_from_root: int | None = None
    candidate_plies_from_root: int | None = None
    reference_termination_reason: CandidateContinuationTermination | None = None
    candidate_termination_reason: CandidateContinuationTermination | None = None
    reference_tactical_sequence_resolved: bool | None = None
    candidate_tactical_sequence_resolved: bool | None = None
    material_effect_difference_for_mover_cp: int | None = None
    only_reference_newly_en_prise: list[str] = Field(default_factory=list)
    only_candidate_newly_en_prise: list[str] = Field(default_factory=list)
    only_reference_newly_pinned: list[str] = Field(default_factory=list)
    only_candidate_newly_pinned: list[str] = Field(default_factory=list)
    only_reference_opened_files: list[str] = Field(default_factory=list)
    only_candidate_opened_files: list[str] = Field(default_factory=list)
    only_reference_pawn_structure_changes: list[str] = Field(default_factory=list)
    only_candidate_pawn_structure_changes: list[str] = Field(default_factory=list)
    king_ring_attack_delta_difference_white: int = 0
    king_ring_attack_delta_difference_black: int = 0
    reference_irreversible_events: list[str] = Field(default_factory=list)
    candidate_irreversible_events: list[str] = Field(default_factory=list)
    proof_scope: str = (
        "Compares evidence-bounded principal-variation endpoints. Unequal continuation "
        "lengths or termination reasons are not a controlled causal experiment."
    )


class CandidateEvidence(BaseModel):
    requested: str
    uci: str
    san: str
    resulting_fen: str
    eval_cp: int | None = None
    eval_mate: int | None = None
    searched_depth: int | None = None
    search_provenance: dict[str, Any] | None = None
    opponent_best_reply: StrongestReplyEvidence | None = None
    tactical_snapshot_after: TacticalSnapshot
    continuation_uci: list[str] = Field(default_factory=list)
    continuation_san: list[str] = Field(default_factory=list)
    continuation_termination_reason: Literal[
        "terminal_position",
        "pv_exhausted",
        "invalid_pv_move",
        "no_pv",
    ] = "no_pv"
    continuation_tactical_sequence_resolved: bool = False
    continuation_proof_status: Literal["principal_variation_only"] = "principal_variation_only"
    position_after: PositionFingerprint | None = None
    position_delta: PositionDelta | None = None
    position_after_reply: PositionFingerprint | None = None
    tactical_after_reply: TacticalSnapshot | None = None
    reply_delta: PositionDelta | None = None
    continuation_endpoint: CandidateContinuationEndpointEvidence | None = None


class CandidatePositionDifference(BaseModel):
    """Compare a candidate's resulting position with the engine reference move.

    Positive ``eval_gap_candidate_minus_reference_for_mover_cp`` means the
    candidate is better than the reference from the root mover's perspective;
    negative means worse. Feature-only lists describe differences between the
    two resulting positions, not causal proof of why the evaluation differs.
    """

    reference_uci: str
    reference_san: str
    candidate_uci: str
    candidate_san: str
    first_divergence_ply: int = 1
    first_divergence: dict[str, str] = Field(default_factory=dict)
    eval_gap_candidate_minus_reference_for_mover_cp: int | None = None
    material_effect_difference_for_mover_cp: int = 0
    reference_reply_is_forcing: bool | None = None
    candidate_reply_is_forcing: bool | None = None
    reference_root_move_irreversible: bool = False
    candidate_root_move_irreversible: bool = False
    reference_root_irreversible_reasons: list[str] = Field(default_factory=list)
    candidate_root_irreversible_reasons: list[str] = Field(default_factory=list)
    only_reference_newly_en_prise: list[str] = Field(default_factory=list)
    only_candidate_newly_en_prise: list[str] = Field(default_factory=list)
    only_reference_newly_pinned: list[str] = Field(default_factory=list)
    only_candidate_newly_pinned: list[str] = Field(default_factory=list)
    only_reference_opened_files: list[str] = Field(default_factory=list)
    only_candidate_opened_files: list[str] = Field(default_factory=list)
    only_reference_pawn_structure_changes: list[str] = Field(default_factory=list)
    only_candidate_pawn_structure_changes: list[str] = Field(default_factory=list)
    only_reference_piece_safety_changes: list[str] = Field(default_factory=list)
    only_candidate_piece_safety_changes: list[str] = Field(default_factory=list)
    only_reference_piece_mobility_changes: list[str] = Field(default_factory=list)
    only_candidate_piece_mobility_changes: list[str] = Field(default_factory=list)
    only_reference_strategic_square_control_changes: list[str] = Field(default_factory=list)
    only_candidate_strategic_square_control_changes: list[str] = Field(default_factory=list)
    only_reference_mechanism_candidates: list[str] = Field(default_factory=list)
    only_candidate_mechanism_candidates: list[str] = Field(default_factory=list)
    only_reference_immediate_reply_forcing_moves: list[str] = Field(default_factory=list)
    only_candidate_immediate_reply_forcing_moves: list[str] = Field(default_factory=list)
    only_reference_root_forcing_threats_if_reply_passes: list[str] = Field(default_factory=list)
    only_candidate_root_forcing_threats_if_reply_passes: list[str] = Field(default_factory=list)
    king_ring_attack_delta_difference_white: int = 0
    king_ring_attack_delta_difference_black: int = 0
    continuation_endpoint_difference: CandidateContinuationEndpointDifference | None = None
    proof_scope: str = (
        "Resulting-position comparison only. Feature differences are deterministic; "
        "they do not by themselves prove which feature caused the engine-evaluation gap."
    )


class ForcedLineEvidence(BaseModel):
    uci: list[str] = Field(default_factory=list)
    san: list[str] = Field(default_factory=list)
    termination_reason: Literal[
        "terminal_position",
        "pv_exhausted",
        "invalid_pv_move",
        "no_pv",
    ] = "no_pv"
    tactical_sequence_resolved: bool = False
    proof_status: Literal["principal_variation_only"] = "principal_variation_only"


class DefenseEvidence(BaseModel):
    rank: int
    uci: str
    san: str
    is_check: bool = False
    is_capture: bool = False
    captured_piece: str | None = None
    resulting_fen: str
    eval_cp: int | None = None
    eval_mate: int | None = None
    searched_depth: int | None = None
    continuation: ForcedLineEvidence


class TacticalProofEvidence(BaseModel):
    mode: Literal["tactical"] = "tactical"
    root_move_uci: str
    root_move_san: str
    root_resulting_fen: str
    root_margin_effective_cp: int | None = None
    proof_status: Literal[
        "exhaustive",
        "sampled_top_defenses",
        "terminal_after_root",
        "principal_variation_only",
    ]
    legal_defense_count: int
    analyzed_defense_count: int
    defenses: list[DefenseEvidence] = Field(default_factory=list)
    inference_boundary: str = (
        "A sampled proof covers only the returned engine-ranked defenses. "
        "Only proof_status=exhaustive means every legal immediate reply was evaluated."
    )


class PositionForensicEvidence(BaseModel):
    detail: Literal["coach", "forensic"]
    position: PositionFingerprint
    tactical_snapshot: TacticalSnapshot
    best_move_uci: str | None = None
    best_move_san: str | None = None
    position_after_best: PositionFingerprint | None = None
    tactical_after_best: TacticalSnapshot | None = None
    best_move_delta: PositionDelta | None = None
    inference_boundary: str = (
        "Static board evidence is deterministic. Tactical-hanging, motif and "
        "overloaded-defender fields have explicit bounded proof scopes and are not "
        "human-process claims."
    )


class ForensicEval(MCPEval):
    """Backward-compatible ``evaluate_position`` result with opt-in evidence."""

    forensics: PositionForensicEvidence | None = None


class TopMovesForensicEvidence(BaseModel):
    detail: Literal["coach", "forensic"]
    position: PositionFingerprint
    tactical_snapshot: TacticalSnapshot
    candidate_comparisons: list[CandidateEvidence] = Field(default_factory=list)
    candidate_differences: list[CandidatePositionDifference] = Field(default_factory=list)
    proof: TacticalProofEvidence | None = None


class ForensicTopMovesResult(TopMovesResult):
    """Backward-compatible ``top_moves`` result with opt-in evidence."""

    forensics: TopMovesForensicEvidence | None = None
    requested_proof_defenses: int | None = None
    clamped_proof_defenses: int | None = None
    forensic_compute_triggered_by: list[str] = Field(default_factory=list)



class ForensicEvidence(BaseModel):
    detail: Literal["coach", "forensic"]
    position_before: PositionFingerprint
    position_after_played: PositionFingerprint
    tactical_before: TacticalSnapshot
    tactical_after_played: TacticalSnapshot
    strongest_reply: StrongestReplyEvidence | None = None
    position_after_reply: PositionFingerprint | None = None
    tactical_after_reply: TacticalSnapshot | None = None
    reply_delta: PositionDelta | None = None
    position_delta: PositionDelta
    mechanism_evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_signatures: list[str] = Field(default_factory=list)
    forced_line: ForcedLineEvidence
    candidate_comparisons: list[CandidateEvidence] = Field(default_factory=list)
    candidate_differences: list[CandidatePositionDifference] = Field(default_factory=list)
    stability: dict[str, Any] = Field(default_factory=dict)
    inference_boundary: str = (
        "Evidence signatures describe board/engine facts. Human thought-process labels "
        "must be inferred by the coaching layer, ideally with the player's self-report."
    )


class ForensicMoveAnalysis(MCPMoveAnalysis):
    """Backward-compatible ``classify_move`` result with opt-in evidence."""

    forensics: ForensicEvidence | None = None
