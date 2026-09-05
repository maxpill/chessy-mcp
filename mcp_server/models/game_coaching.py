"""Structured coaching evidence for full-game analysis.

The models in this module describe engine/board evidence and event selection.
They intentionally stop short of assigning psychological labels such as
"incomplete CCT". The coaching layer can make that inference after combining
these facts with player self-report.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from mcp_server.models.legacy import GameAnalysisResult


class GameSegment(BaseModel):
    start_ply: int
    end_ply: int
    perspective: Literal["white", "black"]
    state: str
    eval_start_effective_cp: int
    eval_end_effective_cp: int
    transition_cause_ply: int | None = None


class AdvantageEvent(BaseModel):
    ply: int
    san: str
    side: Literal["white", "black"]
    perspective: Literal["white", "black"]
    kind: Literal[
        "gained_advantage",
        "lost_advantage",
        "fell_behind",
        "recovered",
        "missed_recovery",
        "missed_conversion",
    ]
    before_effective_cp: int
    after_effective_cp: int
    evidence: dict[str, Any] = Field(default_factory=dict)


class CriticalMoment(BaseModel):
    ply: int
    san: str
    uci: str
    side: Literal["white", "black"]
    move_class: str
    centipawn_loss: int | None = None
    effective_loss: int | None = None
    eval_before_effective_cp: int
    eval_after_effective_cp: int
    best_move_san: str | None = None
    reasons: list[str] = Field(default_factory=list)
    importance_score: float = 0.0
    user_comment_raw: str | None = None
    verification_depth: int | None = None
    verified_move_class: str | None = None
    verified_effective_loss: int | None = None
    classification_stable: bool | None = None
    candidate_gap_effective_cp: int | None = None
    resource_uniqueness: Literal["unknown", "low", "medium", "high"] = "unknown"
    strongest_reply_uci: str | None = None
    strongest_reply_san: str | None = None
    strongest_reply_is_check: bool | None = None
    strongest_reply_is_capture: bool | None = None
    evidence_signatures: list[str] = Field(default_factory=list)


class PositiveMoment(BaseModel):
    ply: int
    san: str
    uci: str
    side: Literal["white", "black"]
    eval_before_effective_cp: int
    eval_after_effective_cp: int
    reason: Literal[
        "best_engine_move_under_pressure",
        "held_difficult_position",
        "converted_advantage_without_slippage",
        "unique_resource",
    ]
    candidate_gap_effective_cp: int | None = None
    resource_uniqueness: Literal["unknown", "low", "medium", "high"] = "unknown"


class RootCauseLink(BaseModel):
    root_cause_ply: int
    materialization_ply: int
    root_cause_san: str
    materialization_san: str
    affected_side: Literal["white", "black"]
    material_swing_cp: int
    plies_later: int
    basis: Literal["first_adverse_material_change_within_6_plies"] = (
        "first_adverse_material_change_within_6_plies"
    )


class FinalPositionAssessment(BaseModel):
    perspective: Literal["white", "black"]
    position_terminal_by_rules: bool
    checkmate: bool = False
    stalemate: bool = False
    forced_mate: bool = False
    mate_distance: int | None = None
    effective_cp: int
    wdl: tuple[int, int, int] | None = None
    side_to_move: Literal["white", "black"]
    legal_move_count: int
    best_move_uci: str | None = None
    best_move_san: str | None = None
    defensive_resources_exist: bool
    reasonable_resource_count: int | None = None
    verification_depth: int | None = None


class GameCoachingEvidence(BaseModel):
    detail: Literal["coach", "forensic"]
    perspective: Literal["white", "black"]
    critical_moments: list[CriticalMoment] = Field(default_factory=list)
    game_segments: list[GameSegment] = Field(default_factory=list)
    advantage_events: list[AdvantageEvent] = Field(default_factory=list)
    positive_moments: list[PositiveMoment] = Field(default_factory=list)
    root_cause_links: list[RootCauseLink] = Field(default_factory=list)
    final_position: FinalPositionAssessment
    scan_depth: int
    verification_depth: int | None = None
    adaptive_escalation_depth: int | None = None
    inference_boundary: str = (
        "Events and signatures are board/engine evidence. Labels for the player's "
        "thought process must be inferred by the coaching layer, ideally with self-report."
    )


class ForensicGameAnalysisResult(GameAnalysisResult):
    """Backward-compatible ``analyze_game`` result with opt-in coaching evidence."""

    coaching: GameCoachingEvidence | None = None
