"""Structured, evidence-first chess coaching models.

These models deliberately contain chess facts rather than natural-language
coaching conclusions.  The MCP server establishes what is on the board,
which forcing moves exist, what the engine's strongest reply is, and how
candidate positions differ.  A coach/LLM can then infer the human process
error without the engine service pretending to read the player's mind.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from mcp_server.models.legacy import MCPMoveAnalysis


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


class PieceEvidence(BaseModel):
    color: Literal["white", "black"]
    piece: str
    square: str
    attackers: int = 0
    defenders: int = 0


class TacticalSnapshot(BaseModel):
    side_to_move: Literal["white", "black"]
    checks: list[ForcingMoveEvidence] = Field(default_factory=list)
    captures: list[ForcingMoveEvidence] = Field(default_factory=list)
    loose_pieces: list[PieceEvidence] = Field(default_factory=list)
    en_prise_pieces: list[PieceEvidence] = Field(default_factory=list)
    pinned_pieces: list[PieceEvidence] = Field(default_factory=list)


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


class PositionDelta(BaseModel):
    material_delta_white: int = 0
    material_delta_black: int = 0
    removed_pieces: list[str] = Field(default_factory=list)
    added_pieces: list[str] = Field(default_factory=list)
    newly_loose_pieces: list[str] = Field(default_factory=list)
    newly_pinned_pieces: list[str] = Field(default_factory=list)
    check_state_changed: bool = False


class CandidateEvidence(BaseModel):
    requested: str
    uci: str
    san: str
    resulting_fen: str
    eval_cp: int | None = None
    eval_mate: int | None = None
    searched_depth: int | None = None
    opponent_best_reply: StrongestReplyEvidence | None = None
    tactical_snapshot_after: TacticalSnapshot


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


class ForensicEvidence(BaseModel):
    detail: Literal["coach", "forensic"]
    position_before: PositionFingerprint
    position_after_played: PositionFingerprint
    tactical_before: TacticalSnapshot
    tactical_after_played: TacticalSnapshot
    strongest_reply: StrongestReplyEvidence | None = None
    position_delta: PositionDelta
    mechanism_evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_signatures: list[str] = Field(default_factory=list)
    forced_line: ForcedLineEvidence
    candidate_comparisons: list[CandidateEvidence] = Field(default_factory=list)
    stability: dict[str, Any] = Field(default_factory=dict)
    inference_boundary: str = (
        "Evidence signatures describe board/engine facts. Human thought-process labels "
        "must be inferred by the coaching layer, ideally with the player's self-report."
    )


class ForensicMoveAnalysis(MCPMoveAnalysis):
    """Backward-compatible ``classify_move`` result with opt-in evidence."""

    forensics: ForensicEvidence
