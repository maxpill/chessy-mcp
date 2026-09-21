"""Response models for the ``ocr_to_pgn`` MCP tool."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OcrCandidate(BaseModel):
    """One candidate PGN produced by a single OCR pass."""

    pass_label: str = Field(
        ..., description="E.g. 'raw', 'pl_hint', 'en_hint_1', 'verifier', 'sanity'"
    )
    language_hint: str | None = Field(None, description='"en", "pl", or null')
    text: str = ""
    error: str | None = None
    parse_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Fraction of plies python-chess parsed."
    )
    score: float = 0.0
    latency_ms: float = 0.0
    rotation: str | None = Field(
        default=None,
        description=(
            "Orientation variant the candidate was generated from, e.g. 'rot0', "
            "'rot90', 'rot-90'. Populated only when ``verify_with_rotation`` was used."
        ),
    )


class OcrMoveEntry(BaseModel):
    """Per-move evidence collected during python-chess validation."""

    ply: int
    raw_token: str
    canonical_token: str
    parsed_ok: bool
    parser_warning: str | None = None
    normalization_kind: Literal["none", "cosmetic", "notation_variant", "semantic"] = "none"
    stockfish_eval_cp: int | None = None


class OcrPgnResult(BaseModel):
    """Response payload for the ``ocr_to_pgn`` MCP tool."""

    canonical_pgn: str
    raw_ocr_text: str
    detected_language: str
    detected_language_confidence: float = Field(0.0, ge=0.0, le=1.0)
    normalization_changes: list[str] = Field(default_factory=list)
    pgn_is_valid: bool
    final_fen: str
    final_move_number: int = 0
    moves: list[OcrMoveEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    candidates: list[OcrCandidate] = Field(default_factory=list)
    selected_candidate_index: int = 0
    verifier_agreement: bool | None = None
    verifier_notes: str | None = None
    verifier_confidence: float | None = None
    ocr_engine_used: str
    ocr_model_version: str
    ocr_pass_count: int
    cache_hit: bool = False
    cache_key: str
    auto_rotation_applied: int = Field(
        default=0,
        description=(
            "Degrees CCW the auto-rotation detector chose. 0 means no rotation "
            "was needed. Only populated when the sidecar's auto-rotation path "
            "ran (i.e. verify_with_rotation=False or the request used rotation_strategy="
            '"auto").'
        ),
    )
    request_duration_ms: float


__all__ = ["OcrCandidate", "OcrMoveEntry", "OcrPgnResult"]
