"""Response / request models for the ``ocr_to_pgn`` MCP tool v2.

The v2 surface replaces the ``image_b64: str`` parameter with a
discriminated :class:`ImageSource` union so callers can supply:

    - inline base64 (preserves the original happy path),
    - a server-fetchable http(s) URL,
    - an absolute filesystem path (``file_uri``) within a sandboxed root.

It also adds:

    - :class:`OcrMetadataHints` — caller-supplied PGN header overrides.
    - :class:`PreprocessingHints` — opt-in preprocessing flags, only
      honored when ``mode="explicit"``.
    - :class:`OcrHeaderField` — value + confidence per PGN header.
    - :class:`OcrUncertainty` — one ambiguous half-move flagged by the
      legal-sequence beam reranker.
    - :class:`OcrValidation` — legality, ply count, FEN, result-consistent
      flag, preprocessing strategy summary.
    - :class:`OcrPgnResult` — the response payload, with verbosity-gated
      ``None`` fields in ``minimal`` / ``compact`` mode.

The old v1 fields (``raw_ocr_text``, ``candidates``, ``moves`` etc.)
remain on :class:`OcrPgnResult` but are ``None`` when ``verbosity`` is
``minimal`` (the new default), so the wire payload is small by default.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


# --- Image source union -----------------------------------------------------


class ImageBase64(BaseModel):
    """Inline base64-encoded image."""

    kind: Literal["base64"] = "base64"
    data: str = Field(..., description="Base64-encoded image bytes")
    mime_type: str | None = Field(
        default=None,
        description="Optional image mime type, e.g. 'image/jpeg', 'image/png'",
    )


class ImageUrl(BaseModel):
    """http(s) URL the server should fetch with httpx."""

    kind: Literal["url"] = "url"
    url: str = Field(..., description="http:// or https:// URL")
    timeout_s: float | None = Field(
        default=None,
        description="Per-fetch timeout in seconds. Defaults to 30.",
    )


class ImageFileUri(BaseModel):
    """Absolute filesystem path inside the sandboxed root (``CHESSY_MCP_FILE_ROOT``)."""

    kind: Literal["file_uri"] = "file_uri"
    path: str = Field(..., description="Absolute path; must resolve under CHESSY_MCP_FILE_ROOT")


class ImageAttachment(BaseModel):
    """Image file reference by attachment ID / file ID."""

    kind: Literal["attachment"] = "attachment"
    file_id: str = Field(..., description="File ID or attachment identifier")


ImageSource = Annotated[
    ImageBase64 | ImageUrl | ImageFileUri | ImageAttachment,
    Field(discriminator="kind"),
]


# --- Hints ------------------------------------------------------------------


class OcrMetadataHints(BaseModel):
    """Caller-supplied overrides for the seven PGN header fields.

    Any field set here overrides the auto-detected value in the
    response. Confidence is bumped to 0.85 to signal "caller supplied".
    """

    white: str | None = None
    black: str | None = None
    round: str | None = None
    date: str | None = None
    event: str | None = None
    site: str | None = None
    result: str | None = None


class PreprocessingHints(BaseModel):
    """Opt-in preprocessing flags, only honored when ``mode="explicit"``."""

    enhance_contrast: bool = False
    verify_with_rotation: bool = False
    denoise: bool = False


# --- Response value objects -------------------------------------------------


class OcrHeaderField(BaseModel):
    """One PGN header field with confidence and source."""

    value: str | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source: Literal["detected", "hint", "merged"] = "detected"


class OcrUncertainty(BaseModel):
    """One ambiguous half-move flagged by the legal-sequence beam reranker."""

    ply: int
    side: Literal["white", "black"]
    selected: str
    selected_confidence: float = Field(..., ge=0.0, le=1.0)
    sequence_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    alternatives: list[tuple[str, float]] = Field(default_factory=list)
    reason: str = "handwriting_ambiguity"
    crop_base64: str | None = None


class OcrValidation(BaseModel):
    """Game-level validation result."""

    legal: bool
    all_moves_legal: bool = True
    unique_legal_path: bool = True
    plies: int = 0
    final_fen: str = ""
    result_consistent: bool = False
    preprocessing_strategy: str = ""


# --- Verbose evidence (verbosity=full only) ---------------------------------


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


# --- Top-level response -----------------------------------------------------


class OcrPgnResult(BaseModel):
    """Response payload for the ``ocr_to_pgn`` MCP tool v2."""

    # Top-level signal — always present.
    status: Literal["ok", "needs_review"]
    canonical_pgn: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    validation: OcrValidation
    uncertainties: list[OcrUncertainty] = Field(default_factory=list)
    metadata: dict[str, OcrHeaderField] = Field(default_factory=dict)

    # Verbosity-gated evidence. ``None`` when suppressed.
    detected_language: str | None = None
    detected_language_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_ocr_text: str | None = None
    normalization_changes: list[str] | None = None
    moves: list[OcrMoveEntry] | None = None
    candidates: list[OcrCandidate] | None = None
    selected_candidate_index: int | None = None
    verifier_agreement: bool | None = None
    verifier_notes: str | None = None
    verifier_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    warnings: list[str] | None = None

    # Always present (diagnostics).
    ocr_engine_used: str
    ocr_model_version: str
    ocr_pass_count: int
    auto_rotation_applied: int = 0
    request_duration_ms: float


__all__ = [
    "ImageAttachment",
    "ImageBase64",
    "ImageFileUri",
    "ImageSource",
    "ImageUrl",
    "OcrCandidate",
    "OcrHeaderField",
    "OcrMetadataHints",
    "OcrMoveEntry",
    "OcrPgnResult",
    "OcrUncertainty",
    "OcrValidation",
    "PreprocessingHints",
]
