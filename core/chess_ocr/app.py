"""FastAPI app for the chess-ocr sidecar.

Endpoints:
    POST /ocr   — multi-pass OCR. Request body:
                    {
                      "image_b64": str,         # base64-encoded image
                      "hint_language": "en"|"pl"|null
                    }
                  Response body:
                    ConsensusResult JSON.
    GET  /health — liveness probe (no auth).
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.chess_ocr.config import OCRSettings
from core.chess_ocr.consensus import ConsensusOrchestrator
from core.chess_ocr.preprocess import detect_image_format


log = logging.getLogger("chessy_mcp.chess_ocr.app")


class OCRRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image (HEIC/JPEG/PNG/WebP/GIF)")
    hint_language: str | None = Field(
        default=None,
        description='Optional language hint: "en" or "pl"',
    )
    # Opt-in preprocessing flags. Each defaults to off so behaviour matches
    # today's single-orientation OCR pipeline out of the box.
    verify_with_rotation: bool | None = Field(
        default=None,
        description=(
            "If True, fire 4 M3 passes at each of 3 orientations (12 total) "
            "and let the verifier pick the upright one. Costs ~3x more API "
            "calls but recovers sideways photos (camera held landscape + "
            "EXIF stripped)."
        ),
    )
    enhance_contrast: bool | None = Field(
        default=None,
        description=(
            "If True, apply CLAHE (tile-based adaptive histogram equalisation) "
            "before sending to M3. Lifts shadows while preserving pen-stroke "
            "gradients."
        ),
    )
    denoise: bool | None = Field(
        default=None,
        description=(
            "If True, apply a bilateral filter (edge-preserving denoise) before sending to M3."
        ),
    )
    # v2 additions
    extract_headers: bool = Field(
        default=True,
        description=(
            "If True, also run a structured header-extraction pass over the "
            "score sheet (White / Black / Round / Date / Event / Site / Result). "
            "Default ON — adds one M3 call but lets callers skip a separate "
            "OCR pass for the metadata block."
        ),
    )
    beam_per_cell: bool = Field(
        default=True,
        description=(
            "If True, aggregate per-cell SAN candidates across all OCR passes "
            "and emit ``cell_candidates`` in the response. The MCP ``ocr_to_pgn`` "
            "tool uses this for the v2 legal-sequence beam search."
        ),
    )
    metadata_hints: dict[str, str] | None = Field(
        default=None,
        description=(
            'Optional caller-supplied header hints (e.g. {"white": "X"}). '
            "Used as a cross-check by the header-extraction prompt; final "
            "merging is done in the MCP tool."
        ),
    )


class OCRCandidateResponse(BaseModel):
    pass_label: str
    language_hint: str | None
    text: str
    error: str | None = None
    parse_rate: float = 0.0
    score: float = 0.0
    latency_ms: float = 0.0
    rotation: str | None = None  # rotation label of the variant, e.g. "rot0"/"rot90"/"rot-90"


class OcrCellCandidateResponse(BaseModel):
    """One candidate SAN at one half-move position (v2 beam-search input)."""

    ply: int
    side: str  # "white" | "black"
    san: str
    score: float


class OcrHeaderFieldResponse(BaseModel):
    """One PGN header field extracted from the score sheet (v2)."""

    value: str | None
    confidence: float


class OCRResponse(BaseModel):
    raw_text: str
    canonical_text: str
    detected_language: str
    candidates: list[OCRCandidateResponse]
    selected_candidate_index: int
    verifier_agreement: bool | None
    verifier_notes: str | None
    verifier_confidence: float | None
    ocr_engine_used: str
    ocr_model_version: str
    pass_count: int
    total_latency_ms: float
    auto_rotation_applied: int = 0  # degrees CCW the auto-detector chose (0 = none)
    # v2 additions — all optional with safe defaults so existing callers keep working.
    cell_candidates: list[OcrCellCandidateResponse] = Field(default_factory=list)
    headers: dict[str, OcrHeaderFieldResponse] = Field(default_factory=dict)


_settings: OCRSettings | None = None
_orchestrator: ConsensusOrchestrator | None = None


def get_settings() -> OCRSettings:
    global _settings
    if _settings is None:
        _settings = OCRSettings()
    return _settings


def get_orchestrator() -> ConsensusOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ConsensusOrchestrator(get_settings())
    return _orchestrator


def reset_singletons_for_tests(
    settings: OCRSettings | None = None,
    orchestrator: ConsensusOrchestrator | None = None,
) -> None:
    """Test hook: replace module singletons."""
    global _settings, _orchestrator
    _settings = settings
    _orchestrator = orchestrator


def create_app() -> FastAPI:
    app = FastAPI(
        title="chessy-mcp chess-ocr sidecar",
        version="0.1.0",
        description="Multi-pass M3 OCR for chess score sheets.",
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        settings = get_settings()
        return {
            "status": "ok",
            "service": "chessy-mcp-chess-ocr",
            "model": settings.model,
            "engine": settings.engine,
        }

    @app.post("/ocr", response_model=OCRResponse)
    async def ocr(req: OCRRequest) -> OCRResponse:
        settings = get_settings()
        if not settings.api_key:
            raise HTTPException(
                status_code=503,
                detail="MISSING_API_KEY: env var `n` must be set",
            )

        try:
            image_bytes = base64.b64decode(req.image_b64, validate=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"INVALID_BASE64: {exc}") from exc

        if len(image_bytes) > settings.max_image_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"IMAGE_TOO_LARGE: decoded {len(image_bytes)} bytes > cap {settings.max_image_bytes}",
            )

        fmt = detect_image_format(image_bytes)
        if fmt == "unknown":
            raise HTTPException(
                status_code=415,
                detail="UNSUPPORTED_IMAGE_FORMAT: must be HEIC/JPEG/PNG/WebP/GIF",
            )

        orchestrator = get_orchestrator()
        try:
            result = await orchestrator.run(
                image_bytes,
                hint_language=req.hint_language,
                rotation_strategy="three" if req.verify_with_rotation else "auto",
                contrast_strategy="clahe" if req.enhance_contrast else "off",
                denoise_strategy="bilateral" if req.denoise else "off",
                extract_headers=req.extract_headers,
                beam_per_cell=req.beam_per_cell,
                metadata_hints=req.metadata_hints,
            )
        except ValueError as exc:
            msg = str(exc)
            if "OCR_TIMEOUT" in msg:
                raise HTTPException(status_code=504, detail=msg) from exc
            if "OCR_UNREACHABLE" in msg:
                raise HTTPException(status_code=502, detail=msg) from exc
            if "OCR_API_ERROR" in msg or "OCR_VERIFY_API_ERROR" in msg:
                raise HTTPException(status_code=502, detail=msg) from exc
            raise HTTPException(status_code=500, detail=msg) from exc

        return OCRResponse(
            raw_text=result.raw_text,
            canonical_text=result.canonical_text,
            detected_language=result.detected_language,
            candidates=[
                OCRCandidateResponse(
                    pass_label=c.get("pass_label", ""),
                    language_hint=c.get("language_hint"),
                    text=c.get("text", ""),
                    error=c.get("error"),
                    parse_rate=c.get("parse_rate", 0.0),
                    score=c.get("score", 0.0),
                    latency_ms=c.get("latency_ms", 0.0),
                    rotation=c.get("rotation"),
                )
                for c in result.candidates
            ],
            selected_candidate_index=result.selected_candidate_index,
            verifier_agreement=result.verifier_agreement,
            verifier_notes=result.verifier_notes,
            verifier_confidence=result.verifier_confidence,
            ocr_engine_used=result.ocr_engine_used,
            ocr_model_version=result.ocr_model_version,
            pass_count=result.pass_count,
            total_latency_ms=result.total_latency_ms,
            auto_rotation_applied=result.auto_rotation_applied,
            cell_candidates=[
                OcrCellCandidateResponse(
                    ply=cc.ply,
                    side=cc.side,
                    san=cc.san,
                    score=cc.score,
                )
                for cc in result.cell_candidates
            ],
            headers={
                name: OcrHeaderFieldResponse(value=h.value, confidence=h.confidence)
                for name, h in result.headers.items()
            },
        )

    return app


app = create_app()


__all__ = [
    "OCRRequest",
    "OCRResponse",
    "app",
    "create_app",
    "get_orchestrator",
    "get_settings",
    "reset_singletons_for_tests",
]
