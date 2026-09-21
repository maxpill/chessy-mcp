"""``ocr_to_pgn`` MCP tool v2.

One tool call per score-sheet photo: the caller passes the image as one
of three sources (``base64`` / ``url`` / ``file_uri``) and the tool
returns a canonical PGN plus the seven PGN headers, per-ply
uncertainties, and validation evidence.

Pipeline:

    1. Resolve image bytes via :mod:`mcp_server.ocr.image_source`.
    2. Decide preprocessing strategy:
         - ``mode="auto"``  (default) — CLAHE + auto-rotation always;
                                 denoise if image SNR < threshold.
         - ``mode="explicit"`` — caller controls the three flags.
    3. POST to the chess-ocr sidecar with ``extract_headers=True`` and
       optional ``metadata_hints`` (caller-supplied header overrides).
    4. Aggregate per-cell SAN candidates (sidecar does this already).
    5. Rerank via legal-sequence beam search
       (:mod:`mcp_server.ocr.beam_search`). Engine tiebreak optional
       and budgeted.
    6. Build canonical PGN with detected/caller-supplied headers.
    7. Replay through python-chess for final validation.
    8. Compose the :class:`OcrPgnResult` with verbosity-gated evidence.

The default verbosity is ``"minimal"`` — only ``status``, ``canonical_pgn``,
``confidence``, ``validation``, ``uncertainties``, ``metadata`` are
populated; forensic evidence (``candidates``, ``moves``, ``raw_ocr_text``)
stays ``None`` until the caller asks for ``"compact"`` / ``"full"``.
"""

from __future__ import annotations

import io
import logging
import math
import time
from typing import Annotated, Any, Literal

import chess
import chess.pgn

from mcp.types import ToolAnnotations
from pydantic import Field, TypeAdapter

from mcp_server._mcp import mcp
from mcp_server.contracts.errors import InvalidArgument, InvalidInput
from mcp_server.metrics import metrics
from mcp_server.models import (
    ImageSource,
    OcrCandidate,
    OcrHeaderField,
    OcrMetadataHints,
    OcrMoveEntry,
    OcrPgnResult,
    OcrUncertainty,
    OcrValidation,
    PreprocessingHints,
)
from mcp_server.ocr import (
    OCRAuthError,
    OCRClient,
    OCRImageTooLarge,
    OCRResponseError,
    OCRTimeout,
    OCRUnreachable,
    OCRUnavailable,
    OCRUnsupportedFormat,
)
from mcp_server.ocr.beam_search import (
    CellCandidate,
    beam_rescore,
)
from mcp_server.ocr.image_source import ResolvedImage, resolve_image
from mcp_server.parsers.san_normalize import normalize_pgn
from mcp_server.tools._common import _tool_error


log = logging.getLogger("chessy_mcp.ocr_to_pgn")


# Pre-built TypeAdapters for the discriminated union payloads.
_IMAGE_ADAPTER = TypeAdapter(ImageSource)
_METADATA_ADAPTER = TypeAdapter(OcrMetadataHints | None)
_PREPROCESSING_ADAPTER = TypeAdapter(PreprocessingHints | None)


_CLIENT_SINGLETON: OCRClient | None = None


def _client() -> OCRClient:
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is None:
        _CLIENT_SINGLETON = OCRClient()
    return _CLIENT_SINGLETON


def reset_singletons_for_tests(
    client: OCRClient | None = None,
) -> None:
    """Test hook: replace module-level OCRClient singleton."""
    global _CLIENT_SINGLETON
    _CLIENT_SINGLETON = client


# --- preprocessing strategy -------------------------------------------------

SNR_DENOISE_THRESHOLD: float = 18.0  # grayscale stdev below this → denoise


def _estimate_snr(image_bytes: bytes) -> float:
    """Estimate image SNR as the standard deviation of the grayscale pixel histogram.

    Cheap O(N) loop on the first few thousand bytes is enough to
    decide whether to enable the bilateral filter. Returns 100.0
    (very high) when we can't decode the bytes (e.g. HEIC without
    pillow-heif installed) so the caller never denoises by accident.
    """
    try:
        from PIL import Image, ImageOps  # local import keeps tests fast

        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.grayscale(img)
        # Downsample for speed.
        img.thumbnail((256, 256))
        hist = img.histogram()
        total = sum(hist)
        if total == 0:
            return 100.0
        mean = sum(i * c for i, c in enumerate(hist)) / total
        variance = sum((i - mean) ** 2 * c for i, c in enumerate(hist)) / total
        return math.sqrt(variance)
    except Exception:
        return 100.0


def _decide_preprocessing(
    mode: Literal["auto", "explicit"],
    hints: PreprocessingHints | None,
    image_bytes: bytes,
) -> tuple[dict[str, bool], str]:
    """Return ``(flags, strategy_label)``.

    ``flags`` is what gets forwarded to the sidecar's
    ``OCRRequest``. ``strategy_label`` is a short string the caller
    can echo back so they know which path ran.
    """
    if mode == "explicit":
        if hints is None:
            hints = PreprocessingHints()
        return (
            {
                "enhance_contrast": hints.enhance_contrast,
                "verify_with_rotation": hints.verify_with_rotation,
                "denoise": hints.denoise,
            },
            "explicit",
        )

    snr = _estimate_snr(image_bytes)
    denoise = snr < SNR_DENOISE_THRESHOLD
    return (
        {
            "enhance_contrast": True,
            "verify_with_rotation": True,
            "denoise": denoise,
        },
        f"auto_clahe_rot_denoise{int(denoise)}_snr{snr:.1f}",
    )


# --- header merge ----------------------------------------------------------


def _merge_header_field(
    detected_value: str | None,
    detected_confidence: float,
    hint_value: str | None,
) -> OcrHeaderField:
    """Build one :class:`OcrHeaderField` from sidecar + caller-hint pair."""
    if hint_value:
        return OcrHeaderField(
            value=hint_value,
            confidence=max(detected_confidence, 0.85),
            source="hint",
        )
    return OcrHeaderField(
        value=detected_value,
        confidence=detected_confidence,
        source="detected",
    )


# --- validation ------------------------------------------------------------


def _validate_path(
    selected_path: tuple[str, ...],
) -> tuple[chess.Board | None, list[OcrMoveEntry], bool]:
    """Replay the selected path through python-chess.

    Returns ``(final_board, move_entries, result_consistent)``. The last
    flag is True iff the movetext ends with one of the canonical result
    tokens AND that token matches the final board state.
    """
    board = chess.Board()
    moves: list[OcrMoveEntry] = []
    if not selected_path:
        return board, moves, False
    for idx, san in enumerate(selected_path, start=1):
        try:
            move = board.parse_san(san)
            canonical = board.san(move)
        except (chess.AmbiguousMoveError, chess.InvalidMoveError, ValueError, IndexError) as exc:
            log.warning("ply %d (%s) failed validation: %s", idx, san, exc)
            return None, moves, False
        moves.append(
            OcrMoveEntry(
                ply=idx,
                raw_token=san,
                canonical_token=canonical,
                parsed_ok=True,
                parser_warning=None,
                normalization_kind="none",
            )
        )
        board.push(move)
        if board.is_game_over(claim_draw=False):
            break

    result_consistent = (
        board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material()
    )
    return board, moves, result_consistent


# --- the tool --------------------------------------------------------------


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=False))
async def ocr_to_pgn(  # pyright: ignore[reportGeneralTypeIssues]
    image: Annotated[
        dict[str, Any],
        Field(
            description=(
                "Image source as a discriminated-union object keyed by `kind`. "
                'Forms: {"kind":"base64","data":"<base64>"}, '
                '{"kind":"url","url":"https://..."}, '
                '{"kind":"file_uri","path":"/abs/path.jpg"}. '
                "Exactly one form is required."
            )
        ),
    ],
    source_language: Annotated[
        Literal["auto", "en", "pl"],
        Field(
            description=(
                'Language hint for the notation. "auto" autodetects Polish vs English '
                '(Polish-distinctive letters H/W/G/S, ":" capture marker, "0-0" castling). '
                "When autodetect confidence is low, M3 is asked a second opinion."
            )
        ),
    ] = "auto",
    metadata: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Optional caller-supplied PGN header overrides. Fields set here "
                "win over OCR detection."
            )
        ),
    ] = None,
    mode: Annotated[
        Literal["auto", "explicit"],
        Field(
            description=(
                'Preprocessing strategy. "auto" (default) applies CLAHE + '
                "auto-rotation always and the bilateral denoise when the "
                "image SNR is below threshold. "
                '"explicit" uses the ``preprocessing`` field verbatim.'
            )
        ),
    ] = "auto",
    preprocessing: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Preprocessing flags. Only honored when ``mode='explicit'``. Ignored otherwise."
            )
        ),
    ] = None,
    resolve_ambiguities: Annotated[
        Literal["strict", "auto", "best_effort"],
        Field(
            description=(
                '"strict" raises on any ambiguous ply (needs caller retry); '
                '"auto" (default) returns needs_review when ambiguity cannot be '
                'resolved silently; "best_effort" always returns the best path '
                "even when low confidence."
            )
        ),
    ] = "auto",
    engine_plausibility: Annotated[
        Literal["off", "tiebreak_only"],
        Field(
            description=(
                'When "tiebreak_only" (default), Stockfish is consulted as a '
                "weak tiebreaker between visually similar candidates whose "
                "OCR scores are within 0.15 of each other. "
                'When "off", the engine is never invoked. Stockfish is NEVER '
                "used to invalidate a move solely on the basis of a bad eval."
            )
        ),
    ] = "tiebreak_only",
    verbosity: Annotated[
        Literal["minimal", "compact", "full"],
        Field(
            description=(
                'Response verbosity. "minimal" (default) returns only '
                "{status, canonical_pgn, confidence, validation, uncertainties, metadata}. "
                '"compact" adds detected language + warnings. '
                '"full" returns the full forensic payload (every OCR candidate, '
                "per-move evidence, raw OCR text)."
            )
        ),
    ] = "minimal",
    strict: Annotated[
        bool,
        Field(
            description=(
                "Legacy flag: when True, reject any ply whose canonical SAN "
                "required a semantic normalization (e.g. Polish capture colon). "
                "Lenient mode (default) records the normalization and continues."
            )
        ),
    ] = False,
) -> OcrPgnResult:
    """OCR a chess score-sheet image and return a validated canonical PGN."""
    t0 = time.time()
    tool_name = "ocr_to_pgn"
    warnings: list[str] = []

    # ---- Step 0: validate the discriminated-union payloads ----
    try:
        image_validated = _IMAGE_ADAPTER.validate_python(image)
        metadata_validated = _METADATA_ADAPTER.validate_python(metadata)
        preprocessing_validated = _PREPROCESSING_ADAPTER.validate_python(preprocessing)
    except Exception as exc:
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, is_error=True)
        raise _tool_error("invalid_argument", str(exc), tool_name) from exc

    # ---- Step 1: resolve image bytes ----
    try:
        resolved: ResolvedImage = await _resolve(image_validated)
    except (InvalidInput, InvalidArgument) as exc:
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, is_error=True)
        raise _tool_error(exc.code, str(exc), tool_name) from exc

    image_bytes = resolved.bytes

    # ---- Step 2: decide preprocessing strategy ----
    pre_flags, strategy_label = _decide_preprocessing(mode, preprocessing_validated, image_bytes)
    if mode == "explicit" and pre_flags == {
        "enhance_contrast": False,
        "verify_with_rotation": False,
        "denoise": False,
    }:
        warnings.append("explicit_mode_all_preprocessing_off")

    # ---- Step 3: call sidecar ----
    hint_language: str | None = None if source_language == "auto" else source_language
    metadata_hints_dict: dict[str, str] | None = (
        metadata_validated.model_dump(exclude_none=True) if metadata_validated else None
    )

    try:
        sidecar_payload = await _call_sidecar(
            image_bytes,
            hint_language,
            verify_with_rotation=pre_flags["verify_with_rotation"],
            enhance_contrast=pre_flags["enhance_contrast"],
            denoise=pre_flags["denoise"],
            metadata_hints=metadata_hints_dict,
        )
    except OCRUnreachable as exc:
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, is_error=True)
        raise _tool_error("ocr_unreachable", str(exc), tool_name) from exc
    except OCRTimeout as exc:
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, is_error=True)
        raise _tool_error("ocr_timeout", str(exc), tool_name) from exc
    except OCRImageTooLarge as exc:
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, is_error=True)
        raise _tool_error("ocr_image_too_large", str(exc), tool_name) from exc
    except OCRUnsupportedFormat as exc:
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, is_error=True)
        raise _tool_error("invalid_argument", str(exc), tool_name) from exc
    except OCRAuthError as exc:
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, is_error=True)
        raise _tool_error("ocr_auth_unavailable", str(exc), tool_name) from exc
    except (OCRUnavailable, OCRResponseError) as exc:
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, is_error=True)
        raise _tool_error("ocr_sidecar_error", str(exc), tool_name) from exc

    # ---- Step 4: extract raw text + language detection + normalization ----
    raw_ocr_text = sidecar_payload.get("raw_text", "") or ""
    norm_result = normalize_pgn(raw_ocr_text, language=source_language)
    canonical_text = norm_result.canonical_text
    detected_language = norm_result.detected_language
    detected_language_confidence = norm_result.detected_language_confidence
    normalization_changes = list(norm_result.normalization_changes)

    if strict and any(":→x" in c for c in normalization_changes):
        raise _tool_error(
            "strict_validation_error",
            "Polish-style colon capture marker detected; pass strict=False or normalize upstream.",
            tool_name,
        )

    # ---- Step 5: build cell candidates + run beam search ----
    raw_cell_candidates = sidecar_payload.get("cell_candidates", []) or []
    cell_candidates: list[CellCandidate] = [
        CellCandidate(
            ply=cc.get("ply", 0),
            side=cc.get("side", "white"),
            san=cc.get("san", ""),
            score=float(cc.get("score", 0.0) or 0.0),
        )
        for cc in raw_cell_candidates
    ]

    beam_result = beam_rescore(
        cell_candidates,
        beam_width=15,
        resolve=resolve_ambiguities,
        engine_plausibility=engine_plausibility,
        engine_eval=None,  # wired to Stockfish pool when caller opts in
        expand_visual=True,
    )

    # ---- Step 6: build canonical PGN ----
    headers_dict = sidecar_payload.get("headers", {}) or {}
    detected_headers = {
        name: (
            headers_dict[name].get("value"),
            float(headers_dict[name].get("confidence", 0.0) or 0.0),
        )
        for name in ("white", "black", "round", "date", "event", "site", "result")
        if name in headers_dict
    }
    metadata_out: dict[str, OcrHeaderField] = {}
    hint_dict = metadata_validated.model_dump(exclude_none=True) if metadata_validated else {}
    for field_name in ("white", "black", "round", "date", "event", "site", "result"):
        detected_value, detected_conf = detected_headers.get(field_name, (None, 0.0))
        hint_value = hint_dict.get(field_name)
        metadata_out[field_name] = _merge_header_field(detected_value, detected_conf, hint_value)

    canonical_pgn = _compose_pgn(metadata_out, canonical_text, beam_result.selected_path)

    # ---- Step 7: final validation through python-chess ----
    final_board, move_entries, result_consistent = _validate_path(beam_result.selected_path)
    plies = len(beam_result.selected_path)
    final_fen = final_board.fen() if final_board is not None else ""
    legal = final_board is not None

    validation = OcrValidation(
        legal=legal,
        all_moves_legal=beam_result.all_moves_legal and legal,
        unique_legal_path=beam_result.unique_legal_path,
        plies=plies,
        final_fen=final_fen,
        result_consistent=result_consistent,
        preprocessing_strategy=strategy_label,
    )

    # ---- Step 8: compose response with verbosity gating ----
    uncertainties: list[OcrUncertainty] = []
    cell_crops = sidecar_payload.get("cell_crops", {}) or {}
    for u in beam_result.uncertainties:
        crop_b64 = cell_crops.get(str(u.ply)) or cell_crops.get(u.ply)
        uncertainties.append(
            OcrUncertainty(
                ply=u.ply,
                side=u.side,
                selected=u.selected,
                selected_confidence=u.selected_confidence,
                sequence_confidence=u.sequence_confidence,
                alternatives=list(u.alternatives),
                reason=u.reason,
                crop_base64=crop_b64 if (verbosity == "full" or beam_result.status == "needs_review") else None,
            )
        )

    status: Literal["ok", "needs_review"] = beam_result.status
    if resolve_ambiguities == "strict" and beam_result.status == "needs_review":
        status = "needs_review"

    candidates = (
        [
            OcrCandidate(
                pass_label=c.get("pass_label", ""),
                language_hint=c.get("language_hint"),
                text=c.get("text", ""),
                error=c.get("error"),
                parse_rate=c.get("parse_rate", 0.0),
                score=c.get("score", 0.0),
                latency_ms=c.get("latency_ms", 0.0),
                rotation=c.get("rotation"),
            )
            for c in sidecar_payload.get("candidates", [])
        ]
        if verbosity in {"compact", "full"}
        else None
    )
    selected_candidate_index = (
        int(sidecar_payload.get("selected_candidate_index", 0))
        if verbosity in {"compact", "full"}
        else None
    )
    verifier_agreement = (
        sidecar_payload.get("verifier_agreement") if verbosity in {"compact", "full"} else None
    )
    verifier_notes = (
        sidecar_payload.get("verifier_notes") if verbosity in {"compact", "full"} else None
    )
    verifier_confidence = (
        sidecar_payload.get("verifier_confidence") if verbosity in {"compact", "full"} else None
    )

    response = OcrPgnResult(
        status=status,
        canonical_pgn=canonical_pgn,
        confidence=beam_result.confidence,
        validation=validation,
        uncertainties=uncertainties,
        metadata=metadata_out,
        detected_language=detected_language if verbosity != "minimal" else None,
        detected_language_confidence=(
            detected_language_confidence if verbosity != "minimal" else None
        ),
        raw_ocr_text=raw_ocr_text if verbosity == "full" else None,
        normalization_changes=normalization_changes if verbosity == "full" else None,
        moves=move_entries if verbosity == "full" else None,
        candidates=candidates,
        selected_candidate_index=selected_candidate_index,
        verifier_agreement=verifier_agreement,
        verifier_notes=verifier_notes,
        verifier_confidence=verifier_confidence,
        warnings=warnings or None,
        ocr_engine_used=sidecar_payload.get("ocr_engine_used", "minimax"),
        ocr_model_version=sidecar_payload.get("ocr_model_version", "minimax/MiniMax-M3"),
        ocr_pass_count=int(sidecar_payload.get("pass_count", 1)),
        auto_rotation_applied=int(sidecar_payload.get("auto_rotation_applied", 0)),
        request_duration_ms=(time.time() - t0) * 1000.0,
    )

    await metrics.record(tool_name, response.request_duration_ms, cache_hit=False)
    return response


# --- helpers ----------------------------------------------------------------


async def _resolve(image: ImageSource) -> ResolvedImage:
    """Dispatch :func:`resolve_image` based on the discriminated union tag."""
    if image.kind == "base64":
        return await resolve_image(base64=image.data, url=None, file_uri=None, attachment_id=None)
    if image.kind == "url":
        return await resolve_image(
            base64=None,
            url=image.url,
            file_uri=None,
            attachment_id=None,
            timeout_s=image.timeout_s,
        )
    if image.kind == "file_uri":
        return await resolve_image(
            base64=None, url=None, file_uri=image.path, attachment_id=None
        )
    if image.kind == "attachment":
        return await resolve_image(
            base64=None, url=None, file_uri=None, attachment_id=image.file_id
        )
    raise AssertionError(f"unhandled image.kind={image.kind!r}")


async def _call_sidecar(
    image_bytes: bytes,
    hint_language: str | None,
    *,
    verify_with_rotation: bool,
    enhance_contrast: bool,
    denoise: bool,
    metadata_hints: dict[str, str] | None,
) -> dict[str, Any]:
    """Call the OCR sidecar (no caching — every request is fresh)."""
    client = _client()
    return await client.ocr(
        image_bytes,
        hint_language=hint_language,
        verify_with_rotation=verify_with_rotation,
        enhance_contrast=enhance_contrast,
        denoise=denoise,
        metadata_hints=metadata_hints,
    )


def _compose_pgn(
    headers: dict[str, OcrHeaderField],
    canonical_movetext: str,
    selected_path: tuple[str, ...],
) -> str:
    """Compose the final canonical PGN.

    Header values come from the merged (detected + hint) set; missing
    values fall back to ``"?"``. Movetext is the selected beam path,
    not the raw OCR movetext — the beam reranker may have changed
    individual SANs (e.g. Polish ``S:f3`` → ``Nxf3``).
    """
    header_order = ("event", "site", "date", "round", "white", "black", "result")
    parts: list[str] = []
    for name in header_order:
        field = headers.get(name)
        value = field.value if field and field.value else "?"
        parts.append(f'[{name.capitalize()} "{value}"]')
    body_moves = " ".join(selected_path) if selected_path else canonical_movetext.strip()
    parts.append("")
    parts.append(body_moves)
    return "\n".join(parts).strip()
