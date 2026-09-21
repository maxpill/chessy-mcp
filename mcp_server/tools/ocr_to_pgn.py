"""``ocr_to_pgn`` MCP tool.

Takes a HEIC/JPEG/PNG/WebP image of a printed or handwritten chess score
sheet, runs multi-pass M3 OCR via the chess-ocr sidecar, auto-detects
Polish vs English notation, normalizes to canonical English SAN,
validates every move with python-chess, optionally cross-checks each
ply with Stockfish, and returns the validated PGN plus per-move
evidence.

Opt-in preprocessing flags enable multi-orientation rotation handling
(``verify_with_rotation``), tile-based contrast enhancement
(``enhance_contrast``) and edge-preserving denoise (``denoise``). All
default to False to preserve today's behaviour for existing callers.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Annotated, Literal

import chess
import chess.pgn
from pydantic import Field

from mcp.types import ToolAnnotations

from mcp_server._mcp import mcp
from mcp_server.contracts.errors import (
    InvalidArgument,
    InvalidInput,
)
from mcp_server.metrics import metrics
from mcp_server.models import OcrPgnResult, OcrCandidate, OcrMoveEntry
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
from mcp_server.parsers.san_normalize import normalize_pgn
from mcp_server.tools._common import _tool_error


log = logging.getLogger("chessy_mcp.ocr_to_pgn")


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


_MAGIC_JPEG = b"\xff\xd8\xff"
_MAGIC_PNG = b"\x89PNG\r\n\x1a\n"
_MAGIC_HEIC_FTYP = b"ftyp"

_VALID_IMAGE_FORMATS = ("jpeg", "png", "webp", "gif", "heic")
_MAX_IMAGE_BYTES = 32 * 1024 * 1024


def _decode_image_b64(image_b64: str) -> bytes:
    """Decode base64 image data and validate the magic bytes."""
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except Exception as exc:
        raise InvalidInput(f"INVALID_BASE64: {exc}") from exc

    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise InvalidInput(
            f"IMAGE_TOO_LARGE: decoded {len(image_bytes)} bytes > cap {_MAX_IMAGE_BYTES}"
        )

    fmt = _detect_format(image_bytes)
    if fmt not in _VALID_IMAGE_FORMATS:
        raise InvalidArgument(f"UNSUPPORTED_IMAGE_FORMAT: must be one of {_VALID_IMAGE_FORMATS}")
    return image_bytes


def _detect_format(image_bytes: bytes) -> str:
    if image_bytes.startswith(_MAGIC_JPEG):
        return "jpeg"
    if image_bytes.startswith(_MAGIC_PNG):
        return "png"
    if image_bytes.startswith(b"GIF"):
        return "gif"
    if image_bytes.startswith(b"RIFF"):
        return "webp"
    if len(image_bytes) >= 12 and image_bytes[4:9] == _MAGIC_HEIC_FTYP:
        brand = image_bytes[8:12]
        if brand.startswith(b"heic") or brand.startswith(b"mif1") or brand.startswith(b"heim"):
            return "heic"
    return "unknown"


async def _call_sidecar(
    image_bytes: bytes,
    hint_language: str | None,
    *,
    verify_with_rotation: bool,
    enhance_contrast: bool,
    denoise: bool,
) -> dict:
    """Call the OCR sidecar (no caching — every request is fresh)."""
    client = _client()
    return await client.ocr(
        image_bytes,
        hint_language=hint_language,
        verify_with_rotation=verify_with_rotation,
        enhance_contrast=enhance_contrast,
        denoise=denoise,
    )


def _validate_ply_by_ply(
    canonical_text: str,
) -> tuple[chess.Board | None, list[OcrMoveEntry], list[str]]:
    """Replay the canonical PGN move-by-move and collect per-ply evidence."""
    board: chess.Board | None = None
    moves: list[OcrMoveEntry] = []
    warnings: list[str] = []

    if not canonical_text or not canonical_text.strip():
        return None, moves, ["empty_canonical_text"]

    wrapped = (
        canonical_text
        if canonical_text.lstrip().startswith("[")
        else '[Event "?"]\n\n' + canonical_text
    )
    try:
        game = chess.pgn.read_game(io.StringIO(wrapped))
    except Exception as exc:
        return None, moves, [f"invalid_pgn: {exc}"]

    if game is None:
        return None, moves, ["python_chess_could_not_parse"]

    board = game.board()
    for ply_idx, move in enumerate(game.mainline_moves(), start=1):
        canonical_san = board.san(move)
        raw_token = canonical_san
        entry = OcrMoveEntry(
            ply=ply_idx,
            raw_token=raw_token,
            canonical_token=canonical_san,
            parsed_ok=True,
            parser_warning=None,
            normalization_kind="none",
        )
        moves.append(entry)
        board.push(move)
        if board.is_game_over(claim_draw=False):
            break

    return board, moves, warnings


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=False))
async def ocr_to_pgn(  # pyright: ignore[reportGeneralTypeIssues]
    image_b64: Annotated[
        str,
        Field(
            description=(
                "Base64-encoded image of a chess score sheet. "
                "Supported formats: HEIC, JPEG, PNG, WebP, GIF. Max 32 MB decoded."
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
    strict: Annotated[
        bool,
        Field(
            description=(
                "When True, reject any ply whose canonical SAN required a semantic "
                "normalization (capture-marker correction, check-marker correction, etc.). "
                "Lenient mode (default) records the normalization in the response and continues."
            )
        ),
    ] = False,
    verify_with_stockfish: Annotated[
        bool,
        Field(
            description=(
                "When True, run a single-ply Stockfish sanity check on every accepted "
                "move at depth 10. Adds ~5-10s per 60-ply game. Useful for catching OCR "
                "transcription errors that look like legal moves but are huge blunders."
            )
        ),
    ] = False,
    verify_with_rotation: Annotated[
        bool,
        Field(
            description=(
                "When True, fire M3 against 3 orientations (0°/90° CW/90° CCW) and "
                "let the verifier pick the upright variant. Adds ~3x the API cost but "
                "recovers photos shot sideways (camera landscape + EXIF stripped)."
            )
        ),
    ] = False,
    enhance_contrast: Annotated[
        bool,
        Field(
            description=(
                "When True, apply CLAHE (tile-based adaptive histogram equalisation) "
                "before sending to M3. Lifts shadows without removing pen-stroke gradients."
            )
        ),
    ] = False,
    denoise: Annotated[
        bool,
        Field(
            description=(
                "When True, apply a bilateral filter (edge-preserving denoise) before "
                "sending to M3. Helps noisy phone photos."
            )
        ),
    ] = False,
    verbosity: Annotated[
        Literal["minimal", "compact", "full", "min", "standard", "default"] | None,
        Field(description='Response verbosity: "full" (default), "compact", or "minimal".'),
    ] = None,
) -> OcrPgnResult:
    """OCR a chess score-sheet image and return a validated canonical PGN.

    The pipeline:
      1. Decode + magic-byte check the input image.
      2. Optional preprocessing (orientation/CLAHE/bilateral) via opt-in flags.
      3. Send it to the chess-ocr sidecar (multi-pass M3 with verifier).
      4. Auto-detect Polish vs English notation.
      5. Normalize to canonical English SAN via the san_normalize module.
      6. Replay ply-by-ply via python-chess for per-move validation.
      7. Optionally cross-check each ply with Stockfish (depth 10).

    Returns an :class:`OcrPgnResult` with the canonical PGN, per-move
    evidence, every OCR candidate, the verifier verdict, and cache-hit /
    rotation metadata.
    """
    t0 = time.time()
    tool_name = "ocr_to_pgn"

    try:
        image_bytes = _decode_image_b64(image_b64)
    except (InvalidInput, InvalidArgument) as exc:
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, is_error=True)
        raise _tool_error(exc.code, str(exc), tool_name) from exc

    hint_language = None if source_language == "auto" else source_language

    try:
        sidecar_payload = await _call_sidecar(
            image_bytes,
            hint_language,
            verify_with_rotation=verify_with_rotation,
            enhance_contrast=enhance_contrast,
            denoise=denoise,
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

    raw_ocr_text = sidecar_payload.get("raw_text", "") or ""
    norm_result = normalize_pgn(raw_ocr_text, language=source_language)
    canonical_text = norm_result.canonical_text
    detected_language = norm_result.detected_language
    detected_confidence = norm_result.detected_language_confidence
    normalization_changes = list(norm_result.normalization_changes)

    final_board, moves, validation_warnings = _validate_ply_by_ply(canonical_text)
    pgn_is_valid = final_board is not None and final_board.is_valid()
    final_fen = final_board.fen() if final_board is not None else ""
    final_move_number = (final_board.fullmove_number - 1) if final_board is not None else 0
    auto_rotation_applied = int(sidecar_payload.get("auto_rotation_applied", 0))

    if strict and any(":→x" in c for c in normalization_changes):
        raise _tool_error(
            "strict_validation_error",
            "Polish-style colon capture marker detected; pass strict=False or normalize upstream.",
            tool_name,
        )

    if verify_with_stockfish and final_board is not None:
        try:
            from mcp_server.engine import _get_analyzer_pool
            from mcp_server.engine.cached_evaluator import evaluate_game_position_cached

            pool = await _get_analyzer_pool(None)
            walk = chess.Board()
            for entry in moves:
                move = walk.parse_san(entry.canonical_token)
                mcp_eval, _ = await evaluate_game_position_cached(
                    walk,
                    depth=10,
                    pool=pool,
                )
                entry.stockfish_eval_cp = mcp_eval.cp if hasattr(mcp_eval, "cp") else None
                walk.push(move)
        except Exception as exc:
            validation_warnings.append(f"stockfish_verify_skipped: {exc}")

    candidates = [
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

    result = OcrPgnResult(
        canonical_pgn=canonical_text,
        raw_ocr_text=raw_ocr_text,
        detected_language=detected_language,
        detected_language_confidence=detected_confidence,
        normalization_changes=normalization_changes,
        pgn_is_valid=pgn_is_valid,
        final_fen=final_fen,
        final_move_number=final_move_number,
        moves=moves,
        warnings=validation_warnings,
        candidates=candidates,
        selected_candidate_index=int(sidecar_payload.get("selected_candidate_index", 0)),
        verifier_agreement=sidecar_payload.get("verifier_agreement"),
        verifier_notes=sidecar_payload.get("verifier_notes"),
        verifier_confidence=sidecar_payload.get("verifier_confidence"),
        ocr_engine_used=sidecar_payload.get("ocr_engine_used", "minimax"),
        ocr_model_version=sidecar_payload.get("ocr_model_version", "minimax/MiniMax-M3"),
        ocr_pass_count=int(sidecar_payload.get("pass_count", 1)),
        cache_hit=False,
        cache_key="",
        auto_rotation_applied=auto_rotation_applied,
        request_duration_ms=(time.time() - t0) * 1000.0,
    )

    await metrics.record(tool_name, result.request_duration_ms, cache_hit=False)
    return result
