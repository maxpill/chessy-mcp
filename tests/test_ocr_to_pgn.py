"""End-to-end tests for the ``ocr_to_pgn`` MCP tool.

Mocks the chess-ocr sidecar via respx and exercises the full pipeline:
image decode → sidecar call → consensus → language detect →
polish normalization → python-chess validation → response.
"""

from __future__ import annotations

import base64
import io

import chess
import chess.pgn
import pytest
import respx

from mcp_server.models import OcrPgnResult
from mcp_server.ocr import OCRClient
from mcp_server.tools import ocr_to_pgn as ocr_module
from mcp_server.tools.ocr_to_pgn import ocr_to_pgn


def _fake_jpeg_bytes() -> bytes:
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def _sidecar_response(
    text: str = "1.e4 e5 2.Sf3 Sc6 1-0",
    language: str = "pl",
    auto_rotation_applied: int = 0,
    pass_count: int = 2,
    pass_label: str = "raw",
    rotation: str | None = None,
) -> dict:
    return {
        "raw_text": text,
        "canonical_text": text,
        "detected_language": language,
        "candidates": [
            {
                "pass_label": pass_label,
                "language_hint": None,
                "text": text,
                "error": None,
                "parse_rate": 1.0,
                "score": 1.0,
                "latency_ms": 200.0,
                "rotation": rotation,
            },
        ],
        "selected_candidate_index": 0,
        "verifier_agreement": True,
        "verifier_notes": "Best match.",
        "verifier_confidence": 0.95,
        "ocr_engine_used": "minimax",
        "ocr_model_version": "minimax/MiniMax-M3",
        "pass_count": pass_count,
        "total_latency_ms": 1500.0,
        "auto_rotation_applied": auto_rotation_applied,
    }


def _install_mock_sidecar() -> respx.MockRouter:
    """Install a fresh OCR client bound to respx."""
    client = OCRClient(url="http://ocr.test:9552", timeout_s=5.0)
    ocr_module.reset_singletons_for_tests(client=client)
    return respx.mock(assert_all_called=False)


# --- happy paths ------------------------------------------------------------


@pytest.mark.asyncio
async def test_ocr_to_pgn_polish_image_round_trip() -> None:
    """A Polish score sheet photo round-trips through OCR + normalize + validation."""
    router = _install_mock_sidecar()
    image_b64 = base64.b64encode(_fake_jpeg_bytes()).decode("ascii")

    with router:
        route = router.post("http://ocr.test:9552/ocr").respond(
            200, json=_sidecar_response("1.e4 e5 2.Sf3 Sc6 3.Gb5 a6 1-0", language="pl")
        )
        result = await ocr_to_pgn(image_b64=image_b64, source_language="auto")

    assert isinstance(result, OcrPgnResult)
    assert result.detected_language == "pl"
    assert "Nf3" in result.canonical_pgn
    assert "Nc6" in result.canonical_pgn
    assert "Bb5" in result.canonical_pgn
    wrapped = '[Event "?"]\n\n' + result.canonical_pgn
    game = chess.pgn.read_game(io.StringIO(wrapped))
    assert game is not None
    assert len(list(game.mainline_moves())) >= 6
    assert route.call_count == 1
    assert result.cache_hit is False  # caching removed
    assert result.auto_rotation_applied == 0


@pytest.mark.asyncio
async def test_ocr_to_pgn_english_image() -> None:
    router = _install_mock_sidecar()
    image_b64 = base64.b64encode(_fake_jpeg_bytes()).decode("ascii")

    with router:
        route = router.post("http://ocr.test:9552/ocr").respond(
            200, json=_sidecar_response("1.e4 e5 2.Nf3 Nc6 1-0", language="en")
        )
        result = await ocr_to_pgn(image_b64=image_b64)

    assert result.detected_language == "en"
    assert "Nf3" in result.canonical_pgn
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_ocr_to_pgn_verify_with_rotation_forwards_flag() -> None:
    """verify_with_rotation=True should set the flag in the sidecar request body."""
    router = _install_mock_sidecar()
    image_b64 = base64.b64encode(_fake_jpeg_bytes()).decode("ascii")

    with router:
        route = router.post("http://ocr.test:9552/ocr").respond(
            200, json=_sidecar_response(auto_rotation_applied=90)
        )
        result = await ocr_to_pgn(
            image_b64=image_b64,
            verify_with_rotation=True,
        )

    assert route.call_count == 1
    sent_body = route.calls.last.request.content.decode("ascii")
    assert "verify_with_rotation" in sent_body
    assert result.auto_rotation_applied == 90


@pytest.mark.asyncio
async def test_ocr_to_pgn_enhance_contrast_forwards_flag() -> None:
    router = _install_mock_sidecar()
    image_b64 = base64.b64encode(_fake_jpeg_bytes()).decode("ascii")

    with router:
        route = router.post("http://ocr.test:9552/ocr").respond(200, json=_sidecar_response())
        await ocr_to_pgn(image_b64=image_b64, enhance_contrast=True)

    sent_body = route.calls.last.request.content.decode("ascii")
    assert "enhance_contrast" in sent_body


@pytest.mark.asyncio
async def test_ocr_to_pgn_denoise_forwards_flag() -> None:
    router = _install_mock_sidecar()
    image_b64 = base64.b64encode(_fake_jpeg_bytes()).decode("ascii")

    with router:
        route = router.post("http://ocr.test:9552/ocr").respond(200, json=_sidecar_response())
        await ocr_to_pgn(image_b64=image_b64, denoise=True)

    sent_body = route.calls.last.request.content.decode("ascii")
    assert "denoise" in sent_body


# --- input validation -------------------------------------------------------


@pytest.mark.asyncio
async def test_ocr_to_pgn_invalid_base64_raises_tool_error() -> None:
    _install_mock_sidecar()
    with pytest.raises(Exception, match="INVALID_INPUT"):
        await ocr_to_pgn(image_b64="@@@not-base64@@@")


@pytest.mark.asyncio
async def test_ocr_to_pgn_unsupported_format_raises_tool_error() -> None:
    _install_mock_sidecar()
    bad_bytes = b"this is not an image at all"
    image_b64 = base64.b64encode(bad_bytes).decode("ascii")
    with pytest.raises(Exception, match="INVALID_ARGUMENT"):
        await ocr_to_pgn(image_b64=image_b64)


# --- sidecar error mapping --------------------------------------------------


@pytest.mark.asyncio
async def test_ocr_to_pgn_sidecar_unreachable() -> None:
    import httpx

    router = _install_mock_sidecar()
    image_b64 = base64.b64encode(_fake_jpeg_bytes()).decode("ascii")

    with router:
        router.post("http://ocr.test:9552/ocr").mock(side_effect=httpx.ConnectError("nope"))
        with pytest.raises(Exception, match=r"ocr_unreachable|UNREACHABLE"):
            await ocr_to_pgn(image_b64=image_b64)


@pytest.mark.asyncio
async def test_ocr_to_pgn_sidecar_503_maps_to_auth_error() -> None:
    router = _install_mock_sidecar()
    image_b64 = base64.b64encode(_fake_jpeg_bytes()).decode("ascii")

    with router:
        router.post("http://ocr.test:9552/ocr").respond(503, text="MISSING_API_KEY")
        with pytest.raises(Exception, match=r"ocr_auth|API_KEY"):
            await ocr_to_pgn(image_b64=image_b64)
