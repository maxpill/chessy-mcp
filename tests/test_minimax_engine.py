"""Tests for the M3 multimodal OCR engine.

Uses respx to mock the M3 HTTP endpoint. Verifies prompt construction,
image encoding, retry/timeout behavior, and the verifier pass.
"""

from __future__ import annotations

import base64

import httpx
import pytest
import respx

from core.chess_ocr.config import OCRSettings
from core.chess_ocr.engines.minimax import (
    MinimaxEngine,
    SYSTEM_PROMPT_RAW,
)


def _fake_settings() -> OCRSettings:
    return OCRSettings(
        api_key="sk-test-123", model="minimax/MiniMax-M3", api_base="https://api.test/v1"
    )


SECRET_PREFIX = "sk-cp-"  # used only in tests; not a real key


def _fake_jpeg_bytes() -> bytes:
    """Return a minimal valid 1x1 JPEG header."""
    # SOI + APP0 (JFIF) + EOI — python-chess doesn't need to decode, M3 does
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def _ok_response(text: str = "1.e4 e5 2.Nf3 Nc6 1-0") -> dict:
    return {
        "id": "test-1",
        "model": "minimax/MiniMax-M3",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def _ok_verifier_response() -> dict:
    return {
        "id": "test-2",
        "model": "minimax/MiniMax-M3",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"best_index": 0, "confidence": 0.85, "notes": "Best match."}',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 200, "completion_tokens": 20},
    }


@pytest.mark.asyncio
async def test_minimax_ocr_happy_path() -> None:
    """A single OCR pass returns the M3 message text."""
    settings = _fake_settings()
    engine = MinimaxEngine(settings)
    image = _fake_jpeg_bytes()

    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.test/v1/chat/completions").respond(
            200, json=_ok_response("1.e4 e5 2.Nf3 Nc6 1-0")
        )
        result = await engine.ocr(image)

    assert "1.e4 e5" in result["text"]
    assert "Nf3" in result["text"]
    assert result["model"] == "minimax/MiniMax-M3"
    assert result["latency_ms"] >= 0
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_minimax_ocr_includes_image_data_url() -> None:
    """The image is base64-encoded into a data URL in the request body."""
    settings = _fake_settings()
    engine = MinimaxEngine(settings)
    image = _fake_jpeg_bytes()

    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.test/v1/chat/completions").respond(
            200, json=_ok_response()
        )
        await engine.ocr(image)

    request_body = route.calls.last.request.content
    import json as _json

    payload = _json.loads(request_body)
    user_msg = payload["messages"][1]
    image_content = user_msg["content"][0]
    assert image_content["type"] == "image_url"
    url = image_content["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    # Decode and verify we get exactly the bytes we sent (HEIC path is
    # exercised in the dedicated HEIC test).
    b64 = url.split(",", 1)[1]
    assert base64.b64decode(b64) == image  # JPEG passthrough


@pytest.mark.asyncio
async def test_minimax_ocr_uses_language_hint_prompt() -> None:
    """When language_hint is provided, the Polish/English-specific prompt is used."""
    settings = _fake_settings()
    engine = MinimaxEngine(settings)
    image = _fake_jpeg_bytes()

    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.test/v1/chat/completions").respond(
            200, json=_ok_response("1.Ge5 Bf6 1-0")
        )
        await engine.ocr(image, language_hint="pl")

    import json as _json

    payload = _json.loads(route.calls.last.request.content)
    system_msg = payload["messages"][0]["content"]
    assert "Polish" in system_msg
    assert "H=Hetman" in system_msg or "G=Goniec" in system_msg


@pytest.mark.asyncio
async def test_minimax_ocr_missing_api_key_raises() -> None:
    """OCR refuses to run when env var ``n`` is empty."""
    settings = OCRSettings(api_key="", model="minimax/MiniMax-M3")
    engine = MinimaxEngine(settings)
    with pytest.raises(ValueError, match="MISSING_API_KEY"):
        await engine.ocr(_fake_jpeg_bytes())


@pytest.mark.asyncio
async def test_minimax_ocr_timeout_translates_to_value_error() -> None:
    """An M3 timeout surfaces as a structured ValueError, not a raw httpx exception."""
    settings = _fake_settings()
    engine = MinimaxEngine(settings)
    image = _fake_jpeg_bytes()

    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.test/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("simulated timeout")
        )
        with pytest.raises(ValueError, match="OCR_TIMEOUT"):
            await engine.ocr(image)


@pytest.mark.asyncio
async def test_minimax_ocr_5xx_surfaces_as_value_error() -> None:
    settings = _fake_settings()
    engine = MinimaxEngine(settings)
    image = _fake_jpeg_bytes()

    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.test/v1/chat/completions").respond(502, text="bad gateway")
        with pytest.raises(ValueError, match="OCR_API_ERROR"):
            await engine.ocr(image)


@pytest.mark.asyncio
async def test_minimax_verify_picks_best_candidate() -> None:
    settings = _fake_settings()
    engine = MinimaxEngine(settings)
    image = _fake_jpeg_bytes()
    candidates = ["1.e4 e5 2.Nf3", "1.Ge5 Bf6", "1.e4 c5 2.Nf3 d6"]

    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.test/v1/chat/completions").respond(
            200, json=_ok_verifier_response()
        )
        verdict = await engine.verify(image, candidates)

    assert verdict["best_index"] == 0
    assert verdict["confidence"] == 0.85
    assert "raw_verifier_text" in verdict
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_minimax_verify_handles_malformed_json() -> None:
    """A verifier that returns non-JSON falls back to best_index=0 with low confidence."""
    settings = _fake_settings()
    engine = MinimaxEngine(settings)
    image = _fake_jpeg_bytes()

    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.test/v1/chat/completions").respond(
            200, json=_ok_response("not json")
        )
        verdict = await engine.verify(image, ["a", "b", "c"])

    assert verdict["best_index"] == 0
    assert verdict["confidence"] == 0.5


@pytest.mark.asyncio
async def test_minimax_aclose_is_idempotent() -> None:
    """Calling aclose() multiple times doesn't raise."""
    engine = MinimaxEngine(_fake_settings())
    await engine.aclose()
    await engine.aclose()


def test_system_prompt_contains_no_secrets() -> None:
    """The system prompt must NOT contain the API key or any secret."""
    assert "sk-" not in SYSTEM_PROMPT_RAW
    assert "api_key" not in SYSTEM_PROMPT_RAW
    assert SECRET_PREFIX not in SYSTEM_PROMPT_RAW
