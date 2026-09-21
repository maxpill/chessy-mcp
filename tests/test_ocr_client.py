"""Tests for the MCP-side OCR client."""

from __future__ import annotations

import httpx
import pytest
import respx

from mcp_server.ocr import (
    OCRAuthError,
    OCRClient,
    OCRImageTooLarge,
    OCRTimeout,
    OCRUnreachable,
    OCRUnavailable,
)


# --- client HTTP layer -------------------------------------------------------


@pytest.mark.asyncio
async def test_ocr_client_happy_path() -> None:
    client = OCRClient(url="http://ocr.test:9552", timeout_s=5.0)
    with respx.mock(assert_all_called=False) as router:
        route = router.post("http://ocr.test:9552/ocr").respond(
            200,
            json={
                "raw_text": "1.e4 e5 2.Nf3 Nc6 1-0",
                "canonical_text": "1.e4 e5 2.Nf3 Nc6 1-0",
                "detected_language": "en",
                "candidates": [],
                "selected_candidate_index": 0,
                "ocr_engine_used": "minimax",
                "ocr_model_version": "minimax/MiniMax-M3",
                "pass_count": 1,
                "total_latency_ms": 123.0,
            },
        )
        result = await client.ocr(b"\xff\xd8\xff\xe0fake-jpeg")

    assert result["raw_text"] == "1.e4 e5 2.Nf3 Nc6 1-0"
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_ocr_client_forwards_preprocessing_flags() -> None:
    """verify_with_rotation / enhance_contrast / denoise are forwarded in the JSON body."""
    client = OCRClient(url="http://ocr.test:9552", timeout_s=5.0)
    with respx.mock(assert_all_called=False) as router:
        route = router.post("http://ocr.test:9552/ocr").respond(
            200,
            json={
                "raw_text": "1.e4",
                "canonical_text": "1.e4",
                "detected_language": "en",
                "candidates": [],
                "selected_candidate_index": 0,
                "ocr_engine_used": "minimax",
                "ocr_model_version": "minimax/MiniMax-M3",
                "pass_count": 1,
                "total_latency_ms": 50.0,
            },
        )
        await client.ocr(
            b"\xff\xd8\xff",
            verify_with_rotation=True,
            enhance_contrast=True,
            denoise=True,
        )

    assert route.call_count == 1
    sent_body = route.calls.last.request.content.decode("ascii")
    assert "verify_with_rotation" in sent_body
    assert "enhance_contrast" in sent_body
    assert "denoise" in sent_body


@pytest.mark.asyncio
async def test_ocr_client_unreachable_raises_typed() -> None:
    client = OCRClient(url="http://ocr.test:9552", timeout_s=5.0)
    with respx.mock(assert_all_called=False) as router:
        router.post("http://ocr.test:9552/ocr").mock(side_effect=httpx.ConnectError("nope"))
        with pytest.raises(OCRUnreachable):
            await client.ocr(b"\xff\xd8\xff")


@pytest.mark.asyncio
async def test_ocr_client_timeout_raises_typed() -> None:
    client = OCRClient(url="http://ocr.test:9552", timeout_s=5.0)
    with respx.mock(assert_all_called=False) as router:
        router.post("http://ocr.test:9552/ocr").mock(side_effect=httpx.TimeoutException("slow"))
        with pytest.raises(OCRTimeout):
            await client.ocr(b"\xff\xd8\xff")


@pytest.mark.asyncio
async def test_ocr_client_503_maps_to_auth_error() -> None:
    client = OCRClient(url="http://ocr.test:9552", timeout_s=5.0)
    with respx.mock(assert_all_called=False) as router:
        router.post("http://ocr.test:9552/ocr").respond(503, text="MISSING_API_KEY")
        with pytest.raises(OCRAuthError):
            await client.ocr(b"\xff\xd8\xff")


@pytest.mark.asyncio
async def test_ocr_client_413_maps_to_image_too_large() -> None:
    client = OCRClient(url="http://ocr.test:9552", timeout_s=5.0)
    with respx.mock(assert_all_called=False) as router:
        router.post("http://ocr.test:9552/ocr").respond(413, text="too big")
        with pytest.raises(OCRImageTooLarge):
            await client.ocr(b"\xff\xd8\xff")


@pytest.mark.asyncio
async def test_ocr_client_5xx_maps_to_unavailable() -> None:
    client = OCRClient(url="http://ocr.test:9552", timeout_s=5.0)
    with respx.mock(assert_all_called=False) as router:
        router.post("http://ocr.test:9552/ocr").respond(502, text="bad gateway")
        with pytest.raises(OCRUnavailable):
            await client.ocr(b"\xff\xd8\xff")
