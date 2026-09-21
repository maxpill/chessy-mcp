"""Tests for :mod:`mcp_server.ocr.image_source`."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest
import respx

from mcp_server.contracts.errors import InvalidArgument, InvalidInput
from mcp_server.ocr.image_source import MAX_IMAGE_BYTES, ResolvedImage, resolve_image


def _fake_jpeg_bytes() -> bytes:
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


# --- base64 source ----------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_image_base64_happy_path() -> None:
    data = _fake_jpeg_bytes()
    encoded = base64.b64encode(data).decode("ascii")

    result = await resolve_image(base64=encoded, url=None, file_uri=None)

    assert isinstance(result, ResolvedImage)
    assert result.bytes == data
    assert result.format == "jpeg"
    assert result.source_label == "base64"


@pytest.mark.asyncio
async def test_resolve_image_base64_invalid_payload() -> None:
    with pytest.raises(InvalidInput, match="INVALID_BASE64"):
        await resolve_image(base64="@@@not-base64@@@", url=None, file_uri=None)


@pytest.mark.asyncio
async def test_resolve_image_base64_unsupported_format() -> None:
    encoded = base64.b64encode(b"this is not an image at all").decode("ascii")
    with pytest.raises(InvalidArgument, match="UNSUPPORTED_IMAGE_FORMAT"):
        await resolve_image(base64=encoded, url=None, file_uri=None)


@pytest.mark.asyncio
async def test_resolve_image_base64_oversized() -> None:
    payload = b"\xff\xd8\xff" + b"\x00" * (MAX_IMAGE_BYTES + 1)
    encoded = base64.b64encode(payload).decode("ascii")
    with pytest.raises(InvalidInput, match="IMAGE_TOO_LARGE"):
        await resolve_image(base64=encoded, url=None, file_uri=None)


# --- url source -------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_image_url_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHESSY_MCP_URL_ALLOWLIST", "example.com")
    with respx.mock(assert_all_called=False) as router:
        route = router.get("http://example.com/score.jpg").respond(200, content=_fake_jpeg_bytes())
        result = await resolve_image(base64=None, url="http://example.com/score.jpg", file_uri=None)
    assert result.bytes == _fake_jpeg_bytes()
    assert result.format == "jpeg"
    assert result.source_label == "url:http://example.com/score.jpg"
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_resolve_image_url_default_allowlist_blocks_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHESSY_MCP_URL_ALLOWLIST", raising=False)
    with pytest.raises(InvalidArgument, match="URL_HOST_NOT_ALLOWED"):
        await resolve_image(base64=None, url="http://evil.example.com/score.jpg", file_uri=None)


@pytest.mark.asyncio
async def test_resolve_image_url_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHESSY_MCP_URL_ALLOWLIST", "example.com")
    with respx.mock(assert_all_called=False) as router:
        router.get("http://example.com/missing.jpg").respond(404, text="nope")
        with pytest.raises(InvalidInput, match="URL_FETCH_HTTP_404"):
            await resolve_image(base64=None, url="http://example.com/missing.jpg", file_uri=None)


@pytest.mark.asyncio
async def test_resolve_image_url_unsupported_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHESSY_MCP_URL_ALLOWLIST", "example.com")
    with pytest.raises(InvalidArgument, match="UNSUPPORTED_URL_SCHEME"):
        await resolve_image(base64=None, url="ftp://example.com/score.jpg", file_uri=None)


@pytest.mark.asyncio
async def test_resolve_image_url_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHESSY_MCP_URL_ALLOWLIST", "example.com")
    with respx.mock(assert_all_called=False) as router:
        router.get("http://example.com/slow.jpg").mock(side_effect=httpx.ConnectTimeout("slow"))
        with pytest.raises(InvalidInput, match=r"URL_FETCH_(TIMEOUT|FAILED)"):
            await resolve_image(
                base64=None,
                url="http://example.com/slow.jpg",
                file_uri=None,
                timeout_s=0.5,
            )


# --- file_uri source --------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_image_file_uri_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHESSY_MCP_FILE_ROOT", str(tmp_path))
    score_path = tmp_path / "score.jpg"
    score_path.write_bytes(_fake_jpeg_bytes())

    result = await resolve_image(base64=None, url=None, file_uri=str(score_path))

    assert result.bytes == _fake_jpeg_bytes()
    assert result.source_label == f"file_uri:{score_path}"


@pytest.mark.asyncio
async def test_resolve_image_file_uri_outside_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHESSY_MCP_FILE_ROOT", str(tmp_path))
    other = tmp_path.parent / "outside.jpg"
    other.write_bytes(_fake_jpeg_bytes())

    with pytest.raises(InvalidArgument, match="FILE_OUTSIDE_SANDBOX"):
        await resolve_image(base64=None, url=None, file_uri=str(other))


@pytest.mark.asyncio
async def test_resolve_image_file_uri_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHESSY_MCP_FILE_ROOT", str(tmp_path))
    with pytest.raises(InvalidInput, match="FILE_NOT_FOUND"):
        await resolve_image(base64=None, url=None, file_uri=str(tmp_path / "missing.jpg"))


# --- argument arbitration ---------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_image_requires_one_source() -> None:
    with pytest.raises(InvalidArgument, match="IMAGE_SOURCE_MISSING"):
        await resolve_image(base64=None, url=None, file_uri=None)


@pytest.mark.asyncio
async def test_resolve_image_rejects_two_sources() -> None:
    encoded = base64.b64encode(_fake_jpeg_bytes()).decode("ascii")
    with pytest.raises(InvalidArgument, match="IMAGE_SOURCE_AMBIGUOUS"):
        await resolve_image(base64=encoded, url="http://example.com/x.jpg", file_uri=None)
