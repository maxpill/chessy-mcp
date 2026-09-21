"""MCP-side HTTP client for the chess-ocr sidecar.

Calls ``POST /ocr`` on the sidecar (default ``http://chess-ocr:9552``)
and converts HTTP / network failures into typed exceptions the tool
layer maps onto structured ``ToolError`` codes.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from mcp_server.config import get_mcp_settings


log = logging.getLogger("chessy_mcp.ocr.client")


OCR_SIDECAR_DEFAULT_URL = "http://chess-ocr:9552"
OCR_SIDECAR_TIMEOUT_S = 180.0


class OCRError(Exception):
    """Base for every typed failure the tool layer maps onto a ToolError code."""


class OCRAuthError(OCRError):
    """503 / 401 — env var ``n`` not set or rejected."""


class OCRUnavailable(OCRError):
    """502 / 5xx — sidecar upstream failure (model provider rejected etc)."""


class OCRTimeout(OCRError):
    """504 — sidecar request exceeded 60s."""


class OCRUnreachable(OCRError):
    """Network-level failure (DNS, connection refused, TLS, etc.)."""


class OCRResponseError(OCRError):
    """200 but the body is not parseable."""


class OCRImageTooLarge(OCRError):
    """413 — decoded image exceeds the sidecar cap."""


class OCRUnsupportedFormat(OCRError):
    """415 — image format not recognized."""


class OCRClient:
    """Async HTTP client for the chess-ocr sidecar."""

    def __init__(
        self,
        url: str | None = None,
        timeout_s: float = OCR_SIDECAR_TIMEOUT_S,
    ) -> None:
        try:
            settings = get_mcp_settings()
            ocr_url = (
                getattr(settings, "ocr_sidecar_url", OCR_SIDECAR_DEFAULT_URL)
                or OCR_SIDECAR_DEFAULT_URL
            )
        except Exception:
            ocr_url = OCR_SIDECAR_DEFAULT_URL
        self._url = (url or ocr_url).rstrip("/")
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    @property
    def url(self) -> str:
        return self._url

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout_s),
                headers={"User-Agent": "chessy-mcp/0.1.0 (ocr client)"},
                http2=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ocr(
        self,
        image_bytes: bytes,
        hint_language: str | None = None,
        *,
        verify_with_rotation: bool = False,
        enhance_contrast: bool = False,
        denoise: bool = False,
        metadata_hints: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send an image to the OCR sidecar and return the parsed response body.

        ``verify_with_rotation``, ``enhance_contrast``, ``denoise`` are
        opt-in preprocessing flags forwarded to the sidecar. They only
        take effect when the sidecar's ``OCRRequest`` accepts them.

        ``metadata_hints`` (v2) are optional caller-supplied PGN header
        overrides (e.g. ``{"white": "X"}``) used as a cross-check by
        the sidecar's header-extraction prompt.
        """
        body = {
            "image_b64": base64.b64encode(image_bytes).decode("ascii"),
            "hint_language": hint_language,
            "verify_with_rotation": verify_with_rotation,
            "enhance_contrast": enhance_contrast,
            "denoise": denoise,
        }
        if metadata_hints:
            body["metadata_hints"] = metadata_hints
        url = self._url + "/ocr"
        client = await self._http()
        try:
            resp = await client.post(url, json=body)
        except httpx.TimeoutException as exc:
            raise OCRTimeout(f"OCR sidecar timed out after {self._timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise OCRUnreachable(f"OCR sidecar unreachable: {exc}") from exc

        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception as exc:
                raise OCRResponseError(f"OCR sidecar returned malformed JSON: {exc}") from exc
        if resp.status_code in (401, 503):
            raise OCRAuthError(
                f"OCR sidecar auth unavailable (HTTP {resp.status_code}): {resp.text[:200]}"
            )
        if resp.status_code == 413:
            raise OCRImageTooLarge(f"OCR sidecar: image too large (HTTP 413): {resp.text[:200]}")
        if resp.status_code == 415:
            raise OCRUnsupportedFormat(
                f"OCR sidecar: unsupported image format (HTTP 415): {resp.text[:200]}"
            )
        if resp.status_code == 504:
            raise OCRTimeout(f"OCR sidecar upstream timeout: {resp.text[:200]}")
        if 500 <= resp.status_code < 600:
            raise OCRUnavailable(f"OCR sidecar returned {resp.status_code}: {resp.text[:200]}")
        raise OCRResponseError(
            f"OCR sidecar returned unexpected status {resp.status_code}: {resp.text[:200]}"
        )


__all__ = [
    "OCR_SIDECAR_DEFAULT_URL",
    "OCR_SIDECAR_TIMEOUT_S",
    "OCRAuthError",
    "OCRClient",
    "OCRImageTooLarge",
    "OCRResponseError",
    "OCRTimeout",
    "OCRUnavailable",
    "OCRUnreachable",
    "OCRUnsupportedFormat",
]
