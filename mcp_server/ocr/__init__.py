"""MCP-side public surface for the chess-ocr sidecar integration."""

from mcp_server.ocr.client import (
    OCR_SIDECAR_DEFAULT_URL,
    OCR_SIDECAR_TIMEOUT_S,
    OCRAuthError,
    OCRClient,
    OCRImageTooLarge,
    OCRResponseError,
    OCRTimeout,
    OCRUnreachable,
    OCRUnavailable,
    OCRUnsupportedFormat,
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
