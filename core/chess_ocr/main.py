"""CLI entry point for the chess-ocr sidecar."""

from __future__ import annotations

import logging

import uvicorn

from core.chess_ocr.config import OCRSettings


def main() -> None:
    settings = OCRSettings()
    log_level = logging.INFO

    if not settings.api_key:
        # Surface this loudly — without `n`, the sidecar can't run.
        logging.basicConfig(level=log_level)
        logging.getLogger("chessy_mcp.chess_ocr.main").warning(
            "MISSING_API_KEY: env var `n` is not set; sidecar will refuse OCR requests."
        )

    uvicorn.run(
        "core.chess_ocr.app:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
