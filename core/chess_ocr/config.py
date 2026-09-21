"""Configuration for the chess-ocr sidecar.

The M3 multimodal API key is read from env var ``n`` (single-letter per
project convention). Never log this value, never echo it in error messages.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OCRSettings(BaseSettings):
    """Configuration for the chess-ocr sidecar."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # M3 multimodal API key. Read from env var ``n`` at runtime only.
    # If missing at startup, the sidecar refuses to boot.
    # The field is also constructible directly (e.g. for tests via
    # ``OCRSettings(api_key="...")``) — ``populate_by_name=True`` allows
    # the constructor kwarg to take precedence over the env var.
    api_key: str = Field(default="", validation_alias="n")

    # M3 model identifier (pinned so a model upgrade can't silently change
    # OCR quality).
    model: str = Field(default="MiniMax-M3", validation_alias="OCR_MODEL")

    # API endpoint. OpenAI-compatible by default.
    api_base: str = Field(
        default="https://api.MiniMax.io/v1",
        validation_alias="OCR_API_BASE",
    )

    # Sidecar network settings.
    host: str = Field(default="0.0.0.0", validation_alias="OCR_HOST")
    port: int = Field(default=9552, validation_alias="OCR_PORT")

    # Image size cap (decoded). 32 MB accommodates iPhone HEIC scans.
    max_image_bytes: int = Field(default=32 * 1024 * 1024, validation_alias="OCR_MAX_IMAGE_BYTES")

    # Per-pass timeouts.
    per_pass_timeout_s: float = Field(default=30.0, validation_alias="OCR_PER_PASS_TIMEOUT_S")

    # Multi-pass consensus settings.
    max_passes: int = Field(default=6, validation_alias="OCR_MAX_PASSES")
    target_verifier_confidence: float = Field(
        default=0.9, validation_alias="OCR_TARGET_VERIFIER_CONFIDENCE"
    )

    # Engine selection. v1: only "minimax" (M3).
    engine: Literal["minimax"] = Field(default="minimax", validation_alias="OCR_ENGINE")

    def __repr__(self) -> str:
        """Redact the API key from repr to avoid leaking it via logs or tracebacks."""
        return (
            f"OCRSettings(api_key=***REDACTED***, model={self.model!r}, "
            f"engine={self.engine!r}, host={self.host!r}, port={self.port!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()
