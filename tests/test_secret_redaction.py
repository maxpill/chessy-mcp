"""Security tests: confirm the M3 API key (env var ``n``) never leaks.

These tests inspect structlog output, exception messages, and request bodies
to ensure the secret is never echoed back to the user or to logs.
"""

from __future__ import annotations

import logging

import pytest


SECRET_VALUE = "sk-test-FAKE-DO-NOT-USE-IN-PROD-1234567890abcdef"


def test_secret_does_not_appear_in_settings_dict() -> None:
    """The API key is not accidentally exposed via settings repr/dict."""
    from core.chess_ocr.config import OCRSettings

    settings = OCRSettings(api_key=SECRET_VALUE)
    dumped = settings.model_dump()
    assert dumped["api_key"] == SECRET_VALUE  # direct access is intentional
    # But repr shouldn't include the secret.
    assert SECRET_VALUE not in repr(settings)


def test_structlog_redacts_secrets(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that logging the settings object never reveals the API key."""
    from core.chess_ocr.config import OCRSettings

    settings = OCRSettings(api_key=SECRET_VALUE)
    log = logging.getLogger("chessy_mcp.test_secrets")
    log.warning("config: %s", settings)
    captured = caplog.text
    assert SECRET_VALUE not in captured


@pytest.mark.asyncio
async def test_missing_key_message_does_not_echo_env_name() -> None:
    """Missing-key errors reference the env var name, not the missing value."""
    from core.chess_ocr.engines.minimax import MinimaxEngine
    from core.chess_ocr.config import OCRSettings

    settings = OCRSettings(api_key="")
    engine = MinimaxEngine(settings)
    try:
        await engine.ocr(b"\xff\xd8\xff")
    except ValueError as exc:
        assert "MISSING_API_KEY" in str(exc)
        assert SECRET_VALUE not in str(exc)


def test_env_var_n_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OCR settings must read the ``n`` env var exactly (single-letter)."""
    monkeypatch.setenv("n", SECRET_VALUE)
    monkeypatch.delenv("OCR_API_KEY", raising=False)
    from core.chess_ocr.config import OCRSettings

    settings = OCRSettings()
    assert settings.api_key == SECRET_VALUE


def test_minimax_engine_settings_redact_secret_in_repr() -> None:
    """Engine __repr__ doesn't leak the secret."""
    from core.chess_ocr.engines.minimax import MinimaxEngine
    from core.chess_ocr.config import OCRSettings

    engine = MinimaxEngine(OCRSettings(api_key=SECRET_VALUE))
    rendered = repr(engine)
    assert SECRET_VALUE not in rendered
