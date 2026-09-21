"""Tests for the multi-pass consensus orchestrator."""

from __future__ import annotations


import httpx
import pytest
import respx

from core.chess_ocr.config import OCRSettings
from core.chess_ocr.consensus import ConsensusOrchestrator
from core.chess_ocr.engines.minimax import MinimaxEngine


def _settings() -> OCRSettings:
    return OCRSettings(
        api_key="sk-test", model="minimax/MiniMax-M3", api_base="https://api.test/v1"
    )


def _image() -> bytes:
    """Real (decodable) JPEG bytes — the consensus pipeline now passes the
    image through PIL's decode path (orientation + CLAHE + bilateral), so
    a fake-JPEG byte sequence that PIL rejects would skip the test's intent.
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (255, 255, 255)).save(buf, format="JPEG")
    return buf.getvalue()


def _ok_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "t",
            "model": "minimax/MiniMax-M3",
            "choices": [{"message": {"role": "assistant", "content": text}}],
        },
        request=httpx.Request("POST", "https://api.test/v1/chat/completions"),
    )


def _verify_response(best: int, conf: float = 0.9, notes: str = "Best match.") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "v",
            "model": "minimax/MiniMax-M3",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f'{{"best_index": {best}, "confidence": {conf}, "notes": "{notes}"}}',
                    }
                }
            ],
        },
        request=httpx.Request("POST", "https://api.test/v1/chat/completions"),
    )


@pytest.mark.asyncio
async def test_consensus_runs_pass1_and_three_language_hints() -> None:
    """The orchestrator fires Pass 1 + 3 language-hinted passes in parallel."""
    settings = _settings()
    engine = MinimaxEngine(settings)
    orchestrator = ConsensusOrchestrator(settings, engine)

    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.test/v1/chat/completions").mock(
            side_effect=[
                _ok_response("1.e4 e5 2.Nf3 Nc6 1-0"),  # raw
                _ok_response("1.e4 e5 2.Sf3 Sc6 1-0"),  # pl hint
                _ok_response("1.e4 e5 2.Nf3 Nc6 1-0"),  # en hint 1
                _ok_response("1.e4 e5 2.Nf3 Nc6 1-0"),  # en hint 2
                _verify_response(best=0, conf=0.95),  # verifier
            ]
        )
        result = await orchestrator.run(_image())

    assert route.call_count == 5
    assert result.pass_count == 2  # Pass 1 + verifier
    assert result.detected_language == "en"
    assert "Nf3" in result.raw_text
    assert len(result.candidates) == 4


@pytest.mark.asyncio
async def test_consensus_falls_back_to_raw_when_all_passes_fail() -> None:
    """When all OCR passes fail, the orchestrator returns the raw error state."""
    settings = _settings()
    engine = MinimaxEngine(settings)
    orchestrator = ConsensusOrchestrator(settings, engine)

    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.test/v1/chat/completions").respond(
            500, text="simulated upstream failure"
        )
        result = await orchestrator.run(_image())

    # The raw text is empty because every pass errored out.
    assert result.raw_text == ""
    assert all(c.get("error") for c in result.candidates)


@pytest.mark.asyncio
async def test_consensus_sanity_pass_when_verifier_uncertain() -> None:
    """If the verifier reports low confidence, a sanity pass fires (Pass 4)."""
    settings = _settings()
    settings = settings.model_copy(update={"target_verifier_confidence": 0.9})
    engine = MinimaxEngine(settings)
    orchestrator = ConsensusOrchestrator(settings, engine)

    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.test/v1/chat/completions").mock(
            side_effect=[
                _ok_response("1.e4 e5 2.Nf3 Nc6 1-0"),
                _ok_response("1.e4 e5 2.Sf3 Sc6 1-0"),
                _ok_response("1.e4 e5 2.Nf3 Nc6 1-0"),
                _ok_response("1.e4 e5 2.Nf3 Nc6 1-0"),
                _verify_response(best=0, conf=0.45),  # low confidence
                _ok_response("1.e4 e5 2.Nf3 Nc6 1-0"),  # sanity
            ]
        )
        result = await orchestrator.run(_image())

    assert route.call_count == 6
    assert result.pass_count == 3  # 1 raw + verifier + sanity


@pytest.mark.asyncio
async def test_consensus_respects_defensive_max_passes_cap() -> None:
    """The defensive cap prevents infinite loops on repeated low-confidence passes."""
    settings = _settings()
    settings = settings.model_copy(update={"max_passes": 3, "target_verifier_confidence": 0.95})
    engine = MinimaxEngine(settings)
    orchestrator = ConsensusOrchestrator(settings, engine)

    with respx.mock(assert_all_called=False) as router:
        route = router.post("https://api.test/v1/chat/completions").mock(
            side_effect=[
                _ok_response("1.e4 e5 2.Nf3 Nc6 1-0"),
                _ok_response("1.e4 e5 2.Sf3 Sc6 1-0"),
                _ok_response("1.e4 e5 2.Nf3 Nc6 1-0"),
                _ok_response("1.e4 e5 2.Nf3 Nc6 1-0"),
                _verify_response(best=0, conf=0.4),  # low
                # No more responses — would be a 6th call, but cap is 3.
            ]
        )
        try:
            await orchestrator.run(_image())
        except Exception:
            pass

    # 4 OCR passes + 1 verifier = 5 calls. Sanity pass would be 6, but max_passes=3 caps it.
    assert route.call_count <= 5


@pytest.mark.asyncio
async def test_consensus_detects_polish_from_raw_text() -> None:
    """When the chosen candidate contains Polish-distinctive letters, language=pl."""
    settings = _settings()
    engine = MinimaxEngine(settings)
    orchestrator = ConsensusOrchestrator(settings, engine)

    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.test/v1/chat/completions").mock(
            side_effect=[
                _ok_response("1.e4 e5 2.Sf3 Sc6 3.Gb5 W:e5 1-0"),  # Polish
                _ok_response("1.e4 e5 2.Sf3 Sc6 3.Gb5 W:e5 1-0"),
                _ok_response("1.e4 e5 2.Nf3 Nc6 1-0"),
                _ok_response("1.e4 e5 2.Nf3 Nc6 1-0"),
                _verify_response(best=0, conf=0.9),
            ]
        )
        result = await orchestrator.run(_image())

    assert result.detected_language == "pl"
    assert "Sf3" in result.raw_text
