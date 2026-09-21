"""End-to-end tests for the ``ocr_to_pgn`` MCP tool v2.

Mocks the chess-ocr sidecar via respx and exercises the full pipeline:
image-source resolution → sidecar call → header extraction → per-cell
candidate aggregation → legal-sequence beam search → PGN composition
→ python-chess validation → verbosity-gated response.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest
import respx

from mcp_server.models import OcrPgnResult
from mcp_server.ocr import OCRClient
from mcp_server.tools import ocr_to_pgn as ocr_module
from mcp_server.tools.ocr_to_pgn import ocr_to_pgn


def _fake_jpeg_bytes() -> bytes:
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def _sidecar_response(
    text: str = "1.e4 e5 2.Sf3 Sc6 1-0",
    *,
    language: str = "pl",
    auto_rotation_applied: int = 0,
    pass_count: int = 2,
    cell_candidates: list[dict] | None = None,
    headers: dict[str, dict] | None = None,
) -> dict:
    if cell_candidates is None:
        # Default: tokenise the CANONICAL (English SAN) movetext into
        # per-cell candidates with uniformly-high scores so beam search
        # picks the canonical path. Sidecar returns already-normalised
        # text, so feed it English SANs.
        from mcp_server.parsers.san_normalize import normalize_pgn

        norm = normalize_pgn(text, language=language).canonical_text
        import re

        cell_candidates = []
        ply = 1
        side = "white"
        for token in norm.split():
            if token in {"1-0", "0-1", "1/2-1/2", "*"}:
                break
            stripped = re.sub(r"^\d+\.{1,3}", "", token)
            if not stripped or (stripped == token and token.rstrip(".").isdigit()):
                continue
            san = stripped or token
            cell_candidates.append({"ply": ply, "side": side, "san": san, "score": 0.9})
            side = "black" if side == "white" else "white"
            ply += 1
    if headers is None:
        headers = {
            "white": {"value": None, "confidence": 0.0},
            "black": {"value": None, "confidence": 0.0},
            "round": {"value": "2", "confidence": 0.96},
            "date": {"value": "2026.09.05", "confidence": 0.99},
            "event": {"value": None, "confidence": 0.0},
            "site": {"value": None, "confidence": 0.0},
            "result": {"value": "1-0", "confidence": 0.95},
        }
    return {
        "raw_text": text,
        "canonical_text": text,
        "detected_language": language,
        "candidates": [
            {
                "pass_label": "raw",
                "language_hint": None,
                "text": text,
                "error": None,
                "parse_rate": 1.0,
                "score": 1.0,
                "latency_ms": 200.0,
                "rotation": "rot0",
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
        "cell_candidates": cell_candidates,
        "headers": headers,
    }


def _install_mock_sidecar() -> respx.MockRouter:
    """Install a fresh OCR client bound to respx."""
    client = OCRClient(url="http://ocr.test:9552", timeout_s=5.0)
    ocr_module.reset_singletons_for_tests(client=client)
    return respx.mock(assert_all_called=False)


def _image_base64() -> dict:
    return {"kind": "base64", "data": base64.b64encode(_fake_jpeg_bytes()).decode("ascii")}


# --- happy paths ------------------------------------------------------------


@pytest.mark.asyncio
async def test_ocr_to_pgn_polish_image_round_trip() -> None:
    """A Polish score sheet photo round-trips through OCR + normalize + beam."""
    router = _install_mock_sidecar()

    with router:
        route = router.post("http://ocr.test:9552/ocr").respond(
            200,
            json=_sidecar_response(
                "1.e4 e5 2.Sf3 Sc6 3.Gb5 a6 1-0",
                language="pl",
            ),
        )
        result = await ocr_to_pgn(image=_image_base64(), source_language="auto")

    assert isinstance(result, OcrPgnResult)
    # Polish notation gets normalised to canonical English SAN.
    assert "Nf3" in result.canonical_pgn
    assert "Nc6" in result.canonical_pgn
    assert "Bb5" in result.canonical_pgn
    assert result.status == "ok"
    assert result.validation.legal is True
    assert result.validation.plies == 6
    assert route.call_count == 1
    # Default verbosity=minimal: heavy forensic fields stay None.
    assert result.candidates is None
    assert result.moves is None


@pytest.mark.asyncio
async def test_ocr_to_pgn_english_image() -> None:
    router = _install_mock_sidecar()

    with router:
        route = router.post("http://ocr.test:9552/ocr").respond(
            200, json=_sidecar_response("1.e4 e5 2.Nf3 Nc6 1-0", language="en")
        )
        result = await ocr_to_pgn(image=_image_base64())

    # Minimal verbosity: detected_language hidden, but canonical_pgn is present.
    assert result.detected_language is None
    assert "Nf3" in result.canonical_pgn
    assert "Nc6" in result.canonical_pgn
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_ocr_to_pgn_full_verbosity_returns_candidates_and_moves() -> None:
    router = _install_mock_sidecar()

    with router:
        router.post("http://ocr.test:9552/ocr").respond(
            200, json=_sidecar_response("1.e4 e5 2.Nf3 Nc6 1-0", language="en")
        )
        result = await ocr_to_pgn(image=_image_base64(), verbosity="full")

    assert result.candidates is not None and len(result.candidates) >= 1
    assert result.moves is not None and len(result.moves) >= 1
    assert result.raw_ocr_text is not None
    assert result.verifier_confidence == 0.95


@pytest.mark.asyncio
async def test_ocr_to_pgn_minimal_verbosity_omits_evidence() -> None:
    router = _install_mock_sidecar()

    with router:
        router.post("http://ocr.test:9552/ocr").respond(
            200, json=_sidecar_response("1.e4 e5 2.Nf3 Nc6 1-0", language="en")
        )
        result = await ocr_to_pgn(image=_image_base64(), verbosity="minimal")

    assert result.detected_language is None
    assert result.candidates is None
    assert result.moves is None
    assert result.raw_ocr_text is None
    assert result.verifier_confidence is None


@pytest.mark.asyncio
async def test_ocr_to_pgn_compact_verbosity_includes_detected_language() -> None:
    router = _install_mock_sidecar()

    with router:
        router.post("http://ocr.test:9552/ocr").respond(
            200, json=_sidecar_response("1.e4 e5 2.Nf3 Nc6 1-0", language="en")
        )
        result = await ocr_to_pgn(image=_image_base64(), verbosity="compact")

    assert result.detected_language == "en"
    # compact still hides the heaviest forensic fields.
    assert result.raw_ocr_text is None
    assert result.moves is None


@pytest.mark.asyncio
async def test_ocr_to_pgn_auto_mode_forwards_clahe_and_rotation() -> None:
    router = _install_mock_sidecar()

    with router:
        route = router.post("http://ocr.test:9552/ocr").respond(
            200, json=_sidecar_response(auto_rotation_applied=90)
        )
        await ocr_to_pgn(image=_image_base64(), mode="auto")

    sent_body = route.calls.last.request.content.decode("ascii")
    assert '"enhance_contrast":true' in sent_body
    assert '"verify_with_rotation":true' in sent_body


@pytest.mark.asyncio
async def test_ocr_to_pgn_explicit_mode_uses_preprocessing_hints() -> None:
    router = _install_mock_sidecar()

    with router:
        route = router.post("http://ocr.test:9552/ocr").respond(200, json=_sidecar_response())
        await ocr_to_pgn(
            image=_image_base64(),
            mode="explicit",
            preprocessing={
                "enhance_contrast": False,
                "verify_with_rotation": True,
                "denoise": False,
            },
        )

    sent_body = route.calls.last.request.content.decode("ascii")
    assert '"enhance_contrast":false' in sent_body
    assert '"verify_with_rotation":true' in sent_body


# --- image source variants --------------------------------------------------


@pytest.mark.asyncio
async def test_ocr_to_pgn_url_source_fetches_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHESSY_MCP_URL_ALLOWLIST", "example.com")
    router = _install_mock_sidecar()

    with respx.mock(assert_all_called=False) as fetch_router:
        fetch_router.get("http://example.com/score.jpg").respond(200, content=_fake_jpeg_bytes())
        with router:
            sidecar_route = router.post("http://ocr.test:9552/ocr").respond(
                200, json=_sidecar_response("1.e4 e5 2.Nf3 Nc6 1-0", language="en")
            )
            result = await ocr_to_pgn(image={"kind": "url", "url": "http://example.com/score.jpg"})

    assert result.status == "ok"
    assert sidecar_route.call_count == 1


@pytest.mark.asyncio
async def test_ocr_to_pgn_file_uri_source_reads_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHESSY_MCP_FILE_ROOT", str(tmp_path))
    score_path = tmp_path / "score.jpg"
    score_path.write_bytes(_fake_jpeg_bytes())

    router = _install_mock_sidecar()
    with router:
        sidecar_route = router.post("http://ocr.test:9552/ocr").respond(
            200, json=_sidecar_response("1.e4 e5 2.Nf3 Nc6 1-0", language="en")
        )
        result = await ocr_to_pgn(image={"kind": "file_uri", "path": str(score_path)})

    assert result.status == "ok"
    assert sidecar_route.call_count == 1


# --- input validation -------------------------------------------------------


@pytest.mark.asyncio
async def test_ocr_to_pgn_invalid_base64_raises_tool_error() -> None:
    _install_mock_sidecar()
    with pytest.raises(Exception, match="base64"):
        await ocr_to_pgn(image={"kind": "base64", "data": "@@@not-base64@@@"})


@pytest.mark.asyncio
async def test_ocr_to_pgn_unsupported_format_raises_tool_error() -> None:
    _install_mock_sidecar()
    bad_bytes = b"this is not an image at all"
    image_b64 = base64.b64encode(bad_bytes).decode("ascii")
    with pytest.raises(Exception, match=r"jpeg|png|webp|gif|heic"):
        await ocr_to_pgn(image={"kind": "base64", "data": image_b64})


# --- resolve_ambiguities modes ---------------------------------------------


@pytest.mark.asyncio
async def test_ocr_to_pgn_strict_ambiguity_raises_needs_review() -> None:
    router = _install_mock_sidecar()

    # Two legal candidates at ply 3 (white): Nf3 and Nc3.
    cell_candidates = [
        {"ply": 1, "side": "white", "san": "e4", "score": 0.9},
        {"ply": 2, "side": "black", "san": "e5", "score": 0.9},
        {"ply": 3, "side": "white", "san": "Nf3", "score": 0.5},
        {"ply": 3, "side": "white", "san": "Nc3", "score": 0.5},
        {"ply": 4, "side": "black", "san": "Nc6", "score": 0.9},
    ]

    with router:
        router.post("http://ocr.test:9552/ocr").respond(
            200,
            json=_sidecar_response(
                "1.e4 e5 2.Nf3 Nc6 1-0",
                language="en",
                cell_candidates=cell_candidates,
            ),
        )
        with pytest.raises(Exception, match="Strict mode"):
            await ocr_to_pgn(image=_image_base64(), resolve_ambiguities="strict")


@pytest.mark.asyncio
async def test_ocr_to_pgn_best_effort_returns_low_confidence_path() -> None:
    router = _install_mock_sidecar()

    cell_candidates = [
        {"ply": 1, "side": "white", "san": "e4", "score": 0.9},
        {"ply": 2, "side": "black", "san": "e5", "score": 0.9},
        {"ply": 3, "side": "white", "san": "Nf3", "score": 0.5},
        {"ply": 3, "side": "white", "san": "Nc3", "score": 0.5},
    ]

    with router:
        router.post("http://ocr.test:9552/ocr").respond(
            200,
            json=_sidecar_response(
                "1.e4 e5 2.Nf3 Nc6 1-0",
                language="en",
                cell_candidates=cell_candidates,
            ),
        )
        result = await ocr_to_pgn(image=_image_base64(), resolve_ambiguities="best_effort")

    # Status may be ok or needs_review depending on whether low_confidence applies.
    assert "Nf3" in result.canonical_pgn or "Nc3" in result.canonical_pgn
    assert any(
        u.reason in {"handwriting_ambiguity", "low_confidence"} for u in result.uncertainties
    )


# --- metadata hints ---------------------------------------------------------


@pytest.mark.asyncio
async def test_ocr_to_pgn_metadata_hints_override_headers() -> None:
    router = _install_mock_sidecar()

    with router:
        router.post("http://ocr.test:9552/ocr").respond(
            200,
            json=_sidecar_response(
                "1.e4 e5 2.Nf3 Nc6 1-0",
                language="en",
                headers={
                    "white": {"value": "OCR White", "confidence": 0.7},
                    "black": {"value": "OCR Black", "confidence": 0.7},
                    "round": {"value": "2", "confidence": 0.96},
                    "date": {"value": "2026.09.05", "confidence": 0.99},
                    "event": {"value": None, "confidence": 0.0},
                    "site": {"value": None, "confidence": 0.0},
                    "result": {"value": "1-0", "confidence": 0.95},
                },
            ),
        )
        result = await ocr_to_pgn(
            image=_image_base64(),
            metadata={
                "white": "Hint White",
                "black": None,
                "round": None,
                "date": None,
                "event": None,
                "site": None,
                "result": None,
            },
        )

    assert result.metadata["white"].value == "Hint White"
    assert result.metadata["white"].source == "hint"
    assert result.metadata["white"].confidence >= 0.85
    assert result.metadata["black"].value == "OCR Black"
    assert result.metadata["black"].source == "detected"


@pytest.mark.asyncio
async def test_ocr_to_pgn_headers_populated_from_sidecar() -> None:
    router = _install_mock_sidecar()

    with router:
        router.post("http://ocr.test:9552/ocr").respond(
            200,
            json=_sidecar_response(
                "1.e4 e5 2.Nf3 Nc6 1-0",
                language="en",
                headers={
                    "white": {"value": "Smith, Alice", "confidence": 0.91},
                    "black": {"value": "Jones, Bob", "confidence": 0.88},
                    "round": {"value": "3", "confidence": 0.95},
                    "date": {"value": "2026.08.12", "confidence": 0.97},
                    "event": {"value": "City Open 2026", "confidence": 0.80},
                    "site": {"value": "Warsaw", "confidence": 0.75},
                    "result": {"value": "1-0", "confidence": 0.99},
                },
            ),
        )
        result = await ocr_to_pgn(image=_image_base64())

    assert result.metadata["white"].value == "Smith, Alice"
    assert result.metadata["black"].value == "Jones, Bob"
    assert result.metadata["round"].value == "3"
    assert result.metadata["date"].value == "2026.08.12"
    assert "Smith, Alice" in result.canonical_pgn
    assert "Jones, Bob" in result.canonical_pgn


# --- sidecar error mapping --------------------------------------------------


@pytest.mark.asyncio
async def test_ocr_to_pgn_sidecar_unreachable() -> None:
    router = _install_mock_sidecar()

    with router:
        router.post("http://ocr.test:9552/ocr").mock(side_effect=httpx.ConnectError("nope"))
        with pytest.raises(Exception, match=r"ocr_unreachable|UNREACHABLE"):
            await ocr_to_pgn(image=_image_base64())


@pytest.mark.asyncio
async def test_ocr_to_pgn_sidecar_503_maps_to_auth_error() -> None:
    router = _install_mock_sidecar()

    with router:
        router.post("http://ocr.test:9552/ocr").respond(503, text="MISSING_API_KEY")
        with pytest.raises(Exception, match=r"ocr_auth|API_KEY"):
            await ocr_to_pgn(image=_image_base64())
