"""M3 multimodal OCR engine.

Wraps the M3 multimodal API (OpenAI-compatible by default) with prompt
templates tuned for chess score-sheet OCR. The returned text is the raw
PGN movetext as OCR'd — no normalization happens here; that lives in
``mcp_server.parsers.san_normalize`` on the MCP side.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

import httpx

from core.chess_ocr.config import OCRSettings


log = logging.getLogger("chessy_mcp.chess_ocr.engines.minimax")


SYSTEM_PROMPT_RAW = """You are a chess score-sheet OCR specialist. Read the photograph and output ONLY the PGN movetext.

STRICT RULES — no exceptions:
1. Output format: PGN with [Tag "value"] headers followed by movetext, e.g. `[Event "?"]\n\n1.e4 e5 2.Nf3 Nc6 1-0`.
2. PRESERVE the original notation: Polish (K/H/W/G/S with `:` or `x` captures) stays Polish; English (K/Q/R/B/N) stays English; digits `0-0` stay as `O-O`.
3. NO analysis, NO commentary, NO reasoning, NO thinking. Output ONLY the PGN.
4. NO markdown code fences. NO backticks. NO `<think>` blocks. NO preamble.
5. Illegible moves: write your best guess followed by `?`.
6. Preserve `+`, `#`, `!`, `?` annotations.
7. First character of your response must be `[`. Last character must be `]`, `0`, `1`, `2`, `*`, or a move letter.

If the image does not contain a chess score sheet, output exactly: NO_SCORE_SHEET."""


SYSTEM_PROMPT_LANGUAGE_HINT = """You are a chess score-sheet OCR specialist. Output ONLY the PGN movetext.

This score sheet uses {language_name} notation. Use the {language_name} piece letters:
{letters}

STRICT RULES:
1. Output format: PGN with [Tag "value"] headers followed by movetext.
2. NO analysis, NO reasoning, NO thinking. NO markdown. NO code fences. NO backticks. NO `<think>`.
3. First character of your response must be `[`. Last character must be `]`, digit, `*`, or move letter.
4. Illegible moves: best guess + `?`.
5. Preserve `+`, `#`, `!`, `?` annotations.

If the image does not contain a chess score sheet, output exactly: NO_SCORE_SHEET."""


def _language_hint_prompt(language: str) -> str:
    """Build a per-language system prompt for Pass 2a/b/c."""
    letters = {
        "en": "English: K=King, Q=Queen, R=Rook, B=Bishop, N=Knight",
        "pl": "Polish: K=Król (King), H=Hetman (Queen), W=Wieża (Rook), G=Goniec (Bishop), S=Skoczek (Knight). Captures may use `:` or `x`.",
    }.get(language, f"{language}: use standard piece letters")
    name = {"en": "English", "pl": "Polish"}.get(language, language)
    return SYSTEM_PROMPT_LANGUAGE_HINT.format(language_name=name, letters=letters)


VERIFIER_PROMPT = """You are a chess-PGN verifier. You will be shown:
1. A photograph of a chess score sheet.
2. Three candidate PGN transcriptions of that sheet (in different notation styles).

Your job: pick the candidate that BEST matches the visible moves. Consider:
- Move sequence correctness (every move number should match a written move).
- Capture notation (`x` or `:`).
- Check/mate markers (`+`, `#`).
- Castling form.
- Piece letter conventions.

Output a JSON object with:
- `best_index`: integer 0, 1, or 2 (which candidate to pick).
- `confidence`: float 0..1 (how sure you are).
- `notes`: short string explaining any differences.

Output ONLY the JSON, no other text."""


class MinimaxEngine:
    """M3 multimodal OCR engine.

    Wraps the OpenAI-compatible chat completions endpoint with image
    inputs (base64-encoded data URLs). Each call fires one M3 request;
    the orchestrator in :mod:`core.chess_ocr.consensus` runs multiple
    passes and picks the best candidate.
    """

    def __init__(self, settings: OCRSettings | None = None) -> None:
        self._settings = settings or OCRSettings()
        self._client: httpx.AsyncClient | None = None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._settings.per_pass_timeout_s),
                headers={"User-Agent": "chessy-mcp-ocr/0.1.0"},
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
        system_prompt: str = SYSTEM_PROMPT_RAW,
        language_hint: str | None = None,
    ) -> dict[str, Any]:
        """Run one OCR pass over the image.

        Args:
            image_bytes: Decoded image bytes (PNG, JPEG, WebP, GIF, HEIC).
                Pre-encoded bytes — engine does NOT re-decode for orientation.
            system_prompt: Optional system prompt override.
            language_hint: Optional language hint ("en" / "pl") to bias the
                OCR's transcription. When provided, the default Polish-style
                system prompt is replaced with the language-specific one.

        Returns:
            ``{"text": str, "model": str, "latency_ms": float,
               "usage": dict, "raw_response": dict}``

        Raises:
            ValueError: On API errors with a structured message.
        """
        settings = self._settings
        if not settings.api_key:
            raise ValueError("MISSING_API_KEY: env var `n` must be set for M3 OCR.")

        # Image is already preprocessed (decode + orientation auto-correct
        # + optional CLAHE/bilateral) by the orchestrator before this call.
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        prompt = system_prompt
        if language_hint:
            prompt = _language_hint_prompt(language_hint)

        body = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {
                            "type": "text",
                            "text": "Transcribe this chess score sheet as PGN. Output ONLY the PGN.",
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            # M3 is a reasoning model — even with `reasoning_effort=low` it
            # burns a lot of tokens on internal "thinking" before emitting
            # the PGN. We budget 64k so even a 60-move game with messy
            # handwriting and per-character disambiguation fits, paired
            # with OCR_PER_PASS_TIMEOUT_S=360 in production.
            "reasoning_effort": "low",
            "max_tokens": 65536,
        }

        url = settings.api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        }

        t0 = time.time()
        client = await self._http()
        try:
            resp = await client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise ValueError(
                f"OCR_TIMEOUT: M3 request exceeded {settings.per_pass_timeout_s}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ValueError(f"OCR_UNREACHABLE: {exc}") from exc

        latency_ms = (time.time() - t0) * 1000.0

        if resp.status_code != 200:
            raise ValueError(f"OCR_API_ERROR: M3 returned {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        text = _extract_message_text(data)
        return {
            "text": text,
            "model": data.get("model", settings.model),
            "latency_ms": latency_ms,
            "usage": data.get("usage", {}),
            "raw_response": data,
        }

    async def verify(
        self,
        image_bytes: bytes,
        candidates: list[str],
    ) -> dict[str, Any]:
        """Ask M3 to pick the best candidate among multiple OCR outputs.

        Returns a dict with ``best_index``, ``confidence``, ``notes`` and
        the raw M3 response for observability.
        """
        if not candidates:
            raise ValueError("OCR_VERIFY_NO_CANDIDATES: at least one candidate is required")

        settings = self._settings
        if not settings.api_key:
            raise ValueError("MISSING_API_KEY: env var `n` must be set for M3 OCR.")

        png_or_native = image_bytes
        b64 = base64.b64encode(png_or_native).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        candidates_text = "\n\n".join(
            f"--- Candidate {idx} ---\n{cand}" for idx, cand in enumerate(candidates)
        )
        user_text = (
            f"Pick the candidate that best matches the score sheet.\n\n```\n{candidates_text}\n```"
        )

        body = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": VERIFIER_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
        }

        url = settings.api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        }

        client = await self._http()
        try:
            resp = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise ValueError(f"OCR_UNREACHABLE: {exc}") from exc

        if resp.status_code != 200:
            raise ValueError(
                f"OCR_VERIFY_API_ERROR: M3 returned {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        raw_text = _extract_message_text(data)
        try:
            verdict = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback: best_index = 0 with low confidence.
            verdict = {
                "best_index": 0,
                "confidence": 0.5,
                "notes": f"Could not parse verifier JSON: {raw_text[:200]}",
            }

        verdict.setdefault("best_index", 0)
        verdict.setdefault("confidence", 0.5)
        verdict.setdefault("notes", "")
        verdict["raw_verifier_text"] = raw_text
        return verdict


def _extract_message_text(response: dict[str, Any]) -> str:
    """Extract the message content from an OpenAI-compatible chat completion response."""
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts).strip()
    return ""
