"""Multi-pass consensus orchestrator for chess score-sheet OCR.

Single-pass OCR is brittle on handwritten Polish score sheets (scribbles,
folds, shadows). The orchestrator fires multiple M3 passes with
different prompts and asks M3 to pick the best:

    Pass 1   — raw OCR with default prompt (no language bias).
    Pass 2a  — Polish-biased prompt (in case the sheet IS Polish).
    Pass 2b  — English-biased prompt (in case the sheet is English).
    Pass 2c  — repeated English-biased for diversity.
    Pass 3   — verifier: show M3 the image + all candidates, pick winner.

The orchestrator runs Pass 1 + Pass 2a/b/c in parallel (asyncio.gather)
to minimize latency. After the verifier, if ``confidence < 0.9`` we run
Pass 4 (sanity sweep) and reassess.

A defensive cap (``max_passes``) prevents infinite loops.

v2 additions:

    - Per-cell candidate aggregation across all passes. The orchestrator
      tokenises every candidate's movetext into (ply, side, san) tuples,
      groups by (ply, side) and emits the union as
      :attr:`ConsensusResult.cell_candidates`. The MCP ``ocr_to_pgn``
      tool consumes this for legal-sequence beam search.

    - Structured header extraction (optional, default ON). One
      deterministic M3 call asking for the seven PGN header fields with
      per-field confidence. Result lives in
      :attr:`ConsensusResult.headers`.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import chess
import chess.pgn

from core.chess_ocr.config import OCRSettings
from core.chess_ocr.engines.minimax import MinimaxEngine
from core.chess_ocr.header_extractor import (
    empty_headers,
    merge_with_hints,
)
from core.chess_ocr.preprocess import (
    ContrastStrategy,
    DenoiseStrategy,
    RotationStrategy,
    decode_image_to_rgb,
)


ConsensusStrategy = Literal["default", "three_orientation"]


log = logging.getLogger("chessy_mcp.chess_ocr.consensus")


LANGUAGE_HINTS: Final = ("pl", "en", "en")  # Pass 2a, 2b, 2c


@dataclass(frozen=True)
class CellCandidate:
    """One per-cell candidate SAN emitted by the orchestrator (v2)."""

    ply: int
    side: str  # "white" | "black"
    san: str
    score: float


@dataclass
class HeaderField:
    """One PGN header field extracted by the structured pass (v2)."""

    value: str | None
    confidence: float


@dataclass
class ConsensusResult:
    """Result of multi-pass OCR consensus."""

    raw_text: str  # Pass 1 OCR output (the "raw" reference)
    canonical_text: str  # Pass 1 with unicode/figure normalization
    detected_language: str  # "pl" | "en" (heuristic on raw_text)
    candidates: list[dict[str, Any]]  # per-pass outputs
    selected_candidate_index: int
    verifier_agreement: bool | None
    verifier_notes: str | None
    verifier_confidence: float | None
    ocr_engine_used: str
    ocr_model_version: str
    pass_count: int
    total_latency_ms: float
    auto_rotation_applied: int = 0  # degrees CCW the auto-detector chose (0 = none)
    # v2 additions
    cell_candidates: list[CellCandidate] = field(default_factory=list)
    headers: dict[str, HeaderField] = field(default_factory=dict)
    cell_crops: dict[int, str] = field(default_factory=dict)


# python-chess legality checker used to score candidates.
def _python_chess_parse_rate(text: str) -> float:
    """Return the fraction of ply tokens that parse cleanly via python-chess."""
    if not text or not text.strip():
        return 0.0
    # Wrap bare movetext in minimal PGN headers so python-chess can parse.
    wrapped = '[Event "?"]\n\n' + text if not text.lstrip().startswith("[") else text
    try:
        game = chess.pgn.read_game(io.StringIO(wrapped))
    except Exception:
        return 0.0
    if game is None:
        return 0.0
    # Tokenize the movetext roughly to count plies.
    tokens = text.split()
    result_tokens = {"1-0", "0-1", "1/2-1/2", "*"}
    ply_tokens = 0
    parsed_plies = 0
    for tok in tokens:
        if tok in result_tokens:
            break
        if tok.rstrip(".").isdigit():
            continue
        if tok in {"$1", "$2", "$3", "$4", "$5", "$6", "$7", "$8"}:
            continue
        ply_tokens += 1
    parsed_plies = len(list(game.mainline_moves()))
    if ply_tokens == 0:
        return 0.0
    return min(1.0, parsed_plies / ply_tokens)


def _detect_polish(text: str) -> bool:
    """Quick Polish indicator check on raw OCR output."""
    distinctive = sum(text.count(letter) for letter in ("W", "G", "H"))
    return distinctive >= 2 or text.count(":") >= 3


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Look for either a PGN-style header (e.g. ``[Event ...]``), a move-number
# marker (``1.``, ``1...``), or a result token. The first match in the
# cleaned text is where the PGN payload begins.
_PGN_START_RE = re.compile(
    r"(\[Event\s|\[\s*[A-Z][A-Za-z]+\s+\"[^\"]*\"\]|(?:^|\s)1\.{1,3}|\b1-0\b|\b0-1\b|\b1/2-1/2\b|\bResult\s)",
    re.MULTILINE,
)


def _strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks M3 emits before the answer."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _extract_pgn_payload(text: str) -> str:
    """Find the PGN payload inside an M3 response that may include reasoning or commentary."""
    cleaned = _strip_think_blocks(text)
    match = _PGN_START_RE.search(cleaned)
    if match is None:
        return cleaned
    start = match.start()
    for ch_idx in range(start, -1, -1):
        if cleaned[ch_idx] == "[":
            return cleaned[ch_idx:].strip()
        if cleaned[ch_idx] == "\n" and ch_idx != start:
            return cleaned[start:].strip()
    return cleaned[start:].strip()


def _normalize_unicode(text: str) -> str:
    """Apply the existing unicode/figure normalizer so M3 output is canonical-ASCII."""
    # Lazy import so the consensus module doesn't pull the MCP server on import.
    from mcp_server.parsers.pgn.unicode import normalize_unicode_pgn_results

    return normalize_unicode_pgn_results(text)


# Result token markers that terminate a movetext's legal-move sequence.
_RESULT_TOKENS: Final[frozenset[str]] = frozenset({"1-0", "0-1", "1/2-1/2", "*"})


def _tokenize_movetext(canonical_text: str) -> list[tuple[int, str, str]]:
    """Tokenise a canonical movetext into ``(ply, side, san)`` tuples.

    Each half-move gets its own ``ply`` (1=white's 1st, 2=black's 1st,
    3=white's 2nd, …). Stops at the first result token. Move-number
    tokens (``1.``, ``1...``) are consumed but not emitted.
    """
    out: list[tuple[int, str, str]] = []
    if not canonical_text:
        return out
    ply = 1
    expecting_white = True
    # Naive but tolerant split — chess pgn tags are stripped upstream.
    for raw in canonical_text.split():
        if raw in _RESULT_TOKENS:
            break
        if raw.rstrip(".").isdigit():
            # Move number — reset side expectation based on dots.
            dots = raw.count(".")
            if dots >= 2:
                expecting_white = False  # black move token may follow
            continue
        if raw in {"$1", "$2", "$3", "$4", "$5", "$6", "$7", "$8"}:
            continue
        side = "white" if expecting_white else "black"
        out.append((ply, side, raw))
        if expecting_white:
            expecting_white = False
        else:
            expecting_white = True
        ply += 1
    return out


def _aggregate_cell_candidates(
    candidates: list[dict[str, Any]],
) -> list[CellCandidate]:
    """Aggregate per-cell SAN candidates across all OCR passes.

    Each pass contributes one tokenised movetext; per (ply, side) we
    collect the unique SANs and sum the pass-level score as a proxy for
    cross-pass agreement. The MCP beam search consumes the resulting
    list (one entry per (ply, san) tuple).
    """
    bucket: dict[tuple[int, str, str], float] = {}
    for cand in candidates:
        text = cand.get("text") or ""
        if not text:
            continue
        norm_text = _normalize_unicode(text)
        pass_score = float(cand.get("score", 0.0) or 0.0)
        # Per-pass score is a multiplier so a high-confidence pass contributes
        # more than a low-confidence one.
        tokenised = _tokenize_movetext(norm_text)
        for ply, side, san in tokenised:
            key = (ply, side, san)
            bucket[key] = bucket.get(key, 0.0) + pass_score
    return [
        CellCandidate(ply=p, side=side, san=san, score=round(score, 4))
        for (p, side, san), score in sorted(bucket.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]


class ConsensusOrchestrator:
    """Orchestrates multi-pass OCR with verifier-driven selection."""

    def __init__(
        self,
        settings: OCRSettings | None = None,
        engine: MinimaxEngine | None = None,
    ) -> None:
        self._settings = settings or OCRSettings()
        self._engine = engine or MinimaxEngine(self._settings)

    async def aclose(self) -> None:
        await self._engine.aclose()

    async def run(
        self,
        image_bytes: bytes,
        hint_language: str | None = None,
        *,
        rotation_strategy: RotationStrategy = "off",
        contrast_strategy: ContrastStrategy = "off",
        denoise_strategy: DenoiseStrategy = "off",
        extract_headers: bool = True,
        beam_per_cell: bool = True,
        metadata_hints: dict[str, str] | None = None,
    ) -> ConsensusResult:
        """Run the multi-pass consensus pipeline.

        Args:
            image_bytes: Raw encoded image bytes (HEIC/JPEG/PNG/WebP/GIF).
            hint_language: Optional hint from caller (``"en"`` or ``"pl"``).
                When provided, it biases the verifier's selection but does
                not change the multi-pass strategy.
            rotation_strategy:
                ``"off"`` (default) — single orientation; today's behaviour.
                ``"auto"`` — detect sheet rotation once, send M3 the
                  corrected image; cheap, helps the common sideways case.
                ``"three"`` — fire all 4 pass-types × 3 orientations = 12
                  M3 calls + 1 verifier; M3 picks the upright variant.
            contrast_strategy: ``"off"`` (default) or ``"clahe"``.
            denoise_strategy:   ``"off"`` (default) or ``"bilateral"``.
            extract_headers:    v2. Default ON. Fire a deterministic M3 call
                for the seven PGN header fields (White/Black/Round/Date/
                Event/Site/Result) with per-field confidence.
            beam_per_cell:      v2. Default ON. Aggregate per-cell SAN
                candidates across all passes and emit
                :attr:`ConsensusResult.cell_candidates`.
            metadata_hints:     v2. Optional caller-supplied header hints
                (e.g. ``{"white": "X"}``) used as a cross-check by the
                header-extraction prompt.

        Returns:
            :class:`ConsensusResult` with the chosen text + per-pass
            evidence, optional per-cell candidates, optional headers.
        """
        settings = self._settings
        t0 = time.time()
        candidates: list[dict[str, Any]] = []

        # ---- Preprocessing ----
        # Apply rotation/contrast/denoise strategies once up front, producing
        # 1-N PNG byte variants. In "off"/"auto" mode there's exactly 1
        # variant; in "three" mode there are 3.
        rotations, variants, detected_rotation = decode_image_to_rgb(
            image_bytes,
            rotation_strategy=rotation_strategy,
            contrast_strategy=contrast_strategy,
            denoise_strategy=denoise_strategy,
        )
        if not variants:
            raise ValueError("OCR_NO_VARIANTS: preprocessing produced no usable image")
        primary_bytes = variants[0]

        # ---- Pass 1 + 2a/b/c language-biased (parallel) ----
        # In multi-orientation mode the engine fires 4× the passes.
        ocr_tasks: list[tuple[str, str | None, bytes]] = []
        for variant_idx, vbytes in enumerate(variants):
            rot_label = f"rot{rotations[variant_idx]}" if len(variants) > 1 else "rot0"
            for lang in (None, "pl", "en", "en"):
                tag = (
                    "raw"
                    if lang is None
                    else (
                        "pl_hint"
                        if lang == "pl"
                        else f"en_hint_{ocr_tasks.count((rot_label, lang, vbytes))}"
                    )
                )
                # Stable & unique pass label per (rotation, language) pair.
                suffix = ""
                if lang == "en":
                    n_en = sum(1 for t in ocr_tasks if t[0].startswith(rot_label) and t[1] == "en")
                    suffix = f"_{n_en + 1}"
                pass_label = f"{rot_label}_{tag}{suffix}"
                ocr_tasks.append((pass_label, lang, vbytes))

        # Fire all tasks in parallel.
        results = await asyncio.gather(
            *(self._engine.ocr(t[2], language_hint=t[1]) for t in ocr_tasks),
            return_exceptions=True,
        )

        for (pass_label, lang, _vbytes), result in zip(ocr_tasks, results, strict=False):
            if isinstance(result, Exception):
                log.warning("consensus pass %s failed: %s", pass_label, result)
                candidates.append(
                    {
                        "pass_label": pass_label,
                        "language_hint": lang,
                        "text": "",
                        "error": str(result),
                        "parse_rate": 0.0,
                        "rotation": pass_label.split("_")[0],
                    }
                )
            elif isinstance(result, dict):
                text_value = result.get("text", "")
                candidates.append(
                    {
                        "pass_label": pass_label,
                        "language_hint": lang,
                        "text": text_value,
                        "error": None,
                        "parse_rate": _python_chess_parse_rate(_normalize_unicode(text_value)),
                        "latency_ms": float(result.get("latency_ms", 0.0) or 0.0),
                        "rotation": pass_label.split("_")[0],
                    }
                )

        # Score each candidate. Slight bias to "raw" pass within its rotation.
        for cand in candidates:
            base_score = cand["parse_rate"]
            is_raw = cand["pass_label"].endswith("_raw")
            cand["score"] = base_score + (0.2 if is_raw else 0.0)

        pass_count = 1

        # ---- Verifier (single pass over the primary variant) ----
        verifier_agreement: bool | None = None
        verifier_notes: str | None = None
        verifier_confidence: float | None = None
        selected_index = max(range(len(candidates)), key=lambda i: candidates[i]["score"])

        non_empty = [c for c in candidates if c.get("text")]
        if non_empty and len(non_empty) >= 2:
            try:
                verdict = await self._engine.verify(
                    primary_bytes,
                    [c["text"] for c in candidates if c.get("text")][:3],
                )
                verifier_confidence = float(verdict.get("confidence", 0.5))
                verifier_notes = verdict.get("notes")
                non_empty_indices = [i for i, c in enumerate(candidates) if c.get("text")]
                if 0 <= int(verdict.get("best_index", 0)) < len(non_empty_indices):
                    selected_index = non_empty_indices[int(verdict["best_index"])]
                verifier_agreement = (
                    candidates[selected_index]["text"]
                    == non_empty[int(verdict.get("best_index", 0))]["text"]
                )
                pass_count += 1
            except Exception as exc:
                log.warning("consensus verifier failed: %s", exc)

        # ---- Sanity sweep on primary variant if confidence still low ----
        if (
            verifier_confidence is not None
            and verifier_confidence < settings.target_verifier_confidence
            and pass_count < settings.max_passes
        ):
            try:
                sanity = await self._engine.ocr(primary_bytes)
                sanity_candidate = {
                    "pass_label": "sanity",
                    "language_hint": None,
                    "text": sanity["text"],
                    "error": None,
                    "parse_rate": _python_chess_parse_rate(_normalize_unicode(sanity["text"])),
                    "latency_ms": sanity.get("latency_ms", 0.0),
                    "score": 0.0,
                    "rotation": "rot0",
                }
                sanity_candidate["score"] = sanity_candidate["parse_rate"]
                candidates.append(sanity_candidate)
                if sanity_candidate["score"] > candidates[selected_index]["score"]:
                    selected_index = len(candidates) - 1
                    verifier_agreement = False
                pass_count += 1
            except Exception as exc:
                log.warning("consensus sanity pass failed: %s", exc)

        chosen = candidates[selected_index]
        raw_text = chosen["text"] or ""
        # M3 may prefix its answer with ``<think>...</think>`` reasoning or
        # narrative commentary. Strip both so callers receive the PGN payload.
        raw_text = _strip_think_blocks(raw_text)
        raw_text = _extract_pgn_payload(raw_text)
        detected_language = "pl" if _detect_polish(raw_text) else "en"
        if hint_language:
            detected_language = hint_language

        canonical_text = _normalize_unicode(raw_text)
        total_latency_ms = (time.time() - t0) * 1000.0

        # ---- v2: per-cell candidate aggregation ----
        cell_candidates: list[CellCandidate] = []
        if beam_per_cell:
            cell_candidates = _aggregate_cell_candidates(candidates)

        # ---- v2: structured header extraction ----
        headers: dict[str, HeaderField] = {}
        if extract_headers:
            try:
                detected = await self._engine.extract_headers(
                    primary_bytes, hint_metadata=metadata_hints
                )
                merged = merge_with_hints(
                    {
                        k: {"value": v["value"], "confidence": v["confidence"]}
                        for k, v in detected.items()
                    },
                    metadata_hints,
                )
                headers = {
                    name: HeaderField(
                        value=info.get("value"), confidence=float(info.get("confidence", 0.0))
                    )
                    for name, info in merged.items()
                }
            except Exception as exc:
                log.warning("header extraction failed: %s", exc)
                headers = {
                    name: HeaderField(value=None, confidence=0.0) for name in empty_headers()
                }

        # Extract cell crops from primary variant for problem forensics
        cell_crops: dict[int, str] = {}
        try:
            from PIL import Image
            from core.chess_ocr.scoresheet import (
                detect_scoresheet_layout,
                extract_all_cell_crops,
            )

            pil_img = Image.open(io.BytesIO(primary_bytes))
            layout = detect_scoresheet_layout(pil_img)
            cell_crops = extract_all_cell_crops(pil_img, layout, max_ply=60)
        except Exception as exc:
            log.debug("could not extract cell crops: %s", exc)

        return ConsensusResult(
            raw_text=raw_text,
            canonical_text=canonical_text,
            detected_language=detected_language,
            candidates=candidates,
            selected_candidate_index=selected_index,
            verifier_agreement=verifier_agreement,
            verifier_notes=verifier_notes,
            verifier_confidence=verifier_confidence,
            ocr_engine_used="minimax",
            ocr_model_version=self._settings.model,
            pass_count=pass_count,
            total_latency_ms=total_latency_ms,
            auto_rotation_applied=detected_rotation,
            cell_candidates=cell_candidates,
            headers=headers,
            cell_crops=cell_crops,
        )


__all__ = ["ConsensusOrchestrator", "ConsensusResult"]
