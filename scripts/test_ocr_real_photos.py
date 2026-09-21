"""Integration test harness — runs the OCR pipeline against real chess score-sheet photos.

Usage:
    uv run python scripts/test_ocr_real_photos.py <directory> [--max N] [--mock]
    uv run python scripts/test_ocr_real_photos.py /Users/max/Desktop/pgny

When `--mock` is passed, the M3 API is not called — instead the harness
synthesizes plausible Polish-movetext responses so the rest of the
pipeline (preprocess + normalize + validate) can be exercised offline.

Output:
    - Per-image JSON report written to tests/real_photos_report/<name>.json
    - A consolidated CSV summary at tests/real_photos_report/summary.csv
    - A human-readable Markdown report at tests/real_photos_report/report.md
"""

from __future__ import annotations

import argparse
import asyncio
import chess
import chess.pgn
import csv
import io
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from core.chess_ocr.config import OCRSettings
from core.chess_ocr.consensus import ConsensusOrchestrator
from core.chess_ocr.engines.minimax import MinimaxEngine
from core.chess_ocr.preprocess import decode_image_to_rgb, detect_image_format
from mcp_server.parsers.san_normalize import (
    normalize_pgn,
)

log = logging.getLogger("chessy_mcp.real_photos_harness")


REPORT_DIR = Path(__file__).parent.parent / "tests" / "real_photos_report"


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Common opening markers for the PGN payload inside an M3 response.
_PGN_START_RE = re.compile(
    r"(\[Event\s|\[\s*[A-Z][A-Za-z]+\s+\"[^\"]*\"\]|(?:^|\s)1\.{1,3}|\b1-0\b|\b0-1\b|\b1/2-1/2\b|\bResult\s)",
    re.MULTILINE,
)


def _strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks M3 emits before the answer."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _extract_pgn_payload(text: str) -> str:
    """Find the actual PGN payload inside an M3 response.

    M3 sometimes prefixes the answer with reasoning (``<think>...</think>``)
    or with narrative commentary. This function strips both and returns the
    PGN-shaped slice that follows.
    """
    cleaned = _strip_think_blocks(text)
    match = _PGN_START_RE.search(cleaned)
    if match is None:
        return cleaned
    # Find the last `[` header before the first move; if no header, slice from the match.
    # Look for the FIRST '[' from the match position back to the start, plus the last header line.
    start = match.start()
    # Walk back to find the start of the line containing the first '[' if any.
    for ch_idx in range(start, -1, -1):
        if cleaned[ch_idx] == "[":
            return cleaned[ch_idx:].strip()
        if cleaned[ch_idx] == "\n" and ch_idx != start:
            # No header on the same line — start at the move number match.
            return cleaned[start:].strip()
    return cleaned[start:].strip()


@dataclass
class ImageReport:
    name: str
    format: str
    size_bytes: int
    width: int | None
    height: int | None
    raw_ocr_text: str = ""
    detected_language: str = ""
    detected_confidence: float = 0.0
    canonical_pgn: str = ""
    pgn_is_valid: bool = False
    final_move_number: int = 0
    pgn_parse_rate: float = 0.0
    pass_count: int = 0
    verifier_agreement: bool | None = None
    verifier_confidence: float | None = None
    total_latency_ms: float = 0.0
    normalization_changes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    sample_moves: list[str] = field(default_factory=list)
    final_fen: str = ""


# ---------------------------------------------------------------------------
# Mock OCR responses — used when --mock is passed or the real API is
# unreachable. Returns Polish/English content matching the image.
# ---------------------------------------------------------------------------
MOCK_POLISH_RESPONSE = """[Event "Wijk aan Zee"]
[White "?"]
[Black "?"]

1.e4 e5 2.Sf3 Sc6 3.Gb5 a6 4.Ga4 Sf6 5.O-O Ge7 6.We1 b5 7.Gb3 d6 8.c3 O-O 9.h3 Gb7 10.Wd1 1/2-1/2"""

MOCK_ENGLISH_RESPONSE = """[Event "Wijk aan Zee"]
[White "?"]
[Black "?"]

1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 d6 8.c3 O-O 9.h3 Bb7 10.d4 1/2-1/2"""


class MockMinimaxEngine:
    """Drops in for MinimaxEngine when --mock is passed. Returns deterministic text."""

    def __init__(self, settings: OCRSettings) -> None:
        self._settings = settings

    async def ocr(self, image_bytes: bytes, **kwargs: Any) -> dict[str, Any]:
        # Detect likely language from filename hint via kwargs.language_hint,
        # else fall back to a Polish default.
        lang_hint = kwargs.get("language_hint")
        text = (
            MOCK_POLISH_RESPONSE
            if (lang_hint == "pl" or lang_hint is None)
            else MOCK_ENGLISH_RESPONSE
        )
        return {
            "text": text,
            "model": "mock",
            "latency_ms": 1.0,
            "usage": {},
            "raw_response": {},
        }

    async def verify(self, image_bytes: bytes, candidates: list[str]) -> dict[str, Any]:
        return {
            "best_index": 0,
            "confidence": 0.9,
            "notes": "mock verify",
            "raw_verifier_text": json.dumps({"best_index": 0, "confidence": 0.9, "notes": "mock"}),
        }

    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Real or mock engine factory
# ---------------------------------------------------------------------------


def make_engine(use_mock: bool, settings: OCRSettings) -> Any:
    if use_mock:
        return MockMinimaxEngine(settings)
    return MinimaxEngine(settings)


# ---------------------------------------------------------------------------
# Image processing pipeline
# ---------------------------------------------------------------------------


def _image_dimensions(image_bytes: bytes) -> tuple[int | None, int | None]:
    """Read image dimensions without fully decoding (fast)."""
    try:
        if detect_image_format(image_bytes) == "heic":
            from pillow_heif.as_plugin import register_heif_opener

            register_heif_opener()
        img = Image.open(io.BytesIO(image_bytes))
        return img.size
    except Exception:
        return (None, None)


async def _process_one(
    image_path: Path,
    engine: Any,
    orchestrator: ConsensusOrchestrator | None,
    *,
    multi_pass: bool,
    hint_language: str | None,
) -> ImageReport:
    """Run the OCR pipeline on a single image and return a structured report."""
    report = ImageReport(
        name=image_path.name,
        format=image_path.suffix.lower().lstrip("."),
        size_bytes=image_path.stat().st_size,
        width=None,
        height=None,
    )

    try:
        image_bytes = image_path.read_bytes()
    except Exception as exc:
        report.error = f"READ_ERROR: {exc}"
        return report

    w, h = _image_dimensions(image_bytes)
    report.width = w
    report.height = h

    # Run the multi-pass pipeline via the orchestrator (which fires Pass 1
    # + parallel language-hinted passes + verifier + optional sanity pass).
    if orchestrator is None:
        orchestrator = ConsensusOrchestrator(engine=engine)

    t0 = time.time()
    try:
        if multi_pass:
            result = await orchestrator.run(image_bytes, hint_language=hint_language)
        else:
            # Single-pass for bulk tests — pre-decode + auto-rotate first so
            # the engine receives a payload M3 can parse.
            from core.chess_ocr.consensus import (
                ConsensusResult as _CR,
                ConsensusOrchestrator as _CO,
            )

            single_orch = _CO(engine=engine)
            _rot, variants, _det = decode_image_to_rgb(image_bytes, rotation_strategy="auto")
            ocr_result = await engine.ocr(variants[0])
            await single_orch.aclose()

            result = _CR(
                raw_text=_strip_think_blocks(ocr_result["text"]),
                canonical_text="",
                detected_language="en",
                candidates=[
                    {
                        "pass_label": "single",
                        "language_hint": None,
                        "text": ocr_result["text"],
                        "error": None,
                        "parse_rate": 0.0,
                        "score": 0.0,
                        "latency_ms": ocr_result.get("latency_ms", 0.0),
                    }
                ],
                selected_candidate_index=0,
                verifier_agreement=None,
                verifier_notes=None,
                verifier_confidence=None,
                ocr_engine_used="minimax",
                ocr_model_version=engine._settings.model,
                pass_count=1,
                total_latency_ms=ocr_result.get("latency_ms", 0.0),
            )
            result.canonical_text = result.raw_text  # populated below
    except Exception as exc:
        report.error = f"OCR_FAILED: {type(exc).__name__}: {exc}"
        report.total_latency_ms = (time.time() - t0) * 1000.0
        return report

    # M3 sometimes returns reasoning (``<think>...</think>``) or narrative
    # commentary before the actual PGN. Strip it so downstream parsing sees
    # the PGN-shaped payload only.
    extracted_text = _extract_pgn_payload(result.raw_text)
    report.raw_ocr_text = extracted_text
    report.detected_language = result.detected_language
    report.pass_count = result.pass_count
    report.verifier_agreement = result.verifier_agreement
    report.verifier_confidence = result.verifier_confidence
    report.canonical_text_for_normalize = result.canonical_text  # type: ignore[attr-defined]

    # Normalize via the Polish-aware normalizer.
    norm_result = normalize_pgn(extracted_text, language="auto")
    report.canonical_pgn = norm_result.canonical_text
    report.detected_language = norm_result.detected_language
    report.detected_confidence = norm_result.detected_language_confidence
    report.normalization_changes = list(norm_result.normalization_changes)

    # Validate via python-chess (per-ply replay).
    wrapped = (
        report.canonical_pgn
        if report.canonical_pgn.lstrip().startswith("[")
        else '[Event "?"]\n\n' + report.canonical_pgn
    )
    try:
        game = chess.pgn.read_game(io.StringIO(wrapped))
        if game is not None:
            report.pgn_is_valid = True
            board = game.board()
            tokens = report.canonical_pgn.split()
            result_tokens = {"1-0", "0-1", "1/2-1/2", "*"}
            ply_tokens = sum(
                1
                for tok in tokens
                if tok not in result_tokens
                and not tok.rstrip(".").isdigit()
                and tok not in {"$1", "$2", "$3", "$4", "$5", "$6", "$7", "$8"}
                and not (tok.startswith("[") and tok.endswith("]"))
            )
            parsed = len(list(game.mainline_moves()))
            report.pgn_parse_rate = (parsed / ply_tokens) if ply_tokens > 0 else 0.0
            for move in game.mainline_moves():
                report.sample_moves.append(board.san(move))
                board.push(move)
                report.final_move_number = board.fullmove_number - 1
            report.final_fen = board.fen()
        else:
            report.warnings.append("python_chess_returned_none")
    except Exception as exc:
        report.warnings.append(f"python_chess_failed: {type(exc).__name__}: {exc}")

    report.total_latency_ms = (time.time() - t0) * 1000.0
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _write_reports(reports: list[ImageReport]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for r in reports:
        path = REPORT_DIR / f"{Path(r.name).stem}.json"
        # Drop internal helper fields.
        data = {k: v for k, v in asdict(r).items() if not k.endswith("_for_normalize")}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _write_summary_csv(reports: list[ImageReport], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "format",
                "size_kb",
                "width",
                "height",
                "detected_language",
                "detected_confidence",
                "pgn_is_valid",
                "pgn_parse_rate",
                "final_move_number",
                "pass_count",
                "verifier_agreement",
                "verifier_confidence",
                "total_latency_ms",
                "normalization_changes_count",
                "error",
            ]
        )
        for r in reports:
            w.writerow(
                [
                    r.name,
                    r.format,
                    round(r.size_bytes / 1024, 1),
                    r.width,
                    r.height,
                    r.detected_language,
                    round(r.detected_confidence, 3),
                    r.pgn_is_valid,
                    round(r.pgn_parse_rate, 3),
                    r.final_move_number,
                    r.pass_count,
                    r.verifier_agreement,
                    r.verifier_confidence,
                    round(r.total_latency_ms, 1),
                    len(r.normalization_changes),
                    r.error or "",
                ]
            )


def _write_markdown_report(reports: list[ImageReport], path: Path, use_mock: bool) -> None:
    total = len(reports)
    successful = sum(1 for r in reports if r.error is None)
    valid_pgns = sum(1 for r in reports if r.pgn_is_valid)
    polish = sum(1 for r in reports if r.detected_language == "pl")
    english = sum(1 for r in reports if r.detected_language == "en")
    avg_latency = sum(r.total_latency_ms for r in reports) / max(1, total)
    avg_parse_rate = sum(r.pgn_parse_rate for r in reports) / max(1, total)
    avg_moves = sum(r.final_move_number for r in reports) / max(1, total)

    p95_latency = 0.0
    if reports:
        sorted_lat = sorted(r.total_latency_ms for r in reports)
        idx = max(0, min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95)))
        p95_latency = sorted_lat[idx]

    lines = [
        "# Real-Photos OCR Integration Report",
        "",
        f"**Mode**: {'MOCK' if use_mock else 'REAL M3'}",
        f"**Total images processed**: {total}",
        f"**Successful OCR**: {successful} ({successful / total * 100:.1f}%)",
        f"**Valid PGN (python-chess legality across every ply)**: {valid_pgns} ({valid_pgns / total * 100:.1f}%)",
        f"**Polish detected**: {polish}",
        f"**English detected**: {english}",
        f"**Average parse rate**: {avg_parse_rate:.3f}",
        f"**Average ply count per game**: {avg_moves:.1f}",
        f"**Average latency (ms)**: {avg_latency:.0f}",
        f"**p95 latency (ms)**: {p95_latency:.0f}",
        "",
        "## Per-image detail",
        "",
        "| Name | Format | KB | Dim | Lang | Conf | Valid | Parse% | Moves | Passes | VConf | Latency | Err |",
        "|------|--------|----|----|------|------|-------|--------|-------|--------|-------|---------|-----|",
    ]
    for r in reports:
        dim = f"{r.width}x{r.height}" if r.width and r.height else "?"
        err = r.error[:30] + "..." if r.error and len(r.error) > 30 else (r.error or "")
        lines.append(
            f"| {r.name} | {r.format} | {r.size_bytes // 1024} | {dim} | "
            f"{r.detected_language or '?'} | {r.detected_confidence:.2f} | "
            f"{'Y' if r.pgn_is_valid else 'N'} | {r.pgn_parse_rate * 100:.0f}% | "
            f"{r.final_move_number} | {r.pass_count} | "
            f"{r.verifier_confidence if r.verifier_confidence is not None else '-'} | "
            f"{r.total_latency_ms:.0f}ms | {err} |"
        )

    lines.extend(
        [
            "",
            "## Sample first 3 moves from each successful OCR",
            "",
        ]
    )
    for r in reports:
        if not r.sample_moves:
            continue
        lines.append(f"### {r.name}")
        lines.append(
            f"Detected language: **{r.detected_language}** ({r.detected_confidence:.2f} confidence)"
        )
        lines.append(f"Sample moves: `{' '.join(r.sample_moves[:3])}` ...")
        lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="OCR integration test on real photos")
    parser.add_argument("directory", type=Path, help="Directory containing photos")
    parser.add_argument("--max", type=int, default=None, help="Max images to process")
    parser.add_argument("--mock", action="store_true", help="Use mock M3 engine")
    parser.add_argument("--hint-language", choices=["en", "pl"], default=None)
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Run a single OCR pass per image (faster; bypasses multi-pass consensus)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of images to OCR in parallel (default 1; forced to 1 with multi-pass consensus).",
    )
    args = parser.parse_args()

    # Multi-pass consensus + parallel image processing is dangerous (would
    # multiply to 18 concurrent M3 calls per image); force single-pass.
    if args.concurrency > 1 and not args.single_pass:
        log.warning("Forcing --single-pass because --concurrency > 1 would multiply M3 calls")
        args.single_pass = True

    if not args.directory.is_dir():
        print(f"ERROR: {args.directory} is not a directory")
        sys.exit(1)

    # Collect candidate image files (HEIC + JPG/PNG).
    extensions = {".heic", ".jpg", ".jpeg", ".png", ".webp", ".gif"}
    images = sorted(
        [p for p in args.directory.iterdir() if p.suffix.lower() in extensions and p.is_file()]
    )
    if args.max:
        images = images[: args.max]

    print(f"Found {len(images)} images in {args.directory}")
    print(f"Mode: {'MOCK' if args.mock else 'REAL M3'}")
    print()

    settings = OCRSettings()
    if not args.mock and not settings.api_key:
        print("ERROR: env var `n` is not set; cannot call real M3.")
        print("Set `n=<your-key>` in .env, or pass --mock to use the offline harness.")
        sys.exit(1)

    engine = make_engine(args.mock, settings)
    orchestrator = ConsensusOrchestrator(settings, engine)

    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    # Retry-once policy on transient M3 failures (529/timeout/unreachable).
    from mcp_server.ocr.client import (
        OCRUnavailable,
        OCRTimeout,
        OCRUnreachable,
        OCRAuthError,
    )

    async def _process_with_retry(image_path: Path) -> ImageReport:
        """Process one image with a single retry on transient errors."""
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                return await _process_one(
                    image_path,
                    engine,
                    orchestrator,
                    multi_pass=not args.single_pass,
                    hint_language=args.hint_language,
                )
            except (TimeoutError, OCRUnavailable, OCRTimeout, OCRUnreachable) as exc:
                last_exc = exc
                if attempt == 1:
                    log.warning(
                        "transient error on %s (attempt 1): %s — retrying in 5s",
                        image_path.name,
                        exc,
                    )
                    await asyncio.sleep(5.0)
                else:
                    log.warning(
                        "transient error on %s (attempt 2): %s — giving up",
                        image_path.name,
                        exc,
                    )
        # Both attempts failed with transient errors; surface as a structured failure.
        return ImageReport(
            name=image_path.name,
            format=image_path.suffix.lstrip("."),
            size_bytes=image_path.stat().st_size,
            width=None,
            height=None,
            error=f"OCR_RETRY_EXHAUSTED: {type(last_exc).__name__}: {last_exc}",
        )

    async def _process_one_bounded(image_path: Path, slot: int) -> ImageReport:
        async with semaphore:
            print(
                f"[{slot:>2}/{len(images)}] {image_path.name} ...",
                flush=True,
            )
            try:
                report = await _process_with_retry(image_path)
            except OCRAuthError as exc:
                report = ImageReport(
                    name=image_path.name,
                    format=image_path.suffix.lstrip("."),
                    size_bytes=image_path.stat().st_size,
                    width=None,
                    height=None,
                    error=f"OCR_AUTH_ERROR: {exc}",
                )
            except Exception as exc:
                report = ImageReport(
                    name=image_path.name,
                    format=image_path.suffix.lstrip("."),
                    size_bytes=image_path.stat().st_size,
                    width=None,
                    height=None,
                    error=f"HARNESS_ERROR: {type(exc).__name__}: {exc}",
                )
            status = "OK" if not report.error else f"ERR: {report.error[:40]}"
            valid = "PGN✓" if report.pgn_is_valid else "PGN✗"
            print(
                f"  → {valid} {report.detected_language or '?'} "
                f"{report.final_move_number}moves {report.total_latency_ms / 1000:.1f}s {status}",
                flush=True,
            )
            return report

    print(f"Concurrency: {args.concurrency} image(s) in flight")
    tasks = [asyncio.create_task(_process_one_bounded(p, i)) for i, p in enumerate(images, start=1)]
    reports = await asyncio.gather(*tasks, return_exceptions=False)

    await engine.aclose()
    await orchestrator.aclose()

    _write_reports(reports)
    _write_summary_csv(reports, REPORT_DIR / "summary.csv")
    _write_markdown_report(reports, REPORT_DIR / "report.md", args.mock)

    # Print consolidated summary.
    total = len(reports)
    valid = sum(1 for r in reports if r.pgn_is_valid)
    print()
    print(f"Done. {valid}/{total} produced valid PGNs.")
    print(f"Per-image JSON: {REPORT_DIR}/<name>.json")
    print(f"Summary CSV: {REPORT_DIR}/summary.csv")
    print(f"Markdown report: {REPORT_DIR}/report.md")


if __name__ == "__main__":
    asyncio.run(main())
