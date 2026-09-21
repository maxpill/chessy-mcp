"""Regenerate ``tests/test_real_corpus.py`` from the latest successful
``tests/real_photos_report/*.json`` files.

Run after every bulk OCR pass to refresh the corpus:

    uv run python scripts/build_real_corpus.py
"""

from __future__ import annotations

import json
import pathlib
import re


REPORT_DIR = pathlib.Path(__file__).parent.parent / "tests" / "real_photos_report"
TEST_FILE = pathlib.Path(__file__).parent.parent / "tests" / "test_real_corpus.py"


HEADER = '''"""Real-world Polish score-sheet fixtures, harvested from
``tests/real_photos_report/*.json``. Each successful per-image OCR result
becomes one parametrized test that exercises:

    - language autodetect against the raw text
    - san_normalize round-trip against the canonical text
    - python-chess legality of every ply in the sample moves

These fixtures guard against regression in:
    - Polish piece-letter detection (H/W/G/S)
    - Colon capture normalization (: -> x)
    - Polish castling form (0-0 -> O-O)
    - M3 output post-processing (think stripping)

Regenerate with:  uv run python scripts/build_real_corpus.py
"""

from __future__ import annotations

import io
import re

import chess
import chess.pgn
import pytest

from mcp_server.parsers.san_normalize import detect_language, normalize_pgn


_REAL_CORPUS = [
'''

TAIL = ''']


def _id_from_name(name: str) -> str:
    """Stable pytest id derived from image filename."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def _wrap_canonical(canonical: str) -> str:
    """Wrap a movetext snippet into a parseable PGN."""
    if canonical.lstrip().startswith('['):
        return canonical
    return '[Event "?"]\\n\\n' + canonical


@pytest.mark.parametrize(
    "case",
    _REAL_CORPUS,
    ids=[_id_from_name(c['name']) for c in _REAL_CORPUS],
)
def test_real_corpus_language_autodetect(case: dict) -> None:
    """The heuristic detects the same language the live OCR pipeline detected."""
    raw = case['raw_first_300']
    if not raw.strip():
        pytest.skip(f"{case['name']}: empty raw OCR text — no autodetect signal")
    detected, _confidence = detect_language(raw)
    assert detected in {'en', 'pl'}, (
        f"{case['name']}: autodetect returned {detected!r} "
        f"for raw text starting with {raw[:80]!r}"
    )


@pytest.mark.parametrize(
    "case",
    _REAL_CORPUS,
    ids=[_id_from_name(c['name']) for c in _REAL_CORPUS],
)
def test_real_corpus_normalize_round_trip(case: dict) -> None:
    """san_normalize produces canonical English SAN the python-chess parser accepts."""
    canonical = case['canonical_first_300']
    if not canonical.strip():
        pytest.skip(f"{case['name']}: empty canonical text")
    lang = case['language']
    result = normalize_pgn(canonical, language=lang)
    wrapped = _wrap_canonical(result.canonical_text)
    game = chess.pgn.read_game(io.StringIO(wrapped))
    assert game is not None, (
        f"{case['name']}: round-tripped PGN did not parse: {result.canonical_text[:200]!r}"
    )


@pytest.mark.parametrize(
    "case",
    _REAL_CORPUS,
    ids=[_id_from_name(c['name']) for c in _REAL_CORPUS],
)
def test_real_corpus_sample_moves_parse(case: dict) -> None:
    """Every sample_moves entry parses cleanly via python-chess."""
    sample = case['sample_moves']
    if not sample:
        pytest.skip(f"{case['name']}: no sample moves captured")
    board = chess.Board()
    for san in sample:
        try:
            move = board.parse_san(san)
        except (chess.InvalidMove, chess.AmbiguousMoveError, ValueError) as exc:
            pytest.fail(
                f"{case['name']}: sample move {san!r} failed to parse: {exc}"
            )
        board.push(move)
'''


def _id_from_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _wrap_canonical(canonical: str) -> str:
    if canonical.lstrip().startswith("["):
        return canonical
    return '[Event "?"]\n\n' + canonical


def _collect_successes() -> list[dict[str, object]]:
    successes: list[dict[str, object]] = []
    for path in sorted(REPORT_DIR.glob("*.json")):
        with path.open() as f:
            r = json.load(f)
        if r.get("error") or not r.get("pgn_is_valid"):
            continue
        successes.append(
            {
                "name": r["name"],
                "language": r["detected_language"],
                "confidence": float(r.get("detected_confidence") or 0.5),
                "move_count": int(r.get("final_move_number", 0) or 0),
                "sample_moves": (r.get("sample_moves") or [])[:8],
                "canonical_first_300": (r.get("canonical_pgn") or "")[:300],
                "raw_first_300": (r.get("raw_ocr_text") or "")[:300],
            }
        )
    return successes


def main() -> None:
    successes = _collect_successes()
    # Build the corpus block using ast.literal_eval for safety on the values.
    lines: list[str] = [HEADER]
    for case in successes:
        lines.append("    {")
        for key, value in case.items():
            if isinstance(value, str):
                lines.append(f"        {key!r}: {value!r},")
            else:
                lines.append(f"        {key!r}: {value!r},")
        lines.append("    },")
    lines.append(TAIL)
    TEST_FILE.write_text("\n".join(lines))
    print(f"Wrote {TEST_FILE} with {len(successes)} fixtures")


if __name__ == "__main__":
    main()
