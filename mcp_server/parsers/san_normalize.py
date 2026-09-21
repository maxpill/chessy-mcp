"""SAN (Standard Algebraic Notation) language normalization.

PGN's canonical piece-letter set is English (``PNBRQK``). Polish uses
different single letters — ``H/W/G/S`` for queen/rook/bishop/knight —
plus the colon ``:`` as a capture marker and ``0-0`` / ``0-0-0`` (digit
form) for castling. When a handwritten Polish score sheet uses these
non-English forms, ``python-chess``'s ``SAN_REGEX`` (which only accepts
``[NBKRQ]``) silently mis-parses or rejects every move.

This module translates Polish SAN tokens to canonical English SAN
**before** they reach ``python-chess``. It also normalizes:

- Castling: ``0-0`` / ``o-o`` / ``O-O-O`` (any case) → ``O-O`` / ``O-O-O``.
- Capture markers: Polish ``:`` → ``x``.
- Promotion forms: ``e8H`` (Polish letter, no ``=``) / ``e8=H`` → ``e8=Q``.
- Check/mate markers: ``X`` (Polish mate convention) → ``#``.

It also auto-detects whether the movetext is Polish or English by
counting distinctive Polish piece letters (H, W, G) and the Polish
``:`` capture marker.

Public surface:

    - :data:`SUPPORTED_LANGUAGES` — frozenset of ``{"en", "pl"}``.
    - :func:`detect_language` — return ``(lang, confidence)`` from PGN text.
    - :func:`normalize_token` — rewrite one SAN token (with per-change list).
    - :func:`normalize_pgn` — orchestrator returning canonical text + metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final


# Roles in canonical English-letter order. Index = piece role.
_PAWN: Final[int] = 0
_KNIGHT: Final[int] = 1
_BISHOP: Final[int] = 2
_ROOK: Final[int] = 3
_QUEEN: Final[int] = 4
_KING: Final[int] = 5

# English piece letters (canonical, what python-chess accepts).
_EN_PIECE_LETTERS: Final[tuple[str, ...]] = ("", "N", "B", "R", "Q", "K")

# Polish piece letters. Pawn is implicit (no letter), so empty string.
# S=Skoczek (knight), G=Goniec (bishop), W=Wieża (rook), H=Hetman (queen),
# K=Król (king).
_PL_PIECE_LETTERS: Final[tuple[str, ...]] = ("", "S", "G", "W", "H", "K")

DEFAULT_LANGUAGE: Final[str] = "en"
SUPPORTED_LANGUAGES: Final[frozenset[str]] = frozenset({"en", "pl"})

# Castling canonical patterns. Both digit-zero and letter-O variants
# (and lowercase) are recognized. Full-match so we never misfire on a
# token that happens to contain "0-0" mid-string.
_CASTLE_QUEENSIDE_RE: Final[re.Pattern[str]] = re.compile(r"^(?:O-O-O|o-o-o|0-0-0)[+#?!]*$")
_CASTLE_KINGSIDE_RE: Final[re.Pattern[str]] = re.compile(r"^(?:O-O|o-o|0-0)[+#?!]*$")

# Pawn-capture prefix: Polish writes "e:d5" instead of "exd5".
_PAWN_CAPTURE_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^([a-h])(:)([a-h][1-8])")

# Polish-distinctive piece letters. These letters are NOT used by English
# notation, so their presence is a strong Polish signal.
# (W, G, H are unique to Polish among the supported languages.)
_PL_DISTINCTIVE_LETTERS: Final[frozenset[str]] = frozenset({"W", "G", "H"})

# Union of every Polish piece letter. Used by ``_starts_with_piece_letter``
# to detect tokens that begin with a Polish piece identifier.
_PL_PIECE_LETTER_CLASS: Final[str] = "".join(_PL_PIECE_LETTERS[1:])


@dataclass(frozen=True)
class NormalizationResult:
    """Per-call summary returned by :func:`normalize_pgn`."""

    canonical_text: str
    detected_language: str
    detected_language_confidence: float
    normalization_changes: tuple[str, ...] = field(default_factory=tuple)


def _pl_piece_to_english(piece_letter: str) -> str | None:
    """Map a Polish piece letter to its English single-letter equivalent.

    Returns ``None`` when the letter is not a Polish piece letter (likely
    an OCR mis-read or already-English token).
    """
    upper = piece_letter.upper()
    for idx in range(1, 6):
        if upper == _PL_PIECE_LETTERS[idx]:
            return (
                _EN_PIECE_LETTERS[idx].lower() if piece_letter.islower() else _EN_PIECE_LETTERS[idx]
            )
    return None


def detect_language(text: str) -> tuple[str, float]:
    """Heuristic PGN-language detector (Polish vs English).

    Returns ``("pl", confidence)`` when Polish indicators are present,
    otherwise ``("en", confidence)``. Confidence is in ``[0.5, 0.99]``.

    Polish indicators (in priority order):
        1. Polish-distinctive piece letters (W, G, H) — strong signal.
        2. Polish ``:`` capture marker — strong signal when count > ``x``.
        3. Digit-form castling (``0-0`` / ``0-0-0``) — weak signal
           (sometimes used in English handwritten sheets too).

    Variations (RAVs in parens) are NOT stripped before scoring — a Polish
    move inside a variation is still a Polish indicator.
    """
    if not text or not text.strip():
        return DEFAULT_LANGUAGE, 0.5

    # Strip headers, comments, and result tokens, but KEEP variations and
    # do NOT strip digits globally — scrubbing "Sf3" → "Sf" would destroy
    # piece-letter evidence.
    cleaned = text
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"\{[^}]*\}", " ", cleaned)
    cleaned = re.sub(r"\b1-0\b|\b0-1\b|\b1/2-1/2\b|\*", " ", cleaned)

    # Polish-distinctive letter counts (W, G, H each weigh 2.0). These
    # letters are used ONLY as Polish piece letters in our supported set,
    # so their presence is unambiguous evidence.
    distinctive_count = sum(
        len(re.findall(re.escape(letter), cleaned, flags=re.IGNORECASE))
        for letter in _PL_DISTINCTIVE_LETTERS
    )

    # Polish S appears as a piece-letter prefix (Sd7, S:f3, etc.).
    s_piece_count = len(re.findall(r"\bS[a-h][1-8]", cleaned))

    # Capture marker evidence — Polish uses ":".
    colon_count = len(re.findall(r":(?=[a-h])", cleaned))
    x_count = len(re.findall(r"[xX](?=[a-h])", cleaned))

    # Digit-form castling (Polish handwritten sheets tend to use 0-0).
    digit_castle_count = len(re.findall(r"\b0-0-?\b", cleaned))

    # Scoring: distinctive letters and colons are decisive.
    polish_score = (
        distinctive_count * 2.0 + colon_count * 1.5 + s_piece_count * 0.5 + digit_castle_count * 0.3
    )
    english_score = x_count * 0.5  # English uses x for captures

    if polish_score == 0 and english_score == 0:
        return DEFAULT_LANGUAGE, 0.5

    if colon_count >= 3 and colon_count > x_count:
        return "pl", 0.9

    if distinctive_count >= 2:
        confidence = min(0.99, 0.7 + 0.05 * distinctive_count)
        return "pl", confidence

    # Polish S is ambiguous (also common in English prose), so we need
    # either multiple S-pieces or a stronger Polish indicator. If there's
    # some Polish evidence and NO English evidence, default to Polish.
    if polish_score > 0 and english_score == 0:
        return "pl", min(0.85, 0.5 + 0.1 * polish_score)

    if polish_score > english_score + 1.0:
        margin = (polish_score - english_score) / max(1.0, polish_score)
        return "pl", min(0.99, 0.6 + 0.3 * margin)

    return DEFAULT_LANGUAGE, 0.5


def normalize_token(
    token: str,
    language: str = DEFAULT_LANGUAGE,
) -> tuple[str, tuple[str, ...]]:
    """Normalize one SAN move token to canonical English SAN.

    Returns ``(canonical_token, applied_changes)``. ``applied_changes``
    is a tuple of human-readable change descriptions (e.g.
    ``"piece_H→Q"``, ``"capture_:→x"``) so the OCR tool can surface
    them in the response.

    The function is conservative: unknown tokens are returned unchanged
    so the caller can still attempt to parse them via ``python-chess``.
    """
    if not token or language == DEFAULT_LANGUAGE:
        return token, ()

    if language != "pl":
        return token, ()

    changes: list[str] = []
    out = token.strip()
    if not out:
        return out, ()

    # Strip leading move-number punctuation (e.g. "1.e4" → "e4")
    out = re.sub(r"^(\d+[\.\:]+|\.+)", "", out).strip()
    if not out:
        return out, ()

    # 1. Castling — handle 0-0 / o-o / O-O / O-O-O with optional + / # / annotation
    # Preserve trailing NAG/annotation glyphs (!, ?, !!, ??).
    annotation_match = re.search(r"[!?]{1,2}$", out)
    annotation_suffix = annotation_match.group(0) if annotation_match else ""
    castling_core = out[: -len(annotation_suffix)] if annotation_suffix else out
    if _CASTLE_QUEENSIDE_RE.match(castling_core):
        suffix = "#" if "#" in castling_core else ("+" if "+" in castling_core else "")
        canonical = f"O-O-O{suffix}{annotation_suffix}"
        if canonical != out:
            changes.append("castling_queenside")
        return canonical, tuple(changes)
    if _CASTLE_KINGSIDE_RE.match(castling_core):
        suffix = "#" if "#" in castling_core else ("+" if "+" in castling_core else "")
        canonical = f"O-O{suffix}{annotation_suffix}"
        if canonical != out:
            changes.append("castling_kingside")
        return canonical, tuple(changes)

    # 2. Pawn-capture prefix: "e:d5" (Polish) → "exd5"
    m = _PAWN_CAPTURE_PREFIX_RE.match(out)
    if m:
        out = f"{m.group(1)}x{m.group(3)}{out[m.end() :]}"
        changes.append("pawn_capture_:→x")

    # 3. Capture marker normalization: ":" anywhere → "x"
    if ":" in out:
        new_out = out.replace(":", "x")
        if new_out != out:
            changes.append("capture_:→x")
            out = new_out

    # 4. Promotion forms: "e8H" / "e8=H" / "e8/H" / "f8(H)" → "e8=Q"
    promote_re = re.compile(r"^(.*?[a-h][18])\s*[\(=]?\s*([HSGWK])\s*\)?\s*([+#?!]*)$")
    m = promote_re.match(out)
    if m:
        prefix = m.group(1)
        promote_letter = m.group(2)
        suffix = m.group(3) or ""
        en_letter = _pl_piece_to_english(promote_letter)
        if en_letter is not None:
            changes.append(f"promotion_{promote_letter}→{en_letter}")
            return f"{prefix}={en_letter}{suffix}", tuple(changes)

    # 5. Piece-letter rewrite at start of token
    if out and out[0] in _PL_PIECE_LETTER_CLASS:
        original_first = out[0]
        en_letter = _pl_piece_to_english(out[0])
        if en_letter is not None and en_letter != original_first:
            out = en_letter + out[1:]
            changes.append(f"piece_{original_first}→{en_letter}")

    # 6. Trailing "X" mate marker (Polish convention) → "#"
    if out.endswith("X") and not out.endswith("xX"):
        out = out[:-1] + "#"
        changes.append("mate_X→#")

    # 7. Lowercase English piece letter at start of token (e.g. "nf3" from OCR)
    # This catches the case where OCR returned English letters as lowercase
    # and the language detector mistook them for a foreign piece letter.
    if out and out[0].islower() and out[0] in "kqrbn":
        out = out[0].upper() + out[1:]
        changes.append(f"uppercase_{out[0].lower()}→{out[0]}")

    return out, tuple(changes)


def _find_protected_regions(text: str) -> list[tuple[int, int]]:
    """Return half-open (start, end) ranges for headers, comments, and variations.

    Token normalization must skip these regions — rewriting inside a comment
    would corrupt player annotations and rewriting inside a header would
    corrupt event names containing chess piece letters.
    """
    regions: list[tuple[int, int]] = []
    for m in re.finditer(r"\[[^\]]*\]", text):
        regions.append((m.start(), m.end()))
    for m in re.finditer(r"\{[^}]*\}", text, flags=re.DOTALL):
        regions.append((m.start(), m.end()))
    depth = 0
    start = -1
    for idx, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                regions.append((start, idx + 1))
                start = -1
    return regions


def _is_in_protected_region(idx: int, ranges: list[tuple[int, int]]) -> bool:
    """Return True if ``idx`` falls inside any (start, end) range (half-open)."""
    for start, end in ranges:
        if start <= idx < end:
            return True
    return False


# Token surface regex — matches a single SAN move token. Engineered to be
# permissive on the optional pieces (so OCR noise like stray spaces or
# punctuation doesn't reject the match) but strict on the mandatory
# destination square so we don't match English words inside prose.
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:[KQRBNHSGW])?"  # optional piece letter (incl. Polish letters)
    r"(?:[a-h])?"  # optional disambiguation file
    r"(?:[1-8])?"  # optional disambiguation rank
    r"(?:[xX:—\-])?"  # optional capture marker
    r"(?:[a-h][1-8])"  # mandatory destination square
    r"(?:=?(?:[KQRBNHSGW]))?"  # optional promotion
    r"(?:[+#]{1,2})?"  # optional check/mate
    r"(?:[!]{1,2}|[?]{1,2})?"  # optional NAG/annotation
)


def normalize_pgn(
    text: str,
    language: str = "auto",
) -> NormalizationResult:
    """Normalize a full PGN text or movetext string to canonical English SAN.

    When ``language="auto"``, runs :func:`detect_language` first and uses
    the detector's verdict. Returns the rewritten text plus the detected
    language and a flattened list of applied changes.

    The function preserves ``[Tag]`` headers, ``{comments}``, and
    ``(variations)`` untouched — only move tokens are rewritten.
    """
    if not text:
        return NormalizationResult(
            canonical_text=text,
            detected_language=DEFAULT_LANGUAGE,
            detected_language_confidence=0.5,
            normalization_changes=(),
        )

    if language == "auto":
        detected, confidence = detect_language(text)
        effective_language = detected
    elif language in SUPPORTED_LANGUAGES:
        effective_language = language
        confidence = 0.99
    else:
        effective_language = DEFAULT_LANGUAGE
        confidence = 0.5

    if effective_language == DEFAULT_LANGUAGE:
        return NormalizationResult(
            canonical_text=text,
            detected_language=effective_language,
            detected_language_confidence=confidence,
            normalization_changes=(),
        )

    protected_regions = _find_protected_regions(text)
    changes: list[str] = []

    pieces: list[str] = []
    last_end = 0
    for m in _TOKEN_RE.finditer(text):
        if _is_in_protected_region(m.start(), protected_regions):
            continue
        raw = m.group(0)
        if re.fullmatch(r"\d+\.{1,3}", raw):
            continue
        if raw in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        canonical, token_changes = normalize_token(raw, effective_language)
        changes.extend(f"@{m.start()}:{c}" for c in token_changes)
        pieces.append(text[last_end : m.start()])
        pieces.append(canonical)
        last_end = m.end()
    pieces.append(text[last_end:])

    canonical_text = "".join(pieces)

    return NormalizationResult(
        canonical_text=canonical_text,
        detected_language=effective_language,
        detected_language_confidence=confidence,
        normalization_changes=tuple(changes),
    )


__all__ = [
    "DEFAULT_LANGUAGE",
    "SUPPORTED_LANGUAGES",
    "NormalizationResult",
    "detect_language",
    "normalize_pgn",
    "normalize_token",
]
