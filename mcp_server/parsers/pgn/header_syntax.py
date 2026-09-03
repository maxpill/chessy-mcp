"""Strict-tag / header-line validation.

Lives separately from the conversational cleaner because the two have very
different goals:

    - :func:`sanitize_malformed_pgn_header_lines` — lenient-mode pre-pass:
      emits ``metadata_warning`` strings the rest of the system surfaces to
      callers, drops malformed lines (replacing them with empty lines so
      offsets are preserved).
    - :func:`validate_strict_header_syntax` — strict-mode gate: raises
      ``STRICT_PGN_ERROR`` if any line that looks tag-like fails
      :func:`is_canonical_tag_line`.

A malformed line between valid tags used to split the tag cluster and
silently discard otherwise valid metadata. The lenient pre-pass inspects only
the pre-move prefix and only activates when that prefix contains at least one
valid PGN tag, so bracket-looking prose in ordinary movetext is left untouched.
"""

from __future__ import annotations

import re
from typing import Final

from mcp_server.parsers.pgn.multiline_tags import (
    is_canonical_tag_line,
    normalize_multiline_tags,
)
from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX
from mcp_server.parsers.pgn.unicode import normalize_unicode_pgn_results
from mcp_server.parsers.pgn_sanitize import _mask_comments_and_escapes

__all__ = [
    "sanitize_malformed_pgn_header_lines",
    "validate_strict_header_syntax",
]


_FIRST_MOVE_RE: Final[re.Pattern[str]] = re.compile(r"\b1\s*[\.\:]\s*[A-Za-z]")


def validate_strict_header_syntax(text: str) -> None:
    """Reject malformed PGN tag lines that lenient cleaning would otherwise discard."""
    normalized = normalize_unicode_pgn_results(text)
    masked = _mask_comments_and_escapes(normalized)
    raw_lines = normalized.splitlines()
    masked_lines = masked.splitlines()
    for index, visible in enumerate(masked_lines):
        if not re.match(r"^\s*\[[A-Za-z0-9_]+\b", visible):
            continue
        raw = raw_lines[index] if index < len(raw_lines) else visible
        if not is_canonical_tag_line(raw):
            raise ValueError(
                f"STRICT_PGN_ERROR: Malformed PGN tag syntax on line {index + 1}: {raw.strip()!r}"
            )


def sanitize_malformed_pgn_header_lines(text: str, strict: bool = False) -> tuple[str, list[str]]:
    """Reject or remove malformed tag-pair lines before PGN extraction.

    Returns ``(cleaned_text, warnings)``. Strict mode raises instead of
    emitting warnings.
    """
    normalized = normalize_multiline_tags(text)
    lines = normalized.splitlines(keepends=True)
    if not lines:
        return normalized, []

    first_move_line = len(lines)
    for idx, line in enumerate(lines):
        if _FIRST_MOVE_RE.search(line):
            first_move_line = idx
            break

    prefix = "".join(lines[:first_move_line])
    if TAG_PAIR_REGEX.search(_mask_comments_and_escapes(prefix)) is None:
        return normalized, []

    # Header-only PGNs (no first-move line) sanitize every line in the prefix;
    # previously the early-return silently dropped malformed headers in
    # header-only inputs.
    scan_end = first_move_line if first_move_line < len(lines) else len(lines)

    warnings: list[str] = []
    for idx in range(scan_end):
        stripped = lines[idx].strip()
        if not stripped.startswith("["):
            continue
        if is_canonical_tag_line(stripped):
            continue
        # Skip lines that mix a valid tag with extra content (e.g.
        # `[Result "*"] *` — a valid tag followed by the game result
        # token on the same line). The legacy contract accepted such
        # mixed lines as part of the conversational PGN dialect.
        if TAG_PAIR_REGEX.search(stripped) is not None:
            continue
        warning = f"Malformed PGN header line ignored: {stripped!r}."
        if strict:
            raise ValueError(f"STRICT_VALIDATION_ERROR: {warning}")
        warnings.append(warning)
        newline = (
            "\r\n" if lines[idx].endswith("\r\n") else ("\n" if lines[idx].endswith("\n") else "")
        )
        lines[idx] = newline

    return "".join(lines), warnings
