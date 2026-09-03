"""PGN input sanitization.

Helpers that strip conversational PGN cruft (semicolon / brace comments,
``%`` escape lines, malformed header lines) before the canonical PGN
extractor runs. Public unprefixed names + underscored aliases kept so
existing call sites and the test suite keep their import paths.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "TAG_PAIR_REGEX",
    "mask_comments_and_escapes",
    "sanitize_brackets_in_variations_and_comments",
    "sanitize_malformed_pgn_header_lines",
    "strip_pgn_escape_lines",
    "unescape_pgn_tag_value",
]

TAG_PAIR_REGEX: Final[re.Pattern[str]] = re.compile(
    r'\[\s*([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\s*\]', re.DOTALL
)




def unescape_pgn_tag_value(val: str | None) -> str | None:
    if val is None:
        return None
    # R4-§E (2026-09-02 ultra audit round 4): strip embedded NUL bytes
    # from header values so callers see the natural string instead of a
    # corrupted version (and downstream code does not have to defend
    # against NULs in metadata).
    return val.replace("\x00", "").replace('\\"', '"').replace("\\\\", "\\")


def mask_comments_and_escapes(text: str) -> str:
    """Mask semicolon comments, % escape lines, and {braced comments} with spaces, preserving string length and linebreaks."""
    chars = list(text)
    n = len(chars)
    i = 0
    in_brace = False
    in_semi = False
    in_quote = False
    escape_next = False
    is_line_start = True

    while i < n:
        ch = chars[i]
        if ch in ("\r", "\n"):
            in_semi = False
            in_quote = False
            escape_next = False
            is_line_start = True
            i += 1
            continue

        if not in_brace and not in_semi:
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == '"':
                in_quote = not in_quote
            elif not in_quote:
                if is_line_start and ch == "%":
                    in_semi = True
                elif ch == ";":
                    in_semi = True
                    chars[i] = " "
                elif ch == "{":
                    in_brace = True
                    chars[i] = " "
        elif in_brace:
            if ch == "}":
                in_brace = False
            chars[i] = " "
        elif in_semi:
            chars[i] = " "

        if ch not in (" ", "\t"):
            is_line_start = False
        i += 1

    return "".join(chars)


def sanitize_brackets_in_variations_and_comments(text: str) -> str:
    """Mask '[' and ']' characters occurring inside variations (...) or comments so PGN readers don't break."""
    chars = list(text)
    n = len(chars)
    i = 0
    in_brace = False
    in_semi = False
    var_depth = 0
    is_line_start = True

    while i < n:
        ch = chars[i]
        if ch in ("\r", "\n"):
            in_semi = False
            is_line_start = True
            i += 1
            continue

        if is_line_start and ch == "%":
            in_semi = True

        if not in_brace and not in_semi:
            if ch == ";":
                in_semi = True
            elif ch == "{":
                in_brace = True
            elif ch == "(":
                var_depth += 1
            elif ch == ")":
                var_depth = max(0, var_depth - 1)
            elif var_depth > 0 and ch in ("[", "]"):
                chars[i] = " "
        elif in_brace:
            if ch == "}":
                in_brace = False
            elif ch in ("[", "]"):
                chars[i] = " "
        elif in_semi:
            if ch in ("[", "]"):
                chars[i] = " "

        if ch not in (" ", "\t"):
            is_line_start = False
        i += 1

    return "".join(chars)


def strip_pgn_escape_lines(text: str) -> str:
    """Strip lines starting with '%' in column 1 (or after whitespace) per PGN standard."""
    return re.sub(r"(?m)^[ \t]*%[^\r\n]*(?:\r?\n)?", "", text)


def sanitize_malformed_pgn_header_lines(text: str, strict: bool = False) -> tuple[str, list[str]]:
    """Reject or remove malformed tag-pair lines before PGN extraction.
    (pgn_sanitize delegates to pgn_canonical for the helpers it shares.)

    The conversational PGN cleaner clusters only syntactically valid tag
    pairs. A malformed line between valid tags used to split the cluster and
    silently discard otherwise valid metadata. We inspect only the pre-move
    prefix and only activate when that prefix contains at least one valid PGN
    tag, so bracket-looking prose in ordinary movetext is left untouched.
    P2 (2026-09-02 ultra audit): the legacy function returned no warnings
    for header-only PGNs (no `1. <move>` marker before the result) and for
    malformed lines that didn't match the regex pre-filter. Both classes now
    surface warnings in lenient mode.
    """
    normalized = _normalize_multiline_tags(text)
    lines = normalized.splitlines(keepends=True)
    if not lines:
        return normalized, []

    first_move_line = len(lines)
    for idx, line in enumerate(lines):
        if re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", line):
            first_move_line = idx
            break

    prefix = "".join(lines[:first_move_line])
    if TAG_PAIR_REGEX.search(_mask_comments_and_escapes(prefix)) is None:
        # No tag block present; nothing to sanitize.
        return normalized, []
    # P2 (2026-09-02 ultra audit): handle header-only PGNs (no first-move
    # line) by sanitizing every line in the prefix. Previously the
    # `first_move_line >= len(lines)` early return silently dropped
    # malformed headers in header-only inputs.
    scan_end = first_move_line if first_move_line < len(lines) else len(lines)

    warnings: list[str] = []
    for idx in range(scan_end):
        stripped = lines[idx].strip()
        if not stripped.startswith("["):
            continue
        if _is_canonical_tag_line(stripped):
            continue
        # Skip lines that mix a valid tag with extra content (e.g.
        # `[Result "*"] *` — a valid tag followed by the game result
        # token on the same line). The legacy contract accepted such
        # mixed lines as part of the conversational PGN dialect.
        if TAG_PAIR_REGEX.search(stripped) is not None:
            continue
        # P2 (2026-09-02 ultra audit): drop the regex pre-filter that was
        # silently dropping warnings for malformed tags that didn't happen
        # to match `^\[\s*[A-Za-z0-9_]+(?:\s|\])`. The canonical-tag-line
        # check above is the sole gate; everything else that looks tag-like
        # but isn't canonical is reported.
        warning = f"Malformed PGN header line ignored: {stripped!r}."
        if strict:
            raise ValueError(f"STRICT_VALIDATION_ERROR: {warning}")
        warnings.append(warning)
        newline = (
            "\r\n" if lines[idx].endswith("\r\n") else ("\n" if lines[idx].endswith("\n") else "")
        )
        lines[idx] = newline

    return "".join(lines), warnings


# Underscored aliases for backwards-compatible import paths.
_sanitize_malformed_pgn_header_lines = sanitize_malformed_pgn_header_lines
_strip_pgn_escape_lines = strip_pgn_escape_lines
_sanitize_brackets_in_variations_and_comments = sanitize_brackets_in_variations_and_comments
_mask_comments_and_escapes = mask_comments_and_escapes
_unescape_pgn_tag_value = unescape_pgn_tag_value
