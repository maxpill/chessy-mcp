"""PGN game extractor — top-level orchestrator.

Layered approach:

    :func:`extract_canonical_pgn_text` — first-pass cleaner: NUL/zero-width /
    Unicode-result normalization, markdown fence ranking, conversational
    preamble/trailer trim, returns raw text the caller can then parse.
    :func:`extract_game` — public entry point; pre-sanitizes headers,
    multi-game-checks, canonicalizes, parses, applies strict-mode surface
    checks. Returns a :class:`chess.pgn.Game`.
    :func:`extract_game_inner` — parsing-only path used by both the public
    ``extract_game` and re-entry from ``parse_pgn_game_candidate`. Implements
    comment-only input shortcut (audit R4-§B), FEN validation, and bare-moves
    fallback (audit P0). Lives in :mod:`mcp_server.parsers.pgn.game_inner`.
    :func:`parse_pgn_game_candidate` — delegate to ``chess.pgn.read_game`
    with preflight checks. Lives in
    :mod:`mcp_server.parsers.pgn.parse_candidate`.
"""

from __future__ import annotations

import re

import chess
import chess.pgn

from mcp_server.parsers.pgn.conversational import clean_conversational_text
from mcp_server.parsers.pgn.game_inner import extract_game_inner
from mcp_server.parsers.pgn.header_syntax import (
    sanitize_malformed_pgn_header_lines,
    validate_strict_header_syntax,
)
from mcp_server.parsers.pgn.multiline_tags import normalize_multiline_tags
from mcp_server.parsers.pgn.multiple_games import check_multiple_games
from mcp_server.parsers.pgn.parse_candidate import parse_pgn_game_candidate
from mcp_server.parsers.pgn.tokens import validate_strict_mainline_surface


__all__ = [
    "extract_canonical_pgn_text",
    "extract_game",
    "extract_game_inner",
    "parse_pgn_game_candidate",
]


_FENCE_RE = re.compile(r"```([a-zA-Z0-9_-]*)\s*([\s\S]*?)\s*```")


def extract_canonical_pgn_text(text: str, *, warnings: list[str] | None = None) -> str:
    """Isolate the canonical PGN text from markdown fences, conversational preambles, and trailers.

    Bug fix (chessy-mcp-deep-audit §14): when NUL / ZWSP / BOM / NBSP are
    silently stripped from the input, append a metadata_warning to the
    caller-supplied warnings list so the sanitization is observable. The
    warnings channel is optional for back-compat with callers that don't
    surface warnings yet.
    """
    from mcp_server.parsers.pgn.unicode import normalize_unicode_pgn_results
    from mcp_server.parsers.pgn_sanitize import _strip_pgn_escape_lines

    stripped: list[str] = []
    if "\x00" in text:
        stripped.append("NUL")
    if "\u200b" in text:
        stripped.append("ZWSP")
    if "\ufeff" in text:
        stripped.append("BOM")
    if "\u00a0" in text:
        stripped.append("NBSP")

    cleaned = (
        text.replace("\x00", "")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .strip()
    )
    if stripped and warnings is not None:
        warnings.append(f"Sanitized input: stripped invisible characters ({', '.join(stripped)}).")
    cleaned = normalize_unicode_pgn_results(cleaned)
    if not cleaned:
        raise ValueError("Empty chess game/PGN input provided")

    fences = list(_FENCE_RE.finditer(cleaned))
    if fences:
        ranked: list[tuple[int, str]] = []
        for m in fences:
            lang = (m.group(1) or "").strip().lower()
            body = m.group(2).strip("`'\" \t\r\n")
            if not body:
                continue
            if lang in ("pgn", "chess"):
                score = 100
            elif re.search(r"\b1\s*[\.\:]\s*[A-Za-z]|\[\s*[A-Za-z0-9_]+\s+\"", body):
                score = 50
            else:
                score = 10
            ranked.append((score, body))

        if ranked:
            ranked.sort(key=lambda x: x[0], reverse=True)
            best_body = ranked[0][1]
            return _strip_pgn_escape_lines(normalize_multiline_tags(best_body))

    cleaned = normalize_multiline_tags(cleaned)
    cleaned_conv = clean_conversational_text(cleaned)
    if cleaned_conv:
        return _strip_pgn_escape_lines(normalize_multiline_tags(cleaned_conv))

    return _strip_pgn_escape_lines(cleaned)


def extract_game(text: str, strict: bool = False) -> chess.pgn.Game:
    """Extract a ``chess.pgn.Game` from raw, dirty, annotated, or conversational text."""
    sanitized, _header_warnings = sanitize_malformed_pgn_header_lines(text, strict=strict)
    check_multiple_games(sanitized)
    canonical = extract_canonical_pgn_text(sanitized)
    game = extract_game_inner(canonical, strict=strict)
    if strict:
        validate_strict_header_syntax(canonical)
        validate_strict_mainline_surface(canonical, game)
    return game
