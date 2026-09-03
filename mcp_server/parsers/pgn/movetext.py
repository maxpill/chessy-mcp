"""Movetext result detection, completion-detection, and prose discrimination.

The result token (``1-0``, ``0-1``, ``1/2-1/2``, ``*``) is a single-token
PGN element — but locating it requires walking past comments, variations,
and the canonical header block first. The two helpers here share the same
header-detection and variation-depth mechanics so they always agree on what
"top-level movetext" means.

:func:`infer_result_from_termination` maps a PGN ``Termination`` header
("white wins on time", "black resigned", …) to the canonical result token.
"""

from __future__ import annotations

import re
from typing import Final

from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX
from mcp_server.parsers.pgn.unicode import normalize_unicode_pgn_results
from mcp_server.parsers.pgn_sanitize import _mask_comments_and_escapes

__all__ = [
    "find_movetext_result",
    "has_completed_game_before",
    "infer_result_from_termination",
    "is_prose_line",
    "truncate_movetext_at_result",
]


_RESULT_TOKENS: Final[tuple[str, ...]] = ("1-0", "0-1", "1/2-1/2", "*")


def _split_header_and_movetext(text: str, masked: str) -> tuple[str, str, str]:
    """Return ``(headers_part, masked_movetext, plain_movetext)``.

    The canonical header block ends at the last consecutive ``[Tag "Value"]``
    line at the start of the document. After that, both the masked and plain
    views of the movetext are returned.
    """
    header_end = 0
    for m in TAG_PAIR_REGEX.finditer(masked):
        if masked[header_end : m.start()].strip() == "":
            header_end = m.end()
        else:
            break
    return text[:header_end], masked[header_end:], text[header_end:]


def find_movetext_result(text: str) -> str | None:
    """Extract the canonical result marker from the top level of movetext."""
    # L-04: normalize Unicode result markers before scanning.
    text = normalize_unicode_pgn_results(text)
    masked = _mask_comments_and_escapes(text)
    _, movetext, _ = _split_header_and_movetext(text, masked)

    var_depth = 0
    i = 0
    while i < len(movetext):
        ch = movetext[i]
        if ch == "(":
            var_depth += 1
        elif ch == ")":
            var_depth = max(0, var_depth - 1)
        elif var_depth == 0:
            for marker in _RESULT_TOKENS:
                if movetext[i : i + len(marker)] == marker:
                    left_ok = i == 0 or movetext[i - 1] in " \t\r\n;"
                    right_idx = i + len(marker)
                    right_ok = right_idx == len(movetext) or movetext[right_idx] in " \t\r\n;"
                    if left_ok and right_ok:
                        return marker
        i += 1
    return None


def truncate_movetext_at_result(text: str) -> str:
    """Cut movetext at the first top-level result marker. Headers preserved."""
    masked = _mask_comments_and_escapes(text)
    headers_part, masked_movetext, movetext = _split_header_and_movetext(text, masked)

    var_depth = 0
    i = 0
    while i < len(masked_movetext):
        ch = masked_movetext[i]
        if ch == "(":
            var_depth += 1
        elif ch == ")":
            var_depth = max(0, var_depth - 1)
        elif var_depth == 0:
            for marker in _RESULT_TOKENS:
                if masked_movetext[i : i + len(marker)] == marker:
                    left_ok = i == 0 or masked_movetext[i - 1] in " \t\r\n;"
                    right_idx = i + len(marker)
                    right_ok = (
                        right_idx == len(masked_movetext)
                        or masked_movetext[right_idx] in " \t\r\n;"
                    )
                    if left_ok and right_ok:
                        return headers_part + movetext[:right_idx]
        i += 1
    return text


_FIRST_MOVE_RE: Final[re.Pattern[str]] = re.compile(r"\b1\s*[\.\:]\s*[A-Za-z]")
_RESULT_RE: Final[re.Pattern[str]] = re.compile(r"(?:^|\s)(?:1-0|0-1|1/2-1/2|\*)(?:\s|$)")


def has_completed_game_before(text: str, pos: int) -> bool:
    """True when the prefix ``text[:pos]`` already contains a result token.

    Used to gate repeated-game detection in :func:`check_multiple_games`.
    """
    prefix = text[:pos]
    first_mv = _FIRST_MOVE_RE.search(prefix)
    if not first_mv:
        return False
    rest = prefix[first_mv.start() :]
    return bool(_RESULT_RE.search(rest))


_PROSE_WORDS: Final[frozenset[str]] = frozenset(
    {
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "think",
        "thought",
        "thinks",
        "thinking",
        "considered",
        "felt",
        "believe",
        "believed",
        "afterwards",
        "afterward",
        "after",
        "before",
        "during",
        "later",
        "then",
        "next",
        "also",
        "better",
        "best",
        "worse",
        "worst",
        "good",
        "bad",
        "nice",
        "great",
        "poor",
        "blunder",
        "mistake",
        "was",
        "were",
        "is",
        "are",
        "am",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "must",
        "will",
        "shall",
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "game",
        "play",
        "played",
        "moves",
        "move",
        "position",
        "opening",
        "variation",
        "because",
        "since",
        "so",
        "but",
        "and",
        "or",
        "if",
        "though",
        "although",
        "instead",
    }
)


def is_prose_line(line: str) -> bool:
    """Heuristic: does this line look like human prose rather than a SAN line?"""
    stripped = line.strip()
    if not stripped:
        return False
    if (
        stripped.startswith("[")
        or re.match(r"^\d+\s*[\.\:]", stripped)
        or stripped.startswith("{")
        or stripped.startswith("(")
        or stripped in _RESULT_TOKENS
    ):
        return False
    words = stripped.split()
    if not words:
        return False
    prose_count = sum(1 for w in words if w.strip(".,;:!?\"'()").lower() in _PROSE_WORDS)
    if len(words) >= 2 and (prose_count >= 2 or (prose_count / len(words)) >= 0.4):
        return True
    first_word = words[0].strip(".,;:!?\"'()").lower()
    if first_word in (
        "afterwards",
        "afterward",
        "what",
        "how",
        "why",
        "note",
        "comment",
        "analysis",
        "thoughts",
        "question",
        "here",
    ):
        return True
    return False


_INFER_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    # Winner-with-cause patterns.
    (r"\bwhite\s+(?:wins?|won)\b.*\b(?:time|resignation|resigns?)\b", "1-0"),
    (r"\bblack\s+(?:wins?|won)\b.*\b(?:time|resignation|resigns?)\b", "0-1"),
    (r"\bwon\s+by\s+white\b", "1-0"),
    (r"\bwon\s+by\s+black\b", "0-1"),
    # Loser patterns (note: "white resigns" means White lost → 0-1).
    (r"\bwhite\s+(?:resign(?:s|ed)?|lost|loses)\b", "0-1"),
    (r"\bblack\s+(?:resign(?:s|ed)?|lost|loses)\b", "1-0"),
    (r"\bwhite(?:'s)?\s+(?:flag|clock).*(?:fell|expired|flagged|out of time)", "0-1"),
    (r"\bblack(?:'s)?\s+(?:flag|clock).*(?:fell|expired|flagged|out of time)", "1-0"),
    (r"\bwhite\s+(?:lost|loses)\s+on\s+time\b", "0-1"),
    (r"\bblack\s+(?:lost|loses)\s+on\s+time\b", "1-0"),
    # Rules infraction.
    (r"\bwhite\b.*\b(?:illegal move|rules? infraction)\b", "0-1"),
    (r"\bblack\b.*\b(?:illegal move|rules? infraction)\b", "1-0"),
)


def infer_result_from_termination(termination: str | None) -> str | None:
    """Map a PGN ``Termination`` header text to a canonical result token."""
    if not termination:
        return None
    t = re.sub(r"\s+", " ", termination.strip().lower())
    if "normal time control" in t:
        return None
    for pattern, result in _INFER_PATTERNS:
        if re.search(pattern, t):
            return result
    return None
