"""Multiline-tag normalization + canonical-tag-line detection.

PGN §2.3 allows a tag value to span multiple lines as long as it sits inside
the quotes; many export tools emit the value on its own line, which python-chess
rejects. :func:`normalize_multiline_tags` collapses those back to single-line
``[Tag "Value"]`` form. :func:`is_canonical_tag_line` is the inverse — it tests
whether a line is already in canonical single-line form.
"""

from __future__ import annotations

import re
from typing import Final

from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX

__all__ = [
    "is_canonical_tag_line",
    "normalize_multiline_tags",
]

_CANONICAL_TAG_LINE_REGEX: Final[re.Pattern[str]] = re.compile(
    r'^(?:\[\s*[A-Za-z0-9_]+\s+"(?:[^"\\]|\\.)*"\s*\]\s*)+$'
)
"""Pattern matching a line that is already a single-line canonical tag block."""


def normalize_multiline_tags(text: str) -> str:
    """Normalize multiline tag pairs ``[Tag\\n "Value"]`` → ``[Tag "Value"]``."""

    def _repl(m: re.Match[str]) -> str:
        tag_name = m.group(1).strip()
        tag_val = m.group(2)
        return f'[{tag_name} "{tag_val}"]'

    return TAG_PAIR_REGEX.sub(_repl, text)


def is_canonical_tag_line(line: str) -> bool:
    """True when ``line`` is already in single-line canonical ``[Tag "Value"]`` form.

    Comment and escape lines (``,;``, ``%``, ``{``) return False.
    """
    stripped = line.strip()
    if (
        not stripped
        or stripped.startswith(";")
        or stripped.startswith("%")
        or stripped.startswith("{")
    ):
        return False
    return bool(_CANONICAL_TAG_LINE_REGEX.match(stripped))
