"""PGN cleaning Steps — composable, named, individually testable text transformations.

Each class wraps an existing ``_name(...)`` helper from
``mcp_server.parsers.pgn_sanitize`` and exposes it through the :class:`Step`
Protocol so PGN cleaning can be expressed as a list of named steps instead of
a chain of free functions.

These steps run in a deterministic, fixed order — ``Pipeline`` builds them
up in :func:`canonical_pgn_pipeline`.
"""

from __future__ import annotations

import re

from mcp_server.parsers.pipeline import Pipeline, Step
from mcp_server.parsers.pgn_sanitize import (
    _mask_comments_and_escapes,
    _sanitize_brackets_in_variations_and_comments,
    _strip_pgn_escape_lines,
)

__all__ = [
    "MaskComments",
    "StripEscapeLines",
    "StripPromotionEq",
    "canonical_pgn_pipeline",
]


class StripEscapeLines(Step):
    """Remove Str-STM escape lines used by some chat-board encodings."""

    def apply(self, text: str) -> str:
        return _strip_pgn_escape_lines(text)


class MaskComments(Step):
    """Replace comments and inline escapes with placeholder space.

    Masks inside ``{...}`` and ``;...`` so chess syntax stays valid while
    preserving original char counts for offset-sensitive downstream code.
    """

    def apply(self, text: str) -> str:
        return _mask_comments_and_escapes(text)


class SanitizeBrackets(Step):
    """Neutralize unbalanced ``{ }`` inside variations / comments.

    The fed-in text has had its comments masked by :class:`MaskComments`
    already; this catches the rest, including stray brackets in NAG /
    castle notation.
    """

    def apply(self, text: str) -> str:
        return _sanitize_brackets_in_variations_and_comments(text)


_PROMOTION_EQ_RE = re.compile(r"(=[QRNB])\s*", re.MULTILINE)


class StripPromotionEq(Step):
    """Drop the ``=`` before a promotion piece (e.g. ``e8=Q`` → ``e8Q``).

    Some PGN exports emit ``=`` separators; python-chess accepts both.
    """

    def apply(self, text: str) -> str:
        return _PROMOTION_EQ_RE.sub(r"\1", text)


def canonical_pgn_pipeline() -> Pipeline:
    """Build the canonical PGN cleaning pipeline.

    Step order:

    1. :class:`StripEscapeLines` — drop non-standard escape lines early.
    2. :class:`StripPromotionEq` — flatten ``=`` promotion syntax.
    3. :class:`MaskComments` — protect downstream bracket counters.
    4. :class:`SanitizeBrackets` — neutralize any remaining unbalanced ``{ }``.
    """
    return Pipeline(
        [
            StripEscapeLines(),
            StripPromotionEq(),
            MaskComments(),
            SanitizeBrackets(),
        ]
    )
