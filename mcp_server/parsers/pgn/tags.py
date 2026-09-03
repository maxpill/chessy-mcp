"""PGN tag-block primitives — single source of truth for ``[Tag "Value"]`` regex.

``TAG_PAIR_REGEX`` is the canonical pattern used by every parser module.
Other modules import from here so a regex tweak only has to happen in one
place. Kept terse on purpose — no helpers, no wrappers — because the regex
itself is the contract with downstream callers.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["TAG_PAIR_REGEX"]


TAG_PAIR_REGEX: Final[re.Pattern[str]] = re.compile(
    r'\[\s*([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\s*\]',
    re.DOTALL,
)
"""Canonical PGN tag-pair regex — captures tag name in group 1, value in group 2."""
