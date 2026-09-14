"""Centralized detail + verbosity validator.

Audit Phase 10 (2026-09-14). The four MCP tools previously each had their own
ad-hoc check for the ``verbosity='minimal' + detail∈{coach,forensic}`` case.
``evaluate_position`` rejected it; ``top_moves`` silently swallowed it; the
two other tools had no verbosity parameter at all. This module is the single
source of truth.

``validate_detail_verbosity`` raises :class:`InvalidArgument` when the
combination is impossible (minimal verbosity cannot serialize coach/forensic
detail), and otherwise returns ``(canonical_detail, canonical_verbosity,
normalization_changes)`` for the caller to attach to the response.
"""

from __future__ import annotations


from mcp_server.contracts.constants import (
    ALLOWED_DETAIL_LEVELS,
    VERBOSITY_MINIMAL,
    normalize_verbosity,
)
from mcp_server.contracts.errors import InvalidArgument

__all__ = ["validate_detail_verbosity"]


def validate_detail_verbosity(
    detail: str | None,
    verbosity: str | None,
) -> tuple[str, str, list[str]]:
    """Validate a (detail, verbosity) pair across all four tools.

    Raises :class:`InvalidArgument` if the combination cannot be serialized.

    Returns ``(canonical_detail, canonical_verbosity, normalization_changes)``.
    ``normalization_changes`` is a list of human-readable strings recording any
    alias / default normalization that happened, suitable for the response
    ``normalization_changes`` field.
    """
    canonical_verbosity, verbosity_reason = normalize_verbosity(verbosity)
    canonical_detail = detail if detail in ALLOWED_DETAIL_LEVELS else "standard"
    if detail is not None and detail not in ALLOWED_DETAIL_LEVELS:
        # If the caller passed something nonsensical, let the per-tool
        # validator surface the proper InvalidDetail error.
        canonical_detail = detail

    if canonical_verbosity == VERBOSITY_MINIMAL and canonical_detail in {
        "coach",
        "forensic",
    }:
        raise InvalidArgument(
            "verbosity='minimal' is incompatible with "
            f"detail={canonical_detail!r}; use 'compact' or 'full' for rich "
            "forensics."
        )

    changes: list[str] = []
    if verbosity_reason:
        changes.append(verbosity_reason)
    if detail is not None and detail != canonical_detail:
        changes.append(f"detail_unknown_{detail}_to_standard")
    return canonical_detail, canonical_verbosity, changes
