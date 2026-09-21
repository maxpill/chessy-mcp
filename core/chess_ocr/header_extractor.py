"""Structured PGN header extraction for chess score sheets.

The score sheet carries metadata above the move table — player names,
round number, date, result mark — that the multi-pass movetext OCR
deliberately ignores. This module asks M3 for those seven fields in a
single, structured call (see :data:`HEADER_EXTRACTOR_PROMPT` in
:mod:`core.chess_ocr.engines.minimax`) and merges the result with any
caller-supplied hints, returning a normalised
``dict[field, {"value": ..., "confidence": ...}]``.

The MCP tool surfaces the merged result to the caller; downstream
code should use the per-field confidence to decide whether to show
the header back to the user or to fall back to a ``"?"`` placeholder.
"""

from __future__ import annotations

from typing import Any


# Canonical seven PGN header fields we extract.
_HEADER_FIELDS: tuple[str, ...] = (
    "white",
    "black",
    "round",
    "date",
    "event",
    "site",
    "result",
)


def _empty() -> dict[str, dict[str, Any]]:
    return {f: {"value": None, "confidence": 0.0} for f in _HEADER_FIELDS}


def merge_with_hints(
    detected: dict[str, dict[str, Any]],
    hints: dict[str, str] | None,
) -> dict[str, dict[str, Any]]:
    """Overlay caller hints on top of detected headers.

    Hints always win for the ``value`` field, but the ``confidence``
    is computed as ``max(detected_conf, 0.85)`` so downstream code
    knows the caller supplied the value (caller knowledge is treated
    as at least moderately reliable).

    Fields not present in :data:`_HEADER_FIELDS` are silently dropped.
    """
    merged = {f: dict(detected.get(f, {"value": None, "confidence": 0.0})) for f in _HEADER_FIELDS}
    if not hints:
        return merged
    for field, value in hints.items():
        if field not in _HEADER_FIELDS:
            continue
        if not value:
            continue
        previous_conf = float(merged[field].get("confidence") or 0.0)
        merged[field] = {
            "value": str(value),
            "confidence": max(previous_conf, 0.85),
            "source": "hint",
        }
    for f in _HEADER_FIELDS:
        if "source" not in merged[f]:
            merged[f]["source"] = "detected"
    return merged


def empty_headers() -> dict[str, dict[str, Any]]:
    """Return the canonical empty shape — useful as a fallback."""
    return _empty()


__all__ = ["empty_headers", "merge_with_hints"]
