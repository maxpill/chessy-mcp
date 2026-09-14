"""PGN header tag resolution — deterministic duplicate-tag policy.

Phase 13 of the 2026-09-14 ultra-audit repair. Before this module the
three call sites (:func:`extract_game_inner`, :func:`parse_pgn_game_candidate`,
:func:`check_multiple_games`) each had their own loop walking the raw
``[Tag "Value"]`` regex. Worse, the duplicate-tag semantics were implicit
and inconsistent:

    - ``Result`` — first occurrence wins (``chess.pgn.read_game`` overwrites
      on every assignment but writes happen in source order, so first wins).
    - ``FEN`` — last occurrence wins (``chess.pgn`` constructs a new board
      from the last ``FEN`` seen while parsing).
    - ``Variant`` — every occurrence was validated, so a single unsupported
      value anywhere poisoned the whole game.
    - ``SetUp`` — same as ``FEN`` because it gates the FEN setup path.

This module centralizes the resolution. Every call site goes through
:func:`resolve_pgn_tags` and gets back a deterministic view:

    - **Lenient mode** (``strict=False``): policy is ``"first"`` for ALL tags.
      When duplicates exist (same or conflicting values) a warning is
      attached listing the occurrences and the chosen value.
    - **Strict mode** (``strict=True``): duplicates with conflicting values
      for ``fen``, ``setup``, ``variant``, or ``result`` raise
      :class:`StrictValidationError` listing the conflicting values.
      Duplicates with the same value are tolerated and warned about.

Both modes preserve audit invariants (FEN counter / variant validation)
because :func:`resolve_pgn_tags` is the single dispatch point — the call
sites only iterate the resolved dict and run the existing validators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mcp_server.contracts.errors import StrictValidationError


__all__ = [
    "STRICT_VALIDATED_KEYS",
    "ResolvedTag",
    "resolve_pgn_tags",
    "rewrite_text_first_wins",
]


# Tags whose duplicate-value semantics are auditable contracts. Strict mode
# rejects conflicting duplicates for these. Lenient mode always uses first-
# occurrence for ALL tags (including these four) — the audit explicitly
# mandates uniform first-wins policy so FEN stops masking its own conflicts.
STRICT_VALIDATED_KEYS: frozenset[str] = frozenset({"fen", "setup", "variant", "result"})


@dataclass(frozen=True)
class ResolvedTag:
    """A single resolved tag — the canonical view callers consume.

    ``key`` is the lowercased PGN tag name (``"fen"``, ``"result"`` ...).
    ``occurrences`` preserves the order of raw values seen, after
    unescape. ``selected`` is the value the pipeline should consume —
    ``None`` if every occurrence was empty. ``policy`` is the resolution
    rule that picked ``selected``; currently always ``"first"`` in lenient
    mode. ``conflict`` is true when multiple occurrences disagreed;
    ``warning`` is the human-readable diagnostic to surface to callers.
    """

    key: str
    occurrences: tuple[str, ...]
    selected: str | None
    policy: Literal["first", "last", "error"]
    conflict: bool
    warning: str | None


def resolve_pgn_tags(header_text: str, strict: bool) -> dict[str, ResolvedTag]:
    """Walk all ``[Tag "Value"]`` pairs in ``header_text`` and resolve duplicates.

    Comments and escapes are masked before the regex so tags embedded inside
    ``{...}`` or after ``;`` are ignored. Values are unescaped (PGN string
    escapes ``\\\\`` and ``\\"``) and NUL-stripped.

    Strict mode raises :class:`StrictValidationError` for any conflicting
    duplicate of a key in :data:`STRICT_VALIDATED_KEYS`. Lenient mode never
    raises on duplicates — it picks first, attaches a warning, and moves on.
    """
    # Deferred imports — these modules sit downstream in the parser package,
    # which means importing them at module load time creates a cycle
    # (pgn_resolver → pgn.tags → pgn.__init__ → pgn.extractor → pgn.game_inner
    # → pgn_resolver). Inside the function call the cycle is resolved.
    from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX
    from mcp_server.parsers.pgn_sanitize import (
        _mask_comments_and_escapes,
        _unescape_pgn_tag_value,
    )

    masked = _mask_comments_and_escapes(header_text)

    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for m in TAG_PAIR_REGEX.finditer(masked):
        key = m.group(1).lower()
        raw_value = m.group(2)
        unescaped = _unescape_pgn_tag_value(raw_value) or ""
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(unescaped)

    resolved: dict[str, ResolvedTag] = {}
    for key in order:
        values = grouped[key]
        occurrences = tuple(values)
        first = values[0]
        conflict = len(set(values)) > 1

        if strict and conflict and key in STRICT_VALIDATED_KEYS:
            raise StrictValidationError(
                f"STRICT_VALIDATION_ERROR: Duplicate {key} tag with conflicting values: {list(occurrences)!r}.",
                key=key,
                occurrences=list(occurrences),
            )

        if len(values) > 1:
            warning = (
                f"Duplicate {key} tag with occurrences {list(occurrences)!r}; "
                f"selected first occurrence {first!r}."
            )
        else:
            warning = None

        resolved[key] = ResolvedTag(
            key=key,
            occurrences=occurrences,
            selected=first if first else None,
            policy="first",
            conflict=conflict,
            warning=warning,
        )
    return resolved


def rewrite_text_first_wins(header_text: str, resolved: dict[str, ResolvedTag]) -> str:
    """Strip duplicate tag pairs after the first occurrence of each key.

    Downstream :func:`chess.pgn.read_game` (and the python-chess internals)
    resolve duplicates by **last-write-wins** for keys like ``FEN`` — which
    contradicts the audit's required first-wins policy. To keep the parse
    output consistent with the resolved dict, callers strip every duplicate
    tag line *except the first* before handing the text to python-chess.

    Tags with a single occurrence are untouched. Tags that never appeared
    (not in ``resolved``) are untouched. The original whitespace and line
    ordering around the kept tags is preserved — only the duplicate lines
    are replaced with empty lines of the same length, so byte offsets and
    downstream regex matchers stay anchored.
    """
    # Deferred import — same cycle-avoidance as in :func:`resolve_pgn_tags`.
    from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX
    from mcp_server.parsers.pgn_sanitize import _mask_comments_and_escapes

    if not any(len(tag.occurrences) > 1 for tag in resolved.values()):
        return header_text

    masked = _mask_comments_and_escapes(header_text)
    first_seen: dict[str, int] = {}
    chars = list(header_text)
    for m in TAG_PAIR_REGEX.finditer(masked):
        key = m.group(1).lower()
        if key not in first_seen:
            first_seen[key] = m.start()
            continue
        for i in range(m.start(), m.end()):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)
