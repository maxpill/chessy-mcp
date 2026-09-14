"""Normalized position input — single source of truth for FEN/PGN provenance.

The audit (2026-09-14 ultra-hardening, Phase 12) requires that any caller
who supplies a partial FEN (1-5 fields) in lenient mode sees exactly which
fields python-chess auto-completed, what the canonical FEN ended up as,
and what kind of input they supplied (FEN vs PGN vs startpos sentinel).
Previously those facts were lost the moment we passed the raw text to
``chess.Board(...)``; callers had no way to tell "the user typed a 5-field
FEN and we filled in fullmove=1" from "the user typed a 6-field FEN".

:class:`NormalizedPositionInput` is the immutable record returned by
:func:`mcp_server.parsers.board_builder.build_normalized_position`. The
fields mirror what an honest observability hook for partial-FEN parsing
needs:

- ``raw_input``: the caller's original text, before any cleaning.
- ``input_kind``: ``"fen"`` / ``"pgn"`` / ``"startpos"`` — lets the
  caller distinguish the three root sources without re-sniffing the text.
- ``raw_fen_fields``: the whitespace-split raw tokens when ``input_kind
  == "fen"``; otherwise ``None``. Useful for callers that want to report
  exactly which fields were supplied.
- ``defaulted_fields``: the names of FEN fields that python-chess
  silently filled in (e.g. ``["side", "castling", "en_passant",
  "halfmove", "fullmove"]`` for a 1-token FEN). Empty for a 6-field FEN,
  for startpos, and for any PGN.
- ``canonical_fen``: the python-chess ``board.fen()`` after parsing,
  canonicalized to 6 fields. ``None`` only if parsing raised before a
  board was constructed.
- ``was_canonicalized``: True iff the raw input FEN text differs from
  the canonical 6-field output. Always False for non-FEN inputs.
- ``normalization_changes``: human-readable reasons why the FEN was
  rewritten (e.g. ``"defaulted_side_to_w"``, ``"stripped_impossible_ep"``).
  Mirrors the ``normalization_changes`` field on :class:`MCPMoveAnalysis`.
- ``history_completeness``: the same ``"complete" / "partial" /
  "incomplete"`` string that flows into the ``history_completeness``
  field on :class:`MCPEval`, so callers don't have to re-derive it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

InputKind = Literal["fen", "pgn", "startpos"]

__all__ = ["InputKind", "NormalizedPositionInput"]


@dataclass(frozen=True)
class NormalizedPositionInput:
    """Immutable record of how a FEN/PGN/startpos input was parsed."""

    raw_input: str
    input_kind: InputKind
    raw_fen_fields: tuple[str, ...] | None = None
    defaulted_fields: tuple[str, ...] = field(default_factory=tuple)
    canonical_fen: str | None = None
    was_canonicalized: bool = False
    normalization_changes: tuple[str, ...] = field(default_factory=tuple)
    history_completeness: str = "incomplete"

    @property
    def is_partial_fen(self) -> bool:
        """True when a FEN was supplied with fewer than 6 fields."""
        return self.input_kind == "fen" and len(self.defaulted_fields) > 0
