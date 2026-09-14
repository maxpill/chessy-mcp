"""Canonical candidate move list — single source of truth for candidate dedupe.

Audit Phase 3 (2026-09-14 ultra audit). Centralizes the SAN/UCI alias handling
that used to be:

- correctly duplicated in ``top_moves.include_moves`` (deduped by SAN)
- incorrectly duplicated in ``classify_move.compare_moves`` (no dedupe)
- inconsistently duplicated between ``evaluate_position.moves`` and the others

Every tool that accepts a list of legal candidates now goes through
:func:`canonicalize_candidates`. The function:

1. Parses each raw string (UCI first if syntactically UCI, else SAN).
2. Canonicalizes to a ``chess.Move`` + canonical SAN.
3. Deduplicates by canonical UCI, preserving the first-seen spelling.
4. Applies the unique-candidate cap *after* dedupe.
5. Returns a frozen list of :class:`CanonicalCandidate` so consumers cannot
   accidentally re-introduce duplicate engine work by re-iterating the raw
   input list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from collections.abc import Iterable

import chess

from mcp_server.contracts.constants import MAX_COMPARE_MOVES, MAX_INCLUDE_MOVES
from mcp_server.contracts.errors import IllegalMove, StrictValidationError
from mcp_server.parsers.move_parser import parse_move_with_details

__all__ = [
    "CanonicalCandidate",
    "canonicalize_candidates",
]


# Syntactic UCI: two squares (with optional promotion piece).
_UCI_RE = re.compile(r"^[a-hA-H][1-8][a-hA-H][1-8][qrbnQRBN]?$")


@dataclass(frozen=True)
class CanonicalCandidate:
    """A unique legal candidate move with provenance.

    ``requested_first`` is the first raw spelling the caller used for this
    move (preserved for diagnostics). ``aliases_seen`` lists every spelling
    that mapped to the same canonical UCI in source order. ``canonical_san``
    is what the engine will see.
    """

    uci: str
    canonical_san: str
    requested_first: str
    aliases_seen: tuple[str, ...] = field(default_factory=tuple)
    normalization_kind: str = "none"
    normalization_changes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def move(self) -> chess.Move:
        return chess.Move.from_uci(self.uci)


def _looks_like_uci(text: str) -> bool:
    return bool(_UCI_RE.match(text.strip()))


def _parse_one(
    board: chess.Board, raw: str, *, strict: bool
) -> tuple[chess.Move, str, str, tuple[str, ...]]:
    """Parse a single move string.

    Returns ``(move, canonical_san, normalization_kind, normalization_changes)``.
    Raises :class:`IllegalMove` for non-legal inputs.
    """
    text = raw.strip()
    if not text:
        raise IllegalMove(
            "empty move string is not a legal candidate",
            input=raw,
        )
    # Fast path: pure UCI. Bypasses the SAN normalization warnings that the
    # general parser emits when input is, say, "0-0" — that's still valid UCI-
    # adjacent but not legal UCI itself.
    if _looks_like_uci(text):
        try:
            move = chess.Move.from_uci(text)
        except (chess.InvalidMoveError, ValueError) as exc:
            raise IllegalMove(f"Move {raw!r} is not UCI: {exc}", input=raw) from exc
        if move not in board.legal_moves:
            raise IllegalMove(
                f"Move {raw!r} (UCI {move.uci()}) is not legal in position {board.fen()!r}",
                input=raw,
            )
        return (
            move,
            board.san(move),
            "none",
            (),
        )

    # Fall back to the SAN/UCI parser which also accepts UCI in non-strict mode.
    try:
        result = parse_move_with_details(board, text, strict=strict)
    except ValueError as exc:
        msg = str(exc)
        if "STRICT" in msg:
            raise StrictValidationError(msg.strip(), input=raw) from exc
        # Pass-through for any pre-existing typed error, plus generic illegal
        # moves.
        if "AMBIGUOUS_SAN" in msg:
            from mcp_server.contracts.errors import AmbiguousSAN

            raise AmbiguousSAN(msg.strip(), input=raw) from exc
        if "GAME_ALREADY_OVER" in msg:
            from mcp_server.contracts.errors import GameAlreadyOver

            raise GameAlreadyOver(msg.strip(), input=raw) from exc
        raise IllegalMove(msg.strip(), input=raw) from exc

    return (
        result.move,
        result.canonical_san,
        result.normalization_kind,
        tuple(result.normalization_changes),
    )


def canonicalize_candidates(
    board: chess.Board,
    raw_strings: Iterable[str] | None,
    *,
    strict: bool = False,
    cap: int = MAX_COMPARE_MOVES,
) -> list[CanonicalCandidate]:
    """Canonicalize and deduplicate a list of candidate move strings.

    Order is preserved on first occurrence of each unique canonical UCI. The
    cap is applied *after* dedup so callers can supply aliases and exact
    duplicates without bumping against the 8-candidate ceiling.
    """
    if not raw_strings:
        return []

    seen: dict[str, CanonicalCandidate] = {}
    order: list[str] = []

    for raw in raw_strings:
        if raw is None:
            continue
        move, canonical_san, kind, changes = _parse_one(board, raw, strict=strict)
        uci = move.uci()
        if uci in seen:
            existing = seen[uci]
            seen[uci] = CanonicalCandidate(
                uci=uci,
                canonical_san=existing.canonical_san,
                requested_first=existing.requested_first,
                aliases_seen=(*existing.aliases_seen, raw),
                normalization_kind=existing.normalization_kind,
                normalization_changes=existing.normalization_changes,
            )
            continue
        seen[uci] = CanonicalCandidate(
            uci=uci,
            canonical_san=canonical_san,
            requested_first=raw,
            aliases_seen=(raw,),
            normalization_kind=kind,
            normalization_changes=changes,
        )
        order.append(uci)

    candidates = [seen[uci] for uci in order]

    if len(candidates) > cap:
        raise IllegalMove(
            f"Too many unique canonical candidates: supports at most {cap} unique "
            f"moves (got {len(candidates)}).",
            unique_count=len(candidates),
            cap=cap,
        )

    return candidates


def normalize_candidate_limit(
    raw_count: int,
    *,
    cap: int = MAX_INCLUDE_MOVES,
    context: str = "include_moves",
) -> int:
    """Validate a raw input count against the documented cap.

    The current policy (post-Phase-5) is: cap applies to *unique canonical
    candidates*, not raw strings. This helper exists so the caller can still
    cheaply reject obviously huge raw lists before doing parse work.
    """
    if raw_count > cap * 4:
        # Heuristic guard: more than 4x the cap means the input is almost
        # certainly malformed (or an attempt to brute-force the dedupe path).
        # Tightening this further would penalize legitimate spam-of-aliases,
        # which the spec explicitly permits.
        raise IllegalMove(
            f"Too many {context}: supports at most {cap} unique canonical "
            f"moves; received {raw_count} raw strings.",
            count=raw_count,
            cap=cap,
        )
    return raw_count
