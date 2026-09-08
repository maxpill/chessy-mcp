"""2026-09-08 audit regression: lenient FEN parsing canonicalizes non-historical EP+halfmove.

Bug 6 (audit F-24) — Live ``evaluate_position`` raised
``[INVALID_FEN] FEN '...c6 5 2' has en-passant target 'c6' but halfmove clock is 5``
in non-strict mode. The :mod:`fen_counters` docstring claimed lenient
mode would permit non-historical EP + non-zero halfmove as a warning,
but the implementation raised unconditionally.

Fix: branch on ``strict``. Strict mode preserves the previous raise
behavior. Lenient mode strips the impossible EP target to ``-`` and
records a structured warning on the caller's ``metadata_warning`` channel.
"""

from __future__ import annotations

import pytest

from mcp_server.parsers.pgn_validate import validate_fen_counters


_AUDIT_FEN = "rnbqkbnr/pp1ppppp/8/2p5/3P4/8/PPP1PPPP/RNBQKBNR w KQkq c6 5 2"


def test_lenient_strips_impossible_ep_target() -> None:
    """The audit's exact FEN: lenient mode strips the EP target and emits a warning."""
    tokens, cleaned, warnings = validate_fen_counters(_AUDIT_FEN, strict=False)
    assert tokens[3] == "-", f"EP target must be stripped in lenient mode; got {tokens[3]!r}"
    assert "c6" not in cleaned.split()[3], (
        f"cleaned FEN must not carry the impossible EP target; got {cleaned!r}"
    )
    assert warnings, "lenient mode must record a warning"


def test_strict_raises_on_impossible_ep_target() -> None:
    """Strict mode preserves the original raise behavior."""
    with pytest.raises(ValueError, match="INVALID_FEN"):
        validate_fen_counters(_AUDIT_FEN, strict=True)


def test_negative_halfmove_still_raises_in_lenient() -> None:
    """Hard impossibilities (negative, unparseable, > MAX) raise even in lenient mode."""
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_fen_counters("8/8/8/8/8/8/8/4K2k w - - -1 1", strict=False)


def test_unparseable_halfmove_still_raises_in_lenient() -> None:
    """Unparseable counters raise even in lenient mode."""
    with pytest.raises(ValueError, match="must be a valid integer"):
        validate_fen_counters(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - x 1", strict=False
        )


def test_well_formed_fen_no_warnings() -> None:
    """Sanity: a well-formed FEN returns empty warnings and untouched tokens."""
    tokens, cleaned, warnings = validate_fen_counters(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", strict=False
    )
    assert warnings == []
    assert tokens[3] == "-"
    assert cleaned == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
