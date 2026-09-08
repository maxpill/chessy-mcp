"""2026-09-08 audit round 2 Bug 8: strict mode must reject underspecified FENs.

Bug 8 — Live ``evaluate_position(fen="8/8/8/8/8/8/8/4K2k", strict=True)``
returned::

    "status":"insufficient_material",
    "canonical_fen":"8/8/8/8/8/8/8/4K2k w - - 0 1",
    "fen_was_canonicalized":true

The audit expected strict mode to reject a 5-field FEN rather than
silently auto-completing the missing fields.

Root cause: ``mcp_server/parsers/board_builder.py`` accepted any
FEN-like input with ``1 <= len(tokens) <= 6`` in both modes. Strict
mode should require all six FEN fields so typos surface as
INVALID_FEN.

Fix: add a strict-mode-only rejection block for FEN-like input with
fewer than 6 whitespace-separated fields. Lenient mode keeps its
python-chess auto-completion behavior.
"""

from __future__ import annotations

import pytest

from mcp_server.parsers.board_builder import build_board


# -----------------------------------------------------------------------------
# Strict mode rejects underspecified FENs (the audit's contract)
# -----------------------------------------------------------------------------


def test_strict_rejects_audit_1_token_fen() -> None:
    """The audit's exact failing input: 1-token FEN in strict mode.

    Pre-fix this returned an auto-completed canonical FEN
    (``…w - - 0 1``). Post-fix it raises INVALID_FEN.
    """
    with pytest.raises(ValueError, match="INVALID_FEN"):
        build_board("8/8/8/8/8/8/8/4K2k", strict=True)


def test_strict_rejects_2_field_fen() -> None:
    """A FEN with placement + side only is rejected in strict mode."""
    with pytest.raises(ValueError, match="INVALID_FEN"):
        build_board("4k3/8/8/8/8/8/8/4K3 w", strict=True)


def test_strict_rejects_3_field_fen() -> None:
    """A FEN missing castling/EP/halfmove/fullmove is rejected in strict mode."""
    with pytest.raises(ValueError, match="INVALID_FEN"):
        build_board("4k3/8/8/8/8/8/8/4K3 w -", strict=True)


def test_strict_rejects_4_field_fen() -> None:
    """A FEN missing halfmove/fullmove is rejected in strict mode."""
    with pytest.raises(ValueError, match="INVALID_FEN"):
        build_board("4k3/8/8/8/8/8/8/4K3 w - -", strict=True)


def test_strict_rejects_5_field_fen() -> None:
    """A FEN missing fullmove is rejected in strict mode (R4-§A's case)."""
    with pytest.raises(ValueError, match="INVALID_FEN"):
        build_board("4k3/8/8/8/8/8/8/4K3 w - - 0", strict=True)


# -----------------------------------------------------------------------------
# Lenient mode keeps the auto-completion contract (no regression)
# -----------------------------------------------------------------------------


def test_lenient_auto_completes_1_token_fen() -> None:
    """Lenient mode still auto-completes the audit's 1-token FEN.

    The audit's lenient behavior is preserved: ``8/8/8/8/8/8/8/4K2k``
    canonicalizes to ``…w - - 0 1`` with python-chess defaults.
    """
    board = build_board("8/8/8/8/8/8/8/4K2k", strict=False)
    assert board.fen() == "8/8/8/8/8/8/8/4K2k w - - 0 1"


def test_lenient_auto_completes_5_field_fen() -> None:
    """Lenient mode auto-completes the R4-§A case (no fullmove)."""
    board = build_board("4k3/8/8/8/8/8/8/4K3 w - -", strict=False)
    assert board.fen() == "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


# -----------------------------------------------------------------------------
# 6-field and 7+ field controls
# -----------------------------------------------------------------------------


def test_strict_accepts_6_field_fen() -> None:
    """A complete 6-field FEN must still pass in strict mode."""
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    board = build_board(fen, strict=True)
    assert board.fen() == fen


def test_lenient_accepts_6_field_fen() -> None:
    """A complete 6-field FEN must still pass in lenient mode."""
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    board = build_board(fen, strict=False)
    assert board.fen() == fen


def test_strict_rejects_7_field_fen() -> None:
    """Existing 7+ field rejection must continue to fire."""
    with pytest.raises(ValueError, match="INVALID_FEN"):
        build_board("4k3/8/8/8/8/8/8/4K3 w - - 0 1 junk", strict=True)


def test_lenient_rejects_7_field_fen() -> None:
    """7+ field rejection in lenient mode (no change)."""
    with pytest.raises(ValueError, match="INVALID_FEN"):
        build_board("4k3/8/8/8/8/8/8/4K3 w - - 0 1 junk", strict=False)


# -----------------------------------------------------------------------------
# Non-FEN-like inputs are not affected
# -----------------------------------------------------------------------------


def test_strict_accepts_bare_moves() -> None:
    """Bare movetext SAN is unaffected by the strict-mode 5-field rejection."""
    board = build_board("1. e4 e5", strict=True)
    assert [m.uci() for m in board.move_stack] == ["e2e4", "e7e5"]


def test_strict_rejects_non_fen_junk() -> None:
    """Truly invalid input still raises — the new rejection path is additive."""
    with pytest.raises(ValueError):
        build_board("not a fen not a pgn", strict=True)
