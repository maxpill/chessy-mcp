"""2026-09-08 audit regression: numeric-only input must raise INVALID_INPUT.

Bug 4 (audit F-09) — Live ``evaluate_position(fen="12345", depth=1)``
returned the starting position with no warning. The audit's expected
behavior: a structured INVALID_INPUT/FEN/PGN error.

Root cause: ``board_builder.build_board`` allowed a one-token input into
the FEN-parse path; non-FEN failures with no ``/`` set ``board=None``;
the bare-movetext PGN fallback then accepted ``"12345"`` as an empty
game. The fix: reject numeric-only text before the PGN fallback so the
caller gets INVALID_INPUT instead of the starting position.
"""

from __future__ import annotations

import pytest

from mcp_server.parsers.board_builder import build_board


@pytest.mark.parametrize("text", ["12345", "42", "999999", "0", "1"])
def test_numeric_only_input_rejected(text: str) -> None:
    """Every numeric-only input must raise with INVALID_INPUT prefix."""
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_board(text)


def test_bare_move_still_accepted() -> None:
    """Bare SAN (``"e4"``) must still parse — control."""
    board = build_board("e4")
    assert board.turn is False  # White moved; Black to move
    assert board.move_stack[-1].uci() == "e2e4"


def test_numbered_move_sequence_still_accepted() -> None:
    """Numbered PGN movetext (``"1. e4 e5"``) must still parse — control."""
    board = build_board("1. e4 e5")
    assert [m.uci() for m in board.move_stack] == ["e2e4", "e7e5"]


def test_full_fen_still_accepted() -> None:
    """Full 6-field FEN must still parse — control."""
    board = build_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert board.turn is True  # White to move


def test_mixed_alphanumeric_rejected_via_normal_path() -> None:
    """A token with letters + digits (e.g. ``"12ab"``) hits the regular PGN path."""
    with pytest.raises(ValueError):
        build_board("12ab")
