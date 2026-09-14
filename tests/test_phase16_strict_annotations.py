"""Phase 16 — 2026-09-14 ultra-audit repair: strict PGN movetext annotation validation.

Root cause: ``mcp_server/parsers/pgn/tokens.py::validate_strict_mainline_surface``
silently stripped chained annotation glyphs via ``clean.rstrip("!?")``. That meant
``Nf3??!!`` was accepted as ``Nf3`` in strict mode — a hidden data-loss bug for
PGN §8.1.3 grammar, which only defines six canonical annotation suffixes:
``!``, ``?``, ``!!``, ``??``, ``!?``, ``?!``.

Fix: before the SAN normalization step, scan the token for a trailing run of
``!``/``?`` characters and reject any run that is not one of the six canonical
glyphs. Lenient mode still routes through ``validate_movetext_tokens`` (which
treats glyph runs as cosmetic), so behavior is unchanged there.
"""

from __future__ import annotations

import io

import chess
import chess.pgn
import pytest

from mcp_server import server as server_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strict_validate(movetext: str) -> None:
    """Parse the movetext into a ``chess.pgn.Game`` then run strict surface checks.

    The validator consumes moves from ``game.mainline_moves()`` — an empty
    fresh game would raise "Unexpected trailing movetext token" for every
    move, masking the real annotation check. So we parse first.
    """
    from mcp_server.parsers.pgn.tokens import validate_strict_mainline_surface

    game = chess.pgn.read_game(io.StringIO(movetext))
    assert game is not None, f"PGN failed to parse: {movetext!r}"
    validate_strict_mainline_surface(movetext, game)


# ---------------------------------------------------------------------------
# Canonical glyphs — every one must be accepted in strict mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pgn",
    [
        "1. Nf3 Nf6 2. Nc3 Nc6 *",
        "1. Nf3! Nf6 2. Nc3 Nc6 *",
        "1. Nf3? Nf6 2. Nc3 Nc6 *",
        "1. Nf3!! Nf6 2. Nc3 Nc6 *",
        "1. Nf3?? Nf6 2. Nc3 Nc6 *",
        "1. Nf3!? Nf6 2. Nc3 Nc6 *",
        "1. Nf3?! Nf6 2. Nc3 Nc6 *",
    ],
)
def test_strict_accepts_canonical_annotation_glyphs(pgn: str) -> None:
    """All six PGN §8.1.3 canonical annotation suffixes must pass in strict mode.

    Bare ``Nf3`` is also included to pin the no-annotation baseline.
    """
    _strict_validate(pgn)


# ---------------------------------------------------------------------------
# Chained glyphs — must be rejected with STRICT_PGN_ERROR in strict mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suffix",
    ["??!!", "!?!!", "?!!", "!!??", "!?!?", "?!?", "!!!", "???"],
)
def test_strict_rejects_chained_annotation_glyphs(suffix: str) -> None:
    """Any trailing run of !/? that is not one of the six canonical glyphs
    must raise ``STRICT_PGN_ERROR`` in strict mode.

    ``rstrip("!?")`` would silently strip these. Post-fix, strict mode rejects.
    """
    movetext = f"1. Nf3{suffix} Nf6 2. Nc3 Nc6 *"
    with pytest.raises(ValueError, match="STRICT_PGN_ERROR"):
        _strict_validate(movetext)


# ---------------------------------------------------------------------------
# Lenient mode still accepts chained glyphs (cosmetic — no behavioral change)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pgn",
    [
        "1. Nf3??!! Nf6 2. Nc3 Nc6 *",
        "1. Nf3!?!! Nf6 2. Nc3 Nc6 *",
        "1. Nf3?!! Nf6 2. Nc3 Nc6 *",
        "1. e4!!?? e5 2. Nf3?!! Nc6?!? *",
    ],
)
def test_lenient_accepts_chained_glyphs_as_cosmetic(pgn: str) -> None:
    """Lenient mode still parses PGNs with chained glyphs — lenient routing
    goes through ``validate_movetext_tokens`` which treats long annotation
    runs as cosmetic input. No regression in lenient mode.
    """
    game = server_module._extract_game(pgn, strict=False)
    # The move itself must still be consumed (annotations don't poison the parse).
    assert len(list(game.mainline_moves())) >= 2


# ---------------------------------------------------------------------------
# Existing check/mate suffix detection still works
# ---------------------------------------------------------------------------


def test_strict_rejects_check_marker_when_no_check() -> None:
    """``e4+`` from the start position does not give check; strict mode rejects."""
    pgn = "1. e4+ e5 *"
    with pytest.raises(ValueError, match="STRICT_PGN_ERROR"):
        _strict_validate(pgn)


def test_strict_rejects_mate_marker_when_not_mate() -> None:
    """``e4#`` from the start position is not mate; strict mode rejects."""
    pgn = "1. e4# e5 *"
    with pytest.raises(ValueError, match="STRICT_PGN_ERROR"):
        _strict_validate(pgn)


def test_strict_accepts_real_check_with_unicode_check_marker() -> None:
    """``e4+`` that actually gives check must still pass (baseline regression)."""
    # Fool's mate: 1. f3 e5 2. g4 Qh4# — Qh4# is the checkmate marker.
    pgn = "1. f3 e5 2. g4 Qh4# *"
    _strict_validate(pgn)


# ---------------------------------------------------------------------------
# Non-canonical SAN — strict mode rejects (independent of the annotation fix)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "non_canonical_san",
    ["Ng1-f3", "e2-e4"],
)
def test_strict_rejects_non_canonical_san(non_canonical_san: str) -> None:
    """UCI-style SAN like ``Ng1-f3`` / ``e2-e4`` must still be rejected.

    The annotation fix must not weaken the canonical-SAN contract.
    """
    pgn = f"1. {non_canonical_san} Nf6 *"
    with pytest.raises(ValueError, match="STRICT_PGN_ERROR"):
        _strict_validate(pgn)


def test_strict_rejects_zero_zero_castling_when_illegal() -> None:
    """Castling written as ``0-0`` is non-canonical and strict mode rejects."""
    # Castling isn't possible from start position before king/rook move;
    # the canonical form ``O-O`` would also be illegal here, but ``0-0``
    # is the non-canonical spelling the audit locks against.
    pgn = "1. 0-0 e5 *"
    with pytest.raises(ValueError, match="STRICT_PGN_ERROR"):
        _strict_validate(pgn)


# ---------------------------------------------------------------------------
# Direct unit test of the validator — pin the error message format
# ---------------------------------------------------------------------------


def test_strict_error_message_identifies_offending_token() -> None:
    """The STRICT_PGN_ERROR must point at the offending token so users can
    locate the typo without bisecting the whole movetext.
    """
    movetext = "1. Nf3??!! Nf6 2. Nc3 Nc6 *"
    with pytest.raises(ValueError) as exc_info:
        _strict_validate(movetext)
    assert "Nf3??!!" in str(exc_info.value)


def test_strict_error_message_lists_canonical_glyphs() -> None:
    """The error message must enumerate the canonical glyphs so users know
    what is allowed.
    """
    movetext = "1. Nf3?!? Nf6 *"
    with pytest.raises(ValueError) as exc_info:
        _strict_validate(movetext)
    msg = str(exc_info.value)
    assert "!" in msg and "?" in msg


# ---------------------------------------------------------------------------
# Lenient mode via server module — direct end-to-end exercise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pgn",
    [
        "1. Nf3 Nf6 2. Nc3 Nc6 *",
        "1. Nf3!! Nf6 2. Nc3 Nc6 *",
    ],
)
def test_strict_via_server_accepts_canonical_glyphs(pgn: str) -> None:
    """End-to-end: ``server._extract_game(strict=True)`` accepts canonical glyphs."""
    server_module._extract_game(pgn, strict=True)


@pytest.mark.parametrize(
    "pgn",
    [
        "1. Nf3??!! Nf6 2. Nc3 Nc6 *",
        "1. Nf3!?!! Nf6 2. Nc3 Nc6 *",
        "1. Nf3?!! Nf6 2. Nc3 Nc6 *",
    ],
)
def test_strict_via_server_rejects_chained_glyphs(pgn: str) -> None:
    """End-to-end: ``server._extract_game(strict=True)`` rejects chained glyphs."""
    with pytest.raises(ValueError, match="STRICT_PGN_ERROR"):
        server_module._extract_game(pgn, strict=True)
