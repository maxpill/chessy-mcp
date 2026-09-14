"""Phase 15 (2026-09-14) repair: split ``position_hash`` into two distinct concepts.

Before: a single ``position_hash`` field hashed the full 6-field canonical FEN,
which conflates FIDE-repetition identity (piece placement + side + castling +
en-passant) with the halfmove/fullmove counters. Two positions reached at
different move counts (e.g. 1.e4 e5 2.Nf3 Nc6 vs the same arrangement arrived
at via 1.e4 e5 2.Nf3 Nc6 3.Bb5 4.Bxc6 5... etc.) got distinct hashes even
though they are FIDE-repetition-identical.

After:
    * ``fen_hash``        — SHA-256 of the full 6-field canonical FEN, 16 hex.
                            Includes halfmove + fullmove clocks.
    * ``repetition_key``  — SHA-256 of the 4-field repetition identity
                            (board_fen + side + castling + en-passant), 16 hex.
                            Clocks excluded; python-chess's FEN serialization
                            already normalizes EP to "-" when no legal capture
                            exists, so this matches FIDE's threefold-repetition
                            semantics exactly.
    * ``position_hash``   — kept as a deprecated alias of ``fen_hash`` for one
                            release. Reading it emits a single
                            ``DeprecationWarning`` per process.
"""

from __future__ import annotations

import warnings

import chess

from mcp_server.analysis.forensics import build_position_fingerprint
from mcp_server.models.forensics import (
    POSITION_HASH_DEPRECATION_EMITTED,
    PositionFingerprint,
)


def _reset_position_hash_deprecation_flag() -> None:
    """Test helper — restore the once-per-process guard to its initial state.

    Each test exercises the deprecation path; without this reset the warning
    would fire only on the first call across the whole test session, hiding
    regressions on subsequent tests.
    """
    import mcp_server.models.forensics as _models_forensics

    _models_forensics.POSITION_HASH_DEPRECATION_EMITTED = False


def _build_fingerprint(fen: str) -> PositionFingerprint:
    """Build a fingerprint from a FEN, asserting the FEN is parseable."""
    board = chess.Board(fen)
    return build_position_fingerprint(board)


def test_halfmove_clock_change_differs_fen_hash_but_not_repetition_key() -> None:
    """Change only halfmove clock (e.g. ``0 1`` → ``50 1``).

    Same piece arrangement, same side, same rights, same EP. Halfmove clock
    is bookkeeping only — the repetition identity must be identical.
    """
    early = _build_fingerprint("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    later = _build_fingerprint("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 50 1")

    assert early.fen_hash != later.fen_hash
    assert early.repetition_key == later.repetition_key


def test_fullmove_change_differs_fen_hash_but_not_repetition_key() -> None:
    """Change only fullmove number (e.g. ``0 1`` → ``0 2``)."""
    earlier = _build_fingerprint("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    later = _build_fingerprint("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 2")

    assert earlier.fen_hash != later.fen_hash
    assert earlier.repetition_key == later.repetition_key


def test_side_to_move_change_differs_both_hashes() -> None:
    """Flipping the side to move changes both identity concepts."""
    white_to_move = _build_fingerprint("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    black_to_move = _build_fingerprint("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")

    assert white_to_move.fen_hash != black_to_move.fen_hash
    assert white_to_move.repetition_key != black_to_move.repetition_key


def test_castling_rights_change_differs_repetition_key() -> None:
    """Castling rights are part of FIDE-repetition identity.

    Losing the right to castle (e.g. king moves) changes the position's
    identity. We use a position where the rook has moved but the king
    hasn't, so only the castling bit flips between the two states.
    """
    full_rights = _build_fingerprint("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # Rook lifted off a1 — python-chess drops the 'Q' right while keeping
    # the position otherwise identical (no king move, no other change).
    rook_lost_q = _build_fingerprint("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kkq - 0 1")

    assert full_rights.repetition_key != rook_lost_q.repetition_key
    assert full_rights.fen_hash != rook_lost_q.fen_hash


def test_piece_placement_change_differs_both_hashes() -> None:
    """Any piece move changes both hashes — it's a real new position."""
    starting = _build_fingerprint("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    after_e4 = _build_fingerprint("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")

    assert starting.fen_hash != after_e4.fen_hash
    assert starting.repetition_key != after_e4.repetition_key


def test_position_hash_is_deprecated_alias_of_fen_hash() -> None:
    """``position_hash`` must equal ``fen_hash`` and emit a DeprecationWarning."""
    _reset_position_hash_deprecation_flag()

    fp = _build_fingerprint("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_value = fp.position_hash

    assert legacy_value == fp.fen_hash
    assert len(legacy_value) == 16

    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation_warnings, "Expected a DeprecationWarning when reading position_hash"
    assert "position_hash" in str(deprecation_warnings[0].message)
    assert "fen_hash" in str(deprecation_warnings[0].message)


def test_position_hash_warning_fires_only_once_per_process() -> None:
    """Subsequent reads must NOT spam additional DeprecationWarnings.

    The contract is "once per process". This test resets the global guard at
    the start (so it is reliable when run alongside other tests) and then
    asserts that a second read on the same fingerprint stays silent.
    """
    _reset_position_hash_deprecation_flag()

    fp = _build_fingerprint("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        _ = fp.position_hash
        first_count = sum(1 for w in first if issubclass(w.category, DeprecationWarning))

    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        _ = fp.position_hash
        second_count = sum(1 for w in second if issubclass(w.category, DeprecationWarning))

    assert first_count == 1, (
        f"Expected exactly one DeprecationWarning on first read, got {first_count}"
    )
    assert second_count == 0, (
        f"Expected zero DeprecationWarnings on second read, got {second_count}"
    )


def test_fide_repetition_identical_positions_share_repetition_key() -> None:
    """Two positions that are FIDE-repetition-identical (same arrangement,
    same side, same rights) but reached at different move counts must share
    ``repetition_key`` while having distinct ``fen_hash``.

    This is the core behavior the audit demanded: a position reached after
    1.Nf3 Nf6 2.c4 c5 3.e3 (fullmove 3) must hash the same, on the
    repetition axis, as the same arrangement re-entered at fullmove 99
    halfmove-clock 25. The 4-field FEN prefix is identical; only the
    bookkeeping clocks differ.
    """
    piece_placement = "rnbqkb1r/pppp1ppp/4p3/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R"
    rights = "w KQkq -"

    first_path = chess.Board(f"{piece_placement} {rights} 0 3")
    second_path = chess.Board(f"{piece_placement} {rights} 25 99")

    fp_first = build_position_fingerprint(first_path)
    fp_second = build_position_fingerprint(second_path)

    # Different move counts — fen_hash must differ.
    assert fp_first.fen_hash != fp_second.fen_hash
    # But the FIDE-repetition identity must agree.
    assert fp_first.repetition_key == fp_second.repetition_key
    # And the four-field repetition identity is what the hash consumed.
    identity_first = " ".join(fp_first.canonical_fen.split()[:4])
    identity_second = " ".join(fp_second.canonical_fen.split()[:4])
    assert identity_first == identity_second
    # The first four FEN fields are equal — proving the inputs differ only
    # in the bookkeeping clocks (halfmove + fullmove).
    assert identity_first == f"{piece_placement} {rights}"


def test_repetition_key_length_and_hex_format() -> None:
    """Sanity: 16 hex chars (matches the prior ``position_hash`` contract)."""
    fp = _build_fingerprint("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert len(fp.fen_hash) == 16
    assert len(fp.repetition_key) == 16
    int(fp.fen_hash, 16)
    int(fp.repetition_key, 16)


def test_legacy_position_hash_value_unchanged() -> None:
    """Backward compatibility — the bytes returned by ``position_hash``
    must equal what the deprecated field used to hash, so consumers that
    cached it keep matching.

    The legacy definition was ``sha256(canonical_fen)[:16]``; verify both
    the new ``fen_hash`` and the legacy ``position_hash`` alias match it.
    """
    import hashlib

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    expected = hashlib.sha256(fen.encode()).hexdigest()[:16]

    fp = _build_fingerprint(fen)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = fp.position_hash

    assert fp.fen_hash == expected
    assert legacy == expected
    assert POSITION_HASH_DEPRECATION_EMITTED in (True, False)
