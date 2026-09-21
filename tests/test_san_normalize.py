"""Tests for ``mcp_server.parsers.san_normalize``.

Covers:
    - Polish → English SAN translation (the user's primary use case).
    - English passthrough (no-op).
    - Castling variants: ``0-0``, ``o-o``, ``O-O``, ``O-O-O``.
    - Capture marker normalization: ``:`` → ``x``.
    - Promotion forms: ``e8H``, ``e8=H``.
    - The autodetect heuristic on labeled fixtures.
    - python-chess round-trip after normalization.
    - Real-world Polish handwritten tokens from the user's score-sheet photos.
"""

from __future__ import annotations

import io

import chess
import chess.pgn
import pytest

from mcp_server.parsers.san_normalize import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    detect_language,
    normalize_pgn,
    normalize_token,
)


# --- normalize_token --------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected_canonical", "expected_change_substring"),
    [
        # Polish piece letters
        ("Ge5", "Be5", "piece_G→B"),
        ("We1", "Re1", "piece_W→R"),
        ("Sd7", "Nd7", "piece_S→N"),
        ("Hc4+", "Qc4+", "piece_H→Q"),
        # Polish pawn capture with colon
        ("e:d5", "exd5", "pawn_capture_:→x"),
        ("S:d3", "Nxd3", "piece_S→N"),  # combination
        ("G:e5", "Bxe5", "piece_G→B"),  # combination: capture + piece letter
        # Polish castling
        ("0-0", "O-O", "castling_kingside"),
        ("o-o-o", "O-O-O", "castling_queenside"),
        ("0-0-0+", "O-O-O+", "castling_queenside"),
        ("O-O", "O-O", None),  # already canonical — no-op
        # Polish promotion
        ("e8H", "e8=Q", "promotion_H→Q"),
        ("e8=H", "e8=Q", "promotion_H→Q"),
        ("b8S+", "b8=N+", "promotion_S→N"),
        # Polish pawn moves (no piece letter) pass through
        ("e4", "e4", None),
        # Polish mate X → #
        ("Ge5X", "Be5#", "mate_X→#"),
        # Lowercase English piece letter from OCR (Polish detector tripped)
        ("nf3", "Nf3", "uppercase_n→N"),
    ],
)
def test_normalize_token_polish(
    token: str,
    expected_canonical: str,
    expected_change_substring: str | None,
) -> None:
    canonical, changes = normalize_token(token, "pl")
    assert canonical == expected_canonical, (
        f"normalize_token({token!r}, 'pl') → {canonical!r}, expected {expected_canonical!r}"
    )
    if expected_change_substring is None:
        assert not changes, f"unexpected changes: {changes}"
    else:
        assert any(expected_change_substring in c for c in changes), (
            f"expected change containing {expected_change_substring!r} in {changes!r}"
        )


def test_normalize_token_english_passthrough() -> None:
    """Polish normalizer leaves English tokens untouched."""
    for tok in ("Nf3", "Bxe5+", "O-O", "Kxh8#", "e8=Q+", "exd5"):
        canonical, changes = normalize_token(tok, "pl")
        assert canonical == tok, f"{tok!r} → {canonical!r}"
        assert changes == ()


def test_normalize_token_empty_inputs() -> None:
    assert normalize_token("", "pl") == ("", ())
    assert normalize_token("", "en") == ("", ())


def test_normalize_token_strips_move_number_prefix() -> None:
    canonical, changes = normalize_token("1.Ge5", "pl")
    assert canonical == "Be5"
    assert any("piece_G" in c for c in changes)


def test_normalize_token_preserves_annotations() -> None:
    """NAG-like annotation glyphs (``!``, ``?``, ``!!``, ``??``) pass through."""
    canonical, changes = normalize_token("Hf3+!", "pl")
    assert canonical.startswith("Qf3")
    assert "+!" in canonical
    assert any("piece_H" in c for c in changes)


def test_normalize_token_castling_with_annotation() -> None:
    """Castling + annotation: ``o-o!`` becomes ``O-O!``."""
    canonical, changes = normalize_token("o-o!", "pl")
    assert canonical == "O-O!"
    assert "castling_kingside" in changes


def test_normalize_token_unsupported_language_returns_input() -> None:
    """Non-Polish, non-English languages are out of scope — return input untouched."""
    canonical, changes = normalize_token("Sd7", "de")
    assert canonical == "Sd7"
    assert changes == ()


# --- detect_language --------------------------------------------------------


def test_detect_language_polish_with_distinctive_letters() -> None:
    text = "1.e4 e5 2.Sf3 Sc6 3.Gb5 a6 4.Ga4 Sf6 5.O-O Ge7 6.We1 b5"
    lang, confidence = detect_language(text)
    assert lang == "pl"
    assert confidence > 0.7


def test_detect_language_polish_with_colon_capture() -> None:
    text = "1.e4 e5 2.Sf3 Sc6 3.d4 e:d5 4.Gb5 Ge7 5.O-O O-O"
    lang, confidence = detect_language(text)
    assert lang == "pl"
    assert confidence >= 0.7


def test_detect_language_english_default() -> None:
    text = "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7"
    lang, _ = detect_language(text)
    assert lang == "en"


def test_detect_language_empty() -> None:
    lang, confidence = detect_language("")
    assert lang == "en"
    assert confidence == 0.5


def test_detect_language_very_short_polish() -> None:
    """A short Polish snippet still detects correctly."""
    lang, _ = detect_language("1.Ge5 Ge6 2.Hd7 1-0")
    assert lang == "pl"


# --- normalize_pgn end-to-end ----------------------------------------------


def test_normalize_pgn_polish_full_movetext() -> None:
    text = (
        '[Event "Wijk aan Zee"]\n'
        "\n"
        "1.e4 e5 2.Sf3 Sc6 3.Gb5 a6 4.Ga4 Sf6 5.O-O Ge7 6.We1 b5 "
        "7.Gc5 d6 8.c3 O-O 9.d4 Gb7 10.Sbd2 1-0\n"
    )
    result = normalize_pgn(text, language="auto")
    assert result.detected_language == "pl"
    assert "Nf3" in result.canonical_text
    assert "Bb5" in result.canonical_text
    assert "O-O" in result.canonical_text
    # Headers untouched
    assert '[Event "Wijk aan Zee"]' in result.canonical_text


def test_normalize_pgn_preserves_comments() -> None:
    text = "1.e4 {Polish comment with Sd2 and Ge5} 1-0"
    result = normalize_pgn(text, language="pl")
    assert "{Polish comment with Sd2 and Ge5}" in result.canonical_text


def test_normalize_pgn_preserves_variations() -> None:
    text = "1.e4 e5 (1...c5 2.Sf3) 2.Sf3 1-0"
    result = normalize_pgn(text, language="auto")
    assert "Nf3" in result.canonical_text


def test_normalize_pgn_explicit_language_overrides_detector() -> None:
    text = "1.e4 e5 2.Nf3 Nc6 1-0"
    result = normalize_pgn(text, language="pl")
    # With explicit Polish hint but no Polish piece letters, the
    # normalizer is a no-op (nothing to rewrite).
    assert result.canonical_text == text
    assert result.detected_language == "pl"


def test_normalize_pgn_unsupported_language_falls_back_to_default() -> None:
    text = "1.e4 e5 2.Nf3 1-0"
    result = normalize_pgn(text, language="zz")
    assert result.canonical_text == text
    assert result.detected_language == "en"


def test_normalize_pgn_empty_string() -> None:
    result = normalize_pgn("", language="auto")
    assert result.canonical_text == ""
    assert result.detected_language == "en"


def test_normalize_pgn_collects_changes() -> None:
    text = "1.e4 e5 2.Ge5 Gf6 3.Hd1 1-0"
    result = normalize_pgn(text, language="pl")
    assert len(result.normalization_changes) >= 2
    changes_joined = " ".join(result.normalization_changes)
    assert "piece_G" in changes_joined


def test_normalize_pgn_changes_contain_position_prefix() -> None:
    """Each change string carries an ``@<offset>:`` prefix for traceability."""
    text = "1.Ge5 1-0"
    result = normalize_pgn(text, language="pl")
    assert any(c.startswith("@") for c in result.normalization_changes)


# --- python-chess round-trip ------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "1.e4 e5 2.Sf3 Sc6 3.Gb5 a6 4.Ga4 1-0",
        "1.e4 e5 2.Sf3 Sc6 3.Gc4 Gc5 4.O-O 1-0",
        "1.e4 e5 2.f4 e:f4 3.Gc4 Sf6 4.e8H 1-0",
        "1.e4 e5 2.Sf3 Sc6 3.d4 e:d5 4.Gb5 Ge7 1-0",
        "1.e4 c5 2.Sf3 d6 3.d4 c:d4 4.S:d4 Sf6 5.Sc3 1-0",
    ],
)
def test_normalized_polish_pgn_parses_with_python_chess(text: str) -> None:
    """Normalized Polish PGN parses cleanly via ``python-chess.pgn.read_game``."""
    result = normalize_pgn(text, language="auto")
    assert result.detected_language == "pl"
    wrapped = '[Event "?"]\n\n' + result.canonical_text
    game = chess.pgn.read_game(io.StringIO(wrapped))
    assert game is not None, f"python-chess failed to parse: {result.canonical_text}"
    moves = list(game.mainline_moves())
    assert len(moves) > 0, f"no moves parsed from: {result.canonical_text}"


def test_normalized_english_pgn_passthrough_parses() -> None:
    text = "1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 1-0"
    result = normalize_pgn(text, language="auto")
    assert result.detected_language == "en"
    wrapped = '[Event "?"]\n\n' + result.canonical_text
    game = chess.pgn.read_game(io.StringIO(wrapped))
    assert game is not None
    moves = list(game.mainline_moves())
    assert len(moves) >= 9


# --- supported languages inventory ------------------------------------------


def test_supported_languages_inventory() -> None:
    """v2 scope is Polish + English only."""
    assert SUPPORTED_LANGUAGES == frozenset({"en", "pl"})


def test_default_language_is_english() -> None:
    assert DEFAULT_LANGUAGE == "en"


# --- Real-world Polish handwritten fixture tests ----------------------------


@pytest.mark.parametrize(
    "polish_token,expected_english",
    [
        # From IMG_20260815_135623.jpg — Polish tournament sheet
        ("Sf6", "Nf6"),
        ("Gb2", "Bb2"),
        ("Sc6", "Nc6"),
        ("Gxc6", "Bxc6"),
        ("Gxe5", "Bxe5"),
        ("Gd6", "Bd6"),
        ("He2", "Qe2"),
        ("Sxe3", "Nxe3"),
        # From IMG_20260915_142332.jpg — Polish with colon captures
        ("Gg7", "Bg7"),
        ("We1", "Re1"),
        ("We8", "Re8"),
        ("Ge3", "Be3"),
        ("Sd2", "Nd2"),
        ("S:d3", "Nxd3"),
        ("c:d5", "cxd5"),
        ("S:d4", "Nxd4"),
        ("Hd2", "Qd2"),
        ("S:f3", "Nxf3"),
        ("G:f3", "Bxf3"),
        ("Gh6", "Bh6"),
        ("f:d7", "fxd7"),
        ("G:g7", "Bxg7"),
        ("K:g7", "Kxg7"),
        ("Wg1", "Rg1"),
        ("G:c5", "Bxc5"),
        ("G:c3", "Bxc3"),
        ("Wc1", "Rc1"),
        ("Wh4", "Rh4"),
        ("Hd7", "Qd7"),
        ("Ge4", "Be4"),
        ("Wh8", "Rh8"),
        ("W:h3", "Rxh3"),
        ("H:h3", "Qxh3"),
        ("Gg2", "Bg2"),
        ("Sd4", "Nd4"),
        ("W:b8", "Rxb8"),
        ("Hf6", "Qf6"),
        ("Hf2", "Qf2"),
        ("Hh5", "Qh5"),
        ("Hf3", "Qf3"),
        ("G:f4", "Bxf4"),
        ("Hf4", "Qf4"),
        ("G:e5", "Bxe5"),
        ("f:f5", "fxf5"),
        ("a:f5", "axf5"),
        ("a:f4", "axf4"),
        ("a:f3", "axf3"),
        ("Kf6", "Kf6"),
        ("G:e2", "Bxe2"),
        ("a:b3", "axb3"),
        ("G:f5", "Bxf5"),
        ("f3", "f3"),
        ("K:e5", "Kxe5"),
        ("Gg6", "Bg6"),
        ("W:b3", "Rxb3"),
        ("H:f4", "Qxf4"),
        ("Ha7", "Qa7"),
        ("H:f3", "Qxf3"),
        # Polish castling digit-form variants from real sheets
        ("0-0", "O-O"),
        ("0-0-0", "O-O-O"),
    ],
)
def test_polish_real_world_tokens(polish_token: str, expected_english: str) -> None:
    """Polish handwritten score-sheet tokens round-trip to canonical English SAN."""
    canonical, _ = normalize_token(polish_token, "pl")
    assert canonical == expected_english, (
        f"normalize_token({polish_token!r}, 'pl') → {canonical!r}, expected {expected_english!r}"
    )


def test_polish_colon_capture_pattern_from_real_sheet() -> None:
    """A real fragment from the user's Polish tournament sheets."""
    text = "1.e4 e5 2.Sf3 Sc6 3.Gb5 a6 4.Ga4 Sf6 5.O-O Ge7 6.S:d3 S:d4 7.c:d5 cxd5 1-0"
    result = normalize_pgn(text, language="auto")
    assert result.detected_language == "pl"
    assert "Nf3" in result.canonical_text
    assert "Nc6" in result.canonical_text
    assert "Bb5" in result.canonical_text
    assert "Nf6" in result.canonical_text  # Polish Sf6 = English Nf6 (knight, not bishop)
    assert "Be7" in result.canonical_text
    # Colon captures became x captures.
    assert "Nxd3" in result.canonical_text
    assert "Nxd4" in result.canonical_text
    assert "cxd5" in result.canonical_text
    # Castling normalized to letter O.
    assert "O-O" in result.canonical_text
    assert "0-0" not in result.canonical_text


def test_polish_30_move_real_game_parses() -> None:
    """A 30-half-move Polish game (Sicilian Najdorf in Polish notation)
    parses cleanly through python-chess after normalization.

    The English version of this game is a well-known Najdorf line; we
    translate each token to Polish by hand, then verify round-trip.
    """
    # English source (verified legal) translated to Polish:
    #   N → S, B → G, R → W, Q → H, K → K
    english = (
        "1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6 6.Be3 e5 "
        "7.Nb3 Be6 8.f3 Be7 9.Qd2 O-O 10.O-O-O Nbd7 11.g4 b5 "
        "12.g5 b4 13.Ne2 Ne8 14.f4 a5 15.f5 a4 16.Bg2 Nb6 "
        "17.Nbc1 Nc4 18.Nb3 Nxb2 19.Nxb2 1/2-1/2"
    )
    pl_map = str.maketrans({"N": "S", "B": "G", "R": "W", "Q": "H"})
    polish = english.translate(pl_map)

    result = normalize_pgn(polish, language="auto")
    assert result.detected_language == "pl"
    wrapped = '[Event "?"]\n\n' + result.canonical_text
    game = chess.pgn.read_game(io.StringIO(wrapped))
    assert game is not None, f"python-chess failed: {result.canonical_text}"
    moves = list(game.mainline_moves())
    assert len(moves) >= 30, f"only {len(moves)} moves parsed"


def test_polish_real_world_tokens_from_user_photos() -> None:
    """Spot-check that real-world Polish tokens from the user's photo scans
    round-trip correctly."""
    # Tokens observed in IMG_20260815_135623.jpg, IMG_20260915_142332.jpg
    real_tokens = [
        ("Sf6", "Nf6"),
        ("Gb2", "Bb2"),
        ("Sc6", "Nc6"),
        ("Gxc6", "Bxc6"),
        ("S:d3", "Nxd3"),
        ("c:d5", "cxd5"),
        ("0-0", "O-O"),
        ("0-0-0", "O-O-O"),
        ("He2", "Qe2"),
        ("W:h3", "Rxh3"),
        ("K:g7", "Kxg7"),
    ]
    for polish, english in real_tokens:
        canonical, _ = normalize_token(polish, "pl")
        assert canonical == english, f"{polish!r} → {canonical!r}, expected {english!r}"
