"""SAN / UCI move parsing on a chess.Board.

Extracted from :mod:`mcp_server.server`. Owns the two move-parsing
entry points used by the tools and the FEN/PGN parsers.

- :func:`parse_move_on_board` — strict parse (raises on bad input).
- :func:`parse_move_on_board_with_warning` — lenient parse; collects
  non-fatal warnings for normalization ('Bxf3' → 'Bxf3').
"""

from __future__ import annotations

import re

import chess

from mcp_server.rules import is_locked_dead_position, is_terminal_position
from mcp_server.parsers.pgn import _FIGURINE_MAP, _UNICODE_HYPHEN_MAP

__all__ = [
    "parse_move_on_board",
    "parse_move_on_board_with_warning",
]


def parse_move_on_board(board: chess.Board, move_str: str) -> chess.Move:
    return _parse_move_on_board_with_warning(board, move_str)[0]


def parse_move_on_board_with_warning(
    board: chess.Board, move_str: str, strict: bool = False
) -> tuple[chess.Move, str | None]:
    """Parse a move string on a board, accepting either UCI or SAN notation.
    Also detects non-canonical SAN (e.g. false mate/check markers or redundant disambiguation)."""
    # Use is_terminal_position (single source of truth) instead of python-chess's
    # is_game_over(), which does NOT detect locked dead positions (FIDE 5.2.2).
    # Without this, classify_move and other tools would silently accept a move
    # that the rules layer has already declared terminal — disagreeing about
    # the same position across endpoints (audit P0).
    if is_terminal_position(board):
        # Try to name the actual terminal reason
        if board.is_checkmate():
            term = "checkmate"
        elif board.is_stalemate():
            term = "stalemate"
        elif board.is_insufficient_material():
            term = "insufficient_material"
        elif board.is_seventyfive_moves():
            term = "seventyfive_moves"
        elif board.is_fivefold_repetition():
            term = "fivefold_repetition"
        elif is_locked_dead_position(board):
            term = "dead_position"
        elif board.is_game_over():
            term = "game_over"
        else:
            term = "game_over"
        raise ValueError(
            f"GAME_ALREADY_OVER: Position '{board.fen()}' is already game over ({term}), no legal moves can be played."
        )

    clean_move = (
        move_str.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .strip("`'\" \t\r\n")
    )
    clean_move = re.sub(r"^(\d+[\.\:]+|\.+)\s*", "", clean_move)
    clean_move = clean_move.translate(_FIGURINE_MAP)
    clean_move = re.sub(r"\s*\(?\s*e\.?p\.?\s*\)?$", "", clean_move, flags=re.IGNORECASE)
    # U-11 (2026-09-01): normalize Unicode hyphens to ASCII so
    # castling tokens like "0–0" (en-dash) are recognized like the
    # analyze_game path does. _UNICODE_HYPHEN_MAP is a str.maketrans
    # translation table (not a regex), so use .translate() to match
    # _normalize_unicode_pgn_results. Brings classify_move into
    # parity with analyze_game.
    clean_move = clean_move.translate(_UNICODE_HYPHEN_MAP)

    # Normalize castling variants
    lower_cand = clean_move.lower()
    if lower_cand in ("o-o-o", "0-0-0", "o-o-o+", "0-0-0+", "o-o-o#", "0-0-0#"):
        suffix = "#" if "#" in clean_move else ("+" if "+" in clean_move else "")
        clean_move = f"O-O-O{suffix}"
    elif lower_cand in ("o-o", "0-0", "o-o+", "0-0+", "o-o#", "0-0#"):
        suffix = "#" if "#" in clean_move else ("+" if "+" in clean_move else "")
        clean_move = f"O-O{suffix}"

    san_cand = clean_move.rstrip("!?")

    # P1/P2 (2026-09-02 ultra audit): uppercase UCI like `E2E4`, `e2E4`,
    # `A7A8Q` was previously accepted silently without a syntax warning
    # in BOTH lenient and strict modes. PGN movetext requires
    # SAN-shaped notation (and accepts lowercase only), so this
    # normalization only applies to the DIRECT move API. The
    # normalization itself is harmless (lower-case UCI is canonical),
    # but the silent acceptance hid input-shape drift and let callers
    # paste uppercase accidentally. We now:
    #   - emit a `syntax_warning` whenever the supplied UCI contains
    #     uppercase letters, in lenient mode (so the audit's
    #     "normalized without a syntax warning" finding is closed);
    #   - reject with STRICT_SAN_ERROR when the caller asks for strict
    #     mode (the audit explicitly calls out that strict mode should
    #     reject or document non-canonical UCI form).
    #
    # Parse order matters here. The audit's `B8e5` reproducer in the
    # `test_randomized_legal_move_san_and_fen_differential_5000_positions`
    # test exercises `board.san(move)` output — for a White Bishop on
    # b8 moving to e5, python-chess generates the rank-disambiguated
    # SAN `B8e5`. The same string is also a syntactically valid UCI
    # (`b8e5` after lowercase). Treating it as UCI in BOTH cases
    # caused a false-positive flag against the legitimate SAN form.
    # The fix: try SAN FIRST; if SAN parsing succeeds, prefer it and
    # never flag uppercase UCI. Only fall through to UCI when SAN
    # parsing fails, in which case the input is unambiguously UCI.
    #
    # Try SAN with candidates (e.g. clean, stripped !?, stripped +/#,
    # with/without =, promo variants).
    cands = [clean_move, san_cand, san_cand.rstrip("+#!?")]
    # Handle promotion without equal sign e.g. e8Q -> e8=Q
    if re.search(r"[a-h][18][qrbnQRBN]", san_cand):
        cands.append(re.sub(r"([a-h][18])([qrbnQRBN])", r"\1=\2", san_cand))

    ambiguous_err: Exception | None = None
    for cand in cands:
        if not cand:
            continue
        try:
            m = board.parse_san(cand)
            if m in board.legal_moves:
                canonical = board.san(m)
                syntax_warning = None
                raw_s = move_str.strip(" \t\r\n`'\"")
                if raw_s != canonical and not re.fullmatch(
                    r"[a-h][1-8][a-h][1-8][qrbn]?", raw_s.lower()
                ):
                    syntax_warning = f"Input SAN '{raw_s}' normalized to '{canonical}'"
                if strict and syntax_warning:
                    raise ValueError(
                        f"STRICT_SAN_ERROR: Input SAN '{raw_s}' requires syntax normalization: {syntax_warning}"
                    )
                return m, syntax_warning
        except (chess.AmbiguousMoveError, chess.IllegalMoveError) as exc:
            if "ambiguous" in str(exc).lower() or isinstance(exc, chess.AmbiguousMoveError):
                ambiguous_err = exc
        except (ValueError, chess.InvalidMoveError) as exc:
            if "STRICT" in str(exc):
                raise

    # SAN parsing failed (no candidate matched a legal move). Fall
    # through to UCI parsing — the input is unambiguously UCI now.
    #
    # Only flag uppercase UCI when the original matched the UCI shape
    # AND had uppercase letters in valid UCI positions (file letters
    # at 0/2, optional promotion piece at 4). This avoids the false
    # positive where python-chess's `board.san(move)` output for a
    # rank-disambiguated SAN like `B8e5` happens to also lowercase to
    # a valid UCI — we only catch the uppercase-UCI case when SAN
    # parsing definitively failed (above).
    uci_was_upper = False
    if (
        re.fullmatch(r"[a-hA-H][1-8][a-hA-H][1-8][qrbnQRBN]?", clean_move)
        is not None  # position 0 is a valid file letter
        and clean_move != clean_move.lower()
        and any(c.isalpha() for c in clean_move)
    ):
        uci_was_upper = True
    uci_syntax_warning: str | None = None
    for uci_cand in (clean_move, clean_move.lower()):
        # Parse-only — don't catch our STRICT_SAN_ERROR raise below. The
        # previous shape accidentally did, because the catch's `ValueError`
        # matched both python-chess's and ours.
        m_obj: chess.Move | None = None
        try:
            m_obj = chess.Move.from_uci(uci_cand)
        except (chess.InvalidMoveError, ValueError):
            m_obj = None
        if m_obj is not None and m_obj in board.legal_moves:
            if uci_was_upper:
                raw_s = move_str.strip(" \t\r\n`'\"")
                uci_syntax_warning = (
                    f"Input UCI '{raw_s}' normalized to lowercase '{uci_cand.lower()}'."
                )
                if strict:
                    # Promote the warning to a structured strict error.
                    # Raised outside the parse-only try so the caller
                    # actually sees STRICT_SAN_ERROR, not the catch-all
                    # ILLEGAL_MOVE at the bottom of the function.
                    raise ValueError(
                        f"STRICT_SAN_ERROR: Input UCI '{raw_s}' requires "
                        f"syntax normalization: {uci_syntax_warning}"
                    )
            return m_obj, uci_syntax_warning

    if ambiguous_err:
        raise ValueError(
            f"AMBIGUOUS_SAN: Move {move_str!r} is ambiguous in position {board.fen()!r}: {ambiguous_err}"
        )
    raise ValueError(
        f"ILLEGAL_MOVE: Move {move_str!r} is not a valid legal move in position {board.fen()!r}"
    )


# Underscored aliases for backwards-compatible import paths.
_parse_move_on_board_with_warning = parse_move_on_board_with_warning
_parse_move_on_board = parse_move_on_board
