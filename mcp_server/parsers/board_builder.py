"""Build a ``chess.Board`` from FEN, PGN, or movetext.

Extracted from :mod:`mcp_server.server`. Owns the three public entry
points:

- :func:`build_board` — bare board construction.
- :func:`history_provenance_for_input` — returns
  ``"complete"``/``"partial"``/``"incomplete"`` for the audit H-01 history
  completeness field on :class:`MCPEval`.
- :func:`build_board_with_metadata` — returns the board PLUS the input
  FEN, the canonical FEN, and a ``fen_was_canonicalized`` flag for the
  audit L-06 observability hook.
"""

from __future__ import annotations

import chess

from mcp_server.parsers.move_parser import _parse_move_on_board_with_warning
from mcp_server.parsers.pgn import _extract_game
from mcp_server.parsers.pgn_validate import (
    _validate_castling_rights,
    _validate_fen_counters,
)
from mcp_server.rules import format_fen_status_errors

__all__ = [
    "build_board",
    "build_board_with_metadata",
    "history_provenance_for_input",
]


def build_board(
    fen_or_pgn: str, moves: list[str] | None = None, strict: bool = False
) -> chess.Board:
    """Build a chess.Board from FEN, PGN, or movetext, optionally replaying additional UCI/SAN moves."""
    cleaned = (
        fen_or_pgn.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\x00", "")
        .strip("`'\" \t\r\n")
    )
    if cleaned.lower() in ("startpos", "initial", "start"):
        board: chess.Board | None = chess.Board()
    else:
        board = None
        tokens = cleaned.split()
        # R4-§A (2026-09-02 ultra audit round 4): a FEN must have at most 6
        # whitespace-separated fields. Inputs that look like FENs (contain
        # `/` and a placement row) but have 7+ fields previously fell
        # through to PGN parsing, producing a misleading "INVALID_PGN: Move
        # token '<placement>' could not be parsed" error. Reject them
        # explicitly here with INVALID_FEN so callers see the real cause.
        if "/" in cleaned and len(tokens) > 6 and not cleaned.startswith("["):
            raise ValueError(
                f"INVALID_FEN: Position '{cleaned}' has {len(tokens)} whitespace-separated "
                f"fields; a FEN has exactly 6 (placement, side, castling, en-passant, "
                f"halfmove, fullmove). The extra trailing field(s) cannot be parsed."
            )
        if 1 <= len(tokens) <= 6 and not cleaned.startswith("[") and not tokens[0].endswith("."):
            if "/" in cleaned:
                tokens, _ = _validate_fen_counters(cleaned, strict)
            try:
                # U-09 (2026-09-01): validate castling rights BEFORE handing
                # the FEN to python-chess. The library raises
                # INVALID_CASTLING_RIGHTS status on rights tokens that don't
                # match the actual rook placement, which previously
                # asymmetrically rejected "K" but accepted "Q" (the audit's
                # symptom). Pre-validation lets non-strict mode silently
                # strip bad chars and re-parse the canonicalized FEN; strict
                # mode raises the same structured error the audit wants.
                cleaned_to_parse = cleaned
                if not strict and "/" in cleaned and len(tokens) >= 3:
                    rights_token = tokens[2]
                    placement_side = " ".join(tokens[:2])
                    try:
                        rights_check_board = chess.Board(f"{placement_side} - - 0 1")
                    except Exception:
                        rights_check_board = None
                    if rights_check_board is not None:
                        validated = _validate_castling_rights(
                            rights_check_board,
                            rights_token,
                            False,
                            cleaned,
                        )
                        if validated != rights_token:
                            tokens[2] = validated if validated else "-"
                            cleaned_to_parse = " ".join(tokens)
                b = chess.Board(cleaned_to_parse)
                if b.is_valid() or b.status() == chess.STATUS_VALID:
                    board = b
                    # U-09 (2026-09-01): in strict mode, run the post-parse
                    # castling-rights validation too. python-chess silently
                    # drops rights that don't match the actual rook
                    # placement (e.g. "Q" with no rook on a1) and only
                    # raises INVALID_CASTLING_RIGHTS when the rights TOKEN
                    # itself conflicts with the king placement — which is
                    # asymmetric and exactly what the audit flagged. Our
                    # explicit check catches both directions.
                    if strict and "/" in cleaned_to_parse:
                        parts = cleaned_to_parse.split()
                        if len(parts) >= 3:
                            rights_token = parts[2]
                            _validate_castling_rights(
                                board,
                                rights_token,
                                True,
                                cleaned_to_parse,
                            )
                elif "/" in cleaned_to_parse:
                    if "INVALID_CASTLING_RIGHTS" in format_fen_status_errors(b.status()):
                        raise ValueError(
                            f"INVALID_CASTLING_RIGHTS: FEN '{cleaned}' "
                            f"references castling rights whose rooks are "
                            f"not on their canonical squares "
                            f"(status={b.status()})."
                        )
                    raise ValueError(
                        f"INVALID_FEN: Position '{cleaned_to_parse}' is not a valid FEN ({format_fen_status_errors(b.status())})."
                    )
            except ValueError as exc:
                if "/" in cleaned or "INVALID_FEN" in str(exc) or "INVALID_FEN" in str(exc)[:0]:
                    if str(exc).startswith("INVALID_FEN:"):
                        raise
                    raise ValueError(
                        f"INVALID_FEN: Position '{cleaned}' could not be parsed as a valid FEN: {exc}"
                    ) from exc
                board = None
            except IndexError as exc:
                if "/" in cleaned:
                    raise ValueError(
                        f"INVALID_FEN: Position '{cleaned}' is not a valid FEN."
                    ) from exc
                board = None

    if board is None:
        game = _extract_game(cleaned, strict=strict)
        board = game.board()
        if not board.is_valid() or board.status() != chess.STATUS_VALID:
            raise ValueError(
                f"INVALID_FEN: Initial position '{board.fen()}' in PGN is not a valid chess position ({format_fen_status_errors(board.status())})."
            )
        for move in game.mainline_moves():
            board.push(move)

    assert board is not None
    for move_str in moves or []:
        move, _ = _parse_move_on_board_with_warning(board, move_str, strict=strict)
        board.push(move)

    return board


def history_provenance_for_input(
    fen_or_pgn: str,
    moves: list[str] | None,
) -> str:
    """Return complete, partial or incomplete history provenance for an input."""
    cleaned = (
        fen_or_pgn.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .strip("`'\" \t\r\n")
    )
    if cleaned.lower() in ("startpos", "initial", "start"):
        return "complete"

    tokens = cleaned.split()
    looks_like_fen = (
        "/" in cleaned
        and 1 <= len(tokens) <= 6
        and not cleaned.startswith("[")
        and not tokens[0].endswith(".")
    )
    if looks_like_fen:
        return "partial" if moves else "incomplete"

    # A PGN/movetext input defines its own game root and therefore carries the
    # complete history represented by that game. Additional suffix moves keep it complete.
    return "complete"


def build_board_with_metadata(
    fen_or_pgn: str, moves: list[str] | None = None, strict: bool = False
) -> tuple[chess.Board, str | None, str, bool]:
    """Build a chess.Board from FEN/PGN/movetext AND return canonicalization metadata (audit L-06).

    Returns:
        (board, input_fen, canonical_fen, fen_was_canonicalized)
        where input_fen is None when the input wasn't a single 6-field FEN
        (it was startpos, a PGN, or unparseable raw movetext).

    The flag tells callers whether python-chess silently rewrote the EP
    target — an honest observability hook for the common case where a
    Lichess export says "...KQkq e3 0 1" with no black pawn on d4,
    and python-chess's Board constructor drops the e3 to "-" because
    no piece can actually capture en passant.
    """
    input_fen: str | None = None
    cleaned = (
        fen_or_pgn.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .strip("`'\" \t\r\n")
    )
    # Try to capture the raw input FEN so callers can see if we rewrote it.
    tokens = cleaned.split()
    if (
        1 <= len(tokens) <= 6
        and not cleaned.startswith("[")
        and not tokens[0].endswith(".")
        and "/" in cleaned
    ):
        input_fen = cleaned

    # Canonicalization is a property of the supplied FEN itself, not of any
    # suffix moves replayed after that FEN. Compare the input against a board
    # parsed before replaying the suffix, then return the final board FEN.
    canonical_input_fen: str | None = None
    if input_fen is not None:
        canonical_input_fen = _build_board(fen_or_pgn, [], strict).fen()

    board = _build_board(fen_or_pgn, moves, strict)
    canonical = board.fen()
    was_canonicalized = bool(input_fen) and input_fen != canonical_input_fen
    return board, input_fen, canonical, was_canonicalized


# Underscored aliases for backwards-compatible import paths.
_build_board_with_metadata = build_board_with_metadata
_history_provenance_for_input = history_provenance_for_input
_build_board = build_board
