"""SAN / UCI move parsing on a chess.Board.

Extracted from :mod:`mcp_server.server`. Owns the two move-parsing
entry points used by the tools and the FEN/PGN parsers.

- :func:`parse_move_on_board` — strict parse (raises on bad input).
- :func:`parse_move_on_board_with_warning` — lenient parse; collects
  non-fatal warnings for normalization ('Bxf3' → 'Bxf3').
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import chess

from mcp_server.rules import is_locked_dead_position, is_terminal_position
from mcp_server.parsers.pgn import _FIGURINE_MAP, _UNICODE_HYPHEN_MAP

__all__ = [
    "MoveParseResult",
    "parse_move_on_board",
    "parse_move_on_board_with_warning",
    "parse_move_with_details",
]


@dataclass(frozen=True)
class MoveParseResult:
    move: chess.Move
    canonical_san: str
    raw_input: str
    warning: str | None
    normalization_kind: str  # "none", "cosmetic", "notation_variant", "semantic"
    normalization_changes: list[str] = field(default_factory=list)


def _detect_san_normalization(
    board: chess.Board, raw_s: str, canonical: str, move: chess.Move
) -> tuple[str, list[str], str | None]:
    """Classify normalization between raw user input and canonical SAN."""
    if raw_s == canonical:
        return "none", [], None

    trimmed = raw_s.strip(" \t\r\n`'\"")
    # If the input is canonical lowercase UCI matching move.uci()
    if trimmed.lower() == move.uci() and re.fullmatch(
        r"[a-hA-H][1-8][a-hA-H][1-8][qrbnQRBN]?", trimmed
    ):
        if raw_s == move.uci():
            return "none", [], None
        changes: list[str] = []
        if raw_s != trimmed:
            changes.append("whitespace_trimmed")
        if trimmed != move.uci():
            changes.append("uppercase_uci_lowercased")
            warning = f"Input UCI '{trimmed}' normalized to lowercase '{move.uci()}'."
        else:
            warning = f"Input UCI '{raw_s}' normalized to '{move.uci()}'."
        return "cosmetic", changes, warning

    changes: list[str] = []
    is_semantic = False

    if raw_s != trimmed:
        changes.append("whitespace_trimmed")

    # 0. Annotation suffixes (!, ?, !?, ?!, !!, ??)
    if re.search(r"[\?!]+$", raw_s):
        changes.append("annotation_suffix_removed")

    # 1. Capture claims
    claimed_capture = "x" in raw_s or ":" in raw_s
    actual_capture = board.is_capture(move)
    if claimed_capture and not actual_capture:
        changes.append("capture_marker_removed")
        is_semantic = True
    elif not claimed_capture and actual_capture:
        changes.append("capture_marker_added")
        is_semantic = True

    # 2. Check & Checkmate claims
    post = board.copy(stack=False)
    post.push(move)
    is_mate = post.is_checkmate()
    is_check = board.gives_check(move) and not is_mate

    claimed_mate = "#" in raw_s
    claimed_check = "+" in raw_s and not claimed_mate

    if claimed_mate and not is_mate:
        changes.append("mate_marker_removed")
        changes.append("mate_suffix_corrected")
        is_semantic = True
    elif not claimed_mate and is_mate:
        changes.append("mate_marker_added")
        is_semantic = True

    if claimed_check and not is_check:
        changes.append("check_marker_removed")
        changes.append("check_suffix_corrected")
        is_semantic = True
    elif not claimed_check and is_check and not claimed_mate:
        changes.append("check_marker_added")
        is_semantic = True

    # 3. Promotion notation
    if "=" not in raw_s and "=" in canonical:
        changes.append("promotion_equals_inserted")
        changes.append("promotion_syntax_normalized")

    # 4. Castling variants
    if "0" in raw_s and ("O-O" in canonical or "O-O-O" in canonical):
        changes.append("castling_zeros_normalized")

    # 5. Hyphenated SAN (e.g. e2-e4)
    if "-" in raw_s and not ("O-O" in raw_s or "0-0" in raw_s):
        changes.append("hyphenated_san_normalized")

    # 6. Leading move numbers
    if re.search(r"^(\d+[\.\:]+|\.+)", raw_s):
        changes.append("move_number_removed")

    # 7. Disambiguation differences
    clean_no_punct = re.sub(r"[+#!?x:=\-]", "", raw_s).strip()
    canon_no_punct = re.sub(r"[+#!?x:=\-]", "", canonical).strip()
    if clean_no_punct != canon_no_punct:
        changes.append("disambiguation_adjusted")

    if is_semantic:
        kind = "semantic"
        if "capture_marker_removed" in changes:
            warning = f"Input SAN '{raw_s}' normalized to '{canonical}': claimed a capture but resolved legal move is non-capturing."
        elif "check_marker_removed" in changes:
            warning = f"Input SAN '{raw_s}' normalized to '{canonical}': claimed check (+) but resolved legal move does not give check."
        elif "mate_marker_removed" in changes:
            warning = f"Input SAN '{raw_s}' normalized to '{canonical}': claimed checkmate (#) but resolved legal move does not deliver checkmate."
        elif "capture_marker_added" in changes:
            warning = f"Input SAN '{raw_s}' normalized to '{canonical}': omitted capture marker."
        elif "check_marker_added" in changes:
            warning = f"Input SAN '{raw_s}' normalized to '{canonical}': omitted check (+) marker."
        elif "mate_marker_added" in changes:
            warning = f"Input SAN '{raw_s}' normalized to '{canonical}': omitted checkmate (#) marker."
        else:
            warning = f"Input SAN '{raw_s}' normalized to '{canonical}': required semantic normalization."
    elif any(
        c in changes
        for c in (
            "promotion_syntax_normalized",
            "castling_zeros_normalized",
            "hyphenated_san_normalized",
            "disambiguation_adjusted",
        )
    ):
        kind = "notation_variant"
        warning = f"Input SAN '{raw_s}' normalized to '{canonical}'"
    elif changes:
        kind = "cosmetic"
        warning = f"Input SAN '{raw_s}' normalized to '{canonical}'"
    else:
        kind = "cosmetic"
        warning = f"Input SAN '{raw_s}' normalized to '{canonical}'"

    return kind, changes, warning


def parse_move_on_board(board: chess.Board, move_str: str) -> chess.Move:
    return _parse_move_on_board_with_warning(board, move_str)[0]


def parse_move_with_details(
    board: chess.Board, move_str: str, strict: bool = False
) -> MoveParseResult:
    """Parse a move string on a board, classifying cosmetic vs semantic normalization."""
    if is_terminal_position(board):
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
    clean_move = clean_move.translate(_UNICODE_HYPHEN_MAP)

    lower_cand = clean_move.lower()
    if lower_cand in ("o-o-o", "0-0-0", "o-o-o+", "0-0-0+", "o-o-o#", "0-0-0#"):
        suffix = "#" if "#" in clean_move else ("+" if "+" in clean_move else "")
        clean_move = f"O-O-O{suffix}"
    elif lower_cand in ("o-o", "0-0", "o-o+", "0-0+", "o-o#", "0-0#"):
        suffix = "#" if "#" in clean_move else ("+" if "+" in clean_move else "")
        clean_move = f"O-O{suffix}"

    san_cand = clean_move.rstrip("!?")

    cands = [clean_move, san_cand, san_cand.rstrip("+#!?")]
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
                raw_s = move_str.strip(" \t\r\n`'\"")
                kind, changes, warning = _detect_san_normalization(board, move_str, canonical, m)
                if strict and warning:
                    req_type = "semantic" if kind == "semantic" else "syntax"
                    raise ValueError(
                        f"STRICT_SAN_ERROR: Input SAN '{raw_s}' requires {req_type} normalization: {warning}"
                    )
                return MoveParseResult(
                    move=m,
                    canonical_san=canonical,
                    raw_input=raw_s,
                    warning=warning,
                    normalization_kind=kind,
                    normalization_changes=changes,
                )
        except (chess.AmbiguousMoveError, chess.IllegalMoveError) as exc:
            if "ambiguous" in str(exc).lower() or isinstance(exc, chess.AmbiguousMoveError):
                ambiguous_err = exc
        except (ValueError, chess.InvalidMoveError) as exc:
            if "STRICT" in str(exc):
                raise

    uci_was_upper = False
    if (
        re.fullmatch(r"[a-hA-H][1-8][a-hA-H][1-8][qrbnQRBN]?", clean_move) is not None
        and clean_move != clean_move.lower()
        and any(c.isalpha() for c in clean_move)
    ):
        uci_was_upper = True

    for uci_cand in (clean_move, clean_move.lower()):
        m_obj: chess.Move | None = None
        try:
            m_obj = chess.Move.from_uci(uci_cand)
        except (chess.InvalidMoveError, ValueError):
            m_obj = None
        if m_obj is not None and m_obj in board.legal_moves:
            raw_s = move_str.strip(" \t\r\n`'\"")
            uci_syntax_warning: str | None = None
            norm_kind = "none"
            norm_changes: list[str] = []
            if uci_was_upper:
                uci_syntax_warning = (
                    f"Input UCI '{raw_s}' normalized to lowercase '{uci_cand.lower()}'."
                )
                norm_kind = "cosmetic"
                norm_changes = ["uppercase_uci_lowercased"]
                if strict:
                    raise ValueError(
                        f"STRICT_SAN_ERROR: Input UCI '{raw_s}' requires "
                        f"syntax normalization: {uci_syntax_warning}"
                    )
            return MoveParseResult(
                move=m_obj,
                canonical_san=board.san(m_obj),
                raw_input=raw_s,
                warning=uci_syntax_warning,
                normalization_kind=norm_kind,
                normalization_changes=norm_changes,
            )

    if ambiguous_err:
        raise ValueError(
            f"AMBIGUOUS_SAN: Move {move_str!r} is ambiguous in position {board.fen()!r}: {ambiguous_err}"
        )
    raise ValueError(
        f"ILLEGAL_MOVE: Move {move_str!r} is not a valid legal move in position {board.fen()!r}"
    )


def parse_move_on_board_with_warning(
    board: chess.Board, move_str: str, strict: bool = False
) -> tuple[chess.Move, str | None]:
    """Parse a move string on a board, accepting either UCI or SAN notation."""
    res = parse_move_with_details(board, move_str, strict=strict)
    return res.move, res.warning


# Underscored aliases for backwards-compatible import paths.
_parse_move_on_board_with_warning = parse_move_on_board_with_warning
_parse_move_on_board = parse_move_on_board
