"""Termination resolution for :func:\`reconcile_result\`.

Extracted from :mod:\`mcp_server.analysis.result_reconciliation\`. Owns
:func:\`resolve_termination\` — cross-checking the PGN ``Termination``
header against the board state and the auto-derived termination. Also
:func:\`is_concurrent_board_match\`, the truth table that decides when
the header and board-trace are concurrent (e.g. ``stalemate\` +
``is_stalemate()\`).
"""

from __future__ import annotations

import chess

from mcp_server.rules import is_locked_dead_position
from mcp_server.tools._common import normalize_termination


def resolve_termination(
    final_board: chess.Board,
    termination_header: str | None,
    auto_termination: str | None,
    result_val: str,
    result_movetext: str | None,
    warnings: list[str],
) -> tuple[str | None, list[str]]:
    """Cross-check the Termination header against the auto-derived termination.

    Mutates ``warnings`` with disagreement findings (preserving the
    audit-equivalent ordering vs. the inline version) and returns the
    surface termination value.
    """
    if auto_termination is not None:
        termination_val = auto_termination
        if termination_header:
            norm_term_hdr = normalize_termination(termination_header)
            if norm_term_hdr != "normal":
                is_concurrent = is_concurrent_board_match(norm_term_hdr, final_board)
                if norm_term_hdr != auto_termination and not is_concurrent:
                    warnings.append(
                        f"Termination header '{termination_header}' disagrees "
                        f"with board outcome '{auto_termination}'; using board outcome."
                    )
        return termination_val, warnings

    if not termination_header:
        return None, warnings

    norm_term_hdr = normalize_termination(termination_header)
    if norm_term_hdr == "normal":
        return "normal", warnings
    if norm_term_hdr in ("checkmate", "stalemate"):
        warnings.append(
            f"Termination header '{termination_header}' contradicts board "
            f"state (position is not {norm_term_hdr})."
        )
        return None, warnings
    if norm_term_hdr == "threefold_repetition":
        if not final_board.is_repetition(3):
            warnings.append(
                f"Termination header '{termination_header}' contradicts "
                f"board state (position is not threefold_repetition)."
            )
            return None, warnings
        return "threefold_repetition", warnings
    if norm_term_hdr == "fifty_moves":
        if not final_board.is_fifty_moves() and final_board.halfmove_clock < 100:
            warnings.append(
                f"Termination header '{termination_header}' contradicts "
                f"board state (position is not fifty_moves)."
            )
            return None, warnings
        return "fifty_moves", warnings
    if norm_term_hdr in (
        "insufficient_material",
        "seventyfive_moves",
        "fivefold_repetition",
        "dead_position",
    ):
        warnings.append(
            f"Termination header '{termination_header}' contradicts "
            f"board state (position is not {norm_term_hdr})."
        )
        return None, warnings
    return norm_term_hdr, warnings


def is_concurrent_board_match(norm_term: str, board: chess.Board) -> bool:
    """True iff ``board`` independently exhibits ``norm_term`` (so header
    and board-trace are concurrent, not contradictory)."""
    if norm_term == "stalemate":
        return board.is_stalemate()
    if norm_term == "seventyfive_moves":
        return board.is_seventyfive_moves()
    if norm_term == "fivefold_repetition":
        return board.is_fivefold_repetition()
    if norm_term == "insufficient_material":
        return board.is_insufficient_material()
    if norm_term == "fifty_moves":
        return board.is_fifty_moves() or board.halfmove_clock >= 100
    if norm_term == "threefold_repetition":
        return board.is_repetition(3)
    if norm_term == "dead_position":
        return is_locked_dead_position(board)
    return False


# Back-compat shims.
_resolve_termination = resolve_termination
_is_concurrent_board_match = is_concurrent_board_match
