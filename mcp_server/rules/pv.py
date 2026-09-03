"""Principal-variation utilities.

``truncate_pv_at_terminal`` stops a PV replay once an automatic terminal is
reached — prevents users from seeing engine continuations that aren't legal
(GUI clients replaying past the terminal would surface dead-board plies).

``validate_mating_possibility`` checks FIDE Articles 5.1.2 (resignation),
6.9 (time forfeit) and 7.5.5 (rules infraction / second illegal move) and
downgrades the declared result to a draw if the declared winner cannot
materially mate by ANY continuation.
"""

from __future__ import annotations

import re

import chess

from mcp_server.rules.dead_position import is_locked_dead_position

__all__ = ["truncate_pv_at_terminal", "validate_mating_possibility"]


def truncate_pv_at_terminal(board: chess.Board, pv_uci: list[str]) -> list[str]:
    """Ensure a PV does not continue past an automatic terminal state."""
    b = board.copy(stack=True)
    truncated: list[str] = []
    for uci in pv_uci:
        try:
            m = chess.Move.from_uci(uci.lower())
            if m not in b.legal_moves:
                break
            truncated.append(m.uci())
            b.push(m)
            if (
                b.is_checkmate()
                or b.is_stalemate()
                or b.is_insufficient_material()
                or b.is_seventyfive_moves()
                or b.is_fivefold_repetition()
                or is_locked_dead_position(b)
            ):
                break
        except Exception:
            break
    return truncated


# Strict regex matching the same predicates as ``normalize_termination`` —
# only an explicit forfeit marker counts as time forfeit.
_TIME_FORFEIT_REGEX = re.compile(
    r"\btime\s*(?:forfeit|expired|exhausted|loss)\b"
    r"|\bout\s+of\s+time\b"
    r"|\bflag\s*(?:fell|fall|dropped)\b"
    r"|\blost\s+on\s+time\b"
    r"|\b(?:white|black)\s+(?:wins?|won)\s+on\s+time\b"
    r"|\bclock\s+(?:flagged|expired)\b"
)

_RULES_INFRACTION_REGEX = re.compile(
    r"\brules?\s+infraction\b|\b(?:second\s+)?illegal\s+move\b|\binfraction\b|\billegal\b"
)


def validate_mating_possibility(
    board: chess.Board,
    result: str | None,
    termination: str | None,
) -> tuple[str | None, list[str]]:
    """Validate resignation / time-forfeit / rules-infraction outcomes under FIDE.

    For Laws 5.1.2 (resign), 6.9 (time forfeit), 7.5.5 (rules infraction): if the
    declared winning player cannot mate by ANY series of legal moves, the game
    is drawn and a metadata warning is generated (the caller surfaces this to
    the client so the UI can explain the downgrade).
    """
    if not termination or not result:
        return result, []

    term_clean = termination.strip().lower()
    is_resignation = "resign" in term_clean
    is_time_forfeit = bool(_TIME_FORFEIT_REGEX.search(term_clean))
    is_rules_infraction = bool(_RULES_INFRACTION_REGEX.search(term_clean))

    if not (is_resignation or is_time_forfeit or is_rules_infraction):
        return result, []

    if is_resignation:
        term_name, article = "Resignation", "5.1.2"
    elif is_time_forfeit:
        term_name, article = "Time forfeit", "6.9"
    else:
        term_name, article = "Rules infraction / second illegal move", "7.5.5"

    from mcp_server.rules.terminal import can_checkmate  # local import to avoid cycle

    warnings: list[str] = []
    if result == "1-0":
        if not can_checkmate(board, chess.WHITE):
            warnings.append(
                f"{term_name} by Black declared 1-0, but White has insufficient "
                f"material to deliver checkmate; normalized to draw (1/2-1/2) "
                f"under FIDE Article {article}."
            )
            return "1/2-1/2", warnings
    elif result == "0-1":
        if not can_checkmate(board, chess.BLACK):
            warnings.append(
                f"{term_name} by White declared 0-1, but Black has insufficient "
                f"material to deliver checkmate; normalized to draw (1/2-1/2) "
                f"under FIDE Article {article}."
            )
            return "1/2-1/2", warnings
    return result, warnings
