"""``_reconcile_trailing_plies`` — count tokens the analyzer ignored after the
board trace ended.

Extracted from :mod:`mcp_server.analysis.game_analyzer` so the
:class:`GameAnalyzer` orchestration stays focused on the high-level
flow. Returns the actual number of unused ply tokens rather than
``len(game.errors)`` (which under-reports — usually 1 even when several
trailing tokens were ignored).
"""

from __future__ import annotations

import re

import chess
import chess.pgn

from mcp_server.parsers import _strip_pgn_escape_lines, _truncate_movetext_at_result


_SAN_TOKEN_PATTERN = (
    r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*"
    r"|O-O-O[\+#\?!]*|O-O[\+#\?!]*)\b"
)


def reconcile_trailing_plies(
    *,
    canonical_pgn: str,
    cleaned_movetext: str,
    moves: list[chess.Move],
    game: chess.pgn.Game,
) -> int:
    """Reconcile ignored trailing plies (audit P2): count tokens after the
    board-trace ends so callers see the actual number of unused ply tokens
    rather than ``len(game.errors)``."""
    ignored_trailing = _count_from_parse_errors(moves, cleaned_movetext, game)
    ignored_trailing += _count_from_truncated_pgn(canonical_pgn)
    return ignored_trailing


def _count_from_parse_errors(
    moves: list[chess.Move],
    cleaned_movetext: str,
    game: chess.pgn.Game,
) -> int:
    """Branch 1: python-chess flagged parse errors — count tokens after the
    last successfully executed ply."""
    if not game.errors:
        return 0
    consumed_plies = len(moves)
    total_ply_tokens = len(re.findall(_SAN_TOKEN_PATTERN, cleaned_movetext))
    trailing_from_errors = max(0, total_ply_tokens - consumed_plies)
    if trailing_from_errors > 0:
        return trailing_from_errors
    return max(0, len(game.errors))


def _count_from_truncated_pgn(canonical_pgn: str) -> int:
    """Branch 2: tokens that survive AFTER the result token in the raw PGN
    (e.g. trailing junk after ``1-0`` / ``*`` that python-chess silently
    accepted). Audit U-15 ensures side markers are caught upstream; this
    branch catches ply tokens themselves."""
    raw_pgn_clean = _strip_pgn_escape_lines(canonical_pgn)
    raw_truncated = _truncate_movetext_at_result(raw_pgn_clean)
    if len(raw_truncated) >= len(raw_pgn_clean):
        return 0
    after_part = raw_pgn_clean[len(raw_truncated) :]
    after_clean = re.sub(r"\{[^{}]*\}", " ", after_part)
    after_clean = re.sub(r";[^\r\n]*", " ", after_clean)
    tokens_after = re.findall(_SAN_TOKEN_PATTERN, after_clean)
    return len(tokens_after)
