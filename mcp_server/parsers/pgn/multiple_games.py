"""Multiple-game detection.

Raises ``MULTIPLE_GAMES`` when the input appears to contain more than one PGN
game. Two detection strategies:

    1. Multiple markdown fenced code blocks (`` ```pgn...``` ``) where at
       least two parsed blocks produce valid games.
    2. Multiple distinct header clusters in plain text — a "cluster" is a
       maximal run of ``[Tag "Value"]`` lines with no non-whitespace gap
       between them. Two clusters are real games only when each is followed
       by a movetext marker (``1. <move>``).

Used by the canonical extraction pipeline to refuse ambiguous inputs that
the rest of the system can't disambiguate.
"""

from __future__ import annotations

import io
import re

import chess
import chess.pgn

from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX
from mcp_server.parsers.pgn_sanitize import (
    _mask_comments_and_escapes,
    _strip_pgn_escape_lines,
    _unescape_pgn_tag_value,
)
from mcp_server.parsers.pgn_validate import _validate_variant

__all__ = ["check_multiple_games"]


_FENCE_RE = re.compile(r"```([a-zA-Z0-9_-]*)\s*([\s\S]*?)\s*```")
_FIRST_MOVE_RE = re.compile(r"\b1\s*[\.\:]\s*[A-Za-z]")


def check_multiple_games(cleaned: str) -> None:
    """Raise ``MULTIPLE_GAMES`` if more than one PGN game is detectable.

    The check covers both markdown-fenced inputs and inline multi-cluster
    text. Variant headers are validated up-front so bad data surfaces here
    rather than in the deeper extractor.
    """
    cleaned_escapes = _strip_pgn_escape_lines(cleaned)
    masked_cleaned = _mask_comments_and_escapes(cleaned_escapes)

    for m in TAG_PAIR_REGEX.finditer(masked_cleaned):
        if m.group(1).lower() == "variant":
            _validate_variant(_unescape_pgn_tag_value(m.group(2)))

    # 1. Multiple markdown fenced code blocks.
    fences = list(_FENCE_RE.finditer(cleaned_escapes))
    if len(fences) > 1:
        tagged_fences = [
            m for m in fences if (m.group(1) or "").strip().lower() in ("pgn", "chess")
        ]
        if tagged_fences:
            blocks_to_check = [m.group(2).strip() for m in tagged_fences]
        else:
            blocks_to_check = [m.group(2).strip() for m in fences]
        valid_games = 0
        for block in blocks_to_check:
            s = io.StringIO(_strip_pgn_escape_lines(block))
            g = chess.pgn.read_game(s)
            if g and (
                list(g.mainline_moves())
                or (
                    len(g.headers) >= 3
                    and any(k in g.headers for k in ("White", "Black", "FEN", "SetUp"))
                )
            ):
                valid_games += 1
        if valid_games > 1:
            raise ValueError(
                "MULTIPLE_GAMES: Multiple PGN games detected in input. This operation only supports analyzing a single game at a time."
            )

    # 2. Multiple games via explicit header clusters.
    tag_matches = list(TAG_PAIR_REGEX.finditer(masked_cleaned))
    if tag_matches:
        clusters: list[list[re.Match[str]]] = []
        curr = [tag_matches[0]]
        for m in tag_matches[1:]:
            if cleaned_escapes[curr[-1].end() : m.start()].strip() == "":
                curr.append(m)
            else:
                clusters.append(curr)
                curr = [m]
        clusters.append(curr)

        valid_header_games = 0
        for cl in clusters:
            after_cl = cleaned_escapes[cl[-1].end() :]
            first_mv = _FIRST_MOVE_RE.search(after_cl)
            has_ident = any(
                m.group(1) in ("White", "Black", "FEN", "SetUp", "Event")
                and _unescape_pgn_tag_value(m.group(2)) not in (None, "?")
                for m in cl
            )
            if has_ident and first_mv:
                mv_pos = first_mv.start()
                has_subsequent_cluster_before_mv = any(
                    other_cl is not cl
                    and cl[-1].end() <= other_cl[0].start() < cl[-1].end() + mv_pos
                    for other_cl in clusters
                )
                if not has_subsequent_cluster_before_mv:
                    valid_header_games += 1
        if valid_header_games > 1:
            raise ValueError(
                "MULTIPLE_GAMES: Multiple PGN games detected in input. This operation only supports analyzing a single game at a time."
            )
