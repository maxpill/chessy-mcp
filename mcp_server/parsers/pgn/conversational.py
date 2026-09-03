"""Conversational PGN cleaner.

Walks a block of freeform text (e.g. ChatGPT's prose-style PGN response,
markdown bullet points around the game, forum quote + game) and extracts the
single best PGN game block.

Strategy:

    1. Strip escape lines (``%``...) so they don't confuse the tag detector.
    2. Mask comments + inline-code so the regex sees whitespace instead.
    3. Find every valid ``[Tag "Value"]`` outside inline-code regions.
    4. Group them into clusters (consecutive-with-blank-gaps-apart).
    5. Score each cluster by (has-direct-move-after, standard-tag-count, size)
       and pick the best cluster.
    6. Trim trailing non-chess prose from the movetext.

The intent is to leave real prose alone and surface the embedded game
when one is clearly present.
"""

from __future__ import annotations

import re

from mcp_server.parsers.pgn.movetext import _RESULT_TOKENS, is_prose_line
from mcp_server.parsers.pgn.multiline_tags import normalize_multiline_tags
from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX
from mcp_server.parsers.pgn.unicode import normalize_movetext_figurines
from mcp_server.parsers.pgn_sanitize import (
    _mask_comments_and_escapes,
    _strip_pgn_escape_lines,
)

__all__ = ["clean_conversational_text"]


_FIRST_MOVE_RE = re.compile(r"\b1\s*[\.\:]\s*[A-Za-z]")
_CHESS_TOKEN_RE = re.compile(r"\b(?:[a-h][1-8]|O-O)")
_CHESS_LINE_RE = re.compile(r"\b\d+\s*[\.\:]|[a-h][1-8]|O-O|1-0|0-1|1/2")


_STD_TAGS: frozenset[str] = frozenset(
    {
        "Event",
        "Site",
        "Date",
        "Round",
        "White",
        "Black",
        "Result",
        "FEN",
        "SetUp",
        "ECO",
        "Opening",
        "Termination",
        "Annotator",
        "WhiteElo",
        "BlackElo",
        "TimeControl",
        "Variant",
    }
)


def _is_tag_inside_inline_code(text: str, m: re.Match[str]) -> bool:
    """Detect `` `[Tag "..."]` `` — tag enclosed in inline backticks."""
    if m.start() > 0 and text[m.start() - 1] == "`":
        if m.end() < len(text) and text[m.end()] == "`":
            return True
    masked = _mask_comments_and_escapes(text)
    line_start = masked.rfind("\n", 0, m.start()) + 1
    prefix_on_line = masked[line_start : m.start()]
    line_end = masked.find("\n", m.end())
    line_end = len(masked) if line_end == -1 else line_end
    suffix_on_line = masked[m.end() : line_end]
    pref_clean = prefix_on_line.strip()
    suff_clean = suffix_on_line.strip()
    if pref_clean.endswith("`") or suff_clean.startswith("`"):
        return True
    # Tags followed by non-tag prose on the same line belong to surrounding
    # prose, not the canonical header block.
    if suff_clean and not suff_clean.startswith("["):
        return True
    return False


def _cluster_score(
    clusters: list[list[re.Match[str]]],
    text: str,
    cluster: list[re.Match[str]],
) -> tuple[bool, int, int]:
    """Score one cluster: ``(has_direct_moves, std_tag_count, cluster_size)``.

    Higher is better — :func:`clean_conversational_text` picks the cluster
    with the lexicographically max tuple.
    """
    h_end = cluster[-1].end()
    after_h = text[h_end:].strip()
    first_mv_after = _FIRST_MOVE_RE.search(after_h)
    has_direct_moves = False
    if first_mv_after:
        mv_pos = h_end + first_mv_after.start()
        # Real cluster only if no other valid tag cluster appears between
        # this cluster and the moves.
        other_cluster_between = any(
            other_cl is not cluster and h_end <= other_cl[0].start() < mv_pos
            for other_cl in clusters
        )
        if not other_cluster_between:
            has_direct_moves = True
    std_count = sum(1 for m in cluster if m.group(1) in _STD_TAGS)
    return (has_direct_moves, std_count, len(cluster))


def clean_conversational_text(text: str) -> str:
    """Extract the canonical PGN block from freeform prose-wrapped input.

    Returns the cleaned text — header block (if found) joined to the
    movetext section. Trailing prose after the result token is trimmed.
    """
    text = _strip_pgn_escape_lines(text)
    text = normalize_multiline_tags(text)
    masked_text = _mask_comments_and_escapes(text)

    # 1. Find valid PGN tag pairs outside inline code and comments.
    tag_matches: list[re.Match[str]] = []
    for m in TAG_PAIR_REGEX.finditer(masked_text):
        if _is_tag_inside_inline_code(masked_text, m):
            continue
        tag_matches.append(m)

    best_header_str = ""
    best_movetext_str = ""
    first_mv_in_full = _FIRST_MOVE_RE.search(masked_text)

    if tag_matches:
        clusters: list[list[re.Match[str]]] = []
        curr_cluster: list[re.Match[str]] = [tag_matches[0]]
        for m in tag_matches[1:]:
            prev_m = curr_cluster[-1]
            gap = text[prev_m.end() : m.start()]
            if gap.strip() == "":
                curr_cluster.append(m)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [m]
        clusters.append(curr_cluster)

        scored = [(c, _cluster_score(clusters, text, c)) for c in clusters]
        scored.sort(key=lambda item: item[1], reverse=True)
        best_cluster = scored[0][0]
        has_direct_moves, std_count, cl_len = scored[0][1]

        if has_direct_moves or (first_mv_in_full is None and (std_count > 0 or cl_len >= 2)):
            h_start = best_cluster[0].start()
            h_end = best_cluster[-1].end()
            best_header_str = text[h_start:h_end].strip()

            after_header = text[h_end:].strip()
            first_move_after = _FIRST_MOVE_RE.search(after_header)
            if first_move_after:
                movetext_candidate = after_header[first_move_after.start() :]
            else:
                movetext_candidate = after_header
            best_movetext_str = movetext_candidate
        elif first_mv_in_full:
            best_movetext_str = text[first_mv_in_full.start() :]
        else:
            best_movetext_str = text
    else:
        best_movetext_str = text[first_mv_in_full.start() :] if first_mv_in_full else text

    # Trim trailing non-chess prose from movetext.
    lines = best_movetext_str.splitlines()
    end_idx = len(lines)
    found_moves = False
    for i, line_item in enumerate(lines):
        stripped = line_item.strip()
        if stripped.startswith(";") or stripped.startswith("%"):
            continue
        if _FIRST_MOVE_RE.search(stripped) or _CHESS_TOKEN_RE.search(stripped):
            found_moves = True
        elif found_moves:
            if any(stripped.startswith(res) or stripped.endswith(res) for res in _RESULT_TOKENS):
                end_idx = i + 1
                break
            if (
                is_prose_line(stripped)
                or stripped.startswith("Here ")
                or stripped.startswith("Note ")
            ):
                end_idx = i
                break
            has_chess = bool(_CHESS_LINE_RE.search(stripped))
            if not has_chess and stripped:
                end_idx = i
                break

    final_movetext = "\n".join(lines[:end_idx]).strip()

    if best_header_str and final_movetext:
        return f"{best_header_str}\n\n{final_movetext}"
    return best_header_str or final_movetext or text


# Re-export figurine normalization under the conversational friendly name so
# downstream imports still resolve if a previous module imported via
# ``clean_conversational_text`` namespace.
normalize_movetext_figurines_default_alias = normalize_movetext_figurines
del normalize_movetext_figurines  # keep public API minimal — accessible via the unicode module
