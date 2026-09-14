"""Phase 14 (2026-09-14) bridge module: semicolon PGN comment capture.

python-chess silently drops ``;`` comments during tokenization. This
module recovers them as ``(ply, text)`` pairs that the analyzer can
attach to the mainline record set so ``analyze_game(detail="coach")``
surfaces them as ``user_comment_raw`` exactly the same way brace
``{...}`` comments do.

The extraction is layered on the same state-machine shape as
``mask_comments_and_escapes`` (skip brace blocks, skip variations, skip
escaped sequences, skip tag-value quotes) so only true mainline
semicolon comments land in the captures list.

Public API:
    extract_semicolon_comments(text) -> list[tuple[int, str]]
    attach_to_game(game, text) -> chess.pgn.Game   # stashes on game._semicolon_comments
"""

from __future__ import annotations

import re

import chess
import chess.pgn

_SAN_MOVE_TOKEN_RE = re.compile(
    r"^(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#?!]*|"
    r"O-O-O[+#?!]*|O-O[+#?!]*|0-0-0[+#?!]*|0-0[+#?!]*)$"
)
_MOVE_NUMBER_TOKEN_RE = re.compile(r"^(\d+)(\.{1,3})$")


def extract_semicolon_comments(text: str) -> list[tuple[int, str]]:
    """Capture ``;`` (and ``%`` escape) PGN comments from the mainline movetext.

    Returns ``(ply, comment_text)`` pairs. The ``ply`` value is the 1-indexed
    mainline ply the comment attaches to:

      * If at least one SAN move preceded the comment on the mainline, attach
        to that most recent SAN ply.
      * Otherwise, if a move number (``N.`` / ``N...``) was seen, attach to the
        ply implied by that move number (``N.`` -> ``2*N-1``, ``N...`` ->
        ``2*N``).
      * Otherwise default to ply 1.

    Only mainline comments are captured. ``;`` / ``%`` inside brace ``{...}``
    blocks, inside variations ``(...)``, inside escaped ``\\X`` sequences, or
    inside tag-value double quotes are skipped. Empty / whitespace-only
    comments are dropped.

    Phase 14 (2026-09-14): python-chess discards ``;`` comments during
    tokenization, so the parse tree's ``node.comment`` never sees them. This
    function recovers the text and tags each comment with the ply the
    analyzer should surface as ``user_comment_raw``.
    """
    captures: list[tuple[int, str]] = []
    chars = list(text)
    n = len(chars)
    i = 0
    in_brace = False
    in_semi = False
    in_quote = False
    escape_next = False
    is_line_start = True
    var_depth = 0

    mainline_ply = 0
    last_move_number: int | None = None
    last_move_dots = 0

    san_buffer: list[str] = []
    semi_buffer: list[str] = []
    semi_open_ply: int | None = None

    def flush_san_buffer() -> None:
        nonlocal mainline_ply
        if not san_buffer:
            return
        tok = "".join(san_buffer)
        san_buffer.clear()
        if _SAN_MOVE_TOKEN_RE.match(tok):
            mainline_ply += 1

    def _attach_ply() -> int:
        # Prefer the move-number hint when no SAN has followed it yet.
        # ``1. e4 e5 2. ; hello`` then ``2. Nf3`` should attach "hello" to ply 3
        # (the upcoming white move), not ply 2 (the last SAN).
        move_number_ply = 0
        if last_move_number is not None:
            move_number_ply = 2 * last_move_number - (1 if last_move_dots == 1 else 0)
        if mainline_ply >= move_number_ply:
            return mainline_ply if mainline_ply > 0 else (move_number_ply or 1)
        return move_number_ply or 1

    while i < n:
        ch = chars[i]

        if ch == "\r" or ch == "\n":
            flush_san_buffer()
            if in_semi and semi_open_ply is not None:
                body = "".join(semi_buffer).strip()
                if body:
                    captures.append((semi_open_ply, body))
                semi_buffer = []
                semi_open_ply = None
            in_semi = False
            in_quote = False
            escape_next = False
            is_line_start = True
            i += 1
            continue

        if not in_brace and not in_semi:
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == '"':
                in_quote = not in_quote
            elif not in_quote:
                if (is_line_start and ch == "%") or ch == ";":
                    in_semi = True
                    semi_buffer = []
                    if var_depth == 0:
                        semi_open_ply = _attach_ply()
                    else:
                        semi_open_ply = None
                elif ch == "{":
                    flush_san_buffer()
                    in_brace = True
                elif ch == "(":
                    flush_san_buffer()
                    var_depth += 1
                elif ch == ")":
                    flush_san_buffer()
                    var_depth = max(0, var_depth - 1)
                elif var_depth > 0:
                    pass
                elif ch in (" ", "\t"):
                    flush_san_buffer()
                else:
                    san_buffer.append(ch)
                    buf = "".join(san_buffer)
                    m = _MOVE_NUMBER_TOKEN_RE.match(buf)
                    if m is not None:
                        last_move_number = int(m.group(1))
                        last_move_dots = len(m.group(2))
            else:
                pass
        elif in_brace:
            if ch == "}":
                in_brace = False
        elif in_semi:
            if semi_open_ply is not None:
                semi_buffer.append(ch)

        if ch not in (" ", "\t"):
            is_line_start = False
        i += 1

    flush_san_buffer()
    if in_semi and semi_open_ply is not None:
        body = "".join(semi_buffer).strip()
        if body:
            captures.append((semi_open_ply, body))

    return captures


def attach_to_game(game: chess.pgn.Game, text: str) -> chess.pgn.Game:
    """Stash ``[(ply, text), ...]`` on ``game._semicolon_comments``.

    The parse tree never sees ``;`` comments; this attribute is the side
    channel the analyzer reads to project them onto ``user_comment_raw``.
    """
    game._semicolon_comments = extract_semicolon_comments(text)  # type: ignore[attr-defined]
    return game


# Back-compat alias for callers that prefer the longer verb.
attach_semicolon_comments = attach_to_game


def get_semicolon_comments(game: chess.pgn.Game) -> list[tuple[int, str]]:
    """Return the captured comments, defaulting to an empty list."""
    return list(getattr(game, "_semicolon_comments", []) or [])
