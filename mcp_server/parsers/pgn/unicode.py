"""Unicode normalization for PGN text.

Tolerates the typographic variants that chess software, online databases, and
tournament exports emit. The two transforms are kept separate so callers that
only need result-marker normalization (e.g. legacy `"1/2-1/2"` ASCII bytes)
don't pay for the figurine translation cost.

Crossover from chess figurines happens in two places:

    - movetext: the inside-the-moves section, where ``♔`` should become ``K``.
      Done by :func:`normalize_movetext_figurines` so headers and ``{comments}``
      keep their original characters.
    - everywhere: a positional scan for ``½-½``, ``½–½``, ``½—½`` is acceptable
      because the result marker is a single token — see :func:`normalize_unicode_pgn_results`.
"""

from __future__ import annotations

from typing import Final

from mcp_server.parsers.pgn_sanitize import _mask_comments_and_escapes

__all__ = [
    "FIGURINE_MAP",
    "UNICODE_HYPHEN_MAP",
    "normalize_movetext_figurines",
    "normalize_unicode_pgn_results",
]


FIGURINE_MAP: Final = str.maketrans(
    {
        "♔": "K",
        "♚": "K",
        "♕": "Q",
        "♛": "Q",
        "♖": "R",
        "♜": "R",
        "♗": "B",
        "♝": "B",
        "♘": "N",
        "♞": "N",
        "♙": "",
        "♟": "",
    }
)
"""Translation table: Unicode chess figurines → ASCII letters (pawns drop)."""

UNICODE_HYPHEN_MAP: Final = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "‐": "-",
        "−": "-",
    }
)
"""Translation table: typographic hyphens → ASCII ``-``."""


def normalize_unicode_pgn_results(text: str) -> str:
    """Normalize Unicode PGN result markers and hyphens to ASCII (audit L-04).

    Tolerates the common Unicode variants emitted by chess software and
    online databases:

        ½-½, ½–½, ½—½  ->  1/2-1/2
        0–1, 0—1        ->  0-1
        1–0, 1—0        ->  1-0

    and en/em dashes in movetext castling/result lines.
    """
    text = text.replace("½", "1/2")
    # Strip a wide zero-width joiner / non-breaking hyphen that some browsers
    # insert in PGN exports; harmless if absent.
    text = text.replace("\u200b", "").replace("\u00a0", " ")
    text = text.translate(UNICODE_HYPHEN_MAP)
    return text


def normalize_movetext_figurines(text: str) -> str:
    """Translate Unicode chess figurines only in the movetext section."""
    masked = _mask_comments_and_escapes(text)
    header_end = 0
    from mcp_server.parsers.pgn.tags import (
        TAG_PAIR_REGEX,
    )  # local import — tags is the canonical owner

    for m in TAG_PAIR_REGEX.finditer(masked):
        if masked[header_end : m.start()].strip() == "":
            header_end = m.end()
        else:
            break

    headers_part = text[:header_end]
    movetext = text[header_end:]

    result: list[str] = []
    in_brace = False
    in_semi = False
    i = 0
    while i < len(movetext):
        ch = movetext[i]
        if ch in ("\r", "\n"):
            in_semi = False
            result.append(ch)
        elif in_semi:
            result.append(ch)
        elif ch == ";":
            in_semi = True
            result.append(ch)
        elif ch == "{" and not in_brace:
            in_brace = True
            result.append(ch)
        elif ch == "}" and in_brace:
            in_brace = False
            result.append(ch)
        elif in_brace:
            result.append(ch)
        else:
            result.append(ch.translate(FIGURINE_MAP))
        i += 1

    return headers_part + "".join(result)
