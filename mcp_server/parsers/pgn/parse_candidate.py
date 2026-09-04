"""``chess.pgn.read_game`` preflight wrapper.

Extracted from :mod:`mcp_server.parsers.pgn.extractor`. Wraps
:func:`chess.pgn.read_game` with the preflight checks needed to
match audit P0 / R4-§B behavior:

  * Variant validation (via :func:`_validate_variant`).
  * FEN counter validation (via :func:`_validate_fen_counters`).
  * Attached-asterisk + bracket sanitization.
  * movetext token validation.
  * Game-errors surface with reasonable error messages.

Returns ``None` on parse failure that *might* succeed elsewhere
(e.g. when the caller can fall back to bare-moves parsing).
"""

from __future__ import annotations

import io
import re

import chess
import chess.pgn

from mcp_server.parsers.pgn.tokens import validate_movetext_tokens
from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX
from mcp_server.parsers.pgn.movetext import truncate_movetext_at_result
from mcp_server.parsers.pgn_sanitize import (
    _mask_comments_and_escapes,
    _sanitize_brackets_in_variations_and_comments,
    _unescape_pgn_tag_value,
)
from mcp_server.parsers.pgn_validate import _validate_variant
from mcp_server.rules import format_fen_status_errors


_ATTACHED_ASTERISK_RE = re.compile(r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)\*")
_ATTACHED_ASTERISK_CASTLE_RE = re.compile(r"\b(O-O-O|O-O)([\+#\?!]*)\*")


def parse_pgn_game_candidate(text: str, strict: bool = False) -> chess.pgn.Game | None:
    """Wrap :func:`chess.pgn.read_game` with preflight checks.

    Returns ``None` on parse failure that *might* succeed elsewhere.
    """
    try:
        masked_for_tags = _mask_comments_and_escapes(text)
        for m in TAG_PAIR_REGEX.finditer(masked_for_tags):
            if m.group(1).lower() == "variant":
                _validate_variant(_unescape_pgn_tag_value(m.group(2)))

        has_real_tags = bool(TAG_PAIR_REGEX.search(masked_for_tags))
        text = truncate_movetext_at_result(text)
        text = _ATTACHED_ASTERISK_RE.sub(r"\1 *", text)
        text = _ATTACHED_ASTERISK_CASTLE_RE.sub(r"\1\2 *", text)
        text_sanitized = _sanitize_brackets_in_variations_and_comments(text)
        text_for_reader = re.sub(
            r"(\[\s*[A-Za-z0-9_]+\s+\"(?:[^\"\\]|\\.)*\"\s*\])\s*(?=\b\d+\s*[\.\:]|[a-h][1-8]|[A-Z])",
            r"\1\n\n",
            text_sanitized,
        )
        game = chess.pgn.read_game(io.StringIO(text_for_reader))
        if game is not None:
            _validate_variant(game.headers.get("Variant"))
            root_b = game.board()
            if not root_b.is_valid() or root_b.status() != chess.STATUS_VALID:
                raise ValueError(
                    f"INVALID_FEN: Initial position '{root_b.fen()}' in PGN is not a valid chess position ({format_fen_status_errors(root_b.status())})."
                )

            moves = list(game.mainline_moves())
            if not moves and not has_real_tags:
                return None

            invalid_tokens = validate_movetext_tokens(text, start_board=game.board(), strict=strict)
            if invalid_tokens:
                error_prefix = "STRICT_PGN_ERROR" if strict else "INVALID_PGN"
                raise ValueError(
                    f"{error_prefix}: Invalid PGN syntax or unrecognized token in movetext: {invalid_tokens[0]!r}"
                )

            if game.errors:
                b = game.board()
                reached_game_over = False
                for node in game.mainline():
                    b.push(node.move)
                    if b.is_game_over(claim_draw=False):
                        reached_game_over = True
                        break
                if not reached_game_over:
                    raise ValueError(
                        f"Invalid PGN syntax or illegal move in game: {game.errors[0]}"
                    )

            return game
    except ValueError:
        raise
    except Exception:
        pass
    return None


# Back-compat shim for callers that imported the underscored name.
