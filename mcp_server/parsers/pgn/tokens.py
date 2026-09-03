"""Strict-movetext validation.

Three responsibilities:

    - :func:`validate_movetext_tokens` — token-by-token validation against
      ``chess.Board.parse_san`` plus NAG range and en-passant marker checks.
      Returns the list of invalid tokens; strict mode fails on any.
    - :func:`strict_top_level_movetext_tokens` — pre-tokenizer that masks
      comments / variations and returns just the top-level movetext tokens.
    - :func:`validate_strict_mainline_surface` — stricter check: requires
      canonical SAN, correct explicit move numbers, and ``+``/``#`` markers
      match actual check/mate state. Used by ``extract_game(strict=True)``.

The promotion-equals helper :func:`strip_promotion_eq` lives here because
both validators (SAN comparison and mainline consume) need it.
"""

from __future__ import annotations

import re
from typing import Final

import chess

from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX
from mcp_server.parsers.pgn.unicode import (
    normalize_movetext_figurines,
    normalize_unicode_pgn_results,
)
from mcp_server.parsers.pgn_sanitize import _mask_comments_and_escapes

__all__ = [
    "strip_promotion_eq",
    "strict_top_level_movetext_tokens",
    "validate_movetext_tokens",
    "validate_strict_mainline_surface",
]


_RESULT_TOKENS: Final[frozenset[str]] = frozenset({"1-0", "0-1", "1/2-1/2", "*"})


def strip_promotion_eq(s: str) -> str:
    """Strip the optional ``=`` in PGN promotion (``e8=Q`` → ``e8Q`` per §8.1.4)."""
    return re.sub(r"=([QRBN])$", r"\1", s)


_ATTACHED_NAG_RE = re.compile(r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)(\$\d+)")
_ATTACHED_NAG_CASTLE_RE = re.compile(r"\b(O-O-O|O-O)([\+#\?!]*)(\$\d+)")
_ATTACHED_ASTERISK_RE = re.compile(r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)\*")
_ATTACHED_ASTERISK_CASTLE_RE = re.compile(r"\b(O-O-O|O-O)([\+#\?!]*)\*")


def _split_attached_annotations(text: str) -> str:
    text = _ATTACHED_NAG_RE.sub(r"\1 \2", text)
    text = _ATTACHED_NAG_CASTLE_RE.sub(r"\1\2 \3", text)
    text = _ATTACHED_ASTERISK_RE.sub(r"\1 *", text)
    text = _ATTACHED_ASTERISK_CASTLE_RE.sub(r"\1\2 *", text)
    return text


def _normalize_castling_and_ep(text: str) -> str:
    text = re.sub(r"\b0-0-0\b", "O-O-O", text)
    text = re.sub(r"\bo-o-o\b", "O-O-O", text, flags=re.IGNORECASE)
    text = re.sub(r"\b0-0\b", "O-O", text)
    text = re.sub(r"\bo-o\b", "O-O", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b([a-h][1-8]|[a-h]x[a-h][1-8])\s+\(?e\.?p\.?\)?(?=\s|$)",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    return text


def validate_movetext_tokens(
    movetext: str,
    start_board: chess.Board | None = None,
    strict: bool = False,
    nag_warnings: list[str] | None = None,
) -> list[str]:
    """Check that all tokens in the active movetext section are valid.

    `nag_warnings`, when provided, receives out-of-range NAG tokens in lenient
    mode (audit P3 INVESTIGATE: ``$999999`` was silently accepted in lenient
    mode). Strict mode returns them in `invalid_tokens` instead.
    """
    t = normalize_movetext_figurines(movetext)
    t = _split_attached_annotations(t)
    t = _normalize_castling_and_ep(t)
    t = TAG_PAIR_REGEX.sub(" ", t)
    t = re.sub(r";[^\r\n]*", " ", t)
    t = re.sub(r"^[ \t]*%[^\r\n]*", " ", t, flags=re.MULTILINE)
    while "{" in t and "}" in t:
        prev = t
        t = re.sub(r"\{[^{}]*\}", " ", t, flags=re.DOTALL)
        if t == prev:
            break
    while "(" in t and ")" in t:
        prev = t
        t = re.sub(r"\([^()]*\)", " ", t, flags=re.DOTALL)
        if t == prev:
            break

    tokens = t.split()
    b = start_board.copy() if start_board else chess.Board()

    first_move_idx = None
    for i, tok in enumerate(tokens):
        clean_tok = tok.rstrip(".,:;!?").lstrip(".,:;!?")
        clean_tok = re.sub(r"\s*\(?\s*e\.?p\.?\s*\)?$", "", clean_tok, flags=re.IGNORECASE).rstrip(
            ".,:;!?"
        )
        if re.match(r"^\d+[\.\:]*$", tok):
            first_move_idx = i
            break
        try:
            b.parse_san(clean_tok)
            first_move_idx = i
            break
        except Exception:
            pass

    if first_move_idx is None:
        return []

    invalid_tokens: list[str] = []
    b = start_board.copy() if start_board else chess.Board()
    for _idx, tok in enumerate(tokens[first_move_idx:], start=first_move_idx):
        if b.is_game_over(claim_draw=False):
            break
        clean_tok = tok.rstrip(".,:;!?").lstrip(".,:;!?")
        clean_tok = re.sub(r"\s*\(?\s*e\.?p\.?\s*\)?$", "", clean_tok, flags=re.IGNORECASE).rstrip(
            ".,:;!?"
        )
        nag_m = re.match(r"^\$([0-9]+)$", clean_tok)
        if nag_m:
            nag_val = int(nag_m.group(1))
            # P3/INVESTIGATE: NAGs outside 0..255 silently dropped before;
            # lenient mode now surfaces the warning via nag_warnings.
            if nag_val > 255:
                if strict:
                    invalid_tokens.append(tok)
                elif nag_warnings is not None:
                    nag_warnings.append(
                        f"NAG value ${nag_val} outside the PGN-supported range 0..255."
                    )
            continue
        clean_tok = re.sub(r"\$[0-9]+$", "", clean_tok)
        if not clean_tok or clean_tok.lower() in (
            "e.p.",
            "e.p",
            "ep",
            "(e.p.)",
            "(e.p)",
            "(ep)",
        ):
            continue
        if re.match(r"^\d+[\.\:]*$", tok) or clean_tok in _RESULT_TOKENS:
            if clean_tok in _RESULT_TOKENS:
                break
            continue
        if re.match(r"^\$[0-9]+$", clean_tok) or clean_tok in (
            "!",
            "?",
            "!!",
            "??",
            "!?",
            "?!",
        ):
            continue
        try:
            m = b.parse_san(clean_tok)
            b.push(m)
        except Exception:
            try:
                m = chess.Move.from_uci(clean_tok)
                if m in b.legal_moves:
                    b.push(m)
                else:
                    invalid_tokens.append(tok)
            except Exception:
                invalid_tokens.append(tok)
    return invalid_tokens


def strict_top_level_movetext_tokens(text: str) -> list[str]:
    """Return top-level movetext tokens with comments/RAVs masked out."""
    normalized = normalize_movetext_figurines(normalize_unicode_pgn_results(text))
    masked = _mask_comments_and_escapes(normalized)

    from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX as _TAG_REGEX

    header_end = 0
    for match in _TAG_REGEX.finditer(masked):
        if masked[header_end : match.start()].strip() == "":
            header_end = match.end()
        else:
            break

    chars = list(masked[header_end:])
    variation_depth = 0
    for i, ch in enumerate(chars):
        if ch == "(":
            variation_depth += 1
            chars[i] = " "
            continue
        if ch == ")":
            chars[i] = " "
            variation_depth = max(0, variation_depth - 1)
            continue
        if variation_depth > 0:
            chars[i] = " "

    top_level = "".join(chars)
    top_level = re.sub(
        r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#?!]*)(\$\d+)",
        r"\1 \2",
        top_level,
    )
    top_level = re.sub(r"\b(O-O-O|O-O)([+#?!]*)(\$\d+)", r"\1\2 \3", top_level)

    def split_move_number(match: re.Match[str]) -> str:
        dots = "..." if match.group(2) else "."
        return f" {match.group(1)}{dots} "

    top_level = re.sub(r"(?<![A-Za-z0-9_])(\d+)\.(\.\.)?", split_move_number, top_level)
    # R4-§C: PGN §8.1 allows whitespace around the move-number dot.
    # Collapse split-digit / split-dot form back into a single move-number
    # token so the strict validator does not see a bare digit as SAN.
    top_level = re.sub(
        r"(?<!\S)(\d+)\s+(\.+)(\s+|$)",
        lambda m: f" {m.group(1)}{m.group(2)} ",
        top_level,
    )
    return top_level.split()


def validate_strict_mainline_surface(text: str, game: chess.pgn.Game) -> None:
    """Require canonical SAN and correct explicit move numbers in strict mode."""
    tokens = strict_top_level_movetext_tokens(text)
    moves = list(game.mainline_moves())
    board = game.board()
    move_index = 0

    for token in tokens:
        clean = token.strip()
        if not clean:
            continue
        if clean in _RESULT_TOKENS:
            break
        nag = re.fullmatch(r"\$(\d+)", clean)
        if nag:
            if int(nag.group(1)) > 255:
                raise ValueError(
                    f"STRICT_PGN_ERROR: NAG {clean!r} is outside the supported PGN range 0..255."
                )
            continue
        if clean in ("!", "?", "!!", "??", "!?", "?!"):
            continue

        number = re.fullmatch(r"(\d+)(\.|\.\.\.)", clean)
        if number:
            supplied = int(number.group(1))
            expected = board.fullmove_number
            expected_dots = "." if board.turn == chess.WHITE else "..."
            if supplied != expected or number.group(2) != expected_dots:
                raise ValueError(
                    "STRICT_PGN_ERROR: Move number mismatch: "
                    f"found {clean!r}, expected {expected}{expected_dots} for the side to move."
                )
            continue

        if clean.lower() in ("e.p.", "e.p", "ep", "(e.p.)", "(e.p)", "(ep)"):
            raise ValueError(
                "STRICT_PGN_ERROR: Explicit en-passant marker requires syntax normalization; "
                "use canonical SAN only."
            )

        if move_index >= len(moves):
            raise ValueError(f"STRICT_PGN_ERROR: Unexpected trailing movetext token {clean!r}.")

        move = moves[move_index]
        canonical = board.san(move)
        supplied_san = clean.rstrip("!?")

        canonical_base = strip_promotion_eq(canonical.rstrip("+#"))
        supplied_base = strip_promotion_eq(supplied_san.rstrip("+#"))
        if supplied_base != canonical_base:
            raise ValueError(
                f"STRICT_PGN_ERROR: Non-canonical SAN: found {clean!r}, expected {canonical!r}."
            )
        test_board = board.copy()
        test_board.push(move)
        is_check = test_board.is_check()
        is_mate = test_board.is_checkmate()
        if supplied_san.endswith("+") and not is_check:
            raise ValueError(
                f"STRICT_PGN_ERROR: SAN {clean!r} marks check ('+') but the move does not give check."
            )
        if supplied_san.endswith("#") and not is_mate:
            raise ValueError(
                f"STRICT_PGN_ERROR: SAN {clean!r} marks mate ('#') but the move is not mate."
            )
        board.push(move)
        move_index += 1

    if move_index != len(moves):
        raise ValueError(
            "STRICT_PGN_ERROR: Strict movetext validation did not consume the complete mainline."
        )
