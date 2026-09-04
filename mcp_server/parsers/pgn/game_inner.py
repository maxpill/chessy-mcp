"""``extract_game_inner\` — the parsing-only entry for :mod:\`extractor\`.

Extracted from :mod:\`mcp_server.parsers.pgn.extractor\`. Handles the
"inner" PGN game extraction: comment-only input shortcut (audit
R4-§B), FEN validation, NUL/zero-width cleanup, unicode normalization,
bare-moves fallback (audit P0).

The header-sanitization and strict-mode surface checks live in
:func:\`extractor.extract_game\`. The chess.pgn preflight wrapper lives
in :mod:\`mcp_server.parsers.pgn.parse_candidate\`.
"""

from __future__ import annotations

import re

import chess
import chess.pgn

from mcp_server.parsers.pgn.parse_candidate import parse_pgn_game_candidate
from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX
from mcp_server.parsers.pgn.unicode import normalize_movetext_figurines
from mcp_server.parsers.pgn_sanitize import (
    _mask_comments_and_escapes,
    _unescape_pgn_tag_value,
)
from mcp_server.parsers.pgn_validate import _validate_fen_counters, _validate_variant


_ATTACHED_NAG_RE = re.compile(r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)(\$\d+)")
_ATTACHED_NAG_CASTLE_RE = re.compile(r"\b(O-O-O|O-O)([\+#\?!]*)(\$\d+)")
_ATTACHED_ASTERISK_RE = re.compile(r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)\*")
_ATTACHED_ASTERISK_CASTLE_RE = re.compile(r"\b(O-O-O|O-O)([\+#\?!]*)\*")


def extract_game_inner(cleaned: str, strict: bool = False) -> chess.pgn.Game:
    """Parsing-only entry — no header sanitization or strict-mode gates.

    Audit invariants preserved: R4-§B (comment-only shortcut), R4-§E (NUL
    / zero-width cleanup), P0 (bare-moves fallback refuses to silently
    substitute a different move).
    """
    masked_cleaned = _mask_comments_and_escapes(cleaned)
    for m in TAG_PAIR_REGEX.finditer(masked_cleaned):
        tag_name = m.group(1).lower()
        if tag_name == "variant":
            _validate_variant(_unescape_pgn_tag_value(m.group(2)))
        if tag_name == "fen":
            fen_val = _unescape_pgn_tag_value(m.group(2))
            if fen_val:
                fen_tokens = fen_val.split()
                if "/" in fen_val and len(fen_tokens) > 6:
                    raise ValueError(
                        f"INVALID_FEN: FEN header value '{fen_val}' has "
                        f"{len(fen_tokens)} whitespace-separated fields; a "
                        f"FEN has exactly 6 (placement, side, castling, "
                        f"en-passant, halfmove, fullmove). The extra "
                        f"trailing field(s) cannot be parsed."
                    )
                _validate_fen_counters(fen_val, strict)

    comment_only = _try_build_comment_only_game(cleaned)
    if comment_only is not None:
        return comment_only

    norm_text = _normalize_text(cleaned)
    masked_norm = _mask_comments_and_escapes(norm_text)
    header_end = _scan_header_end(masked_norm)
    if header_end > 0:
        g = parse_pgn_game_candidate(norm_text, strict=strict)
        if g is not None:
            if not list(g.mainline_moves()):
                has_move_tokens = bool(re.search(r"\b\d+\s*[\.\:]\s*[A-Za-z]", norm_text))
                if has_move_tokens:
                    raise ValueError(
                        f"INVALID_PGN: Could not parse legal moves from movetext '{norm_text[:100]}'."
                    )
            return g

    g = parse_pgn_game_candidate(norm_text, strict=strict)
    if g is not None:
        if not list(g.mainline_moves()):
            has_move_tokens = bool(re.search(r"\b\d+\s*[\.\:]\s*[A-Za-z]", norm_text))
            if has_move_tokens:
                raise ValueError(
                    f"INVALID_PGN: Could not parse legal moves from movetext '{norm_text[:100]}'."
                )
        return g

    for move_match in re.finditer(r"\b1\s*[\.\:]\s*[A-Za-z]", norm_text):
        sub_movetext = norm_text[move_match.start() :]
        try:
            g = parse_pgn_game_candidate(sub_movetext, strict=strict)
            if g is not None and list(g.mainline_moves()):
                return g
        except Exception:
            continue

    return _bare_moves_fallback(norm_text, cleaned)


def _try_build_comment_only_game(cleaned: str) -> chess.pgn.Game | None:
    """R4-§B: comment-only input is a zero-move game with header tags
    extracted and the body result captured.

    Returns ``None\` when the body has actual move tokens or an explicit
    result — those belong on the mainline-parsing path, not the
    comment-only shortcut.
    """
    comment_stripped = _mask_comments_and_escapes(cleaned)
    comment_stripped = re.sub(r"\{[^{}]*\}", " ", comment_stripped, flags=re.DOTALL)
    while "(" in comment_stripped and ")" in comment_stripped:
        prev = comment_stripped
        comment_stripped = re.sub(r"\([^()]*\)", " ", comment_stripped, flags=re.DOTALL)
        if comment_stripped == prev:
            break
    body_start = 0
    for m in TAG_PAIR_REGEX.finditer(comment_stripped):
        if comment_stripped[body_start : m.start()].strip() == "":
            body_start = m.end()
        else:
            break
    body = comment_stripped[body_start:].strip()
    body_tokens = body.split()
    non_result_body_tokens = [t for t in body_tokens if t not in ("1-0", "0-1", "1/2-1/2", "*")]
    has_explicit_result = any(t in ("1-0", "0-1", "1/2-1/2") for t in body_tokens)
    if not non_result_body_tokens and not has_explicit_result:
        return _build_comment_only_game(comment_stripped, body_start, body_tokens)
    return None


def _build_comment_only_game(
    comment_stripped: str,
    body_start: int,
    body_tokens: list[str],
) -> chess.pgn.Game:
    """Compose the zero-move game. Body may be empty (pure comments) or
    contain only result tokens (``*\`, ``1-0\`, etc.)."""
    game = chess.pgn.Game()
    for m in TAG_PAIR_REGEX.finditer(comment_stripped[:body_start]):
        tag_name = m.group(1)
        tag_value = _unescape_pgn_tag_value(m.group(2))
        if tag_name in game.headers:
            continue
        if tag_value is not None:
            game.headers[tag_name] = tag_value
    for tok in body_tokens:
        if tok in ("1-0", "0-1", "1/2-1/2", "*"):
            game.headers["Result"] = tok
            break
    game.headers.setdefault("Result", "*")
    game.comment_only_input = True  # type: ignore[attr-defined]
    return game


def _normalize_text(cleaned: str) -> str:
    """Translate unicode figurines, attach NAGs/asterisks with spaces,
    normalize ``0-0\` / ``o-o\`, strip e.p. markers."""
    norm_text = normalize_movetext_figurines(cleaned)
    norm_text = _ATTACHED_NAG_RE.sub(r"\1 \2", norm_text)
    norm_text = _ATTACHED_NAG_CASTLE_RE.sub(r"\1\2 \3", norm_text)
    norm_text = _ATTACHED_ASTERISK_RE.sub(r"\1 *", norm_text)
    norm_text = _ATTACHED_ASTERISK_CASTLE_RE.sub(r"\1\2 *", norm_text)
    norm_text = re.sub(r"\b0-0-0\b", "O-O-O", norm_text)
    norm_text = re.sub(r"\bo-o-o\b", "O-O-O", norm_text, flags=re.IGNORECASE)
    norm_text = re.sub(r"\b0-0\b", "O-O", norm_text)
    norm_text = re.sub(r"\bo-o\b", "O-O", norm_text, flags=re.IGNORECASE)
    norm_text = re.sub(
        r"\b([a-h][1-8]|[a-h]x[a-h][1-8])\s+\(?e\.?p\.?\)?(?=\s|$)",
        r"\1",
        norm_text,
        flags=re.IGNORECASE,
    )
    return norm_text


def _scan_header_end(masked_norm: str) -> int:
    """Find the index where the movetext section begins."""
    header_end = 0
    first_header = TAG_PAIR_REGEX.search(masked_norm)
    first_mv = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", masked_norm)
    if first_header and (not first_mv or first_header.start() < first_mv.start()):
        header_end = first_header.end()
        for m in TAG_PAIR_REGEX.finditer(masked_norm):
            if m.start() < header_end:
                continue
            if masked_norm[header_end : m.start()].strip() == "":
                header_end = m.end()
            else:
                break
    return header_end


def _bare_moves_fallback(norm_text: str, cleaned: str) -> chess.pgn.Game:
    """Audit P0: bare-moves fallback must NOT silently drop tokens —
    raises on any unrecognized token rather than substituting a different move."""
    tokens = norm_text.split()
    if not tokens:
        raise ValueError(
            f"INVALID_POSITION: Input '{cleaned[:100]}' could not be parsed as a valid FEN, PGN, or move sequence."
        )
    b = chess.Board()
    cur_moves: list[chess.Move] = []
    cur_result: str | None = None
    last_was_result = False
    for t in tokens:
        clean_t = t.rstrip(".,;:!?").lstrip(".,;:!?")
        clean_t = re.sub(
            r"\s*\(?\s*e\.?p\.?\s*\)?$",
            "",
            clean_t,
            flags=re.IGNORECASE,
        ).rstrip(".,:!?")
        if (
            not clean_t
            or clean_t.lower() in ("e.p.", "e.p", "ep", "(e.p.)", "(e.p)", "(ep)")
            or re.match(r"^\d+[\.\:]*$", clean_t)
        ):
            continue
        if clean_t in ("1-0", "0-1", "1/2-1/2", "*"):
            if not last_was_result:
                cur_result = clean_t
                last_was_result = True
            continue
        if last_was_result:
            continue
        try:
            m = b.parse_san(clean_t)
            b.push(m)
            cur_moves.append(m)
            continue
        except Exception:
            pass
        try:
            m = chess.Move.from_uci(clean_t)
            if m in b.legal_moves:
                b.push(m)
                cur_moves.append(m)
                continue
        except Exception:
            pass
        raise ValueError(
            f"INVALID_PGN: Move token {t!r} could not be parsed as a legal "
            f"chess move at this point in the game. The lenient parser "
            f"refuses to substitute a different move."
        )

    game = chess.pgn.Game()
    if cur_result:
        game.headers["Result"] = cur_result
    curr: chess.Board | chess.pgn.GameNode = game
    for m in cur_moves:
        curr = curr.add_variation(m)
    return game
