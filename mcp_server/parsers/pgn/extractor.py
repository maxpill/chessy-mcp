"""PGN game extractor — top-level orchestrator for game-text → ``chess.pgn.Game``.

Layered approach:

    :func:`extract_canonical_pgn_text` — first-pass cleaner: NUL/zero-width /
    Unicode-result normalization, markdown fence ranking, conversational
    preamble/trailer trim, returns raw text the caller can then parse.
    :func:`extract_game` — public entry point; pre-sanitizes headers,
    multi-game-checks, canonicalizes, parses, applies strict-mode surface
    checks. Returns a :class:`chess.pgn.Game`.
    :func:`extract_game_inner` — parsing-only path used by both the public
    ``extract_game`` and re-entry from ``parse_pgn_game_candidate``. Implements
    comment-only input shortcut (audit R4-§B), FEN validation, and bare-moves
    fallback (audit P0).
    :func:`parse_pgn_game_candidate` — delegate to ``chess.pgn.read_game``
    with preflight checks (variant validation, attached-asterisk cleanup,
    bracket sanitization, movetext-token validation).

Strict mode is a flag threaded through each helper; bare-moves fallback in
particular changes shape between lenient and strict invocations.
"""

from __future__ import annotations

import io
import re

import chess
import chess.pgn

from mcp_server.parsers.pgn.conversational import clean_conversational_text
from mcp_server.parsers.pgn.header_syntax import (
    sanitize_malformed_pgn_header_lines,
    validate_strict_header_syntax,
)
from mcp_server.parsers.pgn.multiline_tags import normalize_multiline_tags
from mcp_server.parsers.pgn.multiple_games import check_multiple_games
from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX
from mcp_server.parsers.pgn.tokens import (
    validate_movetext_tokens,
    validate_strict_mainline_surface,
)
from mcp_server.parsers.pgn.unicode import (
    normalize_movetext_figurines,
    normalize_unicode_pgn_results,
)
from mcp_server.parsers.pgn_validate import _validate_fen_counters, _validate_variant
from mcp_server.parsers.pgn_sanitize import (
    _mask_comments_and_escapes,
    _sanitize_brackets_in_variations_and_comments,
    _strip_pgn_escape_lines,
    _unescape_pgn_tag_value,
)
from mcp_server.rules import format_fen_status_errors

__all__ = [
    "extract_canonical_pgn_text",
    "extract_game",
    "extract_game_inner",
    "parse_pgn_game_candidate",
]


_FENCE_RE = re.compile(r"```([a-zA-Z0-9_-]*)\s*([\s\S]*?)\s*```")
_ATTACHED_ASTERISK_RE = re.compile(r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)\*")
_ATTACHED_ASTERISK_CASTLE_RE = re.compile(r"\b(O-O-O|O-O)([\+#\?!]*)\*")
_FIRST_MOVE_RE = re.compile(r"\b1\s*[\.\:]\s*[A-Za-z]")
_ATTACHED_NAG_RE = re.compile(r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)(\$\d+)")
_ATTACHED_NAG_CASTLE_RE = re.compile(r"\b(O-O-O|O-O)([\+#\?!]*)(\$\d+)")


def extract_canonical_pgn_text(text: str) -> str:
    """Isolate the canonical PGN text from markdown fences, conversational preambles, and trailers."""
    # R4-§E: NUL bytes are silently stripped — PGN parsers do not expect them.
    cleaned = (
        text.replace("\x00", "")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .strip()
    )
    cleaned = normalize_unicode_pgn_results(cleaned)
    if not cleaned:
        raise ValueError("Empty chess game/PGN input provided")

    # 1. Enumerate markdown fenced code blocks and rank them.
    fences = list(_FENCE_RE.finditer(cleaned))
    if fences:
        ranked: list[tuple[int, str]] = []
        for m in fences:
            lang = (m.group(1) or "").strip().lower()
            body = m.group(2).strip("`'\" \t\r\n")
            if not body:
                continue
            if lang in ("pgn", "chess"):
                score = 100
            elif re.search(r"\b1\s*[\.\:]\s*[A-Za-z]|\[\s*[A-Za-z0-9_]+\s+\"", body):
                score = 50
            else:
                score = 10
            ranked.append((score, body))

        if ranked:
            ranked.sort(key=lambda x: x[0], reverse=True)
            best_body = ranked[0][1]
            return _strip_pgn_escape_lines(normalize_multiline_tags(best_body))

    cleaned = normalize_multiline_tags(cleaned)
    cleaned_conv = clean_conversational_text(cleaned)
    if cleaned_conv:
        return _strip_pgn_escape_lines(normalize_multiline_tags(cleaned_conv))

    return _strip_pgn_escape_lines(cleaned)


def extract_game(text: str, strict: bool = False) -> chess.pgn.Game:
    """Extract a ``chess.pgn.Game`` from raw, dirty, annotated, or conversational text."""
    sanitized, _header_warnings = sanitize_malformed_pgn_header_lines(text, strict=strict)
    check_multiple_games(sanitized)
    canonical = extract_canonical_pgn_text(sanitized)
    game = extract_game_inner(canonical, strict=strict)
    if strict:
        validate_strict_header_syntax(canonical)
        validate_strict_mainline_surface(canonical, game)
    return game


def parse_pgn_game_candidate(text: str, strict: bool = False) -> chess.pgn.Game | None:
    """Wrap ``chess.pgn.read_game`` with preflight checks.

    Returns ``None`` on parse failure that *might* succeed elsewhere
    (e.g. when the caller can fall back to bare-moves parsing).
    """
    try:
        masked_for_tags = _mask_comments_and_escapes(text)
        for m in TAG_PAIR_REGEX.finditer(masked_for_tags):
            if m.group(1).lower() == "variant":
                _validate_variant(_unescape_pgn_tag_value(m.group(2)))

        has_real_tags = bool(TAG_PAIR_REGEX.search(masked_for_tags))
        text = _truncate_movetext_at_result(text)
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


def _truncate_movetext_at_result(text: str) -> str:
    """Local import shim — delegates to the canonical implementation."""
    from mcp_server.parsers.pgn.movetext import truncate_movetext_at_result

    return truncate_movetext_at_result(text)


def extract_game_inner(cleaned: str, strict: bool = False) -> chess.pgn.Game:
    """Parsing-only entry — no header sanitization or strict-mode gates."""
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

    # R4-§B: comment-only input is a zero-move game.
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
        game.comment_only_input = True  # type: ignore[attr-defined]
        return game

    # Translate unicode figurines and split attached NAGs/asterisks.
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

    masked_norm = _mask_comments_and_escapes(norm_text)
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

    # P0 (2026-09-02 ultra audit): bare-moves fallback must NOT silently drop
    # tokens — that was the bug that let a malformed sequence parse as a
    # different game.
    tokens = norm_text.split()
    if tokens:
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

        if cur_moves or cur_result is not None:
            game = chess.pgn.Game()
            if cur_result:
                game.headers["Result"] = cur_result
            curr: chess.Board | chess.pgn.GameNode = game
            for m in cur_moves:
                curr = curr.add_variation(m)
            return game

    raise ValueError(
        f"INVALID_POSITION: Input '{cleaned[:100]}' could not be parsed as a valid FEN, PGN, or move sequence."
    )
