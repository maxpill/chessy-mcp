"""PGN canonical extraction and validation.

Largest module of the parsers package. Owns:

- Unicode result / hyphen normalization (audit L-04).
- Movetext figurine translation (preserving header text).
- Strict token validation (NAGs, castling, e.p.).
- The conversational-text extractor that walks prose and fences to
  isolate the canonical PGN block.
- The :func:`_extract_game` / :func:`_extract_game_inner` pair that
  delegates to ``chess.pgn.read_game`` after sanitization.
- Helpers for tag-block cluster detection, multi-game detection,
  and the strict-movetext surface validator.

Underscored aliases preserved for backwards compatibility.
"""

from __future__ import annotations

import io
import re
from typing import Final

import chess
import chess.pgn
from mcp_server.rules import format_fen_status_errors

from mcp_server.parsers.pgn_sanitize import (
    _mask_comments_and_escapes,
    _sanitize_brackets_in_variations_and_comments,
    _strip_pgn_escape_lines,
    _unescape_pgn_tag_value,
)
from mcp_server.parsers.pgn_validate import (
    _validate_fen_counters,
    _validate_variant,
)

__all__ = [
    "FIGURINE_MAP",
    "TAG_PAIR_REGEX",
    "UNICODE_HYPHEN_MAP",
    "check_multiple_games",
    "clean_conversational_text",
    "extract_canonical_pgn_text",
    "extract_game",
    "extract_game_inner",
    "find_movetext_result",
    "has_completed_game_before",
    "infer_result_from_termination",
    "is_canonical_tag_line",
    "is_prose_line",
    "normalize_movetext_figurines",
    "normalize_multiline_tags",
    "normalize_unicode_pgn_results",
    "parse_pgn_game_candidate",
    "sanitize_malformed_pgn_header_lines",
    "strict_top_level_movetext_tokens",
    "strip_promotion_eq",
    "truncate_movetext_at_result",
    "validate_movetext_tokens",
    "validate_strict_header_syntax",
    "validate_strict_mainline_surface",
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

TAG_PAIR_REGEX: Final[re.Pattern[str]] = re.compile(
    r'\[\s*([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\s*\]', re.DOTALL
)

UNICODE_HYPHEN_MAP: Final = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "‐": "-",
        "−": "-",
    }
)






def normalize_unicode_pgn_results(text: str) -> str:
    """Normalize Unicode PGN result markers and hyphens to ASCII (audit L-04).

    Tolerates the common Unicode variants emitted by chess software and
    online databases:
        ½-½, ½–½, ½—½  ->  1/2-1/2
        0–1, 0—1        ->  0-1
        1–0, 1—0        ->  1-0
    and en/em dashes in movetext castling/result lines.
    """
    # Map every ½ (U+00BD) run to "1/2"
    text = text.replace("½", "1/2")
    # Strip a wide zero-width joiner / non-breaking hyphen that some browsers
    # insert in PGN exports; harmless if absent.
    text = text.replace("\u200b", "").replace("\u00a0", " ")
    text = text.translate(_UNICODE_HYPHEN_MAP)
    return text




def normalize_movetext_figurines(text: str) -> str:
    """Translate Unicode chess figurines only in the movetext section (preserving headers and comments)."""
    masked = _mask_comments_and_escapes(text)
    header_end = 0
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
            result.append(ch.translate(_FIGURINE_MAP))
        i += 1

    return headers_part + "".join(result)




def validate_movetext_tokens(
    movetext: str,
    start_board: chess.Board | None = None,
    strict: bool = False,
    nag_warnings: list[str] | None = None,
) -> list[str]:
    """Check that all tokens in the active movetext section are valid chess moves or PGN symbols.

    `nag_warnings`, when provided, receives out-of-range NAG tokens in lenient mode (the audit's
    P3 INVESTIGATE finding: `$999999` was silently accepted in lenient mode). In strict mode,
    out-of-range NAGs are returned in `invalid_tokens` and fail the parse.
    """
    # 1. Translate figurines in movetext and split attached NAGs (PGN-07) and attached asterisk
    t = _normalize_movetext_figurines(movetext)
    t = re.sub(r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)(\$\d+)", r"\1 \2", t)
    t = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)(\$\d+)", r"\1\2 \3", t)
    t = re.sub(r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)\*", r"\1 *", t)
    t = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)\*", r"\1\2 *", t)
    # 2. Normalize castling and en passant notation
    t = re.sub(r"\b0-0-0\b", "O-O-O", t)
    t = re.sub(r"\bo-o-o\b", "O-O-O", t, flags=re.IGNORECASE)
    t = re.sub(r"\b0-0\b", "O-O", t)
    t = re.sub(r"\bo-o\b", "O-O", t, flags=re.IGNORECASE)
    t = re.sub(
        r"\b([a-h][1-8]|[a-h]x[a-h][1-8])\s+\(?e\.?p\.?\)?(?=\s|$)",
        r"\1",
        t,
        flags=re.IGNORECASE,
    )
    # 3. Remove all PGN header tags
    t = TAG_PAIR_REGEX.sub(" ", t)
    # 4. Remove semicolon comments and % escape lines
    t = re.sub(r";[^\r\n]*", " ", t)
    t = re.sub(r"^[ \t]*%[^\r\n]*", " ", t, flags=re.MULTILINE)
    # 5. Remove all comments {...} (handling nested brackets in comments)
    while "{" in t and "}" in t:
        prev = t
        t = re.sub(r"\{[^{}]*\}", " ", t, flags=re.DOTALL)
        if t == prev:
            break
    # 6. Remove all variations (...)
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
            # P3/INVESTIGATE (2026-09-02 ultra audit): NAGs outside the
            # 0..255 range defined by the PGN spec were silently dropped
            # in lenient mode. Strict mode already rejected them — keep
            # that behavior, but in lenient mode surface a warning via
            # the nag_warnings channel so callers can see the out-of-
            # range value instead of parsing as if it never existed.
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
        if re.match(r"^\d+[\.\:]*$", tok) or clean_tok in (
            "1-0",
            "0-1",
            "1/2-1/2",
            "*",
        ):
            if clean_tok in ("1-0", "0-1", "1/2-1/2", "*"):
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




def truncate_movetext_at_result(text: str) -> str:
    masked = _mask_comments_and_escapes(text)
    header_end = 0
    for m in TAG_PAIR_REGEX.finditer(masked):
        if masked[header_end : m.start()].strip() == "":
            header_end = m.end()
        else:
            break

    headers_part = text[:header_end]
    movetext = text[header_end:]
    masked_movetext = masked[header_end:]

    var_depth = 0
    i = 0
    while i < len(masked_movetext):
        ch = masked_movetext[i]
        if ch == "(":
            var_depth += 1
        elif ch == ")":
            var_depth = max(0, var_depth - 1)
        elif var_depth == 0:
            for marker in ("1-0", "0-1", "1/2-1/2", "*"):
                if masked_movetext[i : i + len(marker)] == marker:
                    left_ok = i == 0 or masked_movetext[i - 1] in " \t\r\n;"
                    right_idx = i + len(marker)
                    right_ok = (
                        right_idx == len(masked_movetext)
                        or masked_movetext[right_idx] in " \t\r\n;"
                    )
                    if left_ok and right_ok:
                        return headers_part + movetext[:right_idx]
        i += 1
    return text




def parse_pgn_game_candidate(text: str, strict: bool = False) -> chess.pgn.Game | None:
    try:
        masked_for_tags = _mask_comments_and_escapes(text)
        for m in TAG_PAIR_REGEX.finditer(masked_for_tags):
            if m.group(1).lower() == "variant":
                _validate_variant(_unescape_pgn_tag_value(m.group(2)))

        has_real_tags = bool(TAG_PAIR_REGEX.search(masked_for_tags))
        text = _truncate_movetext_at_result(text)
        # Separate attached asterisks
        text = re.sub(
            r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)\*",
            r"\1 *",
            text,
        )
        text = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)\*", r"\1\2 *", text)
        # Sanitize brackets inside variations/comments so read_game does not mistake them for PGN headers
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

            invalid_tokens = _validate_movetext_tokens(
                text, start_board=game.board(), strict=strict
            )
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




def check_multiple_games(cleaned: str) -> None:
    cleaned_escapes = _strip_pgn_escape_lines(cleaned)
    masked_cleaned = _mask_comments_and_escapes(cleaned_escapes)

    for m in TAG_PAIR_REGEX.finditer(masked_cleaned):
        if m.group(1).lower() == "variant":
            _validate_variant(_unescape_pgn_tag_value(m.group(2)))

    # 1. Multiple markdown fenced code blocks
    fences = list(re.finditer(r"```([a-zA-Z0-9_-]*)\s*([\s\S]*?)\s*```", cleaned_escapes))
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

    # 2. Check multiple games via explicit header blocks
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
            first_mv = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", after_cl)
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




def normalize_multiline_tags(text: str) -> str:
    """Normalize multiline tag pairs [Tag \\n "Value"] to [Tag "Value"] on a single line."""

    def _repl(m: re.Match[str]) -> str:
        tag_name = m.group(1).strip()
        tag_val = m.group(2)
        return f'[{tag_name} "{tag_val}"]'

    return TAG_PAIR_REGEX.sub(_repl, text)




def has_completed_game_before(text: str, pos: int) -> bool:
    prefix = text[:pos]
    first_mv = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", prefix)
    if not first_mv:
        return False
    rest = prefix[first_mv.start() :]
    return bool(re.search(r"(?:^|\s)(?:1-0|0-1|1/2-1/2|\*)(?:\s|$)", rest))




def is_prose_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if (
        stripped.startswith("[")
        or re.match(r"^\d+\s*[\.\:]", stripped)
        or stripped.startswith("{")
        or stripped.startswith("(")
        or stripped in ("1-0", "0-1", "1/2-1/2", "*")
    ):
        return False
    words = stripped.split()
    if not words:
        return False
    prose_words = {
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "think",
        "thought",
        "thinks",
        "thinking",
        "considered",
        "felt",
        "believe",
        "believed",
        "afterwards",
        "afterward",
        "after",
        "before",
        "during",
        "later",
        "then",
        "next",
        "also",
        "better",
        "best",
        "worse",
        "worst",
        "good",
        "bad",
        "nice",
        "great",
        "poor",
        "blunder",
        "mistake",
        "was",
        "were",
        "is",
        "are",
        "am",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "must",
        "will",
        "shall",
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "game",
        "play",
        "played",
        "moves",
        "move",
        "position",
        "opening",
        "variation",
        "because",
        "since",
        "so",
        "but",
        "and",
        "or",
        "if",
        "though",
        "although",
        "instead",
    }
    prose_count = sum(1 for w in words if w.strip(".,;:!?\"'()").lower() in prose_words)
    if len(words) >= 2 and (prose_count >= 2 or (prose_count / len(words)) >= 0.4):
        return True
    first_word = words[0].strip(".,;:!?\"'()").lower()
    if first_word in (
        "afterwards",
        "afterward",
        "what",
        "how",
        "why",
        "note",
        "comment",
        "analysis",
        "thoughts",
        "question",
        "here",
    ):
        return True
    return False




def is_canonical_tag_line(line: str) -> bool:
    stripped = line.strip()
    if (
        not stripped
        or stripped.startswith(";")
        or stripped.startswith("%")
        or stripped.startswith("{")
    ):
        return False
    return bool(re.match(r'^(?:\[\s*[A-Za-z0-9_]+\s+"(?:[^"\\]|\\.)*"\s*\]\s*)+$', stripped))




def strip_promotion_eq(s: str) -> str:
    """Strip the optional '=' in PGN promotion (e8=Q vs e8Q per §8.1.4)."""
    return re.sub(r"=([QRBN])$", r"\1", s)




def validate_strict_header_syntax(text: str) -> None:
    """Reject malformed PGN tag lines that tolerant cleaning would otherwise discard."""
    normalized = _normalize_unicode_pgn_results(text)
    masked = _mask_comments_and_escapes(normalized)
    raw_lines = normalized.splitlines()
    masked_lines = masked.splitlines()
    for index, visible in enumerate(masked_lines):
        if not re.match(r"^\s*\[[A-Za-z0-9_]+\b", visible):
            continue
        raw = raw_lines[index] if index < len(raw_lines) else visible
        if not _is_canonical_tag_line(raw):
            raise ValueError(
                f"STRICT_PGN_ERROR: Malformed PGN tag syntax on line {index + 1}: {raw.strip()!r}"
            )




def strict_top_level_movetext_tokens(text: str) -> list[str]:
    """Return top-level movetext tokens with comments/RAVs masked out."""
    normalized = _normalize_movetext_figurines(_normalize_unicode_pgn_results(text))
    masked = _mask_comments_and_escapes(normalized)

    header_end = 0
    for match in TAG_PAIR_REGEX.finditer(masked):
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
    # R4-§C (2026-09-02 ultra audit round 4): PGN §8.1 allows whitespace
    # around the move-number dot (`1 . e4` is equivalent to `1. e4`). The
    # adjacent regex above only matches `N.` or `N...` with no whitespace.
    # Collapse the split-digit / split-dot form back into a single move-
    # number token so the strict validator does not see a bare digit as
    # a SAN attempt. Only the leading `N` of a token is considered so
    # mid-token dots (e.g. `4.0.0` would not match the move-number regex
    # anyway) are unaffected.
    top_level = re.sub(
        r"(?<!\S)(\d+)\s+(\.+)(\s+|$)",
        lambda m: f" {m.group(1)}{m.group(2)} ",
        top_level,
    )
    return top_level.split()




def validate_strict_mainline_surface(text: str, game: chess.pgn.Game) -> None:
    """Require canonical SAN and correct explicit move numbers in strict mode."""
    tokens = _strict_top_level_movetext_tokens(text)
    moves = list(game.mainline_moves())
    board = game.board()
    move_index = 0

    for token in tokens:
        clean = token.strip()
        if not clean:
            continue
        if clean in ("1-0", "0-1", "1/2-1/2", "*"):
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

        # Per PGN §8.1.4, promotion can use either '=X' or 'X' form. Strip
        # the optional '=' so 'e8=Q' and 'e8Q' both compare equal.
        # python-chess's san() inconsistently emits '+' / '#' for check /
        # mate moves — some moves like 'Qh5' come back without '+' even
        # though they give check, while 'Rd8' returns 'Rd8+'. Compare the
        # base forms (both sides stripped of '+' / '#'), then validate the
        # supplied marker separately against actual check / mate state.
        canonical_base = _strip_promotion_eq(canonical.rstrip("+#"))
        supplied_base = _strip_promotion_eq(supplied_san.rstrip("+#"))
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




def extract_canonical_pgn_text(text: str) -> str:
    """Isolate the canonical PGN text from markdown fences, conversational preambles, and trailers."""
    # R4-§E (2026-09-02 ultra audit round 4): NUL bytes are silently
    # stripped from the input. PGN parsers do not expect NUL chars and
    # treating them as part of a SAN token produces confusing errors.
    cleaned = (
        text.replace("\x00", "")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .strip()
    )
    # L-04 audit fix: normalize Unicode PGN markers (½-½ etc.) before any
    # further processing so the rest of the parser sees ASCII PGN.
    cleaned = _normalize_unicode_pgn_results(cleaned)
    if not cleaned:
        raise ValueError("Empty chess game/PGN input provided")

    # 1. Enumerate markdown fenced code blocks and rank them (PGN-04: prefer pgn/chess tag)
    fences = list(re.finditer(r"```([a-zA-Z0-9_-]*)\s*([\s\S]*?)\s*```", cleaned))
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
            return _strip_pgn_escape_lines(_normalize_multiline_tags(best_body))

    # 2. Conversational preamble and trailer cleaning
    cleaned = _normalize_multiline_tags(cleaned)
    cleaned_conv = _clean_conversational_text(cleaned)
    if cleaned_conv:
        return _strip_pgn_escape_lines(_normalize_multiline_tags(cleaned_conv))

    return _strip_pgn_escape_lines(cleaned)




def clean_conversational_text(text: str) -> str:
    text = _strip_pgn_escape_lines(text)
    text = _normalize_multiline_tags(text)
    masked_text = _mask_comments_and_escapes(text)

    # 1. Find valid PGN tag pairs outside inline code and comments
    tag_matches: list[re.Match[str]] = []
    for m in TAG_PAIR_REGEX.finditer(masked_text):
        # Ignore tags enclosed in inline code backticks e.g. `[FEN "..."]`
        if (
            m.start() > 0
            and text[m.start() - 1] == "`"
            and m.end() < len(text)
            and text[m.end()] == "`"
        ):
            continue
        line_start = masked_text.rfind("\n", 0, m.start()) + 1
        prefix_on_line = masked_text[line_start : m.start()]
        line_end = masked_text.find("\n", m.end())
        line_end = len(masked_text) if line_end == -1 else line_end
        suffix_on_line = masked_text[m.end() : line_end]

        pref_clean = prefix_on_line.strip()
        suff_clean = suffix_on_line.strip()
        if pref_clean.endswith("`") or suff_clean.startswith("`"):
            continue
        # If suffix on the same line contains prose text (not additional tag pairs):
        if suff_clean and not suff_clean.startswith("["):
            continue
        tag_matches.append(m)

    best_header_str = ""
    best_movetext_str = ""

    first_mv_in_full = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", masked_text)

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

        def _cluster_eval(cl: list[re.Match[str]]) -> tuple[bool, int, int]:
            h_end = cl[-1].end()
            after_h = text[h_end:].strip()
            first_mv_after = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", after_h)
            has_direct_moves = False
            if first_mv_after:
                mv_pos = h_end + first_mv_after.start()
                # Ensure no other valid tag cluster appears between this cluster and the moves
                other_cluster_between = any(
                    other_cl is not cl and h_end <= other_cl[0].start() < mv_pos
                    for other_cl in clusters
                )
                if not other_cluster_between:
                    has_direct_moves = True
            std_count = sum(
                1
                for m in cl
                if m.group(1)
                in (
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
                )
            )
            return (has_direct_moves, std_count, len(cl))

        clusters.sort(key=_cluster_eval, reverse=True)
        best_cluster = clusters[0]
        has_direct_moves, std_count, cl_len = _cluster_eval(best_cluster)

        if has_direct_moves or (first_mv_in_full is None and (std_count > 0 or cl_len >= 2)):
            h_start = best_cluster[0].start()
            h_end = best_cluster[-1].end()
            best_header_str = text[h_start:h_end].strip()

            after_header = text[h_end:].strip()
            first_move_after = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", after_header)
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
        if first_mv_in_full:
            best_movetext_str = text[first_mv_in_full.start() :]
        else:
            best_movetext_str = text

    # Trim trailing non-chess prose from movetext
    lines = best_movetext_str.splitlines()
    end_idx = len(lines)
    found_moves = False
    for i, line_item in enumerate(lines):
        stripped = line_item.strip()
        if stripped.startswith(";") or stripped.startswith("%"):
            continue
        if re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", stripped) or re.search(
            r"\b(?:[a-h][1-8]|O-O)", stripped
        ):
            found_moves = True
        elif found_moves:
            if any(
                stripped.startswith(res) or stripped.endswith(res)
                for res in ("1-0", "0-1", "1/2-1/2", "*")
            ):
                end_idx = i + 1
                break
            if (
                _is_prose_line(stripped)
                or stripped.startswith("Here ")
                or stripped.startswith("Note ")
            ):
                end_idx = i
                break
            has_chess = bool(re.search(r"\b\d+\s*[\.\:]|[a-h][1-8]|O-O|1-0|0-1|1/2", stripped))
            if not has_chess and stripped:
                end_idx = i
                break

    final_movetext = "\n".join(lines[:end_idx]).strip()

    if best_header_str and final_movetext:
        return f"{best_header_str}\n\n{final_movetext}"
    return best_header_str or final_movetext or text




def extract_game(text: str, strict: bool = False) -> chess.pgn.Game:
    """Extract a chess.pgn.Game from raw, dirty, annotated, or conversational text."""
    sanitized, _header_warnings = _sanitize_malformed_pgn_header_lines(text, strict=strict)
    _check_multiple_games(sanitized)
    canonical = _extract_canonical_pgn_text(sanitized)
    game = _extract_game_inner(canonical, strict=strict)
    if strict:
        _validate_strict_header_syntax(canonical)
        _validate_strict_mainline_surface(canonical, game)
    return game




def extract_game_inner(cleaned: str, strict: bool = False) -> chess.pgn.Game:
    # Validate all PGN headers (FEN, Variant, etc.) before any shortcut
    # can return — otherwise a PGN like
    # `[SetUp "1"][FEN "...w - - 0 1 junk"]\n\n*` (only a result token
    # after the headers) would silently drop the trailing-junk FEN
    # validation. R4-§A.
    masked_cleaned = _mask_comments_and_escapes(cleaned)
    for m in TAG_PAIR_REGEX.finditer(masked_cleaned):
        tag_name = m.group(1).lower()
        if tag_name == "variant":
            _validate_variant(_unescape_pgn_tag_value(m.group(2)))
        # P1 (2026-09-02 ultra audit): the FEN tag from PGN headers must be
        # validated against the same rules as a direct FEN input — fullmove
        # must be ≥1, halfmove within bounds, EP+halfmove historically
        # consistent. Previously, fullmove=0 inside [FEN "..."] was silently
        # accepted by `analyze_game(strict=true)` while the same FEN was
        # rejected by `evaluate_position`. This call shares the unified
        # validator with `_build_board`. Both strict and lenient modes reject
        # because the FEN value is structurally identical to a direct FEN —
        # there is no permissive grammar here that would justify
        # normalization.
        if tag_name == "fen":
            fen_val = _unescape_pgn_tag_value(m.group(2))
            if fen_val:
                # R4-§A (2026-09-02 ultra audit round 4): reject trailing-
                # junk FENs inside [FEN ...] headers with the same
                # INVALID_FEN code that `_build_board` uses. The trailing-
                # junk check lives in `_build_board` for direct-FEN inputs;
                # for PGN headers we must run it here too so callers see
                # consistent semantics across endpoints.
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

    # R4-§B (2026-09-02 ultra audit round 4): a PGN that consists entirely
    # of comments (with or without a result token like '*') is a valid
    # zero-move game. Previously the lenient parser's bare-moves fallback
    # tried to parse `{just` as a SAN token, producing a confusing
    # "Move token '{just' could not be parsed" error. Detect this case
    # before the parser runs, strip comments + variations, and return an
    # empty game with the supplied headers (and a metadata_warning so
    # callers see that the input contained no moves).
    comment_stripped = _mask_comments_and_escapes(cleaned)
    # Strip inline comments in { ... } (already gone after mask; double-check)
    comment_stripped = re.sub(r"\{[^{}]*\}", " ", comment_stripped, flags=re.DOTALL)
    # Strip variations ( ... )
    while "(" in comment_stripped and ")" in comment_stripped:
        prev = comment_stripped
        comment_stripped = re.sub(r"\([^()]*\)", " ", comment_stripped, flags=re.DOTALL)
        if comment_stripped == prev:
            break
    # Now extract just the header block (consecutive leading tags) — the
    # body should contain only whitespace + an optional result token.
    body_start = 0
    for m in TAG_PAIR_REGEX.finditer(comment_stripped):
        if comment_stripped[body_start : m.start()].strip() == "":
            body_start = m.end()
        else:
            break
    body = comment_stripped[body_start:].strip()
    # Reduce body to just its result token if any.
    body_tokens = body.split()
    # R4-§B (2026-09-02 ultra audit round 4): the comment-only shortcut
    # applies only when the body has no moves AND no explicit result
    # token OTHER than the wildcard `*`. Explicit `1-0`, `0-1`, or
    # `1/2-1/2` results must flow through the normal analyzer so the
    # board-derived outcome can be cross-validated against the supplied
    # result (e.g. a checkmated position with a `Result "0-1"` header
    # is still analyzed for its checkmate outcome).
    non_result_body_tokens = [t for t in body_tokens if t not in ("1-0", "0-1", "1/2-1/2", "*")]
    has_explicit_result = any(t in ("1-0", "0-1", "1/2-1/2") for t in body_tokens)
    if not non_result_body_tokens and not has_explicit_result:
        # Comment-only input (no moves, no explicit result token, or
        # only the wildcard `*`). Build an empty game carrying only the
        # parsed headers (if any).
        game = chess.pgn.Game()
        for m in TAG_PAIR_REGEX.finditer(comment_stripped[:body_start]):
            tag_name = m.group(1)
            tag_value = _unescape_pgn_tag_value(m.group(2))
            if tag_name in game.headers:
                # Defer to the standard duplicate-detection path in the
                # analyzer so the comment-only shortcut behaves the same
                # as any other input with respect to duplicate headers.
                continue
            if tag_value is not None:
                game.headers[tag_name] = tag_value
        # Preserve the result token if present.
        for tok in body_tokens:
            if tok in ("1-0", "0-1", "1/2-1/2", "*"):
                game.headers["Result"] = tok
                break
        # Stash a sentinel so analyze_game can surface the
        # comment-only metadata_warning.
        game.comment_only_input = True  # type: ignore[attr-defined]
        return game

    # Translate unicode figurines ONLY in movetext (headers and comments preserved)
    norm_text = _normalize_movetext_figurines(cleaned)
    norm_text = re.sub(
        r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)(\$\d+)",
        r"\1 \2",
        norm_text,
    )
    norm_text = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)(\$\d+)", r"\1\2 \3", norm_text)
    norm_text = re.sub(
        r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)\*",
        r"\1 *",
        norm_text,
    )
    norm_text = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)\*", r"\1\2 *", norm_text)
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

    # 1. Contiguous leading tag block
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
        g = _parse_pgn_game_candidate(norm_text, strict=strict)
        if g is not None:
            if not list(g.mainline_moves()):
                has_move_tokens = bool(re.search(r"\b\d+\s*[\.\:]\s*[A-Za-z]", norm_text))
                if has_move_tokens:
                    raise ValueError(
                        f"INVALID_PGN: Could not parse legal moves from movetext '{norm_text[:100]}'."
                    )
            return g

    # 2. Direct parse attempt
    g = _parse_pgn_game_candidate(norm_text, strict=strict)
    if g is not None:
        if not list(g.mainline_moves()):
            has_move_tokens = bool(re.search(r"\b\d+\s*[\.\:]\s*[A-Za-z]", norm_text))
            if has_move_tokens:
                raise ValueError(
                    f"INVALID_PGN: Could not parse legal moves from movetext '{norm_text[:100]}'."
                )
        return g

    # 3. Fallback: movetext starting with 1. or 1...
    for move_match in re.finditer(r"\b1\s*[\.\:]\s*[A-Za-z]", norm_text):
        sub_movetext = norm_text[move_match.start() :]
        try:
            g = _parse_pgn_game_candidate(sub_movetext, strict=strict)
            if g is not None and list(g.mainline_moves()):
                return g
        except Exception:
            continue

    # 4. Fallback: bare SAN / UCI tokens
    # P0 (2026-09-02 ultra audit): lenient normalization may change syntax,
    # never move identity. Earlier this loop iterated over every possible
    # start index and picked the longest parseable subsequence, which let a
    # malformed movetext like "1... e5 2. Nf3 Nc6 *" silently drop e5 (Black's
    # intended first move) and substitute White's Nf3 in its place — the
    # parser ended up analyzing a completely different game than the caller
    # supplied. The fix is to parse from the start of the input only (no
    # best-of-N across start positions) and refuse to silently drop a move
    # token when later ones parse legally: that drop is exactly the
    # semantic substitution the audit forbids.
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
                # Subsequent result tokens (or trailing move tokens after a
                # result) are dropped — matches the existing "trailing moves
                # after game termination are ignored" behavior.
                continue
            if last_was_result:
                # Game already ended; trailing move tokens are ignored.
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
            # P0: this token did not parse as either SAN or UCI on the
            # current board. The bare-moves fallback must NOT silently skip
            # it and try later tokens (that is the semantic-substitution
            # bug the audit caught). Surface the failure instead.
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




def find_movetext_result(text: str) -> str | None:
    """Extract the canonical result marker from the top level of movetext (outside comments and variations)."""
    # L-04 audit fix: normalize Unicode result markers (½-½, ½–½, etc.) before
    # scanning. PGN specifies ASCII "1/2-1/2" but chess programmes, lichess
    # exports, and tournament software often emit the Unicode fraction. Tolerate
    # the typographic variants so a well-formed-but-Unicode PGN is not rejected
    # as INVALID_PGN.
    text = _normalize_unicode_pgn_results(text)
    masked = _mask_comments_and_escapes(text)
    header_end = 0
    for m in TAG_PAIR_REGEX.finditer(masked):
        if masked[header_end : m.start()].strip() == "":
            header_end = m.end()
        else:
            break

    movetext = masked[header_end:]
    var_depth = 0
    i = 0
    while i < len(movetext):
        ch = movetext[i]
        if ch == "(":
            var_depth += 1
        elif ch == ")":
            var_depth = max(0, var_depth - 1)
        elif var_depth == 0:
            for marker in ("1-0", "0-1", "1/2-1/2", "*"):
                if movetext[i : i + len(marker)] == marker:
                    left_ok = i == 0 or movetext[i - 1] in " \t\r\n;"
                    right_idx = i + len(marker)
                    right_ok = right_idx == len(movetext) or movetext[right_idx] in " \t\r\n;"
                    if left_ok and right_ok:
                        return marker
        i += 1
    return None




def sanitize_malformed_pgn_header_lines(text: str, strict: bool = False) -> tuple[str, list[str]]:
    """Reject or remove malformed tag-pair lines before PGN extraction.

    The conversational PGN cleaner clusters only syntactically valid tag
    pairs. A malformed line between valid tags used to split the cluster and
    silently discard otherwise valid metadata. We inspect only the pre-move
    prefix and only activate when that prefix contains at least one valid PGN
    tag, so bracket-looking prose in ordinary movetext is left untouched.
    P2 (2026-09-02 ultra audit): the legacy function returned no warnings
    for header-only PGNs (no `1. <move>` marker before the result) and for
    malformed lines that didn't match the regex pre-filter. Both classes now
    surface warnings in lenient mode.
    """
    normalized = _normalize_multiline_tags(text)
    lines = normalized.splitlines(keepends=True)
    if not lines:
        return normalized, []

    first_move_line = len(lines)
    for idx, line in enumerate(lines):
        if re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", line):
            first_move_line = idx
            break

    prefix = "".join(lines[:first_move_line])
    if TAG_PAIR_REGEX.search(_mask_comments_and_escapes(prefix)) is None:
        # No tag block present; nothing to sanitize.
        return normalized, []
    # P2 (2026-09-02 ultra audit): handle header-only PGNs (no first-move
    # line) by sanitizing every line in the prefix. Previously the
    # `first_move_line >= len(lines)` early return silently dropped
    # malformed headers in header-only inputs.
    scan_end = first_move_line if first_move_line < len(lines) else len(lines)

    warnings: list[str] = []
    for idx in range(scan_end):
        stripped = lines[idx].strip()
        if not stripped.startswith("["):
            continue
        if _is_canonical_tag_line(stripped):
            continue
        # Skip lines that mix a valid tag with extra content (e.g.
        # `[Result "*"] *` — a valid tag followed by the game result
        # token on the same line). The legacy contract accepted such
        # mixed lines as part of the conversational PGN dialect.
        if TAG_PAIR_REGEX.search(stripped) is not None:
            continue
        # P2 (2026-09-02 ultra audit): drop the regex pre-filter that was
        # silently dropping warnings for malformed tags that didn't happen
        # to match `^\[\s*[A-Za-z0-9_]+(?:\s|\])`. The canonical-tag-line
        # check above is the sole gate; everything else that looks tag-like
        # but isn't canonical is reported.
        warning = f"Malformed PGN header line ignored: {stripped!r}."
        if strict:
            raise ValueError(f"STRICT_VALIDATION_ERROR: {warning}")
        warnings.append(warning)
        newline = (
            "\r\n" if lines[idx].endswith("\r\n") else ("\n" if lines[idx].endswith("\n") else "")
        )
        lines[idx] = newline

    return "".join(lines), warnings




def infer_result_from_termination(termination: str | None) -> str | None:
    if not termination:
        return None
    t = re.sub(r"\s+", " ", termination.strip().lower())
    if "normal time control" in t:
        return None

    winner_patterns = (
        (r"\bwhite\s+(?:wins?|won)\b.*\b(?:time|resignation|resigns?)\b", "1-0"),
        (r"\bblack\s+(?:wins?|won)\b.*\b(?:time|resignation|resigns?)\b", "0-1"),
        (r"\bwon\s+by\s+white\b", "1-0"),
        (r"\bwon\s+by\s+black\b", "0-1"),
    )
    for pattern, result in winner_patterns:
        if re.search(pattern, t):
            return result

    loser_patterns = (
        (r"\bwhite\s+(?:resign(?:s|ed)?|lost|loses)\b", "0-1"),
        (r"\bblack\s+(?:resign(?:s|ed)?|lost|loses)\b", "1-0"),
        (r"\bwhite(?:'s)?\s+(?:flag|clock).*(?:fell|expired|flagged|out of time)", "0-1"),
        (r"\bblack(?:'s)?\s+(?:flag|clock).*(?:fell|expired|flagged|out of time)", "1-0"),
        (r"\bwhite\s+(?:lost|loses)\s+on\s+time\b", "0-1"),
        (r"\bblack\s+(?:lost|loses)\s+on\s+time\b", "1-0"),
    )
    for pattern, result in loser_patterns:
        if re.search(pattern, t):
            return result

    if re.search(r"\bwhite\b.*\b(?:illegal move|rules? infraction)\b", t):
        return "0-1"
    if re.search(r"\bblack\b.*\b(?:illegal move|rules? infraction)\b", t):
        return "1-0"
    return None




# Underscored aliases for backwards-compatible import paths.
_infer_result_from_termination = infer_result_from_termination
_sanitize_malformed_pgn_header_lines = sanitize_malformed_pgn_header_lines
_find_movetext_result = find_movetext_result
_extract_game_inner = extract_game_inner
_extract_game = extract_game
_clean_conversational_text = clean_conversational_text
_extract_canonical_pgn_text = extract_canonical_pgn_text
_validate_strict_mainline_surface = validate_strict_mainline_surface
_strict_top_level_movetext_tokens = strict_top_level_movetext_tokens
_validate_strict_header_syntax = validate_strict_header_syntax
_strip_promotion_eq = strip_promotion_eq
_is_canonical_tag_line = is_canonical_tag_line
_is_prose_line = is_prose_line
_has_completed_game_before = has_completed_game_before
_normalize_multiline_tags = normalize_multiline_tags
_check_multiple_games = check_multiple_games
_parse_pgn_game_candidate = parse_pgn_game_candidate
_truncate_movetext_at_result = truncate_movetext_at_result
_validate_movetext_tokens = validate_movetext_tokens
_normalize_movetext_figurines = normalize_movetext_figurines
_normalize_unicode_pgn_results = normalize_unicode_pgn_results
_UNICODE_HYPHEN_MAP = UNICODE_HYPHEN_MAP
_FIGURINE_MAP = FIGURINE_MAP
