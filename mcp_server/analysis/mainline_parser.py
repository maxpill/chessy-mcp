"""``_parse_mainline`` — walk a PGN game mainline and validate tokens.

Extracted from :mod:`mcp_server.analysis.game_analyzer` so the
:class:`GameAnalyzer` orchestration stays focused on the high-level
flow. Surfaces every audit-relevant warning (`U-15`, `R4-§C`,
`B-01`, `P3`) and preserves the trailing-ply count for the
:class:`GameAnalyzer` reconciliation step.

Public surface:

  * :func:`parse_mainline` — the walker. Returns
    ``(positions, moves, syntax_warnings, ignored_trailing_plies, cleaned_movetext)``.
"""

from __future__ import annotations

import re

import chess
import chess.pgn

from mcp_server.parsers import TAG_PAIR_REGEX, _normalize_movetext_figurines, _strip_promotion_eq
from mcp_server.rules import evaluate_rule_status, format_fen_status_errors


def parse_mainline(
    canonical_pgn: str,
    game: chess.pgn.Game,
    *,
    strict: bool,
) -> tuple[list[chess.Board], list[chess.Move], list[str], int, str]:
    """Walk the game mainline, validating tokens and surfacing strict/non-
    strict normalization warnings. Returns ``(positions, moves, syntax_warnings,
    ignored_trailing_plies, cleaned_movetext)``.

    Audit invariants preserved: U-15, R4-§C, B-01, P3.
    """
    positions: list[chess.Board] = []
    moves: list[chess.Move] = []
    syntax_warnings: list[str] = []
    ignored_trailing_plies = 0

    curr_board = game.board()
    if not curr_board.is_valid() or curr_board.status() != chess.STATUS_VALID:
        raise ValueError(
            f"INVALID_FEN: Initial position '{curr_board.fen()}' in PGN is not a valid chess position ({format_fen_status_errors(curr_board.status())})."
        )

    positions.append(curr_board.copy(stack=True))
    auto_termination: str | None = None
    reached_terminal = False

    initial_rule = evaluate_rule_status(curr_board, history_complete="complete")
    if initial_rule.terminal is not None:
        auto_termination = initial_rule.terminal
        reached_terminal = True
        if strict:
            raise ValueError(
                f"STRICT_PGN_ERROR: Initial FEN '{curr_board.fen()}' is already "
                f"terminal ({initial_rule.terminal}); cannot execute movetext."
            )
        syntax_warnings.append(
            f"Initial FEN is terminal ({initial_rule.terminal}); "
            f"all movetext moves will be ignored."
        )

    for nag_match in re.finditer(r"\$([0-9]+)", canonical_pgn):
        nag_val = int(nag_match.group(1))
        if nag_val > 255:
            syntax_warnings.append(f"NAG value ${nag_val} outside the PGN-supported range 0..255.")

    header_end = _scan_header_block(canonical_pgn)
    movetext_section = canonical_pgn[header_end:]
    movetext_section = _normalize_movetext_spacing(movetext_section)
    cleaned_movetext = _normalize_movetext_figurines(movetext_section)
    cleaned_movetext = _strip_comments_and_ravs(cleaned_movetext)
    cleaned_movetext = _fix_attached_move_numbers(cleaned_movetext)
    if _contains_e_p_marker(cleaned_movetext):
        syntax_warnings.append("En-passant marker 'e.p.' normalized to canonical SAN.")

    movetext_tokens = cleaned_movetext.split()
    tok_idx, expected_fullmove = _find_first_move_token(movetext_tokens, curr_board)

    for node in game.mainline():
        if reached_terminal:
            ignored_trailing_plies += 1
            continue

        move = node.move
        if move not in curr_board.legal_moves:
            ignored_trailing_plies += 1
            reached_terminal = True
            continue

        canonical_san = curr_board.san(move)
        tok_idx = _advance_past_move_number(
            movetext_tokens, tok_idx, expected_fullmove, curr_board, syntax_warnings
        )
        tok_idx = _record_san_normalization_warning(
            movetext_tokens, tok_idx, canonical_san, syntax_warnings
        )

        moves.append(move)
        curr_board.push(move)
        positions.append(curr_board.copy(stack=True))
        if curr_board.turn == chess.WHITE:
            expected_fullmove += 1

        reached_terminal, auto_termination = _detect_termination(
            curr_board, reached_terminal, auto_termination
        )

    return positions, moves, syntax_warnings, ignored_trailing_plies, cleaned_movetext


def _scan_header_block(canonical_pgn: str) -> int:
    """Find the index where the movetext section begins (after the
    contiguous run of ``[Tag "value"]`` lines)."""
    header_end = 0
    first_header = TAG_PAIR_REGEX.search(canonical_pgn)
    first_mv = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", canonical_pgn)
    if first_header and (not first_mv or first_header.start() < first_mv.start()):
        header_end = first_header.end()
        for m in TAG_PAIR_REGEX.finditer(canonical_pgn):
            if m.start() < header_end:
                continue
            if canonical_pgn[header_end : m.start()].strip() == "":
                header_end = m.end()
            else:
                break
    return header_end


def _normalize_movetext_spacing(movetext_section: str) -> str:
    """Insert a space between a move token and an attached NAG / result marker."""
    movetext_section = re.sub(
        r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)(\$\d+)",
        r"\1 \2",
        movetext_section,
    )
    movetext_section = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)(\$\d+)", r"\1\2 \3", movetext_section)
    return movetext_section


def _strip_comments_and_ravs(cleaned_movetext: str) -> str:
    """Remove ``{...}`` and ``;...`` comment syntax plus ``(...)`` RAV groups."""
    while "{" in cleaned_movetext and "}" in cleaned_movetext:
        prev = cleaned_movetext
        cleaned_movetext = re.sub(r"\{[^{}]*\}", " ", cleaned_movetext, flags=re.DOTALL)
        if cleaned_movetext == prev:
            break
    cleaned_movetext = re.sub(r";[^\r\n]*", " ", cleaned_movetext)
    while "(" in cleaned_movetext and ")" in cleaned_movetext:
        prev = cleaned_movetext
        cleaned_movetext = re.sub(r"\([^()]*\)", " ", cleaned_movetext, flags=re.DOTALL)
        if cleaned_movetext == prev:
            break
    return cleaned_movetext


def _fix_attached_move_numbers(cleaned_movetext: str) -> str:
    """Reattach move numbers that lost their dots after comment/RAV stripping."""
    return re.sub(
        r"(?<!\S)(\d+)\s+(\.+)(?=\s|$)",
        lambda m: f" {m.group(1)}{m.group(2)} ",
        cleaned_movetext,
    )


def _contains_e_p_marker(cleaned_movetext: str) -> bool:
    return bool(
        re.search(
            r"(?:^|\s)\(?e\.?p\.?\)?(?=\s|$)",
            cleaned_movetext,
            flags=re.IGNORECASE,
        )
    )


def _find_first_move_token(
    movetext_tokens: list[str],
    curr_board: chess.Board,
) -> tuple[int, int]:
    """Find the index of the first move token and the expected fullmove number."""
    tok_idx = 0
    expected_fullmove = curr_board.fullmove_number
    for i, tok in enumerate(movetext_tokens):
        clean_tok = tok.strip(".,;:!?")
        num_m = re.match(r"^(\d+)[\.\:]*$", clean_tok)
        if num_m:
            tok_idx = i
            break
        try:
            curr_board.parse_san(clean_tok)
            tok_idx = i
            break
        except Exception:
            continue
    return tok_idx, expected_fullmove


def _advance_past_move_number(
    movetext_tokens: list[str],
    tok_idx: int,
    expected_fullmove: int,
    curr_board: chess.Board,
    syntax_warnings: list[str],
) -> int:
    """Skip ``N.`` / ``N...`` markers and result tokens at the start of a ply."""
    while tok_idx < len(movetext_tokens):
        raw_tok = movetext_tokens[tok_idx]
        num_m = re.match(r"^(\d+)(\.+)$", raw_tok)
        if num_m:
            move_num = int(num_m.group(1))
            if move_num != expected_fullmove:
                syntax_warnings.append(
                    f"Move number mismatch: found '{movetext_tokens[tok_idx]}' but expected move {expected_fullmove}."
                )
            expected_dots = "..." if curr_board.turn == chess.BLACK else "."
            actual_dots = num_m.group(2) or ""
            if actual_dots != expected_dots:
                syntax_warnings.append(
                    f"Wrong side marker: found '{movetext_tokens[tok_idx]}' "
                    f"but expected '{expected_dots}' for the side to move."
                )
            tok_idx += 1
            continue
        if raw_tok in ("1-0", "0-1", "1/2-1/2", "*") or re.match(r"^\$[0-9]+$", raw_tok):
            tok_idx += 1
            continue
        break
    return tok_idx


def _record_san_normalization_warning(
    movetext_tokens: list[str],
    tok_idx: int,
    canonical_san: str,
    syntax_warnings: list[str],
) -> int:
    """Compare the input SAN at ``tok_idx`` against the canonical SAN and
    warn on normalization (e.g. ``Nf3`` → ``Nf3`` with disambiguation, or
    ``e.p.`` removal). Skips if the token looks like raw UCI."""
    if tok_idx >= len(movetext_tokens):
        return tok_idx
    raw_tok = movetext_tokens[tok_idx].strip(".,;:!?")
    raw_tok_san = raw_tok.rstrip("!?")
    raw_tok_promotionless = _strip_promotion_eq(raw_tok_san)
    canonical_promotionless = _strip_promotion_eq(canonical_san)
    if raw_tok_promotionless != canonical_promotionless and not re.fullmatch(
        r"[a-h][1-8][a-h][1-8][qrbn]?", raw_tok_san.lower()
    ):
        syntax_warnings.append(
            f"Input SAN '{movetext_tokens[tok_idx]}' normalized to '{canonical_san}'"
        )
    return tok_idx + 1


def _detect_termination(
    curr_board: chess.Board,
    reached_terminal: bool,
    auto_termination: str | None,
) -> tuple[bool, str | None]:
    """Return the updated ``(reached_terminal, auto_termination)`` after pushing
    a ply. Fivefold repetition, rule-status terminal, or stalemate-style draws
    all set ``reached_terminal=True``."""
    if curr_board.is_repetition(5):
        return True, "fivefold_repetition"
    rule_after = evaluate_rule_status(curr_board, history_complete="complete")
    if rule_after.terminal is not None:
        return True, rule_after.terminal
    return reached_terminal, auto_termination
