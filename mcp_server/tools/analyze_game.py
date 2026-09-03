"""``analyze_game`` MCP tool.

Extracted from ``mcp_server.server``. Full PGN game analysis with accuracy
%, ACPL, mistake counts, and turning points. Coordinates PGN parsing,
position evaluation, move scoring, and the :func:`_compute_game_metrics`
aggregator.
"""

from __future__ import annotations

import logging
import re
import time

import chess
import chess.pgn

from core.engines.openings import lookup_opening

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from mcp_server.engine import (
    _build_identity,
    _gather_evaluate_positions_bounded,
    _get_analyzer_pool,
)
from mcp_server.metrics import metrics
from mcp_server.models import GameAnalysisResult, MCPEval
from mcp_server.parsers import (
    TAG_PAIR_REGEX,
    _check_multiple_games,
    _extract_canonical_pgn_text,
    _extract_game_inner,
    _find_movetext_result,
    _infer_result_from_termination,
    _normalize_movetext_figurines,
    _sanitize_malformed_pgn_header_lines,
    _strip_pgn_escape_lines,
    _strip_promotion_eq,
    _truncate_movetext_at_result,
    _unescape_pgn_tag_value,
    _is_valid_pgn_time_control,
    _validate_pgn_date,
    _validate_strict_header_syntax,
    _validate_strict_mainline_surface,
)
from mcp_server.rules import (
    evaluate_rule_status,
    format_fen_status_errors,
    validate_mating_possibility,
)
from mcp_server.server import mcp
from mcp_server.tools._common import (
    _tool_error,
    _validate_requested_depth,
    error_code_for,
    normalize_termination,
)
from mcp_server.tools.game_metrics import _compute_game_metrics

log = logging.getLogger("chessy_mcp.analyze_game")


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def analyze_game(  # pyright: ignore[reportGeneralTypeIssues]
    pgn: str,
    depth: int = 18,
    strict: bool = False,
    ctx: Context | None = None,
) -> GameAnalysisResult:
    """Analyze a full game in PGN format with Stockfish, providing accuracy scores, mistake counts, and metadata.

    Supports standard PGN, annotated PGNs (with comments, NAGs, variations), conversational
    preamble/trailer text, markdown-wrapped PGNs, and bare move lists. Side variations in parentheses
    and comments are ignored for the mainline analysis. `white_acpl` / `black_acpl` report the effective
    ACPL across all plies (including 1000cp mate transitions and draw claim forfeitures), while
    `white_raw_acpl` / `black_raw_acpl` report unweighted raw CPL on non-mate plies.

    Args:
        pgn: PGN string, annotated game, or move text.
        depth: Search depth per move (default 18, clamped 1-30). ``analyze_game`` fans
            one Stockfish search per mainline ply — default 18 trims compute vs the
            previous d14 while staying accurate enough to separate inaccuracy/mistake
            classes. For "find the turning points" mode, d18 is enough. For precise
            post-mortems where borderline decisions matter, push to 20 or selectively
            re-classify borderline plies at d22-24. Avoid going above d24 except for
            3-7 critical positions: nodes scale roughly 5x from d20→d24 and ~10x to d30
            (Stockfish 18 figures), with sharply diminishing Elo per added depth.
        strict: When True, reject non-canonical SAN syntax, move number mismatches, or metadata discrepancies (default False).

    Returns:
         GameAnalysisResult with player accuracy %, ACPL, blunder/mistake counts, turning points, and game metadata.
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="analyze_game")
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    try:
        sanitized_pgn, lexical_header_warnings = _sanitize_malformed_pgn_header_lines(
            pgn, strict=strict
        )
        _check_multiple_games(sanitized_pgn)
        if strict:
            _validate_strict_header_syntax(sanitized_pgn)
        canonical_pgn = _extract_canonical_pgn_text(sanitized_pgn)
        game = _extract_game_inner(canonical_pgn, strict=strict)
        if strict:
            _validate_strict_mainline_surface(canonical_pgn, game)

        positions: list[chess.Board] = []
        moves: list[chess.Move] = []
        syntax_warnings: list[str] = []
        curr_board = game.board()
        if not curr_board.is_valid() or curr_board.status() != chess.STATUS_VALID:
            raise ValueError(
                f"INVALID_FEN: Initial position '{curr_board.fen()}' in PGN is not a valid chess position ({format_fen_status_errors(curr_board.status())})."
            )

        positions.append(curr_board.copy(stack=True))
        auto_termination: str | None = None
        reached_terminal = False
        ignored_trailing_plies = 0

        # U-03 (2026-09-01): if the initial FEN is already terminal (75-move
        # draw, checkmate, stalemate, insufficient material, fivefold
        # repetition, dead position), the movetext's first move is bogus —
        # the board has no legal moves. Strict mode raises a
        # STRICT_PGN_ERROR. Non-strict mode records a syntax_warning and
        # treats every following move as a trailing ply so the analysis
        # surfaces 0 executed plies. Without this check the mainline loop
        # silently advanced `ignored_trailing_plies` without ever telling
        # the caller that the starting position was terminal.
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

        # P3/INVESTIGATE (2026-09-02 ultra audit): NAG values outside the
        # PGN 0..255 range were silently dropped in lenient mode (the
        # `_validate_movetext_tokens` helper only flags them in strict
        # mode). Re-scan the movetext here so lenient callers also see
        # the warning. Strict mode still rejects via the helper's
        # `invalid_tokens` branch; this scan catches the lenient case
        # without regressing strict behavior. Comments and variations are
        # already stripped from `cleaned_movetext` below, so the regex
        # scans the same tokens the strict path consumes.
        for nag_match in re.finditer(r"\$([0-9]+)", canonical_pgn):
            nag_val = int(nag_match.group(1))
            if nag_val > 255:
                if strict:
                    # Strict mode: promote to a metadata_warning so the
                    # final pass at the bottom raises STRICT_PGN_ERROR
                    # (mirrors the existing NAG enforcement path).
                    syntax_warnings.append(
                        f"NAG value ${nag_val} outside the PGN-supported range 0..255."
                    )
                else:
                    syntax_warnings.append(
                        f"NAG value ${nag_val} outside the PGN-supported range 0..255."
                    )

        # Extract headers ONLY from contiguous header block at the start of canonical_pgn
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

        header_section = canonical_pgn[:header_end]
        movetext_section = canonical_pgn[header_end:]

        # Clean movetext for token scanning (strip comments and variations, translate figurines, split attached NAGs)
        movetext_section = re.sub(
            r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)(\$\d+)",
            r"\1 \2",
            movetext_section,
        )
        movetext_section = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)(\$\d+)", r"\1\2 \3", movetext_section)
        cleaned_movetext = _normalize_movetext_figurines(movetext_section)
        if re.search(
            r"(?:^|\s)\(?e\.?p\.?\)?(?=\s|$)",
            cleaned_movetext,
            flags=re.IGNORECASE,
        ):
            syntax_warnings.append("En-passant marker 'e.p.' normalized to canonical SAN.")
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

        # R4-§C (2026-09-02 ultra audit round 4): PGN §8.1 allows whitespace
        # around the move-number dot (`1 . e4` is equivalent to `1. e4`).
        # The downstream tokenizer expects a single `N.` / `N...` token and
        # emits spurious "Input SAN 'N' normalized" warnings otherwise. Collapse
        # digit-then-dots sequences across whitespace before splitting.
        cleaned_movetext = re.sub(
            r"(?<!\S)(\d+)\s+(\.+)(?=\s|$)",
            lambda m: f" {m.group(1)}{m.group(2)} ",
            cleaned_movetext,
        )

        movetext_tokens = cleaned_movetext.split()
        tok_idx = 0
        expected_fullmove = curr_board.fullmove_number

        # Skip leading non-chess tokens to align with the first actual move (PGN-01)
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

            # Advance token index through move number tokens or result tokens
            # NOTE: do NOT strip the trailing dots here — the U-15 side
            # marker check needs the original dot count. The previous
            # code stripped dots which made actual_dots empty for any
            # single-dot or triple-dot token, and the side-marker check
            # would then fire as a false positive.
            while tok_idx < len(movetext_tokens):
                raw_tok = movetext_tokens[tok_idx]
                # U-15 (2026-09-01): the previous pattern `(\.|\.\.)*` was
                # a Python regex footgun — alternation inside a `*` group
                # never extends beyond a single match, so group(2) was
                # always "." regardless of how many dots were in the
                # input. That made the wrong-side-marker check a no-op
                # (the actual and expected dots were always the same).
                # `\.+` captures the full dot run in one shot.
                num_m = re.match(r"^(\d+)(\.+)$", raw_tok)
                if num_m:
                    move_num = int(num_m.group(1))
                    if move_num != expected_fullmove:
                        syntax_warnings.append(
                            f"Move number mismatch: found '{movetext_tokens[tok_idx]}' but expected move {expected_fullmove}."
                        )
                    # U-15 (2026-09-01): also flag wrong-dot count. A black
                    # move (board.turn == BLACK at this point in the
                    # mainline) MUST use "..." (triple dot), not ".".
                    # Strict mode promotes the warning to a STRICT_PGN_ERROR;
                    # the final pass at the bottom raises on any
                    # syntax_warnings in strict mode.
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

            if tok_idx < len(movetext_tokens):
                raw_tok = movetext_tokens[tok_idx].strip(".,;:!?")
                raw_tok_san = raw_tok.rstrip("!?")
                # Round-3 (further super deep): the old check compared the
                # raw SAN to canonical_san verbatim. That rejected the valid
                # PGN §8.1.4 promotion form 'e8Q' (no '=') because
                # canonical_san is 'e8=Q'. Strip the optional '=' from BOTH
                # sides so 'e8=Q' and 'e8Q' both compare equal — same
                # comparison the strict surface validator uses.
                raw_tok_promotionless = _strip_promotion_eq(raw_tok_san)
                canonical_promotionless = _strip_promotion_eq(canonical_san)
                if raw_tok_promotionless != canonical_promotionless and not re.fullmatch(
                    r"[a-h][1-8][a-h][1-8][qrbn]?", raw_tok_san.lower()
                ):
                    syntax_warnings.append(
                        f"Input SAN '{movetext_tokens[tok_idx]}' normalized to '{canonical_san}'"
                    )
                tok_idx += 1

            moves.append(move)
            curr_board.push(move)
            positions.append(curr_board.copy(stack=True))
            if curr_board.turn == chess.WHITE:
                expected_fullmove += 1

            if curr_board.is_repetition(5):
                reached_terminal = True
                auto_termination = "fivefold_repetition"
            else:
                rule_after = evaluate_rule_status(curr_board, history_complete="complete")
                if rule_after.terminal is not None:
                    reached_terminal = True
                    auto_termination = rule_after.terminal

        # Extract headers with TAG_PAIR_REGEX from header_section to handle escaped quotes and robust tag parsing
        # P2 (2026-09-02 ultra audit): the tag name MUST be canonicalized
        # before storage in tags_dict — otherwise [Variant "Standard"] and
        # [variant "Standard"] produced different downstream lookups (Variant
        # returned None from the second form). The metadata pipeline now
        # uses lowercase keys consistently. Downstream code reads both the
        # canonical-key form (lowercase, e.g. "variant") and falls back to
        # python-chess's game.headers for whatever it parsed.
        tags_dict: dict[str, str] = {}
        for tag_m in TAG_PAIR_REGEX.finditer(header_section):
            tag_k = tag_m.group(1).lower()
            tag_v = _unescape_pgn_tag_value(tag_m.group(2))
            if tag_k not in tags_dict and tag_v is not None and tag_v != "?":
                tags_dict[tag_k] = tag_v

        h = game.headers
        white_name = tags_dict.get("white") or (
            _unescape_pgn_tag_value(h.get("White"))
            if h.get("White") and h.get("White") != "?"
            else None
        )
        black_name = tags_dict.get("black") or (
            _unescape_pgn_tag_value(h.get("Black"))
            if h.get("Black") and h.get("Black") != "?"
            else None
        )
        event_name = tags_dict.get("event") or (
            _unescape_pgn_tag_value(h.get("Event"))
            if h.get("Event") and h.get("Event") != "?"
            else None
        )
        site_name = tags_dict.get("site") or (
            _unescape_pgn_tag_value(h.get("Site"))
            if h.get("Site") and h.get("Site") != "?"
            else None
        )
        round_name = tags_dict.get("round") or (
            _unescape_pgn_tag_value(h.get("Round"))
            if h.get("Round") and h.get("Round") != "?"
            else None
        )
        white_elo_val = tags_dict.get("whiteelo") or (
            h.get("WhiteElo") if h.get("WhiteElo") and h.get("WhiteElo") != "?" else None
        )
        black_elo_val = tags_dict.get("blackelo") or (
            h.get("BlackElo") if h.get("BlackElo") and h.get("BlackElo") != "?" else None
        )
        time_control_val = tags_dict.get("timecontrol") or (
            h.get("TimeControl") if h.get("TimeControl") and h.get("TimeControl") != "?" else None
        )
        # P2 (2026-09-02 ultra audit): TimeControl values like "? ",
        # " ?", " ? " were preserved verbatim on the wire — the strict
        # validator allowed them through, but downstream consumers saw
        # a different literal than the canonical sentinel. Strip
        # whitespace and normalize the unknown sentinel to None so the
        # exposed value is consistent across inputs.
        if time_control_val is not None:
            time_control_val = time_control_val.strip()
            if time_control_val == "?":
                time_control_val = None
        variant_val = tags_dict.get("variant") or (
            h.get("Variant") if h.get("Variant") and h.get("Variant") != "?" else None
        )
        date_val = (
            tags_dict.get("date") or tags_dict.get("utcdate") or h.get("Date") or h.get("UTCDate")
        )
        # Round-3 (further super deep): normalize empty / whitespace-only
        # Date tags to None — they all mean "no date" per PGN §7.1. The
        # audit found `[Date ""]` silently accepted (the truthy check above
        # fell through because Python treats "" as falsy) while `[Date " "]`
        # was rejected — inconsistent. Treat them identically.
        if date_val is not None and date_val.strip() in ("", "?", "????.??.??"):
            date_val = None

        result_movetext = _find_movetext_result(canonical_pgn)

        # Extract Result and Termination headers from header_section ONLY
        result_header_raw: str | None = None
        termination_header_val: str | None = None
        for tag_m in TAG_PAIR_REGEX.finditer(header_section):
            tag_k = tag_m.group(1).lower()
            tag_v = _unescape_pgn_tag_value(tag_m.group(2))
            if tag_k == "result" and result_header_raw is None:
                result_header_raw = tag_v
            elif tag_k == "termination" and termination_header_val is None:
                termination_header_val = tag_v

        metadata_warnings: list[str] = list(lexical_header_warnings)

        # R4-§B (2026-09-02 ultra audit round 4): the input was comment-only
        # (no moves after stripping comments + variations). Surface a clear
        # metadata warning in lenient mode so callers see the input was
        # non-empty but contained no moves. Strict mode does NOT raise on
        # this — a comment-only PGN with valid headers is not a metadata
        # inconsistency, so the warning would only confuse the strict
        # validator (which promotes every metadata_warning to a STRICT_PGN_
        # ERROR at the bottom of analyze_game).
        if getattr(game, "comment_only_input", False) and not strict:
            metadata_warnings.append(
                "Input PGN contained only comments (and optionally a result "
                "token) with no moves; returning an empty game."
            )

        # U-14 (2026-09-01): strict mode rejects malformed Date tags.
        # PGN Date is `YYYY.MM.DD`; anything else (e.g. "2026.99.99",
        # "hello", "not.a.date") is a non-canonical value. The legacy
        # behavior silently accepted these and even echoed them back
        # to clients, which is the audit's P2 finding. Strict mode
        # records a metadata_warning; the final strict pass at the
        # bottom of analyze_game raises STRICT_PGN_ERROR on any
        # metadata_warning, so the malformed Date is rejected. The
        # regex is tighter than just `\d{4}\.\d{2}\.\d{2}` — it
        # enforces month 01-12 and day 01-31 so "2026.99.99" is
        # correctly rejected.
        #
        # P2/P3 (2026-09-02 ultra audit): the regex above only catches
        # range issues; it still accepts impossible calendar dates like
        # 2023.02.29, 2026.04.31, 2026.02.31. After the structural
        # check, run the date through Python's `datetime.date`
        # constructor — that raises ValueError for any day that doesn't
        # exist in the given month/year, including the Feb 29 leap-year
        # rule (no Apr 31, no Sep 31, no Feb 30, etc.). In strict mode
        # the impossible date is a metadata_warning that promotes to a
        # STRICT_PGN_ERROR; in lenient mode it is also a warning so
        # downstream callers see that the metadata is suspect, even
        # though parsing continues.
        if date_val is not None:
            # Round-3 (further super deep): the old regex required ALL
            # three components to be concrete digits, which silently
            # rejected the per-component wildcards PGN §7.1 allows
            # (????.09.02, 2026.09.??, 2026.??.02, 2026.??.??). At the
            # same time it accepted ???? and '?' because of the truthy
            # fallback above, which was inconsistent. Validate each
            # component independently:
            #   - YYYY  |  ????   (year)
            #   - MM    |  ??     (month)
            #   - DD    |  ??     (day)
            # Calendar semantics only run when all three components are
            # concrete (Apr 31 / Sep 31 / Feb 29 in non-leap year / etc.).
            _date_err = _validate_pgn_date(date_val)
            if _date_err is not None:
                if strict:
                    metadata_warnings.append(_date_err)
                else:
                    metadata_warnings.append(_date_err)

        CANONICAL_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
        if result_header_raw is not None and result_header_raw != "?":
            if result_header_raw in CANONICAL_RESULTS:
                result_header = result_header_raw
            else:
                metadata_warnings.append(
                    f"Invalid Result header tag '{result_header_raw}'; expected 1-0, 0-1, 1/2-1/2, or *."
                )
                result_header = None
        else:
            result_header = None

        if white_elo_val is not None and white_elo_val != "-":
            if not (white_elo_val.isdigit() and 0 <= int(white_elo_val) <= 4000):
                metadata_warnings.append(
                    f"Invalid WhiteElo header tag '{white_elo_val}'; expected numeric integer rating."
                )
        if black_elo_val is not None and black_elo_val != "-":
            if not (black_elo_val.isdigit() and 0 <= int(black_elo_val) <= 4000):
                metadata_warnings.append(
                    f"Invalid BlackElo header tag '{black_elo_val}'; expected numeric integer rating."
                )
        if time_control_val is not None and not _is_valid_pgn_time_control(time_control_val):
            metadata_warnings.append(f"Invalid TimeControl header tag '{time_control_val}'.")

        eco_header = tags_dict.get("eco") or h.get("ECO")
        opening_header = tags_dict.get("opening") or h.get("Opening")

        # Detect duplicate headers in header block only. P2 (2026-09-02
        # ultra audit): the tag name MUST be canonicalized (lowercased)
        # before duplicate counting — otherwise `[Result "*"]` and
        # `[result "1-0"]` were treated as different tags despite
        # python-chess treating them as the same semantic tag. We also
        # surface value conflicts on the canonical Result tag because
        # the audit flagged that competing values were silently merged.
        tag_counts: dict[str, int] = {}
        tag_values_by_canonical: dict[str, list[str]] = {}
        for tag_m in TAG_PAIR_REGEX.finditer(header_section):
            tag_name_raw = tag_m.group(1)
            tag_value = _unescape_pgn_tag_value(tag_m.group(2))
            tag_name_canonical = tag_name_raw.lower()
            tag_counts[tag_name_canonical] = tag_counts.get(tag_name_canonical, 0) + 1
            if tag_value is not None:
                tag_values_by_canonical.setdefault(tag_name_canonical, []).append(tag_value)
        for tag_name, count in tag_counts.items():
            if count > 1:
                metadata_warnings.append(
                    f"Duplicate PGN tag '[{tag_name}]' detected ({count} occurrences); using canonical tag value."
                )
        # Surface value conflicts on Result / Variant explicitly so the
        # audit's "duplicate detection is not consistently
        # case-insensitive" finding is closed.
        for canonical_name in ("result", "variant"):
            values = tag_values_by_canonical.get(canonical_name) or []
            if len(values) >= 2 and any(v != values[0] for v in values[1:]):
                metadata_warnings.append(
                    f"Conflicting values for PGN tag '{canonical_name}': {values!r}; "
                    f"using the first declared value."
                )

        # Validate SetUp vs FEN tags
        setup_header = h.get("SetUp")
        fen_header = h.get("FEN")
        # P2 (2026-09-02 ultra audit): SetUp tag value domain must be
        # validated. The legacy code special-cased the canonical "1"
        # string and silently accepted every other value (including
        # non-canonical "2", empty string, "true", "false", "01", "-1",
        # and " ") — which the audit showed meant strict mode never
        # rejected malformed SetUp values. Strict mode now rejects any
        # value outside the canonical {"0", "1"} set; lenient mode
        # accepts "1" only (treating everything else as the implicit
        # "SetUp absent" case, with a warning).
        if setup_header is not None:
            if setup_header not in ("0", "1"):
                if strict:
                    metadata_warnings.append(
                        f"Invalid SetUp tag value '{setup_header}': must be exactly '0' or '1'."
                    )
                else:
                    # Lenient: warn but don't reject — preserve
                    # backward compatibility for slightly-malformed
                    # inputs that the caller may not be able to fix.
                    metadata_warnings.append(
                        f"Non-canonical SetUp tag value '{setup_header}': expected '0' or '1'."
                    )
        if setup_header == "1" and not fen_header:
            metadata_warnings.append(
                '[SetUp "1"] tag provided without FEN tag; defaulting to standard starting position.'
            )
        elif fen_header and setup_header != "1":
            metadata_warnings.append(
                'FEN tag provided without [SetUp "1"]; custom position loaded.'
            )

        if game.errors:
            # P2 (2026-09-02 ultra audit): board-detected checkmate path
            # undercounted trailing plies. The legacy code added
            # `len(game.errors)` which is the number of distinct
            # python-chess exceptions raised while parsing — usually 1
            # per movetext that breaks at the first illegal move — rather
            # than the actual number of trailing ply tokens the user
            # wrote. The explicit result-token branch already counted
            # all SAN tokens after the result marker; we now mirror that
            # behavior here, counting remaining movetext tokens after the
            # last successfully executed ply.
            consumed_plies = len(moves)
            tokens_in_movetext = re.findall(
                r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*|O-O-O[\+#\?!]*|O-O[\+#\?!]*)\b",
                cleaned_movetext,
            )
            total_ply_tokens = len(tokens_in_movetext)
            trailing_from_errors = max(0, total_ply_tokens - consumed_plies)
            if trailing_from_errors > 0:
                ignored_trailing_plies += trailing_from_errors
            else:
                # No recoverable move tokens found in the trailing
                # movetext. Fall back to the legacy game.errors
                # count so we never underreport below zero.
                ignored_trailing_plies = max(ignored_trailing_plies, len(game.errors))

        raw_pgn_clean = _strip_pgn_escape_lines(canonical_pgn)
        raw_truncated = _truncate_movetext_at_result(raw_pgn_clean)
        if len(raw_truncated) < len(raw_pgn_clean):
            after_part = raw_pgn_clean[len(raw_truncated) :]
            after_clean = re.sub(r"\{[^{}]*\}", " ", after_part)
            after_clean = re.sub(r";[^\r\n]*", " ", after_clean)
            tokens_after = re.findall(
                r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*|O-O-O[\+#\?!]*|O-O[\+#\?!]*)\b",
                after_clean,
            )
            if tokens_after:
                ignored_trailing_plies += len(tokens_after)

        if ignored_trailing_plies > 0:
            ply_word = "ply" if ignored_trailing_plies == 1 else "plies"
            metadata_warnings.append(
                f"Movetext contained moves after game termination; ignored {ignored_trailing_plies} trailing {ply_word}."
            )

        # Validate game result consistency against final board state
        final_board = positions[-1]
        # positions[] was reconstructed from the complete PGN mainline, so repetition
        # history is authoritative here. Do not downgrade a previously detected fivefold
        # repetition to generic game_over during final result reconciliation.
        rule_final = evaluate_rule_status(final_board, history_complete="complete")
        result_board: str | None = None
        if rule_final.terminal is not None:
            if rule_final.terminal == "checkmate":
                result_board = "1-0" if final_board.turn == chess.BLACK else "0-1"
                auto_termination = "checkmate"
            else:
                result_board = "1/2-1/2"
                auto_termination = rule_final.terminal

        if result_board is not None:
            result_val = result_board
            if result_header and result_header not in ("*", "?") and result_header != result_board:
                metadata_warnings.append(
                    f"Result header '{result_header}' disagrees with board outcome '{result_board}'; using board outcome."
                )
            if (
                result_movetext
                and result_movetext not in ("*", "?")
                and result_movetext != result_board
            ):
                metadata_warnings.append(
                    f"Movetext result '{result_movetext}' disagrees with board outcome '{result_board}'; using board outcome."
                )
        else:
            if result_header_raw and result_movetext and result_header_raw != result_movetext:
                metadata_warnings.append(
                    f"Result header '{result_header_raw}' disagrees with movetext result '{result_movetext}'."
                )

            if result_header and result_header not in ("*", "?"):
                result_val = result_header
            elif result_movetext and result_movetext not in ("*", "?"):
                result_val = result_movetext
            else:
                result_val = result_header or result_movetext or "*"

        # Infer only from explicit winner/loser grammar.
        if result_val == "*" or result_val is None:
            inferred = _infer_result_from_termination(termination_header_val)
            if inferred is not None:
                result_val = inferred

        # Validate Resignation & Time Forfeit & Rules Infraction under FIDE mating possibility rules
        result_val, mate_warnings = validate_mating_possibility(
            final_board, result_val, termination_header_val
        )
        metadata_warnings.extend(mate_warnings)

        # U-14 (2026-09-01): strict-mode Termination validation. The
        # legacy code only flagged Termination when it contradicted
        # the result; arbitrary strings like "foobar" were stored raw
        # without rejection. Strict mode now requires the Termination
        # to either be blank, a known FIDE value, or fall through the
        # normalize_termination mapper; anything else is a metadata
        # warning that strict pass will reject.
        if strict and termination_header_val:
            norm_term = normalize_termination(termination_header_val)
            # If normalize_termination returns None AND the string is
            # not blank/known, it's an unrecognised value.
            if norm_term is None and termination_header_val.strip() not in (
                "",
                "Normal",
                "Time forfeit",
                "Rules infraction",
                "Abandoned",
                "Unterminated",
            ):
                # If normalize_termination returned None but
                # contains a known FIDE term in lowercase, accept.
                lower = termination_header_val.strip().lower()
                if not any(
                    kw in lower
                    for kw in (
                        "resign",
                        "checkmate",
                        "stalemate",
                        "time",
                        "abandon",
                        "rule",
                        "draw",
                        "repetition",
                        "insufficient",
                        "50-move",
                        "75-move",
                    )
                ):
                    metadata_warnings.append(
                        f"Unrecognised Termination tag '{termination_header_val}'."
                    )

        # Check for contradictory metadata
        if termination_header_val:
            norm_term = normalize_termination(termination_header_val)
            if norm_term in (
                "stalemate",
                "insufficient_material",
                "fifty_moves",
                "seventyfive_moves",
                "threefold_repetition",
                "fivefold_repetition",
                "dead_position",
            ) and result_val in ("1-0", "0-1"):
                metadata_warnings.append(
                    f"Contradictory PGN metadata: Termination '{termination_header_val}' contradicts Result '{result_val}'."
                )
            elif norm_term == "checkmate" and result_val in ("1/2-1/2", "*"):
                metadata_warnings.append(
                    f"Contradictory PGN metadata: Termination '{termination_header_val}' contradicts Result '{result_val}'."
                )
            elif norm_term == "unterminated" and result_val in (
                "1-0",
                "0-1",
                "1/2-1/2",
            ):
                metadata_warnings.append(
                    f"Contradictory PGN metadata: Termination '{termination_header_val}' contradicts Result '{result_val}'."
                )

        # Premature draw agreement warning
        if (
            termination_header_val
            and "agreement" in termination_header_val.lower()
            and len(moves) < 2
        ):
            metadata_warnings.append(
                "Draw agreement declared before both players completed at least one move."
            )

        if auto_termination is not None:
            termination_val = auto_termination
            if termination_header_val:
                norm_term_hdr = normalize_termination(termination_header_val)
                if norm_term_hdr == "normal":
                    pass
                else:
                    is_concurrent_draw = norm_term_hdr in (
                        "stalemate",
                        "seventyfive_moves",
                        "fivefold_repetition",
                        "insufficient_material",
                        "fifty_moves",
                        "threefold_repetition",
                        "dead_position",
                    ) and (
                        (norm_term_hdr == "threefold_repetition" and final_board.is_repetition(3))
                        or (
                            norm_term_hdr == "fifty_moves"
                            and (final_board.is_fifty_moves() or final_board.halfmove_clock >= 100)
                        )
                        or (
                            norm_term_hdr == "fivefold_repetition"
                            and final_board.is_fivefold_repetition()
                        )
                        or (
                            norm_term_hdr == "seventyfive_moves"
                            and final_board.is_seventyfive_moves()
                        )
                        or (
                            norm_term_hdr == "insufficient_material"
                            and final_board.is_insufficient_material()
                        )
                        or (
                            norm_term_hdr == "dead_position"
                            and is_locked_dead_position(final_board)
                        )
                        or (norm_term_hdr == "stalemate" and final_board.is_stalemate())
                    )
                    if norm_term_hdr != auto_termination and not is_concurrent_draw:
                        metadata_warnings.append(
                            f"Termination header '{termination_header_val}' disagrees with board outcome '{auto_termination}'; using board outcome."
                        )
        elif termination_header_val:
            norm_term_hdr = normalize_termination(termination_header_val)
            if norm_term_hdr == "normal":
                termination_val = "normal"
            elif norm_term_hdr in ("checkmate", "stalemate"):
                metadata_warnings.append(
                    f"Termination header '{termination_header_val}' contradicts board state (position is not {norm_term_hdr})."
                )
                termination_val = None
            elif norm_term_hdr == "threefold_repetition":
                if not final_board.is_repetition(3):
                    metadata_warnings.append(
                        f"Termination header '{termination_header_val}' contradicts board state (position is not threefold_repetition)."
                    )
                    termination_val = None
                else:
                    termination_val = "threefold_repetition"
            elif norm_term_hdr == "fifty_moves":
                if not final_board.is_fifty_moves() and final_board.halfmove_clock < 100:
                    metadata_warnings.append(
                        f"Termination header '{termination_header_val}' contradicts board state (position is not fifty_moves)."
                    )
                    termination_val = None
                else:
                    termination_val = "fifty_moves"
            elif norm_term_hdr in (
                "insufficient_material",
                "seventyfive_moves",
                "fivefold_repetition",
                "dead_position",
            ):
                metadata_warnings.append(
                    f"Termination header '{termination_header_val}' contradicts board state (position is not {norm_term_hdr})."
                )
                termination_val = None
            else:
                termination_val = norm_term_hdr
        else:
            termination_val = None

        if strict and not moves:
            if syntax_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains syntax normalization or move number mismatch: {syntax_warnings[0]}"
                )
            if metadata_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains metadata inconsistency: {metadata_warnings[0]}"
                )

        is_standard_start = game.board().fen() == chess.STARTING_FEN

        pool = await _get_analyzer_pool(ctx)
        engine_name_str = getattr(pool, "engine_version", getattr(pool, "name", "Stockfish"))

        if not moves:
            detected_opening, detected_eco = (
                lookup_opening([])[:2] if is_standard_start else (None, None)
            )
            return GameAnalysisResult(
                total_plies=0,
                white_accuracy=None,
                black_accuracy=None,
                white_acpl=None,
                black_acpl=None,
                white_raw_acpl=None,
                black_raw_acpl=None,
                white_effective_acpl=None,
                black_effective_acpl=None,
                white_average_effective_loss=None,
                black_average_effective_loss=None,
                white_blunders=0,
                white_mistakes=0,
                white_inaccuracies=0,
                black_blunders=0,
                black_mistakes=0,
                black_inaccuracies=0,
                turning_points=[],
                white=white_name,
                black=black_name,
                event=event_name,
                site=site_name,
                date=date_val,
                round=round_name,
                result=result_val or result_header or "*",
                result_header=result_header,
                result_header_raw=result_header_raw,
                result_movetext=result_movetext,
                result_inferred=result_board,
                white_elo=white_elo_val,
                black_elo=black_elo_val,
                time_control=time_control_val,
                variant=variant_val,
                eco=detected_eco or eco_header,
                opening=detected_opening or opening_header,
                opening_header=opening_header,
                eco_header=eco_header,
                metadata_warnings=metadata_warnings,
                syntax_warnings=syntax_warnings,
                termination=termination_val,
                termination_header=termination_header_val,
                requested_depth=raw_requested_depth,
                searched_depth=0,
                engine="Stockfish",
                engine_version=engine_name_str,
                **_build_identity(pool),
                accuracy_method="win_probability_logistic",
                mate_penalty_policy="1000_cp_mate_transition",
            )

        eval_pairs = await _gather_evaluate_positions_bounded(
            positions,
            depth,
            pool,
            requested_depth=raw_requested_depth,
            history_complete="complete",
        )
        evals: list[MCPEval] = [ep[0] for ep in eval_pairs]
        all_cached = all(ep[1] for ep in eval_pairs)

        (
            white_acc,
            black_acc,
            white_acpl,
            black_acpl,
            white_raw_acpl,
            black_raw_acpl,
            white_avg_eff,
            black_avg_eff,
            (white_blunders, white_mistakes, white_inaccuracies),
            (black_blunders, black_mistakes, black_inaccuracies),
            top_turning_points,
        ) = _compute_game_metrics(positions, moves, evals)

        await metrics.record("analyze_game", (time.time() - t0) * 1000, cache_hit=all_cached)

        uci_moves = [m.uci() for m in moves]
        if is_standard_start:
            detected_opening, detected_eco, _ = lookup_opening(uci_moves)
        else:
            detected_opening, detected_eco = None, None

        final_opening = detected_opening or opening_header
        final_eco = detected_eco or eco_header

        if detected_opening and opening_header:
            det_clean = detected_opening.strip().lower()
            hdr_clean = opening_header.strip().lower()
            det_base = det_clean.split(":")[0].strip()
            hdr_base = hdr_clean.split(":")[0].strip()
            is_parent_child = (
                det_clean.startswith(hdr_clean)
                or hdr_clean.startswith(det_clean)
                or det_base == hdr_base
            )
            if not is_parent_child:
                metadata_warnings.append(
                    f"Opening header '{opening_header}' disagrees with detected opening '{detected_opening}'"
                )
        if (
            detected_eco
            and eco_header
            and detected_eco.strip().upper() != eco_header.strip().upper()
        ):
            metadata_warnings.append(
                f"ECO header '{eco_header}' disagrees with detected ECO '{detected_eco}'"
            )

        if strict:
            if syntax_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains syntax normalization or move number mismatch: {syntax_warnings[0]}"
                )
            if metadata_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains metadata inconsistency: {metadata_warnings[0]}"
                )

        return GameAnalysisResult(
            total_plies=len(moves),
            white_accuracy=white_acc,
            black_accuracy=black_acc,
            white_acpl=white_acpl,
            black_acpl=black_acpl,
            white_raw_acpl=white_raw_acpl,
            black_raw_acpl=black_raw_acpl,
            white_effective_acpl=white_avg_eff,
            black_effective_acpl=black_avg_eff,
            white_average_effective_loss=white_avg_eff,
            black_average_effective_loss=black_avg_eff,
            white_blunders=white_blunders,
            white_mistakes=white_mistakes,
            white_inaccuracies=white_inaccuracies,
            black_blunders=black_blunders,
            black_mistakes=black_mistakes,
            black_inaccuracies=black_inaccuracies,
            turning_points=top_turning_points,
            white=white_name,
            black=black_name,
            event=event_name,
            site=site_name,
            date=date_val,
            round=round_name,
            result=result_val,
            result_header=result_header,
            result_header_raw=result_header_raw,
            result_movetext=result_movetext,
            result_inferred=result_board
            or (
                result_val
                if (result_header_raw in ("*", None) and result_val in ("1-0", "0-1", "1/2-1/2"))
                else None
            ),
            white_elo=white_elo_val,
            black_elo=black_elo_val,
            time_control=time_control_val,
            variant=variant_val,
            eco=final_eco,
            opening=final_opening,
            opening_header=opening_header,
            eco_header=eco_header,
            metadata_warnings=metadata_warnings,
            syntax_warnings=syntax_warnings,
            termination=termination_val,
            termination_header=termination_header_val,
            requested_depth=raw_requested_depth,
            searched_depth=depth,
            engine="Stockfish",
            engine_version=engine_name_str,
            **_build_identity(pool),
            accuracy_method="win_probability_logistic",
            mate_penalty_policy="1000_cp_mate_transition",
        )
    except ToolError:
        await metrics.record("analyze_game", (time.time() - t0) * 1000, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record("analyze_game", (time.time() - t0) * 1000, is_error=True)
        msg = str(exc)
        code = error_code_for(msg)
        raise _tool_error(code=code, message=msg, tool="analyze_game", input=pgn[:100]) from exc
    except Exception as exc:
        await metrics.record("analyze_game", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="analyze_game") from exc


# Middleware re-exports — implementation lives in mcp_server.middleware.
