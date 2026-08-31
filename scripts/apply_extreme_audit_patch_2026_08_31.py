from __future__ import annotations

from pathlib import Path

SERVER = Path("mcp_server/server.py")
MODELS = Path("mcp_server/models.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


server = SERVER.read_text(encoding="utf-8")
models = MODELS.read_text(encoding="utf-8")

if "def _validate_strict_header_syntax(" in server:
    print("extreme audit patch already applied")
    raise SystemExit(0)

# Strict PGN surface validators. They intentionally validate only top-level
# movetext; comments and RAVs are permitted and ignored for mainline analysis.
anchor = '''def _is_canonical_tag_line(line: str) -> bool:
    stripped = line.strip()
    if (
        not stripped
        or stripped.startswith(";")
        or stripped.startswith("%")
        or stripped.startswith("{")
    ):
        return False
    return bool(re.match(r'^(?:\\[\\s*[A-Za-z0-9_]+\\s+"(?:[^"\\\\]|\\\\.)*"\\s*\\]\\s*)+$', stripped))


'''
helpers = anchor + r'''def _validate_strict_header_syntax(text: str) -> None:
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


def _strict_top_level_movetext_tokens(text: str) -> list[str]:
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
    return top_level.split()


def _validate_strict_mainline_surface(text: str, game: chess.pgn.Game) -> None:
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
            raise ValueError(
                f"STRICT_PGN_ERROR: Unexpected trailing movetext token {clean!r}."
            )

        move = moves[move_index]
        canonical = board.san(move)
        supplied_san = clean.rstrip("!?")
        if supplied_san != canonical:
            raise ValueError(
                "STRICT_PGN_ERROR: Non-canonical SAN: "
                f"found {clean!r}, expected {canonical!r}."
            )
        board.push(move)
        move_index += 1

    if move_index != len(moves):
        raise ValueError(
            "STRICT_PGN_ERROR: Strict movetext validation did not consume the complete mainline."
        )


'''
server = replace_once(server, anchor, helpers, "strict helper insertion")

server = replace_once(
    server,
    '''def _extract_game(text: str, strict: bool = False) -> chess.pgn.Game:
    """Extract a chess.pgn.Game from raw, dirty, annotated, or conversational text."""
    _check_multiple_games(text)
    canonical = _extract_canonical_pgn_text(text)
    return _extract_game_inner(canonical, strict=strict)
''',
    '''def _extract_game(text: str, strict: bool = False) -> chess.pgn.Game:
    """Extract a chess.pgn.Game from raw, dirty, annotated, or conversational text."""
    _check_multiple_games(text)
    if strict:
        _validate_strict_header_syntax(text)
    canonical = _extract_canonical_pgn_text(text)
    game = _extract_game_inner(canonical, strict=strict)
    if strict:
        _validate_strict_mainline_surface(canonical, game)
    return game
''',
    "extract game strict propagation",
)

server = replace_once(
    server,
    "        game = _extract_game(cleaned)\n",
    "        game = _extract_game(cleaned, strict=strict)\n",
    "build board strict propagation",
)

old_metadata = '''    board = _build_board(fen_or_pgn, moves, strict)
    canonical = board.fen()
    was_canonicalized = bool(input_fen) and input_fen != canonical
    return board, input_fen, canonical, was_canonicalized
'''
new_metadata = '''    # Canonicalization is a property of the supplied FEN itself, not of any
    # suffix moves replayed after that FEN. Compare the input against a board
    # parsed before replaying the suffix, then return the final board FEN.
    canonical_input_fen: str | None = None
    if input_fen is not None:
        canonical_input_fen = _build_board(fen_or_pgn, [], strict).fen()

    board = _build_board(fen_or_pgn, moves, strict)
    canonical = board.fen()
    was_canonicalized = bool(input_fen) and input_fen != canonical_input_fen
    return board, input_fen, canonical, was_canonicalized
'''
server = replace_once(server, old_metadata, new_metadata, "fen metadata semantics")

old_eval_start = '''    t0 = time.time()
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    verbosity_mode = _resolve_verbosity(verbosity)
    try:
        board = _build_board(fen, moves or [], strict=strict)
        pool = await _get_analyzer_pool(ctx)
'''
new_eval_start = '''    t0 = time.time()
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    try:
        verbosity_mode = _resolve_verbosity(verbosity)
        board, input_fen, canonical_fen, fen_was_canonicalized = _build_board_with_metadata(
            fen, moves or [], strict=strict
        )
        pool = await _get_analyzer_pool(ctx)
'''
server = replace_once(server, old_eval_start, new_eval_start, "evaluate setup")

old_eval_metadata = '''        # L-06: surface input vs canonical FEN so callers can detect when
        # python-chess silently rewrote the EP target or other fields.
        canonical_fen = board.fen()
        cleaned_input = (
            fen.replace("\\u00a0", " ")
            .replace("\\u200b", "")
            .replace("\\ufeff", "")
            .strip("`'\\\" \\t\\r\\n")
        )
        is_fen_input = (
            "/" in cleaned_input
            and 1 <= len(cleaned_input.split()) <= 6
            and not cleaned_input.startswith("[")
            and not cleaned_input.lower().startswith("startpos")
        )
        result = res.model_copy(
            update={
                "requested_depth": raw_requested_depth,
                "input_fen": cleaned_input if is_fen_input else None,
                "canonical_fen": canonical_fen,
                "fen_was_canonicalized": is_fen_input and cleaned_input != canonical_fen,
            }
        )
'''
new_eval_metadata = '''        # L-06: surface input vs canonical FEN. Canonicalization describes
        # parser normalization of the supplied FEN only; replayed suffix moves
        # are reflected in canonical_fen but do not make the input noncanonical.
        result = res.model_copy(
            update={
                "requested_depth": raw_requested_depth,
                "input_fen": input_fen,
                "canonical_fen": canonical_fen,
                "fen_was_canonicalized": fen_was_canonicalized,
            }
        )
'''
server = replace_once(server, old_eval_metadata, new_eval_metadata, "evaluate fen metadata")

# Both evaluate_position and top_moves need verbosity errors inside their try blocks.
server = server.replace(
    '''        code = "invalid_input"
        if "STRICT" in msg:
''',
    '''        code = "invalid_input"
        if "INVALID_VERBOSITY" in msg:
            code = "invalid_verbosity"
        elif "STRICT" in msg:
''',
    2,
)
if server.count('code = "invalid_verbosity"') != 2:
    raise RuntimeError("expected invalid_verbosity mapping in evaluate_position and top_moves")

old_top_start = '''    t0 = time.time()
    raw_requested_depth = depth
    raw_requested_n = n
    depth = max(1, min(depth, 30))
    clamped_n = max(1, min(n, 20))
    n = clamped_n
    verbosity_mode = _resolve_verbosity(verbosity)
    try:
        board = _build_board(fen, moves or [], strict=strict)
        # evaluate_position with explicit moves has full history; naked FEN doesn't.
'''
new_top_start = '''    t0 = time.time()
    raw_requested_depth = depth
    raw_requested_n = n
    depth = max(1, min(depth, 30))
    clamped_n = max(1, min(n, 20))
    n = clamped_n
    try:
        verbosity_mode = _resolve_verbosity(verbosity)
        board, _input_fen, canonical_fen, fen_was_canonicalized = _build_board_with_metadata(
            fen, moves or [], strict=strict
        )
        # evaluate_position with explicit moves has full history; naked FEN doesn't.
'''
server = replace_once(server, old_top_start, new_top_start, "top moves setup")

# Add canonicalization flag to all three TopMovesResult construction paths.
old_top_canonical = '                canonical_fen=board.fen(),\n                engine="Stockfish",\n'
new_top_canonical = '''                canonical_fen=canonical_fen,
                fen_was_canonicalized=fen_was_canonicalized,
                engine="Stockfish",
'''
count = server.count(old_top_canonical)
if count != 2:
    raise RuntimeError(f"top terminal/cache canonical path: expected 2 matches, found {count}")
server = server.replace(old_top_canonical, new_top_canonical, 2)

old_top_final_canonical = '''            canonical_fen=board.fen(),
            engine="Stockfish",
'''
new_top_final_canonical = '''            canonical_fen=canonical_fen,
            fen_was_canonicalized=fen_was_canonicalized,
            engine="Stockfish",
'''
server = replace_once(server, old_top_final_canonical, new_top_final_canonical, "top fresh canonical path")

old_candidate = '''                mcp_eval = MCPEval.from_eval(
                    r,
                    b_cand.fen(),
                    board=b_cand,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                    pv_board=board,
                ).model_copy(
                    update={
                        "post_terminal_status": cand_post_terminal,
                        "candidate_san": cand_san_val,
                        "post_can_claim_draw": cand_can_claim_draw,
                        "post_can_claim_now": cand_can_claim_now,
                        "post_claim_reasons": cand_claim_reasons,
                        "post_claim_moves": cand_claim_moves,
                        "recommended_action": "game_over"
                        if cand_post_terminal is not None
                        else "play_move",
                        "post_position": {
                            "status": cand_post_terminal or "active",
                            "winner": cand_winner if cand_post_terminal == "checkmate" else None,
                            "can_claim_now": cand_can_claim_now,
                            "can_claim_draw": cand_can_claim_draw,
                            "claim_reasons": cand_claim_reasons_now or cand_claim_reasons,
                        },
                    }
                )
                res_list.append(mcp_eval)
'''
new_candidate = '''                identity = _build_identity(pool)
                mcp_eval = MCPEval.from_eval(
                    r,
                    b_cand.fen(),
                    board=b_cand,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                    pv_board=board,
                ).model_copy(
                    update={
                        "build_sha": identity["build_sha"],
                        "engine_config": identity["engine_config"],
                        "post_terminal_status": cand_post_terminal,
                        "candidate_san": cand_san_val,
                        "post_can_claim_draw": cand_can_claim_draw,
                        "post_can_claim_now": cand_can_claim_now,
                        "post_claim_reasons": cand_claim_reasons,
                        "post_claim_moves": cand_claim_moves,
                        "recommended_action": "game_over"
                        if cand_post_terminal is not None
                        else "play_move",
                        "post_position": {
                            "status": cand_post_terminal or "active",
                            "winner": cand_winner if cand_post_terminal == "checkmate" else None,
                            "can_claim_now": cand_can_claim_now,
                            "can_claim_draw": cand_can_claim_draw,
                            "claim_reasons": cand_claim_reasons_now or cand_claim_reasons,
                        },
                    }
                )
                res_list.append(mcp_eval)
'''
server = replace_once(server, old_candidate, new_candidate, "candidate identity")

old_fresh_items = '''        items = [c.model_copy(update={"requested_depth": raw_requested_depth}) for c in res[:n]]
        root_rec_action = _pick_root_recommended_action(items)
'''
new_fresh_items = '''        items = [c.model_copy(update={"requested_depth": raw_requested_depth}) for c in res[:n]]
        if verbosity_mode == VERBOSITY_COMPACT:
            items = [_compact_mcpeval(c) for c in items]
        root_rec_action = _pick_root_recommended_action(items)
'''
server = replace_once(server, old_fresh_items, new_fresh_items, "fresh compact path")

old_terminal_constructor = '''        return (
            MCPEval(
                status=rule_status.terminal,
                winner=rule_status.winner,
'''
new_terminal_constructor = '''        from mcp_server.actions import build_best_action, build_legal_actions

        terminal_best_action = build_best_action(
            recommended_action="game_over",
            rule_status=rule_status,
            engine_eval=None,
            board=b,
            sign=1 if b.turn == chess.WHITE else -1,
        )
        terminal_legal_actions = build_legal_actions(
            rule_status=rule_status,
            engine_eval=None,
            board=b,
            legal_engine_moves=None,
        )
        return (
            MCPEval(
                status=rule_status.terminal,
                winner=rule_status.winner,
'''
server = replace_once(server, old_terminal_constructor, new_terminal_constructor, "terminal typed action setup")

old_terminal_fields = '''                recommended_action="game_over",
                best_action="game_over",
                best_action_type="game_over",
                decision_value={
'''
new_terminal_fields = '''                recommended_action="game_over",
                best_action="game_over",
                best_action_type="game_over",
                best_action_obj=terminal_best_action,
                legal_actions=terminal_legal_actions,
                decision_value={
'''
server = replace_once(server, old_terminal_fields, new_terminal_fields, "terminal typed action fields")

old_analyze_parse = '''        _check_multiple_games(pgn)
        canonical_pgn = _extract_canonical_pgn_text(pgn)
        game = _extract_game_inner(canonical_pgn)
'''
new_analyze_parse = '''        _check_multiple_games(pgn)
        if strict:
            _validate_strict_header_syntax(pgn)
        canonical_pgn = _extract_canonical_pgn_text(pgn)
        game = _extract_game_inner(canonical_pgn, strict=strict)
        if strict:
            _validate_strict_mainline_surface(canonical_pgn, game)
'''
server = replace_once(server, old_analyze_parse, new_analyze_parse, "analyze strict parsing")

old_zero_ply = '''        is_standard_start = game.board().fen() == chess.STARTING_FEN

        pool = await _get_analyzer_pool(ctx)
'''
new_zero_ply = '''        if strict and not moves:
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
'''
server = replace_once(server, old_zero_ply, new_zero_ply, "zero ply strict validation")

old_top_doc = '''        IMPORTANT (audit C-02 / H-03):
          Each candidate in `result` represents a `play_move` action evaluated
          against its POST-CANDIDATE position. Draw-claim actions are reported
          separately at the outer level via `best_action_obj` and
          `legal_actions` — they are NOT mixed into candidate scores.
          Candidate `cp`/`mate` reflects the engine's evaluation of the
          position AFTER the move is played; the post-position status, winner,
          and Lichess URL refer to that post-state.
'''
new_top_doc = '''        IMPORTANT (audit C-02 / H-03):
          Each candidate in `result` represents a `play_move` action. Its
          `best_move`, `pv`, and engine `cp`/`mate` retain the root MultiPV
          action value and notation frame, so PV[0] is the candidate move and
          a mating candidate may retain Stockfish's root mate distance (e.g. 1).
          The candidate `canonical_fen`, terminal status, winner, rule fields,
          and `post_position` describe the board AFTER that candidate is played.
          Automatic terminal draws normalize candidate `cp` to 0. Draw-claim
          actions are reported separately via outer `best_action_obj` and
          `legal_actions`; they are not mixed into the MultiPV candidate list.
'''
server = replace_once(server, old_top_doc, new_top_doc, "top candidate contract docs")

# Update the nearby implementation comment too, so future changes do not regress
# back to the contradictory post-position score description.
server = replace_once(
    server,
    '''            # AUDIT C-02 / H-03: each candidate is a play_move action evaluated
            # AGAINST ITS POST-POSITION. The post-candidate terminal state,
            # winner, and Lichess URL describe the position after the move —
            # NOT a hypothetical claim outcome.
''',
    '''            # AUDIT C-02 / H-03: each candidate is a play_move action.
            # Root MultiPV score/mate/PV stay action-oriented; post-position
            # status, winner, rules and FEN describe the board after the move.
            # Claim actions remain separate from the candidate list.
''',
    "top candidate implementation docs",
)

old_model_desc = '''            "Ranked candidate moves (best first) with evaluation, best_move, pv, "
            "and depth. Empty for terminal positions. Each candidate represents "
            "a play_move action evaluated AGAINST ITS POST-POSITION — claim "
            "actions are reported separately at the outer level (best_action_obj "
            "/ legal_actions). See audit fix C-02/H-03."
'''
new_model_desc = '''            "Ranked play_move candidates (best first). Candidate best_move/pv and "
            "engine cp/mate retain the root MultiPV action frame (PV[0] is the "
            "candidate; mating moves may retain root mate distance), while "
            "canonical_fen, terminal/rule fields and post_position describe the "
            "resulting board. Claim actions are reported separately at the outer "
            "level (best_action_obj / legal_actions)."
'''
models = replace_once(models, old_model_desc, new_model_desc, "TopMovesResult model docs")

SERVER.write_text(server, encoding="utf-8")
MODELS.write_text(models, encoding="utf-8")
print("extreme audit patch applied")
