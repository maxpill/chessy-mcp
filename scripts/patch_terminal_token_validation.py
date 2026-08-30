from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one occurrence, got {count}")
    return text.replace(old, new, 1)


p = Path("mcp_server/server.py")
s = p.read_text(encoding="utf-8")

old = '''def _validate_movetext_tokens(
    movetext: str, start_board: chess.Board | None = None, strict: bool = False
) -> list[str]:
'''
new = '''def _validate_movetext_tokens(
    movetext: str,
    start_board: chess.Board | None = None,
    strict: bool = False,
    allow_trailing_after_terminal: bool = False,
) -> list[str]:
'''
s = replace_once(s, old, new, "validator signature")

old = '''    invalid_tokens: list[str] = []
    b = start_board.copy() if start_board else chess.Board()
    for _idx, tok in enumerate(tokens[first_move_idx:], start=first_move_idx):
        if b.is_game_over(claim_draw=False):
            break
        clean_tok = tok.rstrip(".,:;!?").lstrip(".,:;!?")
        clean_tok = re.sub(r"\\s*\\(?\\s*e\\.?p\\.?\\s*\\)?$", "", clean_tok, flags=re.IGNORECASE).rstrip(
            ".,:;!?"
        )
'''
new = '''    invalid_tokens: list[str] = []
    b = start_board.copy(stack=True) if start_board else chess.Board()
    for _idx, tok in enumerate(tokens[first_move_idx:], start=first_move_idx):
        clean_tok = tok.rstrip(".,:;!?").lstrip(".,:;!?")
        clean_tok = re.sub(r"\\s*\\(?\\s*e\\.?p\\.?\\s*\\)?$", "", clean_tok, flags=re.IGNORECASE).rstrip(
            ".,:;!?"
        )

        # A legal board move can still exist after automatic game termination
        # (75-move rule, fivefold repetition, insufficient/dead positions).
        # python-chess therefore may parse post-terminal movetext without a
        # parser error. Treat any actual move token after our central terminal
        # predicate as invalid unless analyze_game explicitly requested the
        # permissive warn-and-ignore contract.
        if is_terminal_position(b):
            if allow_trailing_after_terminal:
                break
            if re.match(r"^\\d+[\\.\\:]*$", tok):
                continue
            if clean_tok in ("1-0", "0-1", "1/2-1/2", "*"):
                break
            if re.match(r"^\\$[0-9]+$", clean_tok) or clean_tok in (
                "!",
                "?",
                "!!",
                "??",
                "!?",
                "?!",
                "e.p.",
                "e.p",
                "ep",
                "(e.p.)",
                "(e.p)",
                "(ep)",
            ):
                continue
            invalid_tokens.append(tok)
            break
'''
s = replace_once(s, old, new, "terminal token loop")

old = '''            invalid_tokens = _validate_movetext_tokens(
                text, start_board=game.board(), strict=strict
            )
'''
new = '''            invalid_tokens = _validate_movetext_tokens(
                text,
                start_board=game.board(),
                strict=strict,
                allow_trailing_after_terminal=allow_trailing_after_terminal,
            )
'''
s = replace_once(s, old, new, "validator call")

p.write_text(s, encoding="utf-8")


p = Path("tests/test_ultra_stress_v5.py")
s = p.read_text(encoding="utf-8")
marker = "test_residual_rejects_post_seventyfive_move_even_when_board_move_is_legal"
if marker not in s:
    s += r'''

@pytest.mark.asyncio
async def test_residual_rejects_post_seventyfive_move_even_when_board_move_is_legal():
    pgn = """[SetUp \"1\"]
[FEN \"7k/8/8/8/8/8/R7/K7 w - - 149 76\"]
[Result \"1/2-1/2\"]

76. Ra3 Kg7 1/2-1/2
"""
    with pytest.raises(ToolError, match="INVALID_PGN"):
        await server_module.evaluate_position(pgn, depth=1)
    permissive = await server_module.analyze_game(pgn, depth=1, strict=False)
    assert permissive.total_plies == 1
    assert any("after game termination" in w for w in permissive.metadata_warnings)
    with pytest.raises(ToolError, match="INVALID_PGN"):
        await server_module.analyze_game(pgn, depth=1, strict=True)


@pytest.mark.asyncio
async def test_residual_rejects_post_insufficient_material_promotion_move():
    pgn = """[SetUp \"1\"]
[FEN \"7k/P7/8/8/8/8/8/K7 w - - 0 1\"]
[Result \"1/2-1/2\"]

1. a8=B Kg7 1/2-1/2
"""
    with pytest.raises(ToolError, match="INVALID_PGN"):
        await server_module.top_moves(pgn, n=1, depth=1)
    permissive = await server_module.analyze_game(pgn, depth=1, strict=False)
    assert permissive.total_plies == 1
    assert any("after game termination" in w for w in permissive.metadata_warnings)


@pytest.mark.asyncio
async def test_residual_rejects_post_fivefold_move_that_is_still_board_legal():
    pgn = """[Result \"1/2-1/2\"]

1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8
5. Nf3 Nf6 6. Ng1 Ng8 7. Nf3 Nf6 8. Ng1 Ng8
9. Nf3 1/2-1/2
"""
    with pytest.raises(ToolError, match="INVALID_PGN"):
        await server_module.evaluate_position(pgn, depth=1)
    permissive = await server_module.analyze_game(pgn, depth=1, strict=False)
    assert permissive.total_plies == 16
    assert permissive.termination == "fivefold_repetition"
    assert any("after game termination" in w for w in permissive.metadata_warnings)
'''
    p.write_text(s, encoding="utf-8")
