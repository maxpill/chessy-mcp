from pathlib import Path

# R-19: accepting the optional e.p. marker is intentional, but normalization
# must be visible to callers and to strict-mode validation instead of happening
# silently between the PGN reader and token validator.
server_path = Path("mcp_server/server.py")
server_text = server_path.read_text()
old_ep_hook = '''        cleaned_movetext = _normalize_movetext_figurines(movetext_section)\n        while "{" in cleaned_movetext and "}" in cleaned_movetext:\n'''
new_ep_hook = '''        cleaned_movetext = _normalize_movetext_figurines(movetext_section)\n        if re.search(\n            r"(?:^|\\s)\\(?e\\.?p\\.?\\)?(?=\\s|$)",\n            cleaned_movetext,\n            flags=re.IGNORECASE,\n        ):\n            syntax_warnings.append(\n                "En-passant marker 'e.p.' normalized to canonical SAN."\n            )\n        while "{" in cleaned_movetext and "}" in cleaned_movetext:\n'''
if old_ep_hook not in server_text:
    raise SystemExit("analyze_game movetext normalization hook not found")
server_path.write_text(server_text.replace(old_ep_hook, new_ep_hook, 1))

path = Path("tests/test_mcp_ultra_audit_fixtures_2026_08_28.py")
text = path.read_text()

replacements = [
(
'''@pytest.mark.asyncio
async def test_r_18_en_passant_legal_capture():
    """R-18: e4 a6 e5 d5 exd6 must be parsed as en passant."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = "1. e4 a6 2. e5 d5 3. exd6 *"
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 5
    assert "exd6" in [tp.san for tp in res.turning_points] or res.total_plies == 5
''',
'''@pytest.mark.asyncio
async def test_r_18_en_passant_legal_capture():
    """R-18: e4 a6 e5 d5 exd6 must execute a real en-passant capture."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = "1. e4 a6 2. e5 d5 3. exd6 *"
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 5

    final_board = server_module._build_board(pgn, [])
    assert final_board.piece_at(chess.D6) == chess.Piece(chess.PAWN, chess.WHITE)
    assert final_board.piece_at(chess.D5) is None
    assert final_board.move_stack[-1].uci() == "e5d6"
'''
),
(
'''@pytest.mark.asyncio
async def test_r_19_en_passant_e_p_notation():
    """R-19: 'exd6 e.p.' should parse with normalization warning."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = "1. e4 a6 2. e5 d5 3. exd6 e.p. *"
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 5
    # exd6 e.p. normalized to exd6 — should be in syntax_warnings
    assert any("normalized" in w.lower() for w in res.syntax_warnings) or res.total_plies == 5
''',
'''@pytest.mark.asyncio
async def test_r_19_en_passant_e_p_notation():
    """R-19: 'exd6 e.p.' must normalize and still execute en passant."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = "1. e4 a6 2. e5 d5 3. exd6 e.p. *"
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 5
    assert any("normalized" in w.lower() for w in res.syntax_warnings)

    final_board = server_module._build_board(pgn, [])
    assert final_board.piece_at(chess.D6) == chess.Piece(chess.PAWN, chess.WHITE)
    assert final_board.piece_at(chess.D5) is None
'''
),
(
'''@pytest.mark.asyncio
async def test_r_21_castle_aliases():
    """R-21: O-O and 0-0 must both parse as castling."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e1g1")  # type: ignore
    for san in ("O-O", "0-0"):
        pgn = f"1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. {san} {san.replace('O', 'O')} *"
        try:
            res = await server_module.analyze_game(pgn, depth=8)
            assert res.total_plies >= 4
        except Exception:
            pass
''',
'''@pytest.mark.asyncio
async def test_r_21_castle_aliases():
    """R-21: O-O and 0-0 must both parse and move king/rook correctly."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e1g1")  # type: ignore
    for san in ("O-O", "0-0"):
        # Both castles are legal: Black first clears g8 with ...Nf6 and f8 with ...Be7.
        pgn = f"1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. {san} Be7 5. Re1 {san} *"
        res = await server_module.analyze_game(pgn, depth=8)
        assert res.total_plies == 10

        final_board = server_module._build_board(pgn, [])
        assert final_board.piece_at(chess.G1) == chess.Piece(chess.KING, chess.WHITE)
        assert final_board.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)
        assert final_board.piece_at(chess.G8) == chess.Piece(chess.KING, chess.BLACK)
        assert final_board.piece_at(chess.F8) == chess.Piece(chess.ROOK, chess.BLACK)
'''
),
(
'''@pytest.mark.asyncio
async def test_r_44_black_ranking():
    """R-44: for black-to-move, candidates sorted from black's perspective."""
    server_module._analyzer_pool = _FlatPool(cp=20, best_move="e7e5")  # type: ignore
    res = await server_module.top_moves(
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
        n=3,
        depth=10,
    )
    # Top candidate is best for Black (cp positive from white POV = bad for black,
    # so first candidate should be the one with cp negative or mate against white)
    assert len(res.result) >= 1
''',
'''@pytest.mark.asyncio
async def test_r_44_black_ranking():
    """R-44: black-to-move candidates must be ranked by Black utility."""

    class BlackRankingPool(_FlatPool):
        async def top_moves(
            self, board: chess.Board, n: int = 3, depth: int = 14
        ) -> list[Eval]:
            candidates = [
                Eval(cp=50, best_move="e7e5", pv=["e7e5"], depth=depth),
                Eval(cp=-80, best_move="d7d5", pv=["d7d5"], depth=depth),
                Eval(cp=10, best_move="g8f6", pv=["g8f6"], depth=depth),
            ]
            return candidates[:n]

    server_module._analyzer_pool = BlackRankingPool(cp=0, best_move="d7d5")  # type: ignore
    res = await server_module.top_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 100 1",
        n=3,
        depth=10,
    )
    assert [c.best_move for c in res.result] == ["d7d5", "g8f6", "e7e5"]
    assert [c.cp for c in res.result] == [-80, 10, 50]
'''
),
]

for old, new in replacements:
    if old not in text:
        raise SystemExit(f"expected regression block not found: {old.splitlines()[1]}")
    text = text.replace(old, new, 1)

path.write_text(text)
