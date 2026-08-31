from pathlib import Path

legacy = Path("tests/test_mcp_ultra_audit_fixtures_2026_08_28.py")
s = legacy.read_text(encoding="utf-8")
if "from mcp.server.mcpserver.exceptions import ToolError" not in s:
    needle = "import chess\nimport pytest\n"
    if needle not in s:
        raise RuntimeError("legacy import anchor not found")
    s = s.replace(
        needle,
        "import chess\nimport pytest\nfrom mcp.server.mcpserver.exceptions import ToolError\n",
        1,
    )
old = '''    with pytest.raises(ValueError, match="INVALID_VERBOSITY"):
        await server_module.evaluate_position("startpos", depth=10, verbosity="unknown-mode")
'''
new = '''    with pytest.raises(ToolError, match="INVALID_VERBOSITY"):
        await server_module.evaluate_position("startpos", depth=10, verbosity="unknown-mode")
'''
if old in s:
    s = s.replace(old, new, 1)
elif new not in s:
    raise RuntimeError("legacy verbosity expectation anchor not found")
legacy.write_text(s, encoding="utf-8")

extreme = Path("tests/test_extreme_audit_2026_08_31.py")
e = extreme.read_text(encoding="utf-8")
marker = "test_strict_rejects_uci_mainline_across_primary_pgn_and_analyze_game"
if marker not in e:
    e += r'''

@pytest.mark.asyncio
async def test_strict_rejects_uci_mainline_across_primary_pgn_and_analyze_game() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    movetext = "1. e2e4 e7e5 *"
    with pytest.raises(Exception, match=r"\[STRICT_VALIDATION_ERROR\]"):
        await server_module.evaluate_position(movetext, depth=2, strict=True)
    with pytest.raises(Exception, match=r"\[STRICT_VALIDATION_ERROR\]"):
        await server_module.analyze_game(movetext, depth=2, strict=True)


@pytest.mark.asyncio
async def test_strict_rejects_zero_character_castling_notation() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. 0-0 Be7 *"
    with pytest.raises(Exception, match=r"\[STRICT_VALIDATION_ERROR\]"):
        await server_module.analyze_game(pgn, depth=2, strict=True)


@pytest.mark.asyncio
async def test_nonstrict_still_accepts_uci_and_zero_castling_compatibility_inputs() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    uci = await server_module.evaluate_position("1. e2e4 e7e5 *", depth=2, strict=False)
    assert uci.status == "active"
    castling = await server_module.analyze_game(
        "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. 0-0 Be7 *", depth=2, strict=False
    )
    assert castling.total_plies == 8
'''
extreme.write_text(e, encoding="utf-8")
print("legacy expectation and strict notation regressions patched")
