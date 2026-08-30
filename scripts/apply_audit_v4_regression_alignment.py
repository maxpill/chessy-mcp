from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, got {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# Full PGN replay has complete history, so automatic fivefold must be detected
# as fivefold rather than falling through python-chess is_game_over() to a
# generic game_over status.
replace_once(
    Path("mcp_server/server.py"),
    "            rule_after = evaluate_rule_status(curr_board)\n",
    '            rule_after = evaluate_rule_status(curr_board, history_complete="complete")\n',
    "analyze_game fivefold provenance",
)

# Cache regression must follow the production semantic file set instead of
# freezing the old four-file list. The audit deliberately expanded this list.
replace_once(
    Path("tests/test_mcp_2026_08_28_post_fix_v2.py"),
    '    from mcp_server.cache import _LOGIC_HASH\n',
    '    from mcp_server.cache import _LOGIC_FILES, _LOGIC_HASH\n',
    "logic hash imports",
)
replace_once(
    Path("tests/test_mcp_2026_08_28_post_fix_v2.py"),
    '    for rel in ("mcp_server/cache.py", "mcp_server/rules.py", "mcp_server/models.py", "mcp_server/server.py"):\n',
    "    for rel in _LOGIC_FILES:\n",
    "logic hash file set",
)

# The standalone repository documents its live tool surface in README. Never
# depend on a developer-specific absolute path outside the checkout.
replace_once(
    Path("tests/test_mcp_server.py"),
    '    doc_path = Path("/Users/max/Desktop/projects/chessy/docs/chatgpt-setup.md")\n    assert doc_path.exists()\n    content = doc_path.read_text()\n',
    '    doc_path = Path(__file__).resolve().parent.parent / "README.md"\n    assert doc_path.exists()\n    content = doc_path.read_text(encoding="utf-8")\n',
    "portable docs path",
)

# Direct MCPEval construction cannot infer whether an arbitrary board stack is
# complete or partial. Tests that construct a game from startpos know the
# provenance and must pass it explicitly.
replace_once(
    Path("tests/test_mcp_server.py"),
    '    ev_rep = MCPEval.from_eval(Eval(cp=0, best_move="g1f3", pv=["g1f3"], depth=14), b_rep.fen(), board=b_rep)\n',
    '    ev_rep = MCPEval.from_eval(\n        Eval(cp=0, best_move="g1f3", pv=["g1f3"], depth=14),\n        b_rep.fen(),\n        board=b_rep,\n        history_complete="complete",\n    )\n',
    "explicit repetition provenance",
)

# FEN plus continuation is partial history by design. It can prove a repetition
# within the supplied suffix but cannot disprove a repetition before that FEN.
replace_once(
    Path("tests/test_mcp_ultra_audit_fixtures_2026_08_28.py"),
    '    assert res.history_completeness == "complete"\n    assert res.repetition_status == "threefold_claimable"\n',
    '    assert res.history_completeness == "partial"\n    assert res.repetition_status == "threefold_claimable"\n',
    "FEN suffix partial history",
)

# startpos is a known game root even with zero plies, so history is complete.
replace_once(
    Path("tests/test_mcp_ultra_audit_fixtures_2026_08_28.py"),
    'async def test_invariant_i08_naked_fen_repetition_unknown():\n    """I-08: naked FEN must have repetition_status=\'unknown\' (not \'none\')."""\n    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore\n    res = await server_module.evaluate_position("startpos", depth=10)\n    assert res.history_completeness == "incomplete"\n    assert res.repetition_status == "unknown"\n',
    'async def test_invariant_i08_startpos_zero_ply_history_complete():\n    """I-08: startpos is a known root, so zero-ply history is complete."""\n    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore\n    res = await server_module.evaluate_position("startpos", depth=10)\n    assert res.history_completeness == "complete"\n    assert res.repetition_status == "none"\n',
    "startpos provenance regression",
)

# Unknown verbosity was an audit defect: silently treating it as full hides
# caller mistakes. The corrected public contract rejects it.
replace_once(
    Path("tests/test_mcp_ultra_audit_fixtures_2026_08_28.py"),
    '    # Unknown falls back to full\n    res = await server_module.evaluate_position("startpos", depth=10, verbosity="unknown-mode")\n    assert res.lichess_url is not None\n',
    '    # Unknown values are rejected instead of silently changing semantics.\n    with pytest.raises(ValueError, match="INVALID_VERBOSITY"):\n        await server_module.evaluate_position("startpos", depth=10, verbosity="unknown-mode")\n',
    "unknown verbosity rejection",
)

print("audit v4 regression alignment applied")
