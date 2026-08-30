from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        if new in text:
            print(f"{label}: already applied")
            return
        raise SystemExit(f"{label}: expected block not found")
    if count != 1:
        raise SystemExit(f"{label}: expected one occurrence, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"{label}: applied")


# 1. Cache build identity must consume the BUILD_SHA injected by Docker/Compose.
cache = ROOT / "mcp_server/cache.py"
replace_once(
    cache,
    '''def _git_sha() -> str:\n    """Best-effort git HEAD sha for the working tree; 'unknown' if not a git repo."""\n    try:\n''',
    '''def _git_sha() -> str:\n    """Return the deployed build SHA, falling back to the local git HEAD."""\n    env_sha = os.environ.get("BUILD_SHA") or os.environ.get("CHESSY_BUILD_SHA")\n    if env_sha and env_sha.strip():\n        return env_sha.strip()\n    try:\n''',
    "cache BUILD_SHA",
)

# 2. If repetition cannot be disproved because history is incomplete/partial,
# the response must explicitly say that a move stack is required.
rules = ROOT / "mcp_server/rules.py"
replace_once(
    rules,
    "    requires_stack = repetition_proven\n",
    "    requires_stack = repetition_proven or repetition_status == \"unknown\"\n",
    "unresolved repetition metadata",
)

# 3. Winner-oriented time-forfeit wording is still a time forfeit and must
# participate in FIDE mating-possibility normalization.
replace_once(
    rules,
    '''            r"|\\blost\\s+on\\s+time\\b"\n            r"|\\bclock\\s+(?:flagged|expired)\\b",\n''',
    '''            r"|\\blost\\s+on\\s+time\\b"\n            r"|\\b(?:white|black)\\s+(?:wins?|won)\\s+on\\s+time\\b"\n            r"|\\bclock\\s+(?:flagged|expired)\\b",\n''',
    "mating validation winner-on-time grammar",
)

server = ROOT / "mcp_server/server.py"
replace_once(
    server,
    '''        r"|\\blost\\s+on\\s+time\\b"\n        r"|\\bclock\\s+(?:flagged|expired)\\b",\n''',
    '''        r"|\\blost\\s+on\\s+time\\b"\n        r"|\\b(?:white|black)\\s+(?:wins?|won)\\s+on\\s+time\\b"\n        r"|\\bclock\\s+(?:flagged|expired)\\b",\n''',
    "termination winner-on-time grammar",
)
replace_once(
    server,
    '''        (r"\\bwhite\\s+wins?\\b.*\\b(?:time|resignation|resigns?)\\b", "1-0"),\n        (r"\\bblack\\s+wins?\\b.*\\b(?:time|resignation|resigns?)\\b", "0-1"),\n''',
    '''        (r"\\bwhite\\s+(?:wins?|won)\\b.*\\b(?:time|resignation|resigns?)\\b", "1-0"),\n        (r"\\bblack\\s+(?:wins?|won)\\b.*\\b(?:time|resignation|resigns?)\\b", "0-1"),\n''',
    "result inference won/wins grammar",
)

# 4. Remove the unsound classify_move fast path. A root PV tail is not an
# evaluation of the immediate post-move position, and at finite depth the root
# score/mate distance is not a valid substitute either.
text = server.read_text(encoding="utf-8")
start_marker = "            # Fast path: when the played move is the engine's canonical best,"
end_marker = "            score = score_played_move("
if start_marker in text:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    replacement = '''            # Correctness first: eval_after must describe the immediate\n            # post-move position. Reusing the root PV tail or root score can\n            # misstate finite-depth CP and mate distance. Engine/cache layers\n            # remain responsible for performance reuse.\n            eval_after, _ = await _evaluate_game_position_cached(\n                board_after,\n                depth,\n                pool,\n                requested_depth=raw_requested_depth,\n                history_complete=history_complete,\n            )\n\n'''
    text = text[:start] + replacement + text[end:]
    server.write_text(text, encoding="utf-8")
    print("server classify immediate post-position: applied")
elif "Correctness first: eval_after must describe the immediate" in text:
    print("server classify immediate post-position: already applied")
else:
    raise SystemExit("server classify immediate post-position: markers not found")

# 5. The transport-independent classifier must also return a real post-move
# evaluation for the engine's best move, while preserving loss=0 by definition.
analysis = ROOT / "core/engines/analysis.py"
text = analysis.read_text(encoding="utf-8")
best_start = '''    if move_uci.lower() == (eval_before.best_move or "").lower():\n'''
terminal_marker = '''    if board_after.is_checkmate():\n'''
if best_start in text:
    start = text.index(best_start)
    end = text.index(terminal_marker, start)
    text = text[:start] + '''    is_engine_best = move_uci.lower() == (eval_before.best_move or "").lower()\n\n''' + text[end:]
    analysis.write_text(text, encoding="utf-8")
    print("core best-move synthetic eval block: removed")
elif "    is_engine_best = move_uci.lower() == (eval_before.best_move or \"\").lower()" in text:
    print("core best-move synthetic eval block: already removed")
else:
    raise SystemExit("core best-move synthetic eval block: marker not found")

text = analysis.read_text(encoding="utf-8")
replace_once(
    analysis,
    '''    eval_after = eval_after_lookup\n    played_pv = [move_uci] + (eval_after.pv or [])\n\n    if eval_before.mate is not None and eval_after.mate is not None:\n''',
    '''    eval_after = eval_after_lookup\n    played_pv = [move_uci] + (eval_after.pv or [])\n\n    if is_engine_best:\n        return eval_after, played_pv, 0\n\n    if eval_before.mate is not None and eval_after.mate is not None:\n''',
    "core best move keeps real eval_after with zero loss",
)

text = analysis.read_text(encoding="utf-8")
start_marker = '''    if (\n        move.uci().lower() == (eval_before.best_move or "").lower()\n'''
end_marker = '''\n    best_san: str | None = None\n'''
if start_marker in text:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    replacement = '''    if board_after.is_checkmate() or board_after.is_game_over(claim_draw=False):\n        eval_after, played_pv, loss = _played_eval_after(\n            move=move,\n            board_after=board_after,\n            eval_before=eval_before,\n            eval_after_lookup=None,\n            sign=sign,\n            actual_depth=actual_depth,\n        )\n    else:\n        post_eval = await backend.evaluate(board_after, depth=actual_depth)\n        eval_after, played_pv, loss = _played_eval_after(\n            move=move,\n            board_after=board_after,\n            eval_before=eval_before,\n            eval_after_lookup=post_eval,\n            sign=sign,\n            actual_depth=actual_depth,\n        )\n'''
    text = text[:start] + replacement + text[end:]
    analysis.write_text(text, encoding="utf-8")
    print("core classify always searches active post-position: applied")
elif "    if board_after.is_checkmate() or board_after.is_game_over(claim_draw=False):" in text:
    print("core classify always searches active post-position: already applied")
else:
    raise SystemExit("core classify active post-position block: marker not found")

# 6. Keep subprocess and TCP configuration feature-equivalent for WDL/Syzygy.
analyzer = ROOT / "core/engines/analyzer.py"
replace_once(
    analyzer,
    '''        depth: int = 12,\n        threads: int = 2,\n        hash_mb: int = 128,\n    ) -> Analyzer:\n        transport, engine = await chess.engine.popen_uci(path)\n        await engine.configure({"Threads": threads, "Hash": hash_mb})\n        return cls(transport, engine, depth)\n''',
    '''        depth: int = 12,\n        threads: int = 2,\n        hash_mb: int = 128,\n        show_wdl: bool = False,\n        syzygy_path: str | None = None,\n    ) -> Analyzer:\n        transport, engine = await chess.engine.popen_uci(path)\n        options: dict[str, int | str] = {"Threads": threads, "Hash": hash_mb}\n        if show_wdl:\n            options["UCI_ShowWDL"] = "true"\n        if syzygy_path:\n            options["SyzygyPath"] = syzygy_path\n            options["SyzygyProbeLimit"] = 7\n        await engine.configure(options)\n        return cls(transport, engine, depth)\n''',
    "local analyzer WDL/Syzygy config",
)

pool = ROOT / "core/engines/pool.py"
replace_once(
    pool,
    '''        threads: int = 1,\n        hash_mb: int = 128,\n        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT,\n    ) -> AnalyzerPool:\n        async def factory() -> object:\n            return await Analyzer.create(\n                path,\n                depth=depth,\n                threads=threads,\n                hash_mb=hash_mb,\n            )\n''',
    '''        threads: int = 1,\n        hash_mb: int = 128,\n        show_wdl: bool = False,\n        syzygy_path: str | None = None,\n        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT,\n    ) -> AnalyzerPool:\n        async def factory() -> object:\n            return await Analyzer.create(\n                path,\n                depth=depth,\n                threads=threads,\n                hash_mb=hash_mb,\n                show_wdl=show_wdl,\n                syzygy_path=syzygy_path,\n            )\n''',
    "AnalyzerPool forwards WDL/Syzygy",
)

replace_once(
    server,
    '''        depth=14,\n        threads=threads,\n        hash_mb=cfg.hash_mb,\n    )\n''',
    '''        depth=14,\n        threads=threads,\n        hash_mb=cfg.hash_mb,\n        show_wdl=cfg.show_wdl,\n        syzygy_path=cfg.syzygy_path or None,\n    )\n''',
    "local pool factory WDL/Syzygy parity",
)

# 7. Durable regression coverage for the follow-up findings.
tests = ROOT / "tests/test_audit_v4_followup.py"
tests.write_text('''from __future__ import annotations\n\nimport os\nfrom types import SimpleNamespace\nfrom typing import Any\n\nimport chess\nimport pytest\n\nfrom core.engines.analysis import classify_move as core_classify_move\nfrom core.engines.analyzer import Analyzer\nfrom core.engines.pool import AnalyzerPool\nfrom core.engines.types import Eval\nfrom mcp_server import cache as cache_module\nfrom mcp_server import server as server_module\nfrom mcp_server.rules import evaluate_rule_status, validate_mating_possibility\n\n\ndef test_cache_build_sha_uses_injected_environment(monkeypatch: pytest.MonkeyPatch):\n    monkeypatch.setenv("BUILD_SHA", "0123456789abcdef")\n    assert cache_module._git_sha() == "0123456789abcdef"\n\n\ndef test_unknown_repetition_explicitly_requires_move_stack():\n    status = evaluate_rule_status(chess.Board(chess.STARTING_FEN), history_complete="incomplete")\n    assert status.repetition_status == "unknown"\n    assert status.requires_move_stack is True\n    assert status.history_dependent_status is True\n    assert status.fen_sufficient_for_status is False\n\n\n@pytest.mark.parametrize("text", ["White wins on time", "Black wins on time", "White won on time", "Black won on time"])\ndef test_winner_oriented_time_text_normalizes_to_time_forfeit(text: str):\n    assert server_module.normalize_termination(text) == "time_forfeit"\n\n\n@pytest.mark.parametrize(\n    ("text", "result"),\n    [\n        ("White won on time", "1-0"),\n        ("Black won on time", "0-1"),\n        ("White won by resignation", "1-0"),\n        ("Black won by resignation", "0-1"),\n    ],\n)\ndef test_past_tense_winner_grammar(text: str, result: str):\n    assert server_module._infer_result_from_termination(text) == result\n\n\ndef test_winner_oriented_time_forfeit_still_applies_fide_mating_rule():\n    board = chess.Board("7k/8/8/8/8/8/2B5/K7 w - - 0 1")\n    result, warnings = validate_mating_possibility(board, "1-0", "White wins on time")\n    assert result == "1/2-1/2"\n    assert warnings\n\n\n@pytest.mark.asyncio\nasync def test_core_best_move_uses_real_immediate_post_evaluation():\n    class Backend:\n        name = "audit"\n\n        def __init__(self) -> None:\n            self.calls: list[str] = []\n\n        async def evaluate(\n            self,\n            board: chess.Board,\n            depth: int | None = None,\n            root_moves: list[chess.Move] | None = None,\n        ) -> Eval:\n            self.calls.append(board.fen())\n            if len(self.calls) == 1:\n                return Eval(cp=40, best_move="e2e4", pv=["e2e4", "e7e5"], depth=2)\n            return Eval(cp=25, best_move="e7e5", pv=["e7e5"], depth=2)\n\n        async def top_moves(self, board: chess.Board, n: int = 3, depth: int | None = None) -> list[Eval]:\n            return []\n\n        async def close(self) -> None:\n            return None\n\n    backend = Backend()\n    board = chess.Board()\n    move = chess.Move.from_uci("e2e4")\n    out = await core_classify_move(backend, board, move, depth=2)\n    expected = board.copy(stack=True)\n    expected.push(move)\n    assert len(backend.calls) == 2\n    assert backend.calls[1] == expected.fen()\n    assert out.eval_after.cp == 25\n    assert out.centipawn_loss == 0\n\n\n@pytest.mark.asyncio\nasync def test_local_analyzer_pool_forwards_wdl_and_syzygy(monkeypatch: pytest.MonkeyPatch):\n    seen: dict[str, Any] = {}\n\n    class FakeAnalyzer:\n        name = "fake"\n\n        async def close(self) -> None:\n            return None\n\n    async def fake_create(\n        path: str,\n        *,\n        depth: int = 12,\n        threads: int = 2,\n        hash_mb: int = 128,\n        show_wdl: bool = False,\n        syzygy_path: str | None = None,\n    ) -> FakeAnalyzer:\n        seen.update(\n            path=path,\n            depth=depth,\n            threads=threads,\n            hash_mb=hash_mb,\n            show_wdl=show_wdl,\n            syzygy_path=syzygy_path,\n        )\n        return FakeAnalyzer()\n\n    monkeypatch.setattr(Analyzer, "create", fake_create)\n    pool = await AnalyzerPool.create(\n        "/fake/stockfish",\n        1,\n        depth=9,\n        threads=3,\n        hash_mb=96,\n        show_wdl=True,\n        syzygy_path="/tb",\n    )\n    try:\n        assert seen == {\n            "path": "/fake/stockfish",\n            "depth": 9,\n            "threads": 3,\n            "hash_mb": 96,\n            "show_wdl": True,\n            "syzygy_path": "/tb",\n        }\n    finally:\n        await pool.close()\n\n\n@pytest.mark.asyncio\nasync def test_server_best_move_eval_after_is_immediate_post_position():\n    path = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")\n    if not os.path.isfile(path):\n        pytest.skip("Stockfish not installed")\n\n    old_pool = server_module._analyzer_pool\n    pool = await AnalyzerPool.create(path, 1, depth=5, threads=1, hash_mb=16)\n    server_module._analyzer_pool = pool\n    await server_module._cache.clear()\n    try:\n        root = await server_module.evaluate_position("startpos", depth=5)\n        assert root.best_move is not None\n        result = await server_module.classify_move("startpos", root.best_move, depth=5)\n        board = chess.Board()\n        board.push(chess.Move.from_uci(root.best_move))\n        assert result.eval_after.canonical_fen == board.fen()\n        assert result.is_engine_best is True\n        assert result.effective_loss == 0\n    finally:\n        await server_module._cache.clear()\n        await pool.close()\n        server_module._analyzer_pool = old_pool\n''', encoding="utf-8")
print("follow-up regression file written")
