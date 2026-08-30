from __future__ import annotations

import os
from typing import Any

import chess
import pytest

from core.engines.analysis import classify_move as core_classify_move
from core.engines.analyzer import Analyzer
from core.engines.pool import AnalyzerPool
from core.engines.types import Eval
from mcp_server import cache as cache_module
from mcp_server import server as server_module
from mcp_server.rules import evaluate_rule_status, validate_mating_possibility


def test_cache_build_sha_uses_injected_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BUILD_SHA", "0123456789abcdef")
    assert cache_module._git_sha() == "0123456789abcdef"


def test_unknown_repetition_explicitly_requires_move_stack():
    status = evaluate_rule_status(chess.Board(chess.STARTING_FEN), history_complete="incomplete")
    assert status.repetition_status == "unknown"
    assert status.requires_move_stack is True
    assert status.history_dependent_status is True
    assert status.fen_sufficient_for_status is False


@pytest.mark.parametrize("text", ["White wins on time", "Black wins on time", "White won on time", "Black won on time"])
def test_winner_oriented_time_text_normalizes_to_time_forfeit(text: str):
    assert server_module.normalize_termination(text) == "time_forfeit"


@pytest.mark.parametrize(
    ("text", "result"),
    [
        ("White won on time", "1-0"),
        ("Black won on time", "0-1"),
        ("White won by resignation", "1-0"),
        ("Black won by resignation", "0-1"),
    ],
)
def test_past_tense_winner_grammar(text: str, result: str):
    assert server_module._infer_result_from_termination(text) == result


def test_winner_oriented_time_forfeit_still_applies_fide_mating_rule():
    board = chess.Board("7k/8/8/8/8/8/2B5/K7 w - - 0 1")
    result, warnings = validate_mating_possibility(board, "1-0", "White wins on time")
    assert result == "1/2-1/2"
    assert warnings


@pytest.mark.asyncio
async def test_core_best_move_uses_real_immediate_post_evaluation():
    class Backend:
        name = "audit"

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def evaluate(
            self,
            board: chess.Board,
            depth: int | None = None,
            root_moves: list[chess.Move] | None = None,
        ) -> Eval:
            self.calls.append(board.fen())
            if len(self.calls) == 1:
                return Eval(cp=40, best_move="e2e4", pv=["e2e4", "e7e5"], depth=2)
            return Eval(cp=25, best_move="e7e5", pv=["e7e5"], depth=2)

        async def top_moves(self, board: chess.Board, n: int = 3, depth: int | None = None) -> list[Eval]:
            return []

        async def close(self) -> None:
            return None

    backend = Backend()
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    out = await core_classify_move(backend, board, move, depth=2)
    expected = board.copy(stack=True)
    expected.push(move)
    assert len(backend.calls) == 2
    assert backend.calls[1] == expected.fen()
    assert out.eval_after.cp == 25
    assert out.centipawn_loss == 0


@pytest.mark.asyncio
async def test_local_analyzer_pool_forwards_wdl_and_syzygy(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    class FakeAnalyzer:
        name = "fake"

        async def close(self) -> None:
            return None

    async def fake_create(
        path: str,
        *,
        depth: int = 12,
        threads: int = 2,
        hash_mb: int = 128,
        show_wdl: bool = False,
        syzygy_path: str | None = None,
    ) -> FakeAnalyzer:
        seen.update(path=path, depth=depth, threads=threads, hash_mb=hash_mb, show_wdl=show_wdl, syzygy_path=syzygy_path)
        return FakeAnalyzer()

    monkeypatch.setattr(Analyzer, "create", fake_create)
    pool = await AnalyzerPool.create(
        "/fake/stockfish", 1, depth=9, threads=3, hash_mb=96, show_wdl=True, syzygy_path="/tb"
    )
    try:
        assert seen == {
            "path": "/fake/stockfish",
            "depth": 9,
            "threads": 3,
            "hash_mb": 96,
            "show_wdl": True,
            "syzygy_path": "/tb",
        }
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_server_best_move_eval_after_is_immediate_post_position():
    path = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
    if not os.path.isfile(path):
        pytest.skip("Stockfish not installed")

    old_pool = server_module._analyzer_pool
    pool = await AnalyzerPool.create(path, 1, depth=5, threads=1, hash_mb=16)
    server_module._analyzer_pool = pool
    await server_module._cache.clear()
    try:
        root = await server_module.evaluate_position("startpos", depth=5)
        assert root.best_move is not None
        result = await server_module.classify_move("startpos", root.best_move, depth=5)
        board = chess.Board()
        board.push(chess.Move.from_uci(root.best_move))
        assert result.eval_after.canonical_fen == board.fen()
        assert result.is_engine_best is True
        assert result.effective_loss == 0
    finally:
        await server_module._cache.clear()
        await pool.close()
        server_module._analyzer_pool = old_pool
