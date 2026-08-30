from __future__ import annotations

import os
from pathlib import Path

import chess
import pytest

from core.engines.pool import AnalyzerPool
from mcp_server import server as server_module


@pytest.fixture
async def real_stockfish_pool():
    path = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
    if not Path(path).is_file():
        pytest.skip(f"Stockfish binary not available at {path}")

    old_pool = server_module._analyzer_pool
    pool = await AnalyzerPool.create(
        path,
        1,
        depth=6,
        threads=1,
        hash_mb=32,
    )
    await server_module._cache.clear()
    server_module._analyzer_pool = pool
    try:
        yield pool
    finally:
        await server_module._cache.clear()
        await pool.close()
        server_module._analyzer_pool = old_pool


@pytest.mark.asyncio
async def test_real_stockfish_evaluate_position_contract(real_stockfish_pool):
    result = await server_module.evaluate_position("startpos", depth=6, verbosity="compact")

    assert result.status == "active"
    assert result.history_completeness == "complete"
    assert result.repetition_status == "none"
    assert result.best_move is not None
    assert chess.Move.from_uci(result.best_move) in chess.Board().legal_moves
    assert result.searched_depth == 6
    assert result.cp is not None or result.mate is not None


@pytest.mark.asyncio
async def test_real_stockfish_top_moves_are_distinct_legal_candidates(real_stockfish_pool):
    result = await server_module.top_moves("startpos", n=3, depth=6)

    assert result.status == "active"
    assert result.returned_n == 3
    assert len(result.result) == 3

    board = chess.Board()
    moves = [item.best_move for item in result.result]
    assert all(move is not None for move in moves)
    assert len(set(moves)) == 3
    assert all(chess.Move.from_uci(move) in board.legal_moves for move in moves if move)


@pytest.mark.asyncio
async def test_real_stockfish_classify_its_cached_best_move_as_best(real_stockfish_pool):
    root = await server_module.evaluate_position("startpos", depth=7)
    assert root.best_move is not None

    result = await server_module.classify_move(
        "startpos",
        root.best_move,
        depth=7,
        action_type="play_move",
    )

    assert result.played == root.best_move
    assert result.is_engine_best is True
    assert result.is_best_engine_move is True
    assert result.effective_loss == 0
    assert result.played_action_obj is not None
    assert result.played_action_obj["type"] == "play_move"
    assert result.classification_verified is True


@pytest.mark.asyncio
async def test_real_stockfish_analyze_short_game(real_stockfish_pool):
    result = await server_module.analyze_game(
        "1. e4 e5 2. Nf3 Nc6",
        depth=4,
    )

    assert result.total_plies == 4
    assert result.searched_depth == 4
    assert result.result == "*"
    assert result.white_accuracy is not None
    assert result.black_accuracy is not None
    assert 0.0 <= result.white_accuracy <= 100.0
    assert 0.0 <= result.black_accuracy <= 100.0


@pytest.mark.asyncio
async def test_real_stockfish_claim_policy_matches_across_tools(real_stockfish_pool):
    # White is materially lost but has an immediate 50-move claim. Both public
    # position tools must prefer the procedural draw action over playing a move.
    fen = "7k/8/8/8/8/8/R6q/K7 w - - 100 51"

    evaluated = await server_module.evaluate_position(fen, depth=6)
    candidates = await server_module.top_moves(fen, n=3, depth=6)

    assert evaluated.can_claim_now is True
    assert candidates.can_claim_now is True
    assert "fifty_moves" in evaluated.claim_reasons_now
    assert "fifty_moves" in candidates.claim_reasons_now
    assert evaluated.recommended_action == "claim_draw"
    assert candidates.recommended_action == "claim_draw"
    assert evaluated.best_action_obj is not None
    assert candidates.best_action_obj is not None
    assert evaluated.best_action_obj["type"] == "claim_draw"
    assert candidates.best_action_obj["type"] == "claim_draw"
