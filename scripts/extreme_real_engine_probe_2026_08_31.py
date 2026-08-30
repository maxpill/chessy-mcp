from __future__ import annotations

import asyncio
import random

import chess

from mcp_server import server as server_module


AUTO_TERMINAL = {
    "checkmate",
    "stalemate",
    "insufficient_material",
    "seventyfive_moves",
    "dead_position",
}


def expected_fen_status(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_seventyfive_moves():
        return "seventyfive_moves"
    return "active"


def generated_positions(count: int) -> list[chess.Board]:
    rng = random.Random(31082026)
    board = chess.Board()
    out: list[chess.Board] = []
    seen: set[str] = set()
    while len(out) < count:
        if board.is_game_over(claim_draw=False) or len(board.move_stack) > 90:
            board = chess.Board()
        legal = list(board.legal_moves)
        if not legal:
            board = chess.Board()
            continue
        board.push(legal[rng.randrange(len(legal))])
        # Skip history-only automatic repetitions because a naked FEN cannot
        # encode them. Other automatic states are FEN-detectable.
        if board.is_fivefold_repetition():
            board = chess.Board()
            continue
        key = board.fen()
        if key not in seen:
            seen.add(key)
            out.append(board.copy(stack=True))
    return out


async def main() -> None:
    await server_module._cache.clear()
    await server_module.close_analyzer_pool()

    positions = generated_positions(200)
    tool_calls = 0
    candidate_checks = 0
    status_checks = 0

    for board in positions:
        fen = board.fen()
        expected_status = expected_fen_status(board)

        ev = await server_module.evaluate_position(fen, depth=2, verbosity="compact")
        tool_calls += 1
        assert ev.status == expected_status, (fen, expected_status, ev.status)
        status_checks += 1
        if ev.status == "active":
            assert ev.best_move is not None
            assert chess.Move.from_uci(ev.best_move) in board.legal_moves
        else:
            assert ev.best_move is None
            assert ev.best_action_obj is not None
            assert ev.best_action_obj["type"] == "game_over"

        top = await server_module.top_moves(fen, n=5, depth=2, verbosity="compact")
        tool_calls += 1
        assert top.status == expected_status
        status_checks += 1
        assert top.returned_n == len(top.result)
        if expected_status != "active":
            assert top.result == []
            continue

        legal_count = board.legal_moves.count()
        assert len(top.result) == min(5, legal_count)
        for candidate in top.result:
            assert candidate.best_move is not None
            move = chess.Move.from_uci(candidate.best_move)
            assert move in board.legal_moves
            assert candidate.candidate_san == board.san(move)
            after = board.copy(stack=True)
            after.push(move)
            assert candidate.canonical_fen == after.fen()
            assert candidate.build_sha == top.build_sha
            assert candidate.engine_config == top.engine_config
            candidate_checks += 1

    # Real-engine state isolation check, using fields that should be fully
    # deterministic even if shallow Stockfish scores fluctuate slightly.
    a = positions[10].fen()
    b = positions[150].fen()
    a1 = await server_module.evaluate_position(a, depth=3, verbosity="compact")
    _ = await server_module.evaluate_position(b, depth=3, verbosity="compact")
    a2 = await server_module.evaluate_position(a, depth=3, verbosity="compact")
    tool_calls += 3
    assert a1.canonical_fen == a2.canonical_fen
    assert a1.status == a2.status
    assert a1.best_move == a2.best_move

    await server_module.close_analyzer_pool()
    print(f"EXTREME_REAL_ENGINE_POSITIONS={len(positions)}")
    print(f"EXTREME_REAL_ENGINE_TOOL_CALLS={tool_calls}")
    print(f"EXTREME_REAL_ENGINE_STATUS_COMPARISONS={status_checks}")
    print(f"EXTREME_REAL_ENGINE_CANDIDATE_COMPARISONS={candidate_checks}")


if __name__ == "__main__":
    asyncio.run(main())
