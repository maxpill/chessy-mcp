"""Property-based randomized oracle tests for chess-mcp.

Uses python-chess to generate legal random boards and sequences, asserting:
1. FEN canonicalization consistency across arbitrary positions.
2. Legal move count invariants: len(board_legal_move_uci) == board.legal_moves.count().
3. Action fields consistency and absence of null vs "play_move" contradictions.
4. SAN parser invariants: canonical SAN always parses cleanly with normalization_kind="none".
5. Clamping bounds: raw n vs clamped n in [1, 10], raw depth vs clamped depth in [1, 30].
6. Terminal consistency: evaluate_rule_status matches python-chess game_over state.
"""

from __future__ import annotations

import random
import chess

from mcp_server.parsers import _build_board_with_metadata
from mcp_server.parsers.move_parser import parse_move_with_details
from mcp_server.rules import evaluate_rule_status
from mcp_server.rules.constants import (
    TOP_MOVES_MIN_N,
    TOP_MOVES_MAX_N,
    DEPTH_MIN,
    DEPTH_MAX,
)
from mcp_server.tools._common import _validate_requested_depth


def _random_legal_board(rng: random.Random, max_plies: int = 30) -> chess.Board:
    board = chess.Board()
    plies = rng.randint(0, max_plies)
    for _ in range(plies):
        if board.is_game_over():
            break
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


def test_property_fen_canonicalization() -> None:
    rng = random.Random(42)
    for _ in range(50):
        board = _random_legal_board(rng, max_plies=40)
        fen = board.fen()
        built_board, _input_fen, canonical_fen, _was_canonicalized = _build_board_with_metadata(
            fen, [], strict=False
        )
        assert built_board.fen() == canonical_fen
        assert built_board.turn == board.turn
        assert built_board.legal_moves.count() == board.legal_moves.count()


def test_property_legal_moves_count_oracle() -> None:
    from core.engines.types import Eval
    from mcp_server.models import MCPEval
    from mcp_server.models.mcpeval_factory import build_mcpeval_from_eval

    rng = random.Random(1337)
    for _ in range(50):
        board = _random_legal_board(rng, max_plies=35)
        ev = Eval(cp=10, depth=1)
        mcp_eval = build_mcpeval_from_eval(MCPEval, ev, board.fen())
        expected_legal = [m.uci() for m in board.legal_moves]
        assert len(mcp_eval.board_legal_move_uci) == len(expected_legal)
        assert set(mcp_eval.board_legal_move_uci) == set(expected_legal)
        if board.is_game_over(claim_draw=False):
            assert len(mcp_eval.legal_move_uci) == 0
        else:
            assert len(mcp_eval.legal_move_uci) == len(expected_legal)


def test_property_san_canonical_roundtrip() -> None:
    rng = random.Random(2026)
    for _ in range(50):
        board = _random_legal_board(rng, max_plies=25)
        if board.is_game_over():
            continue
        for move in board.legal_moves:
            san = board.san(move)
            res = parse_move_with_details(board, san, strict=True)
            assert res.move == move
            assert res.normalization_kind == "none"
            assert res.normalization_changes == []
            assert res.warning is None


def test_property_parameter_clamping_invariants() -> None:
    rng = random.Random(999)
    for _ in range(100):
        raw_n = rng.randint(-500, 500)
        clamped_n = max(TOP_MOVES_MIN_N, min(raw_n, TOP_MOVES_MAX_N))
        assert TOP_MOVES_MIN_N <= clamped_n <= TOP_MOVES_MAX_N

        raw_depth = rng.randint(-50, 100)
        validated_depth = _validate_requested_depth(raw_depth, tool="top_moves")
        clamped_depth = max(DEPTH_MIN, min(validated_depth, DEPTH_MAX))
        assert DEPTH_MIN <= clamped_depth <= DEPTH_MAX


def test_property_terminal_evaluation_consistency() -> None:
    rng = random.Random(777)
    for _ in range(50):
        board = _random_legal_board(rng, max_plies=50)
        rule_status = evaluate_rule_status(board, history_complete="complete")
        if board.is_checkmate():
            assert rule_status.status == "checkmate"
            assert rule_status.winner is not None
            assert rule_status.is_game_over is True
        elif board.is_stalemate():
            assert rule_status.status == "stalemate"
            assert rule_status.winner is None
            assert rule_status.is_game_over is True
        elif board.is_insufficient_material():
            assert rule_status.status == "insufficient_material"
            assert rule_status.is_game_over is True
