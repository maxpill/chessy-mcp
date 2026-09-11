"""Regression tests for the 309-call adversarial QA hardening.

Validates Invariants A-H across:
1. P0-1: Forced / engine-best move never blunder (zero regret).
2. P0-2: Reflexive practical equivalence for engine-best move.
3. P0-3: Authoritative board-terminal checkmate over contradictory PGN Result header.
4. P1-4: Machine PGN directives stripped and not treated as player self-reports.
5. P1-5: Fivefold repetition metadata coherence (not FEN sufficient without history).
6. P2-8: Reject non-positive proof_defenses in tactical proof mode.
7. P2-9: Nested engine_eval.requested_depth parity with root requested_depth.
8. P2-10: top_moves forensic compute transparency and clean minimal projection.
"""

from __future__ import annotations

import chess
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import server as server_module
from mcp_server.engine.retry import reset_breaker
from mcp_server.rules.status import evaluate_rule_status
from mcp_server.tools.analyze_game import analyze_game
from mcp_server.tools.classify_move import classify_move
from mcp_server.tools.evaluate_position import evaluate_position
from mcp_server.tools.top_moves import top_moves


@pytest.fixture(autouse=True)
async def _cleanup():
    yield
    reset_breaker()
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_p0_1_engine_best_move_never_blunder() -> None:
    # Legal's Mate position after 1. e4 e5 2. Nf3 d6 3. Bc4 Bg4 4. Nc3 g6 5. Nxe5 Bxd1 6. Bxf7+
    # Black king has only one legal move: Ke7
    fen = "rn1qkbnr/ppp2B1p/3p2p1/4N3/4P3/2N5/PPPP1PPP/R1BbK2R b KQkq - 0 6"
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    assert len(legal_moves) == 1
    forced_move = legal_moves[0].uci()

    res = await classify_move(fen=fen, move=forced_move, depth=1)
    # Invariant A & B: engine best / only legal move must never be a blunder/mistake
    assert res.move_class == "best", f"Expected 'best' but got {res.move_class}"
    assert res.effective_loss == 0, f"Expected 0 loss but got {res.effective_loss}"
    assert res.is_best_action is True
    assert res.is_best_engine_move is True
    assert res.action_equivalent is True


@pytest.mark.asyncio
async def test_p0_2_engine_best_move_is_practically_equivalent() -> None:
    # Forensic classify_move on forced king move
    fen = "rn1qkbnr/ppp2B1p/3p2p1/4N3/4P3/2N5/PPPP1PPP/R1BbK2R b KQkq - 0 6"
    res = await classify_move(fen=fen, move="Ke7", depth=4, detail="forensic")
    assert res.is_engine_best is True
    assert res.is_best_action is True
    assert res.action_equivalent is True
    if res.forensics:
        pe = res.forensics.stability.get("practical_equivalence")
        assert pe is not None
        assert pe["status"] == "equivalent", f"Expected 'equivalent', got {pe['status']}"
        assert pe["practical_equivalent"] is True



@pytest.mark.asyncio
async def test_p0_3_contradictory_pgn_header_board_outcome_authoritative() -> None:
    # Fool's Mate where PGN header says [Result "1-0"] but board terminal is checkmate 0-1
    pgn = '[Event "Fool\'s Mate Test"]\n[Result "1-0"]\n\n1. f3 e5 2. g4 Qh4# 0-1'
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    # Top-level and coaching termination must agree that Black won
    assert res.result == "0-1"
    assert res.termination == "checkmate"
    assert res.coaching is not None
    assert res.coaching.termination.winner_side == "black"
    assert res.coaching.termination.loser_side == "white"
    assert res.coaching.termination.final_board_checkmate is True


@pytest.mark.asyncio
async def test_p1_4_machine_directives_not_treated_as_player_self_report() -> None:
    # PGN comments containing only clock / eval machine annotations
    pgn = """[Event "Test"]
1. e4 {[%clk 0:05:00] [%eval +0.20]} e5 {[%clk 0:04:55]}
2. Nf3 {[%clk 0:04:58]} Nc6 {[%clk 0:04:50]} 1/2-1/2"""
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    if res.coaching:
        for cm in res.coaching.critical_moments:
            assert "player_self_report" not in cm.reasons
            assert cm.user_comment_raw is None or not cm.user_comment_raw.startswith("[%clk")


@pytest.mark.asyncio
async def test_p1_4_mixed_human_and_machine_comment_preserves_human_text() -> None:
    # PGN comment containing both machine directive and human text
    pgn = """[Event "Test"]
1. e4 {[%clk 0:05:00] I wanted to control the center} e5 1/2-1/2"""
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    if res.coaching and res.coaching.critical_moments:
        cm = res.coaching.critical_moments[0]
        if cm.user_comment_raw:
            assert "[%clk" not in cm.user_comment_raw
            assert "I wanted to control the center" in cm.user_comment_raw


def test_p1_5_fivefold_repetition_metadata_coherent() -> None:
    # Setup fivefold repetition on a board with move history
    b = chess.Board()
    moves = ["g1f3", "g8f6", "f3g1", "f6g8"] * 5
    for m in moves:
        b.push_uci(m)
    assert b.is_fivefold_repetition()

    status = evaluate_rule_status(b, history_complete="complete")
    assert status.terminal == "fivefold_repetition"
    assert status.requires_move_stack is True
    assert status.fen_sufficient_for_status is False
    assert status.repetition_sufficient_without_history is False


@pytest.mark.asyncio
async def test_p2_8_reject_non_positive_proof_defenses_in_tactical_mode() -> None:
    with pytest.raises(ToolError) as exc_info:
        await top_moves(fen="startpos", proof_mode="tactical", proof_defenses=0)
    assert "INVALID_ARGUMENT" in str(exc_info.value).upper() or "INVALID_PROOF_DEFENSES" in str(exc_info.value).upper()


@pytest.mark.asyncio
async def test_p2_9_nested_depth_provenance_parity() -> None:
    res = await evaluate_position("startpos", depth=1, verbosity="full")
    assert res.requested_depth == 1
    if res.engine_eval:
        assert res.engine_eval["requested_depth"] == 1
    if res.eval_block and res.eval_block.engine_eval:
        assert res.eval_block.engine_eval["requested_depth"] == 1


@pytest.mark.asyncio
async def test_p2_10_top_moves_minimal_projection_strips_heavy_forensics() -> None:
    res = await top_moves(
        fen="startpos",
        n=1,
        depth=1,
        verbosity="minimal",
        include_moves=["e4", "d4"],
    )
    assert len(res.result) >= 2
    assert res.forensics is None
    assert "include_moves" in res.forensic_compute_triggered_by
