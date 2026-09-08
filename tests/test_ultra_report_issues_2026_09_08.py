"""Regression suite for issues identified in 2026-09-08 Chess MCP Ultra Test Report."""

from __future__ import annotations

import pytest

from mcp_server import server as server_module
from mcp_server.tools.analyze_game import analyze_game
from mcp_server.tools.classify_move import classify_move
from mcp_server.tools.top_moves import top_moves


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_fools_mate_coaching_perspective_white_and_black():
    """Fool's Mate: 1. f3 e5 2. g4 Qh4# 0-1.
    White has been checkmated.
    White perspective: final state must be loss (effective_cp = -100000).
    Black perspective: final state must be win (effective_cp = +100000).
    """
    fools_pgn = "1. f3 e5 2. g4 Qh4# 0-1"

    coach_white = await analyze_game(pgn=fools_pgn, depth=10, detail="coach", perspective="white")
    assert coach_white.coaching is not None
    assert coach_white.coaching.final_position.checkmate is True
    assert coach_white.coaching.final_position.effective_cp == -100000, (
        f"White was checkmated; expected effective_cp -100000, got {coach_white.coaching.final_position.effective_cp}"
    )

    coach_black = await analyze_game(pgn=fools_pgn, depth=10, detail="coach", perspective="black")
    assert coach_black.coaching is not None
    assert coach_black.coaching.final_position.checkmate is True
    assert coach_black.coaching.final_position.effective_cp == 100000, (
        f"Black checkmated White; expected effective_cp +100000, got {coach_black.coaching.final_position.effective_cp}"
    )


@pytest.mark.asyncio
async def test_scholars_mate_coaching_perspective_white_and_black():
    """Scholar's Mate: 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0.
    White delivered checkmate.
    White perspective: final state must be win (effective_cp = +100000).
    Black perspective: final state must be loss (effective_cp = -100000).
    """
    scholars_pgn = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"

    coach_white = await analyze_game(pgn=scholars_pgn, depth=10, detail="coach", perspective="white")
    assert coach_white.coaching is not None
    assert coach_white.coaching.final_position.checkmate is True
    assert coach_white.coaching.final_position.effective_cp == 100000

    coach_black = await analyze_game(pgn=scholars_pgn, depth=10, detail="coach", perspective="black")
    assert coach_black.coaching is not None
    assert coach_black.coaching.final_position.checkmate is True
    assert coach_black.coaching.final_position.effective_cp == -100000


@pytest.mark.asyncio
async def test_classify_move_scholars_mate_practical_equivalence():
    """White plays Qxf7# (checkmate).
    Must be is_engine_best=True, practical_equivalent=True, status=equivalent.
    Must NOT contain MATE_STATUS_DETERIORATED.
    """
    scholars_moves = ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"]
    res = await classify_move(
        fen="startpos",
        moves=scholars_moves,
        move="Qxf7#",
        detail="forensic",
    )
    assert res.is_engine_best is True
    assert res.forensics is not None
    practical = res.forensics.stability["practical_equivalence"]
    assert practical["practical_equivalent"] is True
    assert practical["status"] == "equivalent"
    assert "MATE_STATUS_DETERIORATED" not in practical["reason_codes"]
    assert practical["mate_after"] == "white_mates"
    assert practical["mate_deterioration_for_mover"] is False


@pytest.mark.asyncio
async def test_classify_move_fools_mate_practical_equivalence():
    """Black plays Qh4# (checkmate).
    Must be is_engine_best=True, practical_equivalent=True, status=equivalent.
    Must NOT contain MATE_STATUS_DETERIORATED.
    """
    fools_moves = ["f3", "e5", "g4"]
    res = await classify_move(
        fen="startpos",
        moves=fools_moves,
        move="Qh4#",
        detail="forensic",
    )
    assert res.is_engine_best is True
    assert res.forensics is not None
    practical = res.forensics.stability["practical_equivalence"]
    assert practical["practical_equivalent"] is True
    assert practical["status"] == "equivalent"
    assert "MATE_STATUS_DETERIORATED" not in practical["reason_codes"]
    assert practical["mate_after"] == "black_mates"
    assert practical["mate_deterioration_for_mover"] is False


@pytest.mark.asyncio
async def test_classify_move_winning_terminal_move_suppresses_failed_position_update():
    """Winning terminal move (checkmate) must not retain FAILED_POSITION_UPDATE_CANDIDATE."""
    scholars_moves = ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"]
    res = await classify_move(
        fen="startpos",
        moves=scholars_moves,
        move="Qxf7#",
        detail="forensic",
    )
    assert res.forensics is not None
    assert "FAILED_POSITION_UPDATE_CANDIDATE" not in res.forensics.evidence_signatures


@pytest.mark.asyncio
async def test_analyze_game_max_critical_moments_clamping():
    """max_critical_moments is clamped to 1..7 rather than raising INVALID_ARGUMENT."""
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
    res_0 = await analyze_game(pgn=pgn, depth=14, detail="coach", max_critical_moments=0)
    assert res_0.coaching is not None
    assert len(res_0.coaching.critical_moments) <= 1

    res_99 = await analyze_game(pgn=pgn, depth=14, detail="coach", max_critical_moments=99)
    assert res_99.coaching is not None
    assert len(res_99.coaching.critical_moments) <= 7


@pytest.mark.asyncio
async def test_top_moves_root_legal_move_uci_populated():
    """Root TopMovesResult must populate legal_move_uci consistent with legal_move_count."""
    res = await top_moves(fen="startpos", n=3, depth=10)
    assert res.legal_move_count == 20
    assert res.board_legal_move_count == 20
    assert len(res.legal_move_uci) == 20
    assert "e2e4" in res.legal_move_uci

