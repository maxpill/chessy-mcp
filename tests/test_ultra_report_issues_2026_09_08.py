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


@pytest.mark.asyncio
async def test_fools_mate_events_and_segments_invariants():
    """Checkmate delivery invariants:
    - Mating player cannot have lost_advantage, missed_conversion, fell_behind, missed_recovery
    - Checkmated player cannot have recovered, gained_advantage
    - Loser-relative terminal eval is always negative; winner-relative always positive
    """
    fools_pgn = "1. f3 e5 2. g4 Qh4# 0-1"

    coach_white = await analyze_game(pgn=fools_pgn, depth=10, detail="coach", perspective="white")
    assert coach_white.coaching is not None
    assert coach_white.coaching.final_position.effective_cp < 0
    # Checkmated player (White) cannot have recovered or gained advantage after Qh4#
    white_kinds = [e.kind for e in coach_white.coaching.advantage_events if e.ply == 4]
    assert "recovered" not in white_kinds
    assert "gained_advantage" not in white_kinds

    coach_black = await analyze_game(pgn=fools_pgn, depth=10, detail="coach", perspective="black")
    assert coach_black.coaching is not None
    assert coach_black.coaching.final_position.effective_cp > 0
    # Mating player (Black) cannot have lost advantage or fallen behind after Qh4#
    black_kinds = [e.kind for e in coach_black.coaching.advantage_events if e.ply == 4]
    assert "lost_advantage" not in black_kinds
    assert "missed_conversion" not in black_kinds
    assert "fell_behind" not in black_kinds
    assert "missed_recovery" not in black_kinds


@pytest.mark.asyncio
async def test_zero_loss_comment_does_not_gain_high_engine_loss():
    """F-003: Player comment retains move as critical, but does NOT synthesize high_engine_loss."""
    pgn = "1. e4 {best move self report} e5 *"
    res = await analyze_game(pgn=pgn, depth=10, detail="coach", perspective="white")
    assert res.coaching is not None
    moments = [m for m in res.coaching.critical_moments if m.ply == 1]
    assert len(moments) == 1
    m = moments[0]
    assert "player_self_report" in m.reasons
    assert "high_engine_loss" not in m.reasons
    assert (m.effective_loss or 0) < 100
    assert (m.centipawn_loss or 0) < 100


@pytest.mark.asyncio
async def test_malformed_fen_raises_invalid_position_input():
    """F-004: Parameter named fen receiving non-FEN non-PGN raises INVALID_POSITION_INPUT."""
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp_server.tools.evaluate_position import evaluate_position

    with pytest.raises(ToolError) as exc_info:
        await evaluate_position(fen="not a fen", depth=10)
    err = str(exc_info.value)
    assert "[INVALID_POSITION]" in err
    assert "fen_parse: failed" in err
    assert "pgn_parse:" in err


@pytest.mark.asyncio
async def test_uniform_illegal_move_taxonomy():
    """F-005: Illegal moves across moves, include_moves, and compare_moves all raise ILLEGAL_MOVE."""
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp_server.tools.evaluate_position import evaluate_position

    # 1. Root moves
    with pytest.raises(ToolError) as exc1:
        await evaluate_position(fen="startpos", moves=["e2e5"], depth=10)
    assert "[ILLEGAL_MOVE]" in str(exc1.value)

    # 2. top_moves include_moves
    with pytest.raises(ToolError) as exc2:
        await top_moves(fen="startpos", include_moves=["e2e5"], depth=10)
    assert "[ILLEGAL_MOVE]" in str(exc2.value)

    # 3. classify_move compare_moves
    with pytest.raises(ToolError) as exc3:
        await classify_move(fen="startpos", move="e4", compare_moves=["e2e5"], depth=10)
    assert "[ILLEGAL_MOVE]" in str(exc3.value)


@pytest.mark.asyncio
async def test_candidate_parameter_count_limit():
    """F-005: >8 candidate parameters raise INVALID_PARAMETER_COUNT."""
    from mcp.server.mcpserver.exceptions import ToolError

    # 9 candidate moves
    nine_moves = ["e4", "d4", "c4", "Nf3", "Nc3", "f4", "g3", "b3", "a3"]

    with pytest.raises(ToolError) as exc1:
        await top_moves(fen="startpos", include_moves=nine_moves, depth=10)
    assert "[INVALID_ARGUMENT]" in str(exc1.value)
    assert "supports at most 8 moves" in str(exc1.value)

    with pytest.raises(ToolError) as exc2:
        await classify_move(fen="startpos", move="e4", compare_moves=nine_moves, depth=10)
    assert "[INVALID_ARGUMENT]" in str(exc2.value)
    assert "at most 8 candidates are allowed" in str(exc2.value)


@pytest.mark.asyncio
async def test_top_moves_action_semantics_and_post_position():
    """F-006: Candidate exposes root_candidate_action while post_position exposes post_position_recommended_action."""
    res = await top_moves(fen="startpos", n=2, depth=10)
    assert len(res.result) >= 1
    cand = res.result[0]
    assert cand.recommended_action == "play_move"
    assert cand.root_candidate_action == "play_move"
    assert cand.post_position is not None
    assert "recommended_action" in cand.post_position
    assert "post_position_recommended_action" in cand.post_position


