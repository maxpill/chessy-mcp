"""Regression suite for all confirmed findings in the 2026-09-11 audit.

Covers:
- P0: Forced win outcome dominance over active cp in classify_move
- P1: Root candidate action preservation on automatic draws in top_moves
- P1: Outcome consistency between played and best on identical actions in classify_move
- P1: board_legal_move_uci cross-tool parity on terminal boards & minimal serialization
- P1: analyze_game final_position draw policy parity with evaluate_position and top_moves
- P2: proof_defenses symmetric clamping in top_moves
- P2: raw max_critical_moments=0 preservation in analyze_game
- P2: candidate search provenance score_comparability metadata
- P3: zero-ply PGN "1." empty_game_reason assignment
"""

from __future__ import annotations

import chess
import pytest

from core.engines.types import MoveClass
from mcp_server import server as server_module
from mcp_server.server import analyze_game, classify_move, evaluate_position, top_moves


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_classify_forced_mate_dominates_active_cp():
    """P0: A forced mate discovered for the mover must dominate a merely active cp move."""
    fen = "8/P7/8/8/8/8/5K2/7k w - - 0 1"
    res = await classify_move(fen, move="a8=Q+", depth=3, strict=True, detail="standard")
    assert res.is_engine_best is True
    assert res.is_best_action is True
    assert res.move_class == MoveClass.BEST
    assert res.played_outcome == "win"
    assert res.best_outcome == "win"
    assert res.played_canonical_value == 100000
    assert res.best_canonical_value == 100000


def test_score_cp_to_mate_forced_win_dominance_unit():
    """P0: score_played_move must classify a move discovering mate as BEST over a non-mating cp root move."""
    from mcp_server.analysis.move_grading import score_played_move
    from mcp_server.models import MCPEval
    from core.engines.types import Eval

    b_before = chess.Board("8/P7/8/8/8/8/5K2/7k w - - 0 1")
    move = chess.Move.from_uci("a7a8q")
    b_after = b_before.copy(stack=True)
    b_after.push(move)

    ev_before = MCPEval.from_eval(
        Eval(cp=818, mate=None, best_move="f2f1", pv=["f2f1"], depth=3),
        b_before.fen(),
        board=b_before,
    )
    ev_after = MCPEval.from_eval(
        Eval(cp=None, mate=1, best_move="h1g1", pv=["h1g1"], depth=3),
        b_after.fen(),
        board=b_after,
    )

    score = score_played_move(b_before, move, ev_before, ev_after, board_after=b_after)
    assert score.is_best_action is True
    assert score.is_best_engine_move is True
    assert score.move_class == MoveClass.BEST
    assert score.effective_loss == 0



@pytest.mark.asyncio
async def test_top_moves_terminal_transition_preserves_root_play_action():
    """P1: A legal root move causing an automatic draw must have root_candidate_action='play_move'."""
    # 1. Capture into insufficient material
    fen_insufficient = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
    res_insufficient = await top_moves(fen_insufficient, depth=6)
    kxe2_cands = [c for c in res_insufficient.result if (c.best_move or "").lower() == "e1e2"]
    assert len(kxe2_cands) == 1
    cand = kxe2_cands[0]
    assert cand.root_candidate_action == "play_move"
    assert cand.recommended_action == "play_move"
    assert cand.best_action_obj is not None and cand.best_action_obj.get("type") == "play_move"
    assert cand.post_position is not None
    assert cand.post_position.get("status") == "insufficient_material"
    assert cand.post_position.get("recommended_action") == "game_over"

    # 2. Move reaching 75-move automatic draw
    fen_75 = "8/8/8/8/8/8/R4K2/7k w - - 149 75"
    res_75 = await top_moves(fen_75, depth=6)
    ra1_cands = [c for c in res_75.result if (c.best_move or "").lower() == "a2a1"]
    assert len(ra1_cands) == 1
    cand_75 = ra1_cands[0]
    assert cand_75.root_candidate_action == "play_move"
    assert cand_75.recommended_action == "play_move"
    assert cand_75.best_action_obj is not None and cand_75.best_action_obj.get("type") == "play_move"
    assert cand_75.post_position is not None
    assert cand_75.post_position.get("status") == "seventyfive_moves"
    assert cand_75.post_position.get("recommended_action") == "game_over"


@pytest.mark.asyncio
async def test_classify_same_action_has_same_outcome():
    """P1: When played move and best action are the same action, their outcomes must agree."""
    fen = "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
    res = await classify_move(fen, move="Kxe2", depth=6)
    assert res.is_best_action is True
    assert res.is_engine_best is True
    assert res.played_outcome == "draw"
    assert res.best_outcome == "draw"
    assert res.same_outcome is True
    assert res.played_canonical_value == res.best_canonical_value


@pytest.mark.asyncio
async def test_board_legal_moves_cross_tool_parity():
    """P1: Terminal boards with geometric moves must report identical board_legal_move_uci across tools."""
    fen = "8/8/8/8/8/8/5K2/7k w - - 100 51"
    b = chess.Board(fen)
    expected_geometry = sorted([m.uci() for m in b.legal_moves])
    assert len(expected_geometry) == 6

    ev = await evaluate_position(fen)
    assert sorted(ev.board_legal_move_uci or []) == expected_geometry

    tm = await top_moves(fen)
    assert sorted(tm.board_legal_move_uci or []) == expected_geometry


@pytest.mark.asyncio
async def test_minimal_serializer_does_not_fake_empty_sets():
    """P1: Minimal mode must not emit [] for omitted move arrays on active positions."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tm_min = await top_moves(fen, verbosity="minimal")
    cand = tm_min.result[0]
    # In minimal mode, omitted arrays should serialize as None, NOT []
    assert cand.legal_move_uci is None
    assert cand.board_legal_move_uci is None


@pytest.mark.asyncio
async def test_analyze_game_threefold_recommendation_matches_policy():
    """P1: analyze_game final position must follow the shared draw-claim policy."""
    moves = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]
    startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    ev = await evaluate_position(startpos, moves=moves)
    assert ev.can_claim_now is True
    assert ev.recommended_action == "claim_draw"

    tm = await top_moves(startpos, moves=moves)
    assert tm.can_claim_now is True
    assert tm.recommended_action == "claim_draw"

    pgn = "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8"
    ag = await analyze_game(pgn, detail="coach")
    assert ag.coaching is not None
    assert ag.coaching.final_position.can_claim_now is True
    assert ag.coaching.final_position.recommended_action == "claim_draw"


@pytest.mark.asyncio
async def test_proof_defenses_symmetric_clamping():
    """P2: proof_defenses validates minimum >= 1 and clamps maximum to 8."""
    from mcp.server.mcpserver.exceptions import ToolError

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    with pytest.raises((ToolError, ValueError)) as exc_info:
        await top_moves(fen, proof_mode="tactical", proof_defenses=0)
    assert "INVALID_ARGUMENT" in str(exc_info.value).upper()

    res_high = await top_moves(fen, proof_mode="tactical", proof_defenses=9)
    assert res_high.requested_proof_defenses == 9
    assert res_high.clamped_proof_defenses == 8


@pytest.mark.asyncio
async def test_requested_max_critical_moments_preserves_zero():
    """P2: max_critical_moments=0 must preserve requested=0."""
    pgn = "1. e4 e5 2. Nf3 Nc6"
    res = await analyze_game(pgn, max_critical_moments=0)
    assert res.requested_max_critical_moments == 0
    assert res.clamped_max_critical_moments == 1


@pytest.mark.asyncio
async def test_zero_ply_pgn_has_explicit_semantics():
    """P3: Bare move number '1.' must return total_plies=0 and empty_game_reason='no_mainline_moves'."""
    res = await analyze_game("1.")
    assert res.total_plies == 0
    assert res.empty_game_reason == "no_mainline_moves"


@pytest.mark.asyncio
async def test_candidate_search_provenance_comparability():
    """P2: Candidate search provenance exposes score_comparability metadata."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res = await top_moves(fen, n=2, include_moves=["c4"])
    # Root MultiPV candidates
    root_cands = [c for c in res.result if c.search_provenance and c.search_provenance.get("kind") == "multipv_root"]
    assert len(root_cands) >= 1
    for c in root_cands:
        assert c.search_provenance.get("score_comparability") == "same_root_multipv"

    # Included candidates
    inc_cands = [c for c in res.result if (c.best_move or "").lower() == "c2c4"]
    assert len(inc_cands) >= 1
    for c in inc_cands:
        assert c.search_provenance.get("score_comparability") in ("same_root_multipv", "independent_search")
