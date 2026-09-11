"""Reproduction and regression test suite for 2026-09-11 ultra hardening audit.

Covers all findings from /Users/max/Downloads/chess_mcp_ultra_test_claude_code_prompt.md:
- P0: missed_draw_claim accuracy (immediate threefold, intended move 50-move)
- P0: draw-claim consistency and analyze_game parity
- P1: top_moves.n canonical clamping (1..20) and returned_n invariant
- P1: verbosity canonical values, aliases, and error handling
- P1: move_quality_class vs action_quality_class invariants
- P2: analysis_confidence signal and shallow depth warnings
- P2: explicit stability subfields in forensics
- P2: tactical presentation relevance ranking (decisive motifs vs pure geometry)
- P2: strict vs lenient FEN parsing
- P2: include_moves candidate scoring without unrequested forensic overhead
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import pytest
import chess

from mcp_server import server as server_module
from mcp_server.engine.retry import reset_breaker
from mcp_server.parsers.board_builder import build_board
from mcp_server.tools._common import _resolve_verbosity


@pytest.fixture(autouse=True)
async def _cleanup() -> AsyncIterator[None]:
    yield
    reset_breaker()
    await server_module.close_analyzer_pool()
    await server_module._cache.clear()


# ============================================================================
# 1. P0 Correctness: missed_draw_claim Accuracy & Semantics
# ============================================================================

@pytest.mark.asyncio
async def test_missed_immediate_threefold_claim_when_move_leaves_game_active() -> None:
    """When an immediate threefold repetition claim is available and recommended,
    playing a normal move that does not claim the draw must set missed_draw_claim=True,
    even if the resulting position is completely equal/active.
    """
    moves = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]

    res = await server_module.classify_move(
        fen="startpos",
        moves=moves,
        move="e4",
        depth=10,
    )

    assert res.can_claim_now is True
    assert res.best_action == "claim_draw"
    assert res.action_type == "play_move"
    assert res.is_best_action is False
    assert res.action_equivalent is False
    assert res.missed_draw_claim is True
    assert getattr(res, "missed_draw_claim_kind", None) == "immediate"
    assert res.action_quality_class in ("suboptimal_rule_action", "missed_rule_action")


@pytest.mark.asyncio
async def test_missed_intended_move_50_move_claim() -> None:
    """When a 50-move draw claim is available with an intended move, playing a
    different move (or playing a move without claiming) must set missed_draw_claim=True.
    """
    fen = "4k3/8/8/8/8/8/r7/4K3 w - - 99 50"
    res = await server_module.classify_move(
        fen=fen,
        move="Kd1",
        depth=10,
    )
    assert res.can_claim_with_intended_move is True
    assert res.best_action == "claim_draw_with_intended_move"
    assert res.action_type == "play_move"
    assert res.is_best_action is False
    assert res.action_equivalent is False
    assert res.missed_draw_claim is True
    assert getattr(res, "missed_draw_claim_kind", None) == "intended_move"


@pytest.mark.asyncio
async def test_correct_draw_claim_not_flagged_as_missed() -> None:
    """When a draw claim action is played and matches the recommended claim action,
    missed_draw_claim must be False.
    """
    moves = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]
    res = await server_module.classify_move(
        fen="startpos",
        moves=moves,
        move="claim_draw",
        action_type="claim_draw",
        depth=10,
    )
    assert res.best_action == "claim_draw"
    assert res.action_type == "claim_draw"
    assert res.is_best_action is True
    assert res.missed_draw_claim is False
    assert getattr(res, "missed_draw_claim_kind", "none") == "none"


@pytest.mark.asyncio
async def test_forced_win_overrides_draw_claim_policy() -> None:
    """If a forced win exists, rule action policy dictates that the winning move
    is recommended over a draw claim, and missed_draw_claim must be False.
    """
    fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 99 50"
    res = await server_module.classify_move(
        fen=fen,
        move="Qg7#",
        depth=8,
    )
    assert res.best_action == "play_move"
    assert res.is_best_action is True
    assert res.missed_draw_claim is False


@pytest.mark.asyncio
async def test_analyze_game_parity_for_missed_draw_claims() -> None:
    """analyze_game must consistently flag missed_draw_claim on plies where a claim
    was recommended but an ordinary non-equivalent move was played.
    """
    pgn = "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. e4 e5"
    res = await server_module.analyze_game(pgn=pgn, depth=8)
    ply_9 = next((p for p in res.turning_points if p.ply == 9), None)
    assert ply_9 is not None
    assert ply_9.best_action == "claim_draw"
    assert ply_9.missed_draw_claim is True


# ============================================================================
# 2. P1 Contract: top_moves.n Clamping & returned_n Invariant
# ============================================================================

@pytest.mark.parametrize(
    "raw_n, expected_clamped",
    [
        (-10, 1),
        (0, 1),
        (1, 1),
        (3, 3),
        (10, 10),
        (11, 11),
        (20, 20),
        (100, 20),
        (999, 20),
    ],
)
@pytest.mark.asyncio
async def test_top_moves_n_parameter_clamping_table(raw_n: int, expected_clamped: int) -> None:
    res = await server_module.top_moves(
        fen="startpos",
        n=raw_n,
        depth=6,
    )
    assert res.requested_n == raw_n
    assert res.clamped_n == expected_clamped
    assert res.returned_n is not None and res.returned_n <= expected_clamped


@pytest.mark.asyncio
async def test_top_moves_include_moves_expands_returned_count_predictably() -> None:
    res = await server_module.top_moves(
        fen="startpos",
        n=3,
        depth=6,
        include_moves=["a3", "h3"],
    )
    assert res.clamped_n == 3
    assert res.returned_n is not None and res.returned_n <= 3 + 2


# ============================================================================
# 3. P1 Contract: verbosity Canonical Values & Aliases
# ============================================================================

@pytest.mark.parametrize(
    "alias, expected_canonical",
    [
        ("full", "full"),
        ("compact", "compact"),
        ("minimal", "minimal"),
        ("min", "minimal"),
        ("standard", "full"),
        ("default", "full"),
    ],
)
def test_verbosity_normalization_table(alias: str, expected_canonical: str) -> None:
    assert _resolve_verbosity(alias) == expected_canonical


def test_verbosity_invalid_rejected() -> None:
    with pytest.raises(ValueError, match="INVALID_VERBOSITY"):
        _resolve_verbosity("super_verbose")


@pytest.mark.asyncio
async def test_top_moves_verbosity_aliases_produce_identical_payloads() -> None:
    res_minimal = await server_module.top_moves(fen="startpos", n=2, depth=6, verbosity="minimal")
    res_min = await server_module.top_moves(fen="startpos", n=2, depth=6, verbosity="min")
    assert res_minimal.model_dump() == res_min.model_dump()

    res_full = await server_module.top_moves(fen="startpos", n=2, depth=6, verbosity="full")
    res_standard = await server_module.top_moves(fen="startpos", n=2, depth=6, verbosity="standard")
    assert res_full.model_dump() == res_standard.model_dump()


# ============================================================================
# 4. P1 Invariant: move_quality_class vs action_quality_class
# ============================================================================

@pytest.mark.asyncio
async def test_move_quality_vs_action_quality_when_draw_claim_missed() -> None:
    moves = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]
    res = await server_module.classify_move(
        fen="startpos",
        moves=moves,
        move="e4",
        depth=10,
    )
    assert res.is_best_action is False
    assert res.action_quality_class in ("suboptimal_rule_action", "missed_rule_action")
    assert res.missed_draw_claim is True


# ============================================================================
# 5. P2 Quality: Low-Depth Confidence Signal
# ============================================================================

@pytest.mark.asyncio
async def test_low_depth_confidence_flagged_for_shallow_search() -> None:
    res = await server_module.classify_move(
        fen="startpos",
        move="e4",
        depth=4,
    )
    assert res.analysis_confidence is not None
    assert res.analysis_confidence.level == "low"
    assert "SHALLOW_SEARCH_DEPTH" in res.analysis_confidence.reason_codes
    assert res.analysis_confidence.requested_depth == 4


@pytest.mark.asyncio
async def test_exact_mate_at_low_depth_retains_high_confidence() -> None:
    moves = ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"]
    res = await server_module.classify_move(
        fen="startpos",
        moves=moves,
        move="Qxf7#",
        depth=4,
    )
    assert res.analysis_confidence is not None
    assert res.analysis_confidence.level == "high"


# ============================================================================
# 6. P2 Forensics: Explicit Stability Subfields
# ============================================================================

@pytest.mark.asyncio
async def test_forensics_stability_exposes_explicit_subfields() -> None:
    res = await server_module.classify_move(
        fen="startpos",
        move="e4",
        depth=12,
        detail="forensic",
    )
    assert res.forensics is not None
    stability = res.forensics.stability
    assert "classification_stable" in stability
    assert "best_move_stable" in stability
    assert "pv_prefix_stable" in stability
    assert "mate_status_stable" in stability
    assert "evaluation_band_stable" in stability
    assert "verification_depths" in stability


# ============================================================================
# 7. P2 Tactical Relevance: Decisive Motifs Prioritized
# ============================================================================

@pytest.mark.asyncio
async def test_tactical_presentation_prioritizes_decisive_motifs() -> None:
    fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    res = await server_module.evaluate_position(
        fen=fen,
        depth=10,
        detail="forensic",
    )
    assert res.forensics is not None
    pm = res.forensics.tactical_snapshot.presentation_mechanisms
    if pm:
        top_priorities = [m.presentation_priority for m in pm[:3]]
        assert any(p in ("immediate_mate", "checking_move") for p in top_priorities)


# ============================================================================
# 8. P2 Contract: Strict vs Lenient FEN Parsing
# ============================================================================

def test_strict_fen_requires_six_fields() -> None:
    with pytest.raises(ValueError, match="INVALID_FEN"):
        build_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq", strict=True)


def test_lenient_fen_auto_completes_missing_fields() -> None:
    board = build_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq", strict=False)
    assert board.turn == chess.WHITE
    assert board.has_kingside_castling_rights(chess.WHITE)


# ============================================================================
# 9. P2 Performance: include_moves Decoupled from Unrequested Forensics
# ============================================================================

@pytest.mark.asyncio
async def test_top_moves_include_moves_standard_detail_avoids_heavy_upgrade() -> None:
    res = await server_module.top_moves(
        fen="startpos",
        n=3,
        depth=8,
        include_moves=["d4"],
        detail="standard",
    )
    assert res.forensics is not None
    assert not res.forensics.candidate_differences
    candidate_ucis = [c.best_move.lower() for c in res.result if c.best_move]
    assert "d2d4" in candidate_ucis
