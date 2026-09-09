"""Regression tests for 2026-09-09 forensic bug report and TimeControl fixes."""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval
from mcp_server import server as server_module
from mcp_server.analysis.game_threat_forensics import critical_forcing_threat_delta
from mcp_server.parsers.pgn_validate.time_control import is_valid_pgn_time_control
from mcp_server.tools.analyze_game import analyze_game
from mcp_server.tools.classify_move import classify_move
from mcp_server.tools.evaluate_position import evaluate_position
from mcp_server.tools.top_moves import top_moves


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


# ------------------------------------------------------------------------------
# TimeControl validation tests
# ------------------------------------------------------------------------------


def test_time_control_accepts_fischer_zero_increment():
    assert is_valid_pgn_time_control("600+0") is True
    assert is_valid_pgn_time_control("180+0") is True
    assert is_valid_pgn_time_control("60+0") is True
    assert is_valid_pgn_time_control("300+5") is True
    assert is_valid_pgn_time_control("40/7200:900+30") is True
    assert is_valid_pgn_time_control("40/7200:600+0") is True


def test_time_control_rejects_zero_base_or_invalid():
    assert is_valid_pgn_time_control("0+0") is False
    assert is_valid_pgn_time_control("0+1") is False
    assert is_valid_pgn_time_control("0/600") is False
    assert is_valid_pgn_time_control("*0") is False
    assert is_valid_pgn_time_control("0") is False


# ------------------------------------------------------------------------------
# Threat semantic transitions tests (H3)
# ------------------------------------------------------------------------------


def test_threat_delta_detects_check_to_mate_transition():
    # 1. f3 e5
    b_before = chess.Board()
    b_before.push_san("f3")
    b_before.push_san("e5")
    # White plays 2. g4 (blunders mate in 1)
    b_after = b_before.copy(stack=True)
    b_after.push_san("g4")

    delta = critical_forcing_threat_delta(b_before, b_after)
    assert "strengthened_opponent_forcing_moves" in delta
    assert "forcing_move_semantic_transitions" in delta

    strengthened_ucis = [m.uci for m in delta["strengthened_opponent_forcing_moves"]]
    assert "d8h4" in strengthened_ucis

    transitions = delta["forcing_move_semantic_transitions"]
    q_trans = next((t for t in transitions if t["uci"] == "d8h4"), None)
    assert q_trans is not None
    assert q_trans["transition"] == "check_to_mate"
    assert q_trans["before"]["is_check"] is True
    assert q_trans["before"]["is_mate"] is False
    assert q_trans["after"]["is_check"] is True
    assert q_trans["after"]["is_mate"] is True
    assert "OPPONENT_FORCING_MOVE_STRENGTHENED" in delta["signatures"]


# ------------------------------------------------------------------------------
# C1: Mating move root action semantics
# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c1_root_action_mating_move_is_play_move():
    # Position after 1. f3 e5 2. g4
    fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
    res = await top_moves(fen=fen, n=1, depth=10)
    assert res.status == "active"
    assert res.recommended_action == "play_move"
    assert res.returned_n >= 1

    mate_cand = res.result[0]
    assert mate_cand.candidate_san == "Qh4#"
    assert mate_cand.root_candidate_action == "play_move"
    assert mate_cand.recommended_action == "play_move"
    assert mate_cand.executable_move == "d8h4"
    assert mate_cand.best_action_obj is not None
    assert mate_cand.best_action_obj["type"] == "play_move"
    assert mate_cand.best_action_obj["move"]["uci"] == "d8h4"
    assert mate_cand.post_terminal_status == "checkmate"
    assert mate_cand.post_position is not None
    assert mate_cand.post_position["status"] == "checkmate"
    assert mate_cand.post_position["recommended_action"] == "game_over"


# ------------------------------------------------------------------------------
# C2: 50-move claim reasons preserved in compact and minimal
# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c2_claim_reasons_preserved_in_compact_and_minimal():
    # Valid nonterminal position at halfmove clock 100
    fen = "8/8/8/8/8/4k3/8/4K2R w - - 100 51"
    compact_res = await evaluate_position(fen=fen, verbosity="compact")
    assert compact_res.can_claim_now is True
    assert compact_res.can_claim_draw is True
    assert "fifty_moves" in compact_res.claim_reasons_now
    assert "fifty_moves" in compact_res.claim_reasons

    minimal_res = await evaluate_position(fen=fen, verbosity="minimal")
    assert minimal_res.can_claim_now is True
    assert minimal_res.can_claim_draw is True
    assert "fifty_moves" in minimal_res.claim_reasons_now
    assert "fifty_moves" in minimal_res.claim_reasons


# ------------------------------------------------------------------------------
# C3: Missed draw claim in classify_move
# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c3_classify_move_missed_claim_action_class(monkeypatch):
    class _MockBestPool:
        name = "MockBestPool"
        engine_version = "MockBestPool"

        async def evaluate(self, board, *, depth=14, root_moves=None):
            return Eval(cp=0, best_move="h1f1", pv=["h1f1"], depth=depth)

        async def close(self):
            pass

    monkeypatch.setattr(server_module, "_analyzer_pool", _MockBestPool())
    # Position where 50-move claim is available now
    fen = "8/8/8/8/8/4k3/8/4K2R w - - 100 51"
    # White plays legal engine-best move instead of claiming draw
    res = await classify_move(fen=fen, move="Rf1", action_type="play_move")
    assert res.move_class.value == "best"
    assert res.is_engine_best is True
    assert res.best_action == "claim_draw"
    assert res.is_best_action is False
    assert res.missed_draw_claim is True
    assert res.action_class == "missed_rule_action"
    assert res.action_quality_class == "missed_rule_action"
    assert res.move_quality_class == "best"


# ------------------------------------------------------------------------------
# H1: include_moves deduplication with engine best
# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_h1_include_moves_engine_best_preserved():
    res = await top_moves(fen=chess.STARTING_FEN, n=1, depth=10, include_moves=["e4"])
    assert res.returned_n >= 1
    assert any(c.best_move.lower() == "e2e4" for c in res.result)
    assert any(
        a.get("move", {}).get("uci", "").lower() == "e2e4"
        for a in res.legal_actions
        if a.get("type") == "play_move"
    )
    assert res.included_move_count == 1


# ------------------------------------------------------------------------------
# H2: Coach mode mate strongest reply
# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_h2_coach_mode_mate_strongest_reply():
    pgn = "1. f3 e5 2. g4 Qh4#"
    res = await analyze_game(pgn=pgn, detail="coach", perspective="white")
    assert res.coaching is not None
    assert len(res.coaching.critical_moments) > 0

    g4_moment = next((m for m in res.coaching.critical_moments if m.san == "g4"), None)
    assert g4_moment is not None
    assert g4_moment.strongest_reply_uci == "d8h4"
    assert g4_moment.strongest_reply_san == "Qh4#"
    assert g4_moment.strongest_reply_is_check is True
    assert g4_moment.strongest_reply_is_mate_in_one is True


# ------------------------------------------------------------------------------
# M3: Terminal top_moves preserves include_moves diagnostics
# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m3_terminal_top_moves_preserves_include_moves():
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    res = await top_moves(fen=fen, n=1, include_moves=["Kh7"])
    assert res.status == "checkmate"
    assert res.requested_include_moves == ["Kh7"]
    assert res.ignored_include_moves == ["Kh7"]
    assert res.ignored_include_moves_reason == "terminal_position"


# ------------------------------------------------------------------------------
# H4: Intended draw claim serialization consistency
# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_h4_intended_draw_claim_consistency():
    # White king alone vs Black king + rook at halfmove 99; Kh2 claims 50-move draw
    fen = "8/8/8/8/8/8/r4k2/7K w - - 99 50"
    full_ev = await evaluate_position(fen=fen, depth=10, verbosity="full")
    compact_ev = await evaluate_position(fen=fen, depth=10, verbosity="compact")
    assert full_ev.can_claim_draw is True
    assert full_ev.can_claim_with_intended_move is True
    assert full_ev.recommended_action == "claim_draw_with_intended_move"
    assert full_ev.claim_move is not None
    assert full_ev.claim_move_san is not None
    assert full_ev.claim_move_uci is not None

    assert compact_ev.can_claim_draw is True
    assert compact_ev.can_claim_with_intended_move is True
    assert compact_ev.recommended_action == "claim_draw_with_intended_move"
    assert compact_ev.claim_move == full_ev.claim_move
    assert compact_ev.claim_move_san == full_ev.claim_move_san
    assert compact_ev.claim_move_uci == full_ev.claim_move_uci


# ------------------------------------------------------------------------------
# M1: Tactical mechanism relevance scoring & noise gating
# ------------------------------------------------------------------------------


def test_m1_opening_pawn_discovered_attack_filtered_from_presentation():
    from mcp_server.analysis.position_integrity import build_rich_tactical_snapshot

    b = chess.Board()
    snap = build_rich_tactical_snapshot(b)

    # All mechanism candidates must have a valid relevance field
    valid_relevance = {"tactically_actionable", "engine_supported", "pure_geometry"}
    for m in snap.mechanism_candidates:
        assert m.relevance in valid_relevance

    # Opening pawn moves (like e2e4 discovering Qd1/Bf1) must NOT be in presentation_mechanisms
    for m in snap.presentation_mechanisms:
        if m.mechanism == "discovered_attack_candidate":
            assert m.relevance != "pure_geometry"
            assert "pawn" not in (m.actor or "") or m.evidence.get("is_check") or m.evidence.get("is_capture")


# ------------------------------------------------------------------------------
# M2: Verbosity enum aliases synchronization
# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m2_verbosity_aliases_accepted():
    fen = chess.STARTING_FEN
    for verbosity in ("full", "compact", "minimal", "min", "standard", "default"):
        res = await evaluate_position(fen=fen, depth=6, verbosity=verbosity)
        assert res is not None

    for verbosity in ("full", "compact", "minimal", "min", "standard", "default"):
        res_top = await top_moves(fen=fen, n=1, depth=6, verbosity=verbosity)
        assert res_top is not None
