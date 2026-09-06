from __future__ import annotations

import chess

from mcp_server.analysis.game_critical_forensics import enrich_game_critical_forensics
from mcp_server.analysis.game_threat_forensics import critical_forcing_threat_delta
from mcp_server.models import MCPEval
from mcp_server.models.game_coaching import (
    CriticalMoment,
    FinalPositionAssessment,
    GameCoachingEvidence,
)


ROOT_FEN = "6k1/4r3/8/8/8/8/4Q3/6K1 w - - 0 1"


def _after(board: chess.Board, uci: str) -> chess.Board:
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    post = board.copy(stack=True)
    post.push(move)
    return post


def test_critical_delta_separates_resolved_baseline_from_new_reply() -> None:
    before = chess.Board(ROOT_FEN)
    after = _after(before, "e2f2")

    delta = critical_forcing_threat_delta(before, after)

    assert delta["baseline_available"] is True
    baseline = {item.uci for item in delta["opponent_forcing_threat_candidates_if_pass_before"]}
    actual = {item.uci for item in delta["opponent_forcing_moves_after_played"]}
    newly = {item.uci for item in delta["newly_enabled_opponent_forcing_moves_after_played"]}
    resolved = {item.uci for item in delta["resolved_opponent_forcing_threat_candidates"]}

    assert "e7e2" in baseline
    assert "e7e1" in actual
    assert "e7e1" in newly
    assert "e7e2" in resolved
    assert "NEW_OPPONENT_FORCING_REPLY_AFTER_MOVE" in delta["signatures"]
    assert "NEW_OPPONENT_CHECK_AFTER_MOVE" in delta["signatures"]
    assert "RESOLVED_OPPONENT_FORCING_THREAT_CANDIDATE" in delta["signatures"]


def test_persistent_exact_forcing_threat_is_failed_update_candidate() -> None:
    before = chess.Board(ROOT_FEN)
    after = _after(before, "g1h2")

    delta = critical_forcing_threat_delta(before, after)

    unresolved = {
        item.uci for item in delta["unresolved_exact_opponent_forcing_threat_candidates"]
    }
    assert "e7e2" in unresolved
    assert "FAILED_FORCING_THREAT_UPDATE_CANDIDATE" in delta["signatures"]


def test_check_position_refuses_null_baseline_but_keeps_actual_replies() -> None:
    before = chess.Board("k7/8/8/8/8/8/6r1/6K1 w - - 0 1")
    assert before.is_check()
    after = _after(before, "g1f1")

    delta = critical_forcing_threat_delta(before, after)

    assert delta["baseline_available"] is False
    assert delta["baseline_reason"] == "player_in_check_before_move"
    assert delta["newly_enabled_opponent_forcing_moves_after_played"] == []
    assert "NEW_OPPONENT_FORCING_REPLY_AFTER_MOVE" not in delta["signatures"]


def test_full_game_critical_moment_populates_threat_fields_and_corpus() -> None:
    before = chess.Board(ROOT_FEN)
    after = _after(before, "g1h2")
    moment = CriticalMoment(
        ply=1,
        san="Kh2",
        uci="g1h2",
        side="white",
        move_class="blunder",
        effective_loss=500,
        eval_before_effective_cp=0,
        eval_after_effective_cp=-500,
        strongest_reply_uci="e7e2",
        strongest_reply_san="Rxe2+",
        strongest_reply_is_check=True,
        strongest_reply_is_capture=True,
        evidence_signatures=["MISSED_FORCING_REPLY_CANDIDATE"],
    )
    coaching = GameCoachingEvidence(
        detail="forensic",
        perspective="white",
        critical_moments=[moment],
        final_position=FinalPositionAssessment(
            perspective="white",
            position_terminal_by_rules=False,
            effective_cp=-500,
            side_to_move="black",
            legal_move_count=after.legal_moves.count(),
            defensive_resources_exist=True,
        ),
        scan_depth=18,
    )
    evals = [MCPEval(cp=0), MCPEval(cp=-500, best_move="e7e2", pv=["e7e2"])]

    enriched = enrich_game_critical_forensics(
        coaching,
        positions=[before, after],
        evals=evals,
    )

    critical = enriched.critical_moments[0]
    assert critical.opponent_forcing_threat_baseline_available is True
    assert "e7e2" in {item.uci for item in critical.opponent_forcing_moves_after_played}
    assert "FAILED_FORCING_THREAT_UPDATE_CANDIDATE" in critical.evidence_signatures
    assert "failed_forcing_threat_update_candidate" in critical.failure_evidence_categories
    assert critical.causal_trace is not None
    assert "opponent_forcing_threat_delta" in critical.causal_trace
    assert enriched.failure_corpus is not None
    bucket = next(
        item
        for item in enriched.failure_corpus.buckets
        if item.category == "failed_forcing_threat_update_candidate"
    )
    assert bucket.plies == [1]
    assert "FAILED_FORCING_THREAT_UPDATE_CANDIDATE" in bucket.supporting_signatures
