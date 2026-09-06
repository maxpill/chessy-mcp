from __future__ import annotations

import chess

from mcp_server.analysis.game_analyzer import _finalize_coaching_evidence
from mcp_server.analysis.game_critical_forensics import enrich_game_critical_forensics
from mcp_server.models import MCPEval
from mcp_server.models.game_coaching import (
    CriticalMoment,
    FinalPositionAssessment,
    GameCoachingEvidence,
)


def _line(*sans: str) -> tuple[list[chess.Board], list[chess.Move]]:
    board = chess.Board()
    positions = [board.copy(stack=True)]
    moves: list[chess.Move] = []
    for san in sans:
        move = board.parse_san(san)
        moves.append(move)
        board.push(move)
        positions.append(board.copy(stack=True))
    return positions, moves


def _coaching(moment: CriticalMoment, final_board: chess.Board) -> GameCoachingEvidence:
    return GameCoachingEvidence(
        detail="forensic",
        perspective="white",
        critical_moments=[moment],
        final_position=FinalPositionAssessment(
            perspective="white",
            position_terminal_by_rules=final_board.is_game_over(claim_draw=False),
            checkmate=final_board.is_checkmate(),
            stalemate=final_board.is_stalemate(),
            forced_mate=False,
            effective_cp=-500,
            side_to_move="white" if final_board.turn == chess.WHITE else "black",
            legal_move_count=final_board.legal_moves.count(),
            defensive_resources_exist=(
                not final_board.is_game_over(claim_draw=False)
                and final_board.legal_moves.count() > 0
            ),
        ),
        scan_depth=18,
    )


def test_critical_move_allowing_mate_gets_exact_mate_and_causal_trace() -> None:
    positions, moves = _line("f3", "e5", "g4")
    final_board = positions[-1]
    moment = CriticalMoment(
        ply=3,
        san="g4",
        uci=moves[2].uci(),
        side="white",
        move_class="blunder",
        effective_loss=1000,
        eval_before_effective_cp=0,
        eval_after_effective_cp=-100000,
        strongest_reply_uci="d8h4",
        strongest_reply_san="Qh4#",
        strongest_reply_is_check=True,
        strongest_reply_is_capture=False,
    )
    evals = [MCPEval(cp=0) for _ in positions]
    evals[3] = MCPEval(mate=-1, best_move="d8h4", pv=["d8h4"])

    enriched = enrich_game_critical_forensics(
        _coaching(moment, final_board),
        positions=positions,
        evals=evals,
    )

    critical = enriched.critical_moments[0]
    assert critical.strongest_reply_is_mate_in_one is True
    assert any(
        item["uci"] == "d8h4"
        for item in critical.opponent_mate_in_one_moves_after_played
    )
    assert "OPPONENT_MATE_IN_ONE_AFTER_MOVE" in critical.evidence_signatures
    assert "STRONGEST_REPLY_IS_MATE_IN_ONE" in critical.evidence_signatures
    assert critical.causal_trace is not None
    assert critical.causal_trace["plies_traced"] == 1
    assert critical.causal_trace["steps"][0]["san"] == "Qh4#"
    assert "CRITICAL_CAUSAL_POSITION_DELTA_TRACE_AVAILABLE" in critical.evidence_signatures


def test_critical_move_can_record_missed_mate_without_psychological_claim() -> None:
    positions, moves = _line("f3", "e5", "g4", "Nc6")
    final_board = positions[-1]
    moment = CriticalMoment(
        ply=4,
        san="Nc6",
        uci=moves[3].uci(),
        side="black",
        move_class="mistake",
        effective_loss=300,
        eval_before_effective_cp=-500,
        eval_after_effective_cp=-100,
    )
    coaching = _coaching(moment, final_board).model_copy(update={"perspective": "black"})
    evals = [MCPEval(cp=0) for _ in positions]

    enriched = enrich_game_critical_forensics(
        coaching,
        positions=positions,
        evals=evals,
    )

    critical = enriched.critical_moments[0]
    assert critical.played_move_was_mate_in_one is False
    assert any(item["uci"] == "d8h4" for item in critical.mate_in_one_moves_before)
    assert "MATE_IN_ONE_AVAILABLE_BEFORE_MOVE" in critical.evidence_signatures
    assert "MISSED_MATE_IN_ONE_CANDIDATE" in critical.evidence_signatures
    assert "proof" not in " ".join(critical.evidence_signatures).lower()


def test_finalization_aggregates_new_critical_signatures_for_cross_game_corpus() -> None:
    positions, moves = _line("f3", "e5", "g4")
    final_board = positions[-1]
    moment = CriticalMoment(
        ply=3,
        san="g4",
        uci=moves[2].uci(),
        side="white",
        move_class="blunder",
        effective_loss=1000,
        eval_before_effective_cp=0,
        eval_after_effective_cp=-100000,
        strongest_reply_uci="d8h4",
        strongest_reply_san="Qh4#",
        strongest_reply_is_check=True,
    )
    evals = [MCPEval(cp=0) for _ in positions]
    evals[3] = MCPEval(mate=-1, best_move="d8h4", pv=["d8h4"])
    enriched = enrich_game_critical_forensics(
        _coaching(moment, final_board),
        positions=positions,
        evals=evals,
    )

    finalized = _finalize_coaching_evidence(
        "[Result \"*\"]\n\n1. f3 e5 2. g4 *",
        enriched,
    )

    assert finalized.critical_evidence_signature_counts[
        "OPPONENT_MATE_IN_ONE_AFTER_MOVE"
    ] == 1
    assert finalized.critical_evidence_signature_counts[
        "CRITICAL_CAUSAL_POSITION_DELTA_TRACE_AVAILABLE"
    ] == 1
