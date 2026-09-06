from __future__ import annotations

import chess

from mcp_server.analysis.game_critical_forensics import (
    _failure_corpus,
    enrich_game_critical_forensics,
)
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


def _coaching(
    moment: CriticalMoment,
    final_board: chess.Board,
    *,
    perspective: str,
) -> GameCoachingEvidence:
    return GameCoachingEvidence(
        detail="forensic",
        perspective=perspective,  # type: ignore[arg-type]
        critical_moments=[moment],
        final_position=FinalPositionAssessment(
            perspective=perspective,  # type: ignore[arg-type]
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


def test_failed_immediate_mate_threat_becomes_cross_game_bucket() -> None:
    positions, moves = _line("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6")
    final_board = positions[-1]
    moment = CriticalMoment(
        ply=6,
        san="Nf6",
        uci=moves[5].uci(),
        side="black",
        move_class="blunder",
        effective_loss=900,
        eval_before_effective_cp=0,
        eval_after_effective_cp=-900,
        user_comment_raw="nie zobaczylem Qxf7#",
    )
    evals = [MCPEval(cp=0) for _ in positions]

    enriched = enrich_game_critical_forensics(
        _coaching(moment, final_board, perspective="black"),
        positions=positions,
        evals=evals,
    )

    critical = enriched.critical_moments[0]
    assert "FAILED_MATE_THREAT_UPDATE_CANDIDATE" in critical.evidence_signatures
    assert (
        "immediate_mate_threat_update_failure_candidate"
        in critical.failure_evidence_categories
    )
    assert enriched.failure_corpus is not None
    bucket = next(
        item
        for item in enriched.failure_corpus.buckets
        if item.category == "immediate_mate_threat_update_failure_candidate"
    )
    assert bucket.count == 1
    assert bucket.plies == [6]
    assert bucket.self_report_overlap_count == 1
    assert bucket.self_reported_plies == [6]
    assert bucket.supporting_signatures == ["FAILED_MATE_THREAT_UPDATE_CANDIDATE"]


def test_failure_corpus_normalizes_signatures_without_diagnosing_process() -> None:
    first = CriticalMoment(
        ply=12,
        san="e5",
        uci="e4e5",
        side="white",
        move_class="blunder",
        effective_loss=320,
        eval_before_effective_cp=20,
        eval_after_effective_cp=-300,
        user_comment_raw="nie widzialem odpowiedzi",
        evidence_signatures=[
            "MISSED_FORCING_REPLY_CANDIDATE",
            "PAWN_MOVE_FORCING_PUNISHMENT",
        ],
        failure_evidence_categories=[
            "missed_forcing_reply_candidate",
            "pawn_move_forcing_punishment",
        ],
    )
    second = CriticalMoment(
        ply=20,
        san="Re1",
        uci="f1e1",
        side="white",
        move_class="mistake",
        effective_loss=180,
        eval_before_effective_cp=-20,
        eval_after_effective_cp=-200,
        evidence_signatures=[],
        failure_evidence_categories=[],
    )

    corpus = _failure_corpus([first, second])

    assert corpus.major_error_critical_moments == 2
    assert corpus.categorized_critical_moments == 1
    assert corpus.uncategorized_major_error_plies == [20]
    forcing = next(
        item
        for item in corpus.buckets
        if item.category == "missed_forcing_reply_candidate"
    )
    assert forcing.count == 1
    assert forcing.plies == [12]
    assert forcing.self_report_overlap_count == 1
    assert "not a diagnosis" in forcing.inference_boundary
