from __future__ import annotations

from collections import Counter

from mcp_server.analysis.game_analyzer import _finalize_coaching_evidence
from mcp_server.models.game_coaching import (
    CriticalMoment,
    FinalPositionAssessment,
    GameCoachingEvidence,
)


PGN = """[Event \"Finalize\"]
[Result \"1-0\"]
[Termination \"Black resigned\"]

1. e4 e5 1-0
"""


def _coaching() -> GameCoachingEvidence:
    final = FinalPositionAssessment(
        perspective="white",
        position_terminal_by_rules=False,
        effective_cp=180,
        side_to_move="white",
        legal_move_count=29,
        best_move_uci="g1f3",
        best_move_san="Nf3",
        defensive_resources_exist=True,
    )
    moment = CriticalMoment(
        ply=2,
        san="e5",
        uci="e7e5",
        side="black",
        move_class="mistake",
        eval_before_effective_cp=0,
        eval_after_effective_cp=180,
        reasons=["largest_error", "player_self_report"],
        user_comment_raw="I missed the reply",
        evidence_signatures=[
            "FORCING_CAPTURE_REPLY",
            "MISSED_FORCING_REPLY_CANDIDATE",
        ],
    )
    return GameCoachingEvidence(
        detail="forensic",
        perspective="white",
        critical_moments=[moment],
        final_position=final,
        scan_depth=18,
    )


def test_finalize_coaching_attaches_termination_and_stable_aggregates() -> None:
    result = _finalize_coaching_evidence(PGN, _coaching())

    assert result.termination is not None
    assert result.termination.status == "explicit_resignation"
    assert result.termination.objectively_forced is False
    assert result.critical_evidence_signature_counts == {
        "FORCING_CAPTURE_REPLY": 1,
        "MISSED_FORCING_REPLY_CANDIDATE": 1,
    }
    assert result.critical_reason_counts == {
        "largest_error": 1,
        "player_self_report": 1,
    }
    assert result.self_reported_critical_plies == [2]


def test_aggregates_are_equivalent_to_counter_semantics() -> None:
    result = _finalize_coaching_evidence(PGN, _coaching())
    expected = Counter(result.critical_moments[0].evidence_signatures)
    assert result.critical_evidence_signature_counts == dict(sorted(expected.items()))
