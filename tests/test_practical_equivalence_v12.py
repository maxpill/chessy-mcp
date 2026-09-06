from __future__ import annotations

import chess

from core.engines.types import MoveClass
from mcp_server.analysis.forensics import (
    build_position_delta,
    build_position_fingerprint,
    build_tactical_snapshot,
)
from mcp_server.analysis.practical_equivalence import (
    apply_practical_equivalence,
    build_practical_equivalence_evidence,
)
from mcp_server.models import MCPEval
from mcp_server.models.forensics import (
    ForcedLineEvidence,
    ForensicEvidence,
    ForensicMoveAnalysis,
)


def _result(
    *,
    before_wdl: tuple[int, int, int] | None,
    after_wdl: tuple[int, int, int] | None,
    effective_loss: int,
    move_class: MoveClass = MoveClass.INACCURACY,
    before_mate: int | None = None,
    after_mate: int | None = None,
    signatures: list[str] | None = None,
    stability: dict | None = None,
) -> tuple[ForensicMoveAnalysis, chess.Board]:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    after = board.copy(stack=True)
    after.push(move)

    evidence = ForensicEvidence(
        detail="forensic",
        position_before=build_position_fingerprint(board),
        position_after_played=build_position_fingerprint(after),
        tactical_before=build_tactical_snapshot(board),
        tactical_after_played=build_tactical_snapshot(after),
        position_delta=build_position_delta(board, after),
        evidence_signatures=signatures or [],
        forced_line=ForcedLineEvidence(),
        stability=stability or {},
    )
    result = ForensicMoveAnalysis(
        played=move.uci(),
        played_san="e4",
        move_class=move_class,
        is_engine_best=False,
        is_best_engine_move=False,
        centipawn_loss=effective_loss,
        effective_loss=effective_loss,
        eval_before=MCPEval(
            cp=0,
            mate=before_mate,
            best_move="d2d4",
            wdl=before_wdl,
        ),
        eval_after=MCPEval(
            cp=-effective_loss,
            mate=after_mate,
            best_move="e7e5",
            wdl=after_wdl,
        ),
        action_type="play_move",
        played_outcome="active",
        best_outcome="active",
        forensics=evidence,
    )
    return result, board


def test_small_wdl_loss_can_be_practically_equivalent_even_outside_cp_fallback_band() -> None:
    result, board = _result(
        before_wdl=(400, 400, 200),
        after_wdl=(390, 410, 200),
        effective_loss=80,
    )

    practical = build_practical_equivalence_evidence(result, mover=board.turn)

    assert practical["wdl_loss_percentage_points"] == 0.5
    assert practical["effective_loss_cp"] == 80
    assert practical["practical_equivalent"] is True
    assert practical["status"] == "equivalent"
    assert practical["coach_priority"] == "negligible"
    assert practical["reason_codes"] == ["WDL_LOSS_WITHIN_EQUIVALENCE_BAND"]


def test_concrete_forcing_punishment_blocks_equivalence_despite_small_wdl_delta() -> None:
    result, board = _result(
        before_wdl=(400, 400, 200),
        after_wdl=(390, 410, 200),
        effective_loss=120,
        move_class=MoveClass.MISTAKE,
        signatures=["FORCING_REPLY_MATERIAL_LOSS_IMMEDIATE"],
    )

    practical = build_practical_equivalence_evidence(result, mover=board.turn)

    assert practical["practical_equivalent"] is False
    assert practical["status"] == "not_equivalent"
    assert practical["coach_priority"] == "high"
    assert practical["tactical_punishment_evidence"] is True
    assert practical["reason_codes"] == ["CONCRETE_FORCING_PUNISHMENT_EVIDENCE"]


def test_new_forced_mate_against_mover_is_never_practically_equivalent() -> None:
    result, board = _result(
        before_wdl=None,
        after_wdl=None,
        effective_loss=0,
        before_mate=None,
        after_mate=-2,
    )

    practical = build_practical_equivalence_evidence(result, mover=board.turn)

    assert practical["mate_deterioration_for_mover"] is True
    assert practical["practical_equivalent"] is False
    assert practical["coach_priority"] == "high"
    assert practical["reason_codes"] == ["MATE_STATUS_DETERIORATED"]


def test_wdl_gray_zone_stays_indeterminate_instead_of_fabricating_equivalence() -> None:
    result, board = _result(
        before_wdl=(500, 300, 200),
        after_wdl=(450, 360, 190),
        effective_loss=45,
    )

    practical = build_practical_equivalence_evidence(result, mover=board.turn)

    assert practical["wdl_loss_percentage_points"] == 2.0
    assert practical["practical_equivalent"] is True

    result2, board2 = _result(
        before_wdl=(500, 300, 200),
        after_wdl=(450, 320, 230),
        effective_loss=45,
    )
    practical2 = build_practical_equivalence_evidence(result2, mover=board2.turn)
    assert practical2["wdl_loss_percentage_points"] == 4.0
    assert practical2["practical_equivalent"] is None
    assert practical2["status"] == "indeterminate"
    assert practical2["coach_priority"] == "low"


def test_verified_stability_evidence_overrides_initial_coaching_priority_basis() -> None:
    result, board = _result(
        before_wdl=(600, 250, 150),
        after_wdl=(250, 300, 450),
        effective_loss=220,
        move_class=MoveClass.MISTAKE,
        stability={
            "verification_performed": True,
            "verified_wdl_loss_percentage_points": 0.7,
            "verified_effective_loss": 12,
            "verified_class": "good",
            "verified_mate_before": "no_mate",
            "verified_mate_after": "no_mate",
        },
    )

    enriched = apply_practical_equivalence(result, mover=board.turn)

    assert enriched.forensics is not None
    practical = enriched.forensics.stability["practical_equivalence"]
    assert practical["evidence_basis"] == "verified"
    assert practical["move_class_used"] == "good"
    assert practical["wdl_loss_percentage_points"] == 0.7
    assert practical["practical_equivalent"] is True
    assert enriched.forensics.stability["practical_equivalent"] is True
    assert enriched.forensics.stability["coach_priority"] == "negligible"
