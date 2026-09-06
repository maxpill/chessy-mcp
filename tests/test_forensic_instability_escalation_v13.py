from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval, MoveClass
from mcp_server.analysis.classification_stability import verify_forensic_classification_stability
from mcp_server.analysis.forensics import (
    build_position_delta,
    build_position_fingerprint,
    build_tactical_snapshot,
)
from mcp_server.models import MCPEval
from mcp_server.models.forensics import ForcedLineEvidence, ForensicEvidence, ForensicMoveAnalysis


def _result(*, tactical: bool = False) -> tuple[ForensicMoveAnalysis, chess.Board, chess.Move]:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    after = board.copy(stack=True)
    after.push(move)

    eval_before = MCPEval(
        cp=0,
        best_move="d2d4",
        pv=["d2d4", "d7d5", "c2c4"],
        depth=20,
        searched_depth=20,
        wdl=(500, 0, 500),
    )
    eval_after = MCPEval(
        cp=-220,
        best_move="e7e5",
        pv=["e7e5", "g1f3", "b8c6"],
        depth=20,
        searched_depth=20,
        wdl=(250, 0, 750),
    )
    evidence = ForensicEvidence(
        detail="forensic",
        position_before=build_position_fingerprint(board),
        position_after_played=build_position_fingerprint(after),
        tactical_before=build_tactical_snapshot(board),
        tactical_after_played=build_tactical_snapshot(after),
        position_delta=build_position_delta(board, after),
        evidence_signatures=(
            ["FORCING_REPLY_MATERIAL_LOSS_IMMEDIATE"] if tactical else []
        ),
        forced_line=ForcedLineEvidence(),
        stability={},
    )
    result = ForensicMoveAnalysis(
        played=move.uci(),
        played_san="e4",
        move_class=MoveClass.MISTAKE,
        centipawn_loss=220,
        effective_loss=220,
        eval_before=eval_before,
        eval_after=eval_after,
        action_type="play_move",
        forensics=evidence,
    )
    return result, board, move


class _StableClassShiftedWDLPool:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, int]] = []

    async def evaluate(self, board: chess.Board, *, depth: int) -> Eval:
        self.calls.append((board.turn, depth))
        if board.turn == chess.WHITE:
            return Eval(
                cp=0,
                best_move="d2d4",
                pv=["d2d4", "d7d5", "c2c4"],
                depth=depth,
                wdl=(500, 0, 500),
            )
        return Eval(
            cp=-220,
            best_move="e7e5",
            pv=["e7e5", "g1f3", "b8c6"],
            depth=depth,
            wdl=(450, 0, 550),
        )


class _StableNumericChangedPVPool:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, int]] = []

    async def evaluate(self, board: chess.Board, *, depth: int) -> Eval:
        self.calls.append((board.turn, depth))
        if board.turn == chess.WHITE:
            return Eval(
                cp=0,
                best_move="d2d4",
                pv=["d2d4", "g8f6", "c2c4"],
                depth=depth,
                wdl=(500, 0, 500),
            )
        return Eval(
            cp=-220,
            best_move="e7e5",
            pv=["e7e5", "f2f4", "b8c6"],
            depth=depth,
            wdl=(250, 0, 750),
        )


@pytest.mark.asyncio
async def test_large_wdl_shift_escalates_even_when_class_best_and_mate_are_stable() -> None:
    result, board, move = _result()
    pool = _StableClassShiftedWDLPool()

    verified = await verify_forensic_classification_stability(
        result,
        board,
        played_move=move,
        pool=pool,
        depth=20,
        history_complete="incomplete",
    )

    assert verified.forensics is not None
    stability = verified.forensics.stability
    assert stability["verification_status"] == "escalated"
    assert stability["class_stable"] is True
    assert stability["best_move_stable"] is True
    assert stability["mate_status_stable"] is True
    assert "wdl_loss_shift" in stability["first_verification_disagreement_reasons"]
    assert stability["evaluation_stability_basis"] == "wdl_loss_percentage_points"
    assert stability["wdl_loss_shift_percentage_points"] == 20.0
    assert stability["verification_converged"] is True
    assert stability["stable"] is False
    assert [depth for _turn, depth in pool.calls] == [24, 24, 26, 26]


@pytest.mark.asyncio
async def test_tactical_pv_prefix_drift_escalates_without_class_or_numeric_change() -> None:
    result, board, move = _result(tactical=True)
    pool = _StableNumericChangedPVPool()

    verified = await verify_forensic_classification_stability(
        result,
        board,
        played_move=move,
        pool=pool,
        depth=20,
        history_complete="incomplete",
    )

    assert verified.forensics is not None
    stability = verified.forensics.stability
    assert stability["verification_status"] == "escalated"
    assert stability["class_stable"] is True
    assert stability["evaluation_magnitude_stable"] is True
    assert stability["tactical_context_for_pv_stability"] is True
    assert stability["tactical_pv_stable"] is False
    assert "tactical_pv_prefix_changed" in stability["first_verification_disagreement_reasons"]
    assert stability["verification_converged"] is True
    assert [depth for _turn, depth in pool.calls] == [24, 24, 26, 26]


@pytest.mark.asyncio
async def test_quiet_pv_drift_does_not_pay_escalation_cost() -> None:
    result, board, move = _result(tactical=False)
    pool = _StableNumericChangedPVPool()

    verified = await verify_forensic_classification_stability(
        result,
        board,
        played_move=move,
        pool=pool,
        depth=20,
        history_complete="incomplete",
    )

    assert verified.forensics is not None
    stability = verified.forensics.stability
    assert stability["verification_status"] == "verified"
    assert stability["tactical_context_for_pv_stability"] is False
    assert stability["tactical_pv_stable"] is None
    assert stability["first_verification_disagreement_reasons"] == []
    assert stability["escalation_depth"] is None
    assert stability["stable"] is True
    assert [depth for _turn, depth in pool.calls] == [24, 24]
