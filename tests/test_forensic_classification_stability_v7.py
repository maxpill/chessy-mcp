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
from mcp_server.models.forensics import (
    ForcedLineEvidence,
    ForensicEvidence,
    ForensicMoveAnalysis,
)


def _forensic_result(*, move_class: MoveClass) -> tuple[ForensicMoveAnalysis, chess.Board, chess.Move]:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    after = board.copy(stack=True)
    after.push(move)

    eval_before = MCPEval(
        cp=0,
        best_move="d2d4",
        pv=["d2d4"],
        depth=20,
        searched_depth=20,
        wdl=(330, 340, 330),
    )
    eval_after = MCPEval(
        cp=-220,
        best_move="e7e5",
        pv=["e7e5"],
        depth=20,
        searched_depth=20,
        wdl=(150, 300, 550),
    )
    evidence = ForensicEvidence(
        detail="forensic",
        position_before=build_position_fingerprint(board),
        position_after_played=build_position_fingerprint(after),
        tactical_before=build_tactical_snapshot(board),
        tactical_after_played=build_tactical_snapshot(after),
        position_delta=build_position_delta(board, after),
        forced_line=ForcedLineEvidence(),
        stability={},
    )
    result = ForensicMoveAnalysis(
        played=move.uci(),
        played_san="e4",
        move_class=move_class,
        centipawn_loss=220 if move_class != MoveClass.BEST else 0,
        effective_loss=220 if move_class != MoveClass.BEST else 0,
        eval_before=eval_before,
        eval_after=eval_after,
        action_type="play_move",
        forensics=evidence,
    )
    return result, board, move


class _DepthCorrectingPool:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, int]] = []

    async def evaluate(self, board: chess.Board, *, depth: int) -> Eval:
        self.calls.append((board.turn, depth))
        if board.turn == chess.WHITE:
            return Eval(
                cp=0,
                best_move="e2e4",
                pv=["e2e4"],
                depth=depth,
                wdl=(340, 340, 320),
            )
        return Eval(
            cp=0,
            best_move="e7e5",
            pv=["e7e5"],
            depth=depth,
            wdl=(340, 340, 320),
        )


@pytest.mark.asyncio
async def test_forensic_stability_researches_error_and_records_changed_classification() -> None:
    result, board, move = _forensic_result(move_class=MoveClass.MISTAKE)
    pool = _DepthCorrectingPool()

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
    assert stability["verification_performed"] is True
    assert stability["verification_status"] == "escalated"
    assert stability["initial_depth"] == 20
    assert stability["initial_class"] == "mistake"
    assert stability["verified_class"] == "best"
    assert stability["class_stable"] is False
    assert stability["best_move_stable"] is False
    assert stability["verification_converged"] is True
    assert stability["verification_depth"] == 26
    assert stability["stable"] is False
    assert [depth for _turn, depth in pool.calls] == [24, 24, 26, 26]


@pytest.mark.asyncio
async def test_forensic_stability_does_not_spend_extra_search_on_best_move() -> None:
    result, board, move = _forensic_result(move_class=MoveClass.BEST)
    pool = _DepthCorrectingPool()

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
    assert stability["verification_performed"] is False
    assert stability["verification_status"] == "low_coaching_priority_class"
    assert stability["initial_class"] == "best"
    assert pool.calls == []
