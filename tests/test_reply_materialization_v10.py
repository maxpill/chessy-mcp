from __future__ import annotations

import chess
import pytest

from core.engines.types import MoveClass
from mcp_server.analysis.forensics import enrich_move_analysis
from mcp_server.analysis.reply_materialization import apply_reply_materialization_evidence
from mcp_server.models import MCPEval, MCPMoveAnalysis


class _UnusedPool:
    async def evaluate(self, board: chess.Board, *, depth: int, root_moves=None):
        raise AssertionError("coach-mode materialization evidence must not add engine searches")


def _analysis(
    *,
    board: chess.Board,
    played: chess.Move,
    reply: str,
    pv: list[str],
    move_class: MoveClass = MoveClass.BLUNDER,
    cpl: int = 300,
) -> MCPMoveAnalysis:
    return MCPMoveAnalysis(
        played=played.uci(),
        played_san=board.san(played),
        move_class=move_class,
        centipawn_loss=cpl,
        eval_before=MCPEval(
            cp=0,
            best_move=played.uci(),
            pv=[played.uci()],
            depth=12,
            searched_depth=12,
        ),
        eval_after=MCPEval(
            cp=-cpl,
            best_move=reply,
            pv=pv,
            depth=12,
            searched_depth=12,
        ),
        best_move_san=board.san(played),
    )


async def _enrich(
    board: chess.Board,
    played: chess.Move,
    result: MCPMoveAnalysis,
):
    base = await enrich_move_analysis(
        result,
        board_before=board,
        played_move=played,
        pool=_UnusedPool(),
        depth=12,
        detail="coach",
    )
    return apply_reply_materialization_evidence(base, board, played_move=played)


@pytest.mark.asyncio
async def test_forcing_check_can_materialize_loss_three_plies_later() -> None:
    board = chess.Board("3q2k1/8/8/8/8/8/PP6/6K1 w - - 0 1")
    played = chess.Move.from_uci("a2a3")
    result = _analysis(
        board=board,
        played=played,
        reply="d8b6",
        pv=["d8b6", "g1h1", "b6b2"],
        cpl=300,
    )

    enriched = await _enrich(board, played, result)

    assert enriched.forensics is not None
    profile = next(
        item
        for item in enriched.forensics.mechanism_evidence
        if item.get("mechanism") == "reply_failure_profile"
    )
    assert profile["reply"] == "Qb6+"
    assert profile["reply_type"] == "check"
    assert profile["returned_pv_starts_with_strongest_reply"] is True
    assert profile["immediate_material_swing_for_mover_cp"] == 0
    assert profile["first_material_loss_ply_for_mover"] == 3
    assert profile["loss_realized_within_plies"] == 3
    assert profile["first_material_loss_cp"] == -100
    assert [
        step["cumulative_material_change_for_mover_cp"]
        for step in profile["material_trajectory_for_mover"]
    ] == [0, 0, -100]
    assert "DELAYED_MATERIALIZATION_AFTER_FORCING_REPLY" in enriched.forensics.evidence_signatures
    assert "FORCING_REPLY_MATERIAL_LOSS_IMMEDIATE" not in enriched.forensics.evidence_signatures


@pytest.mark.asyncio
async def test_forcing_capture_marks_material_loss_on_first_reply() -> None:
    board = chess.Board("4k3/4r3/8/8/8/8/4Q3/4K3 w - - 0 1")
    played = chess.Move.from_uci("e2e7")
    result = _analysis(
        board=board,
        played=played,
        reply="e8e7",
        pv=["e8e7"],
        cpl=900,
    )

    enriched = await _enrich(board, played, result)

    assert enriched.forensics is not None
    profile = next(
        item
        for item in enriched.forensics.mechanism_evidence
        if item.get("mechanism") == "reply_failure_profile"
    )
    assert profile["first_material_loss_ply_for_mover"] == 1
    assert profile["loss_realized_within_plies"] == 1
    assert profile["final_material_change_for_mover_cp"] == -900
    assert "FORCING_REPLY_MATERIAL_LOSS_IMMEDIATE" in enriched.forensics.evidence_signatures
    assert "DELAYED_MATERIALIZATION_AFTER_FORCING_REPLY" not in enriched.forensics.evidence_signatures


@pytest.mark.asyncio
async def test_misaligned_returned_pv_does_not_invent_deeper_materialization() -> None:
    board = chess.Board("3q2k1/8/8/8/8/8/PP6/6K1 w - - 0 1")
    played = chess.Move.from_uci("a2a3")
    result = _analysis(
        board=board,
        played=played,
        reply="d8b6",
        pv=["d8a5"],
        cpl=300,
    )

    enriched = await _enrich(board, played, result)

    assert enriched.forensics is not None
    profile = next(
        item
        for item in enriched.forensics.mechanism_evidence
        if item.get("mechanism") == "reply_failure_profile"
    )
    assert profile["returned_pv_starts_with_strongest_reply"] is False
    assert profile["returned_line_plies_walked"] if "returned_line_plies_walked" in profile else True
    assert profile["first_material_loss_ply_for_mover"] is None
    assert len(profile["material_trajectory_for_mover"]) == 1
    assert profile["material_trajectory_for_mover"][0]["uci"] == "d8b6"
    assert "REPLY_PV_ALIGNMENT_UNAVAILABLE" in enriched.forensics.evidence_signatures
