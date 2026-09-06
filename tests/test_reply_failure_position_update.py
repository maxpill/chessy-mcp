from __future__ import annotations

import chess
import pytest

from core.engines.types import MoveClass
from mcp_server.analysis.forensics import enrich_move_analysis
from mcp_server.models import MCPEval, MCPMoveAnalysis


class _UnusedPool:
    async def evaluate(self, board: chess.Board, *, depth: int, root_moves=None):
        raise AssertionError("coach-mode evidence in these tests must not add engine searches")


def _analysis(
    *,
    board: chess.Board,
    played: chess.Move,
    move_class: MoveClass,
    cp_before: int = 0,
    cp_after: int = 0,
    reply: str | None = None,
    pv: list[str] | None = None,
    cpl: int | None = 0,
    wdl_before: tuple[int, int, int] | None = None,
    wdl_after: tuple[int, int, int] | None = None,
) -> MCPMoveAnalysis:
    return MCPMoveAnalysis(
        played=played.uci(),
        played_san=board.san(played),
        move_class=move_class,
        centipawn_loss=cpl,
        eval_before=MCPEval(
            cp=cp_before,
            best_move=played.uci(),
            pv=[played.uci()],
            depth=12,
            searched_depth=12,
            wdl=wdl_before,
        ),
        eval_after=MCPEval(
            cp=cp_after,
            best_move=reply,
            pv=list(pv or ([] if reply is None else [reply])),
            depth=12,
            searched_depth=12,
            wdl=wdl_after,
        ),
        best_move_san=board.san(played),
    )


@pytest.mark.asyncio
async def test_reply_failure_profile_marks_immediate_forcing_material_realization() -> None:
    board = chess.Board("4k3/4r3/8/8/8/8/4Q3/4K3 w - - 0 1")
    played = chess.Move.from_uci("e2e7")
    result = _analysis(
        board=board,
        played=played,
        move_class=MoveClass.BLUNDER,
        cp_after=-900,
        reply="e8e7",
        pv=["e8e7"],
        cpl=900,
    )

    enriched = await enrich_move_analysis(
        result,
        board_before=board,
        played_move=played,
        pool=_UnusedPool(),
        depth=12,
        detail="coach",
    )

    assert enriched.forensics is not None
    profile = next(
        item
        for item in enriched.forensics.mechanism_evidence
        if item.get("mechanism") == "reply_failure_profile"
    )
    assert profile["reply"] == "Kxe7"
    assert profile["reply_type"] == "capture"
    assert profile["opponent_first_pv_move_forcing"] is True
    assert profile["immediate_material_swing_for_mover_cp"] == -900
    assert profile["first_irreversible_event_ply"] == 1
    assert profile["realization_kind"] == "immediate_material"
    assert "IMMEDIATE_MATERIAL_PUNISHMENT" in enriched.forensics.evidence_signatures
    assert "MISSED_FORCING_REPLY_CANDIDATE" in enriched.forensics.evidence_signatures
    assert "share_of_total_loss_after_first_reply" not in profile


@pytest.mark.asyncio
async def test_previous_opponent_move_can_be_flagged_as_urgent_change_left_unaddressed() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("d7d5")
    played = chess.Move.from_uci("a2a3")
    result = _analysis(
        board=board,
        played=played,
        move_class=MoveClass.GOOD,
        reply="d5e4",
        pv=["d5e4"],
        cpl=20,
    )

    enriched = await enrich_move_analysis(
        result,
        board_before=board,
        played_move=played,
        pool=_UnusedPool(),
        depth=12,
        detail="coach",
    )

    assert enriched.forensics is not None
    update = next(
        item
        for item in enriched.forensics.mechanism_evidence
        if item.get("mechanism") == "position_update_after_opponent_move"
    )
    assert update["history_available"] is True
    assert update["opponent_move_uci"] == "d7d5"
    assert update["opponent_move_san"] == "d5"
    assert "white_pawn@e4" in update["newly_attacked_user_pieces"]
    assert "white_pawn@e4" in update["newly_exposed_user_pieces"]
    assert update["opponent_move_created_urgent_change"] is True
    assert update["played_move_addresses_change"] is False
    assert "white_pawn@e4" in update["unresolved_urgent_targets_after_played_move"]
    assert "FAILED_POSITION_UPDATE_CANDIDATE" in enriched.forensics.evidence_signatures


@pytest.mark.asyncio
async def test_position_update_candidate_clears_when_played_move_resolves_new_exposure() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("d7d5")
    played = chess.Move.from_uci("e4d5")
    result = _analysis(
        board=board,
        played=played,
        move_class=MoveClass.GOOD,
        reply=None,
        pv=[],
        cpl=0,
    )

    enriched = await enrich_move_analysis(
        result,
        board_before=board,
        played_move=played,
        pool=_UnusedPool(),
        depth=12,
        detail="coach",
    )

    assert enriched.forensics is not None
    update = next(
        item
        for item in enriched.forensics.mechanism_evidence
        if item.get("mechanism") == "position_update_after_opponent_move"
    )
    assert update["opponent_move_created_urgent_change"] is True
    assert update["played_move_addresses_change"] is True
    assert update["unresolved_urgent_targets_after_played_move"] == []
    assert "FAILED_POSITION_UPDATE_CANDIDATE" not in enriched.forensics.evidence_signatures


@pytest.mark.asyncio
async def test_naked_fen_does_not_invent_previous_move_process_evidence() -> None:
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    played = chess.Move.from_uci("a2a3")
    result = _analysis(
        board=board,
        played=played,
        move_class=MoveClass.GOOD,
        reply="d5e4",
        pv=["d5e4"],
        cpl=20,
    )

    enriched = await enrich_move_analysis(
        result,
        board_before=board,
        played_move=played,
        pool=_UnusedPool(),
        depth=12,
        detail="coach",
    )

    assert enriched.forensics is not None
    update = next(
        item
        for item in enriched.forensics.mechanism_evidence
        if item.get("mechanism") == "position_update_after_opponent_move"
    )
    assert update["history_available"] is False
    assert "FAILED_POSITION_UPDATE_CANDIDATE" not in enriched.forensics.evidence_signatures


@pytest.mark.asyncio
async def test_stability_marks_small_good_wdl_loss_as_practically_equivalent() -> None:
    board = chess.Board()
    played = chess.Move.from_uci("e2e4")
    result = _analysis(
        board=board,
        played=played,
        move_class=MoveClass.GOOD,
        cp_before=20,
        cp_after=10,
        reply="e7e5",
        pv=["e7e5"],
        cpl=10,
        wdl_before=(400, 300, 300),
        wdl_after=(395, 310, 295),
    )

    enriched = await enrich_move_analysis(
        result,
        board_before=board,
        played_move=played,
        pool=_UnusedPool(),
        depth=12,
        detail="coach",
    )

    assert enriched.forensics is not None
    stability = enriched.forensics.stability
    assert stability["wdl_loss_percentage_points"] == pytest.approx(0.0)
    assert stability["forcing_punishment"] is False
    assert stability["practical_equivalent"] is True
    assert stability["coach_priority"] == "negligible"
