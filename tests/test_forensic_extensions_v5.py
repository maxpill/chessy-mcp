from __future__ import annotations

import chess

from mcp_server.analysis.forensic_extensions import (
    build_adaptive_forcing_resolution,
    strongest_reply_followup_evidence,
    strongest_reply_geometry_evidence,
)


def test_adaptive_forcing_resolution_stops_after_immediate_material_punishment() -> None:
    board = chess.Board()
    for san in ["e4", "e5", "Qh5", "Nc6", "Qxe5+"]:
        board.push(board.parse_san(san))

    evidence = build_adaptive_forcing_resolution(
        board,
        ["c6e5"],
        mover=chess.WHITE,
    )

    assert evidence["termination_reason"] == "material_resolution"
    assert evidence["tactical_sequence_resolved"] is True
    assert evidence["first_material_loss_ply_for_mover"] == 1
    assert evidence["plies_consumed"] == 1
    step = evidence["moves"][0]
    assert step["san"] == "Nxe5"
    assert step["is_capture"] is True
    assert step["cumulative_material_change_for_mover_cp"] <= -900


def test_strongest_reply_null_move_probe_surfaces_new_forcing_followup() -> None:
    board = chess.Board("3q2k1/8/8/8/8/8/7P/6K1 b - - 0 1")

    evidence = strongest_reply_followup_evidence(board, "d8h4")

    assert evidence is not None
    assert evidence["pass_hypothesis_available"] is True
    assert evidence["has_forcing_followup_if_pass"] is True
    followups = evidence["forcing_followups_if_opponent_passes"]
    qxh2 = next(item for item in followups if item["uci"] == "h4h2")
    assert qxh2["is_capture"] is True
    assert qxh2["is_check"] is True
    assert any(item["uci"] == "h4h2" for item in evidence["new_followups_vs_pre_reply"])


def test_discovered_check_reply_geometry_is_deterministic() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4B3/4R1K1 w - - 0 1")

    evidence = strongest_reply_geometry_evidence(board, "e2f3")

    discovered = next(item for item in evidence if item["mechanism"] == "discovered_check")
    assert discovered["trigger_san"].endswith("+")
    assert discovered["discovered_checker_squares"] == ["e1"]
    assert discovered["target"] == "black_king@e8"


def test_skewer_reply_geometry_reports_front_and_rear_targets_without_claiming_win() -> None:
    board = chess.Board("4r1k1/3q4/8/8/8/8/4B3/6K1 w - - 0 1")

    evidence = strongest_reply_geometry_evidence(board, "e2b5")

    skewer = next(item for item in evidence if item["mechanism"] == "skewer_candidate")
    assert skewer["front_target"] == "black_queen@d7"
    assert skewer["rear_target"] == "black_rook@e8"
    assert "does not prove" in skewer["proof_scope"]
