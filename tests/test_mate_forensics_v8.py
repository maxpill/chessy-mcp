from __future__ import annotations

import chess

from core.engines.types import MoveClass
from mcp_server.analysis.forensics import (
    build_position_delta,
    build_position_fingerprint,
    build_tactical_snapshot,
)
from mcp_server.analysis.mate_forensics import (
    apply_mate_forensics,
    mate_in_one_moves,
    mate_in_one_threats_if_pass,
)
from mcp_server.models import MCPEval
from mcp_server.models.forensics import (
    ForcedLineEvidence,
    ForensicEvidence,
    ForensicMoveAnalysis,
    StrongestReplyEvidence,
)


def _board_after(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def _result_for_move(
    board: chess.Board,
    move: chess.Move,
    *,
    strongest_reply: StrongestReplyEvidence | None = None,
) -> ForensicMoveAnalysis:
    after = board.copy(stack=True)
    after.push(move)
    evidence = ForensicEvidence(
        detail="coach",
        position_before=build_position_fingerprint(board),
        position_after_played=build_position_fingerprint(after),
        tactical_before=build_tactical_snapshot(board),
        tactical_after_played=build_tactical_snapshot(after),
        strongest_reply=strongest_reply,
        position_delta=build_position_delta(board, after),
        forced_line=ForcedLineEvidence(),
    )
    return ForensicMoveAnalysis(
        played=move.uci(),
        played_san=board.san(move),
        move_class=MoveClass.BLUNDER,
        eval_before=MCPEval(cp=0),
        eval_after=MCPEval(cp=0),
        action_type="play_move",
        forensics=evidence,
    )


def test_mate_in_one_scan_finds_fools_mate() -> None:
    board = _board_after("f3", "e5", "g4")

    mates = mate_in_one_moves(board)

    qh4 = next(item for item in mates if item["uci"] == "d8h4")
    assert qh4["san"] == "Qh4#"
    assert qh4["piece"] == "black_queen@d8"
    assert qh4["resulting_fen"].split()[1] == "w"


def test_missed_mate_in_one_is_explicit_evidence_not_process_claim() -> None:
    board = _board_after("f3", "e5", "g4")
    played = chess.Move.from_uci("b8c6")
    result = _result_for_move(board, played)

    upgraded = apply_mate_forensics(result, board, played_move=played)

    assert upgraded.forensics is not None
    signatures = upgraded.forensics.evidence_signatures
    assert "MATE_IN_ONE_AVAILABLE_BEFORE_MOVE" in signatures
    assert "MISSED_MATE_IN_ONE_CANDIDATE" in signatures
    assert "PLAYED_MATE_IN_ONE" not in signatures
    scan = next(
        item
        for item in upgraded.forensics.mechanism_evidence
        if item.get("mechanism") == "mate_in_one_scan"
    )
    assert scan["played_move_was_mate_in_one"] is False
    assert any(item["uci"] == "d8h4" for item in scan["mate_in_one_moves_before"])
    assert "does not infer why" in scan["proof_scope"]


def test_move_allowing_fools_mate_marks_opponent_mate_and_strongest_reply() -> None:
    board = _board_after("f3", "e5")
    played = board.parse_san("g4")
    after = board.copy(stack=True)
    after.push(played)
    reply = chess.Move.from_uci("d8h4")
    after_reply = after.copy(stack=True)
    reply_san = after.san(reply)
    after_reply.push(reply)
    strongest_reply = StrongestReplyEvidence(
        uci=reply.uci(),
        san=reply_san,
        is_check=True,
        is_capture=False,
        is_forcing=True,
        resulting_fen=after_reply.fen(),
        eval_after_reply_mate=-1,
    )
    result = _result_for_move(board, played, strongest_reply=strongest_reply)

    upgraded = apply_mate_forensics(result, board, played_move=played)

    assert upgraded.forensics is not None
    signatures = upgraded.forensics.evidence_signatures
    assert "OPPONENT_MATE_IN_ONE_AFTER_MOVE" in signatures
    assert "STRONGEST_REPLY_IS_MATE_IN_ONE" in signatures
    scan = next(
        item
        for item in upgraded.forensics.mechanism_evidence
        if item.get("mechanism") == "mate_in_one_scan"
    )
    assert scan["strongest_reply_is_mate_in_one"] is True
    assert any(item["uci"] == "d8h4" for item in scan["opponent_mate_in_one_moves_after_played"])


def test_played_mate_in_one_is_recorded_as_positive_fact() -> None:
    board = _board_after("f3", "e5", "g4")
    played = chess.Move.from_uci("d8h4")
    result = _result_for_move(board, played)

    upgraded = apply_mate_forensics(result, board, played_move=played)

    assert upgraded.forensics is not None
    signatures = upgraded.forensics.evidence_signatures
    assert "PLAYED_MATE_IN_ONE" in signatures
    assert "MISSED_MATE_IN_ONE_CANDIDATE" not in signatures


def test_scholars_mate_threat_is_visible_under_hypothetical_pass() -> None:
    board = _board_after("e4", "e5", "Bc4", "Nc6", "Qh5")

    threats, available, reason = mate_in_one_threats_if_pass(board)

    assert available is True
    assert reason is None
    qxf7 = next(item for item in threats if item["uci"] == "h5f7")
    assert qxf7["san"] == "Qxf7#"


def test_mate_threat_update_reports_when_defense_addresses_immediate_mate() -> None:
    board = _board_after("e4", "e5", "Bc4", "Nc6", "Qh5")
    played = board.parse_san("g6")
    result = _result_for_move(board, played)

    upgraded = apply_mate_forensics(result, board, played_move=played)

    assert upgraded.forensics is not None
    signatures = upgraded.forensics.evidence_signatures
    assert "OPPONENT_MATE_IN_ONE_THREAT_IF_PASS" in signatures
    assert "IMMEDIATE_MATE_THREAT_ADDRESSED" in signatures
    assert "FAILED_MATE_THREAT_UPDATE_CANDIDATE" not in signatures
    scan = next(
        item
        for item in upgraded.forensics.mechanism_evidence
        if item.get("mechanism") == "mate_in_one_scan"
    )
    assert scan["played_move_addresses_immediate_mate_threat"] is True
