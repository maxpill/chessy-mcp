from __future__ import annotations

import chess

from core.engines.types import MoveClass
from mcp_server.analysis.causal_trace import (
    apply_causal_position_trace,
    build_causal_position_trace,
)
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


def test_causal_trace_records_materialization_and_position_identity() -> None:
    root = chess.Board("4k3/4r3/8/8/8/8/4Q3/4K3 w - - 0 1")
    played = chess.Move.from_uci("e2e7")
    root.push(played)

    trace = build_causal_position_trace(root, ["e8e7"])

    assert trace["plies_traced"] == 1
    assert trace["termination_reason"] == "pv_exhausted"
    step = trace["steps"][0]
    assert step["side"] == "black"
    assert step["san"] == "Kxe7"
    assert step["captured_piece"] == "white_queen"
    assert step["delta"]["material_delta_white_cp"] == -900
    assert step["delta"]["material_delta_black_cp"] == 0
    assert "material_changed" in step["causal_flags"]
    assert len(step["position_after_hash"]) == 16
    assert "does not prove" in trace["proof_scope"]


def _rich_opening_result() -> tuple[ForensicMoveAnalysis, chess.Board, chess.Move]:
    board = chess.Board()
    played = chess.Move.from_uci("e2e4")
    after = board.copy(stack=True)
    after.push(played)
    evidence = ForensicEvidence(
        detail="coach",
        position_before=build_position_fingerprint(board),
        position_after_played=build_position_fingerprint(after),
        tactical_before=build_tactical_snapshot(board),
        tactical_after_played=build_tactical_snapshot(after),
        position_delta=build_position_delta(board, after),
        mechanism_evidence=[
            {
                "mechanism": "adaptive_forcing_resolution",
                "plies_consumed": 2,
            }
        ],
        forced_line=ForcedLineEvidence(
            uci=["e7e5", "g1f3", "b8c6"],
            san=["e5", "Nf3", "Nc6"],
        ),
    )
    result = ForensicMoveAnalysis(
        played="e2e4",
        played_san="e4",
        move_class=MoveClass.GOOD,
        eval_before=MCPEval(cp=20, best_move="e2e4", pv=["e2e4"]),
        eval_after=MCPEval(cp=15, best_move="e7e5", pv=["e7e5", "g1f3", "b8c6"]),
        action_type="play_move",
        forensics=evidence,
    )
    return result, board, played


def test_causal_trace_stops_at_adaptive_resolution_and_adds_evidence_signature() -> None:
    result, board, played = _rich_opening_result()

    upgraded = apply_causal_position_trace(result, board, played_move=played)

    assert upgraded.forensics is not None
    traces = [
        item
        for item in upgraded.forensics.mechanism_evidence
        if item.get("mechanism") == "causal_position_delta_trace"
    ]
    assert len(traces) == 1
    trace = traces[0]
    assert trace["pv_plies_available"] == 3
    assert trace["plies_traced"] == 2
    assert trace["termination_reason"] == "trace_limit"
    assert [step["san"] for step in trace["steps"]] == ["e5", "Nf3"]
    assert "CAUSAL_POSITION_DELTA_TRACE_AVAILABLE" in upgraded.forensics.evidence_signatures
