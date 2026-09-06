from __future__ import annotations

import json

import chess

import mcp_server.analysis.tactical_snapshot_extensions as snapshot_extensions
from mcp_server.analysis.position_integrity import build_rich_tactical_snapshot
from mcp_server.middleware.request_cost import estimate_mcp_request_cost


def _rpc(arguments: dict[str, object]) -> bytes:
    return json.dumps(
        {
            "params": {
                "name": "classify_move",
                "arguments": arguments,
            }
        }
    ).encode()


def test_forensic_classify_admission_cost_reserves_multi_depth_stability_searches() -> None:
    standard = estimate_mcp_request_cost(
        _rpc({"fen": "startpos", "move": "e4", "depth": 20})
    )
    coach = estimate_mcp_request_cost(
        _rpc({"fen": "startpos", "move": "e4", "depth": 20, "detail": "coach"})
    )
    forensic = estimate_mcp_request_cost(
        _rpc({"fen": "startpos", "move": "e4", "depth": 20, "detail": "forensic"})
    )
    compare_upgrade = estimate_mcp_request_cost(
        _rpc(
            {
                "fen": "startpos",
                "move": "e4",
                "depth": 20,
                "compare_moves": ["d4", "Nf3"],
            }
        )
    )

    assert standard < coach < forensic < compare_upgrade
    # At d20 the stability verifier can consume before/after pairs at d24 and d26.
    # Admission control must reserve at least that additional search mass instead
    # of charging the pre-stability forensic estimate.
    stability_reserve = (2 * 24 + 2 * 26) / 14.0
    pre_stability_forensic = standard + (20 * 3) / 14.0
    assert forensic >= pre_stability_forensic + stability_reserve


def test_local_exchange_node_budget_fails_closed(monkeypatch) -> None:
    board = chess.Board("4k3/8/2p5/3p4/2P1P3/8/8/4K3 w - - 0 1")
    base = build_rich_tactical_snapshot(board)

    monkeypatch.setattr(snapshot_extensions, "MAX_LOCAL_EXCHANGE_NODES", 0)
    snapshot = snapshot_extensions.extend_tactical_snapshot(board, base)

    assert not any(
        item.reason == "local_capture_exchange_profitable"
        for item in snapshot.tactically_hanging_candidates
    )
