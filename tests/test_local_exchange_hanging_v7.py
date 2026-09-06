from __future__ import annotations

import chess

from mcp_server.analysis.position_integrity import build_rich_tactical_snapshot
from mcp_server.analysis.tactical_snapshot_extensions import extend_tactical_snapshot


def _snapshot(fen: str):
    board = chess.Board(fen)
    return extend_tactical_snapshot(board, build_rich_tactical_snapshot(board))


def test_defended_pawn_can_be_locally_hanging_after_full_recapture_sequence() -> None:
    # Either white pawn can take d5. Black has the nominal recapture ...cxd5,
    # but the other white pawn recaptures again, so the local capture-only tree
    # guarantees White a pawn. This is exactly the "it was defended" failure
    # that an immediate-recapture-only detector cannot explain.
    snapshot = _snapshot("4k3/8/2p5/3p4/2P1P3/8/8/4K3 w - - 0 1")

    candidates = [
        item
        for item in snapshot.tactically_hanging_candidates
        if item.reason == "local_capture_exchange_profitable"
        and item.target.square == "d5"
    ]

    assert candidates
    candidate = next(item for item in candidates if item.capture.uci == "e4d5")
    assert candidate.nominal_defenders >= 1
    assert candidate.legal_immediate_recaptures
    assert candidate.local_exchange_gain_cp == 100
    assert candidate.local_exchange_tree_complete is True
    assert candidate.local_exchange_line_uci[:3] == ["e4d5", "c6d5", "c4d5"]
    assert candidate.local_exchange_line_san[:3] == ["exd5", "cxd5", "cxd5"]
    assert "capture-only" in candidate.proof_scope


def test_equal_defended_exchange_is_not_promoted_to_hanging_candidate() -> None:
    # exd5 cxd5 merely exchanges pawns. The defender works, so the local
    # minimax gain is zero and the richer detector must not call d5 hanging.
    snapshot = _snapshot("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1")

    assert not any(
        item.reason == "local_capture_exchange_profitable"
        and item.target.square == "d5"
        and item.capture.uci == "e4d5"
        for item in snapshot.tactically_hanging_candidates
    )


def test_pinned_defender_keeps_stronger_immediate_recapture_reason() -> None:
    # Existing invariant: Ne7 geometrically defends Bf5 but cannot legally
    # recapture Qxf5 because it is absolutely pinned to Ke8 by Re1.
    snapshot = _snapshot("4k3/4n3/8/5b2/8/8/2Q5/4R1K1 w - - 0 1")

    assert any(
        item.capture.uci == "c2f5"
        and item.reason == "defended_but_no_legal_immediate_recapture"
        and item.legal_immediate_recaptures == []
        for item in snapshot.tactically_hanging_candidates
    )
