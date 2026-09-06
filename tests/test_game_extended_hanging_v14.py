from __future__ import annotations

import chess

from mcp_server.analysis.game_critical_forensics import _new_extended_hanging_targets


def test_game_critical_hanging_includes_local_exchange_profitability() -> None:
    before = chess.Board("4k3/8/2pp4/8/2P1P3/8/8/4K3 b - - 0 1")
    move = chess.Move.from_uci("d6d5")
    assert move in before.legal_moves
    after = before.copy(stack=True)
    after.push(move)

    newly = _new_extended_hanging_targets(before, after, perspective="black")

    # 1.exd5 cxd5 2.cxd5 wins a pawn in the bounded local capture tree.
    assert "black_pawn@d5" in newly
