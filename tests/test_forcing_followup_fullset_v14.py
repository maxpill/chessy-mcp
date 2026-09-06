from __future__ import annotations

import chess
import pytest

from mcp_server.analysis import forensic_extensions


def _item(key: str) -> dict[str, object]:
    return {
        "uci": key,
        "san": key,
        "is_check": False,
        "is_capture": True,
        "promotion": None,
        "captured_piece": "black_pawn",
    }


def test_followup_novelty_uses_complete_set_before_wire_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = chess.Board()
    common = [_item(f"common-{index}") for index in range(16)]
    new_late = _item("new-after-cap")
    calls = 0

    def fake_forcing_moves(_board: chess.Board) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return list(common)
        return [*common, new_late]

    monkeypatch.setattr(forensic_extensions, "_forcing_moves", fake_forcing_moves)

    evidence = forensic_extensions.strongest_reply_followup_evidence(board, "e2e4")

    assert evidence is not None
    assert evidence["forcing_followup_count"] == 17
    assert evidence["new_followup_count"] == 1
    assert evidence["has_new_forcing_followup_if_pass"] is True
    assert evidence["presentation_truncated"]["followups"] is True
    assert evidence["new_followups_vs_pre_reply"] == [new_late]
