"""2026-09-07 audit regression: terminal outcome_perspective must be explicit.

Bug §7 — for checkmate positions ``best_action_obj`` was::

    {type: "game_over", outcome: "win", reason: "checkmate"}

The literal ``outcome`` field is ambiguous: ``"win"`` could mean "the mover
wins" or "White wins" depending on consumer interpretation.

Fix: ``GameOverAction`` now carries an optional ``outcome_perspective`` field
set to the side that won (``"white"`` / ``"black"``) or ``"draw"`` for draw
terminals. The ``outcome`` field stays for backward compatibility.
"""

from __future__ import annotations

import pytest

from mcp_server.domain.action import build_best_action, parse_action


def _rule_status(
    *,
    terminal: str | None,
    winner: str | None = None,
    can_claim_now: bool = False,
    can_claim_with_intended_move: bool = False,
    claim_reasons_now: list[str] | None = None,
):
    return type(
        "RS",
        (),
        {
            "terminal": terminal,
            "winner": winner,
            "can_claim_now": can_claim_now,
            "can_claim_with_intended_move": can_claim_with_intended_move,
            "claim_reasons_now": claim_reasons_now or [],
            "claim_reasons": ["fifty_moves"],
            "intended_claim_ucis": [],
            "intended_claim_sans": [],
            "intended_claim_reasons_by_uci": {},
            "claim_move_uci": None,
            "claim_move_san": None,
            "claim_move": None,
        },
    )()


def test_game_over_white_wins_carries_perspective_white() -> None:
    """White checkmates → outcome_perspective == "white"."""
    rs = _rule_status(terminal="checkmate", winner="white")
    out = build_best_action("play_move", rs)
    assert out["type"] == "game_over"
    assert out["outcome"] == "win"
    assert out["reason"] == "checkmate"
    assert out.get("outcome_perspective") == "white", (
        f"White checkmate must set outcome_perspective=white; got {out!r}"
    )


def test_game_over_black_wins_carries_perspective_black() -> None:
    """Black checkmates → outcome_perspective == "black"."""
    rs = _rule_status(terminal="checkmate", winner="black")
    out = build_best_action("play_move", rs)
    assert out["type"] == "game_over"
    assert out["outcome"] == "loss"
    assert out["outcome_perspective"] == "black"


def test_game_over_draw_carries_perspective_draw() -> None:
    """Stalemate / 75-move / fivefold → outcome_perspective == "draw"."""
    rs = _rule_status(terminal="stalemate")
    out = build_best_action("play_move", rs)
    assert out["type"] == "game_over"
    assert out["outcome"] == "draw"
    assert out["outcome_perspective"] == "draw"


def test_game_over_action_schema_accepts_perspective() -> None:
    """parse_action round-trips a typed GameOverAction with explicit perspective."""
    payload = {
        "type": "game_over",
        "outcome": "win",
        "reason": "checkmate",
        "outcome_perspective": "white",
    }
    parsed = parse_action(payload)
    assert parsed.type == "game_over"
    assert parsed.outcome == "win"  # type: ignore[union-attr]
    assert parsed.reason == "checkmate"  # type: ignore[union-attr]
    assert parsed.outcome_perspective == "white"  # type: ignore[union-attr]


def test_game_over_schema_back_compat_no_perspective() -> None:
    """Backward compat: existing payloads without outcome_perspective still parse."""
    payload = {"type": "game_over", "outcome": "draw", "reason": "stalemate"}
    parsed = parse_action(payload)
    assert parsed.type == "game_over"
    assert parsed.outcome_perspective is None
