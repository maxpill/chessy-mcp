"""2026-09-08 audit regression: history_dependent_status decoupled from repetition_status="unknown".

Bug 3 — Live ``evaluate_position(fen="7k/8/8/8/8/8/4K2P/R7 w - - 100 51", depth=2)``
returned::

    "history_block":{"..."repetition_sufficient_without_history":true,
                    "history_dependent_status":true,"requires_move_stack":true,
                    "fen_sufficient_for_status":false,...}

Root cause: ``mcp_server/rules/status.py:127`` computed
``requires_stack = repetition_proven or repetition_status == "unknown"``
and copied that value into ``history_dependent_status`` /
``requires_move_stack`` / the inverse into ``fen_sufficient_for_status``.
The "unknown" branch fires whenever no PGN was supplied — which
incorrectly marked an immediate FEN-provable 50-move claim as
history-dependent.

Fix: ``requires_stack = repetition_proven``. Only a threefold claim has
actually been proven requires the move stack. ``repetition_sufficient_without_history``
already carries the repetition-knowledge signal on a separate axis.
"""

from __future__ import annotations

import chess

from mcp_server.rules.status import evaluate_rule_status


def test_audit_fen_50_move_immediate_claim_is_history_independent() -> None:
    """Audit's exact failing FEN: halfmove 100, immediate 50-move claim.

    Expected after fix:
      * ``history_dependent_status is False`` (50-move claim is FEN-provable)
      * ``requires_move_stack is False``
      * ``fen_sufficient_for_status is True``
      * ``repetition_sufficient_without_history is True``
      * ``can_claim_draw is True``
      * ``claim_reasons_now == ["fifty_moves"]``
    """
    board = chess.Board("7k/8/8/8/8/8/4K2P/R7 w - - 100 51")
    status = evaluate_rule_status(board, history_complete="incomplete")

    assert status.history_dependent_status is False, (
        f"50-move claim is FEN-provable; got history_dependent_status="
        f"{status.history_dependent_status}"
    )
    assert status.requires_move_stack is False
    assert status.fen_sufficient_for_status is True
    assert status.repetition_sufficient_without_history is True
    assert status.can_claim_draw is True
    assert status.claim_reasons_now == ["fifty_moves"]


def test_repetition_unknown_no_claim_history_independent() -> None:
    """Startpos with incomplete history: no claim reason, FEN-sufficient.

    Pre-fix this asserted ``requires_move_stack is True`` /
    ``history_dependent_status is True`` (the bug). Post-fix: both False.
    """
    status = evaluate_rule_status(chess.Board(chess.STARTING_FEN), history_complete="incomplete")
    assert status.repetition_status == "unknown"
    assert status.history_dependent_status is False
    assert status.requires_move_stack is False
    assert status.fen_sufficient_for_status is True
    assert status.repetition_sufficient_without_history is True
    assert not status.can_claim_draw


def test_threefold_repetition_still_requires_history_stack() -> None:
    """Regression guard: actual threefold claim still requires the move stack.

    1.Nf3 Nf6 2.Ng1 Ng8 3.Nf3 Nf6 4.Ng1 Ng8 (White to move, threefold
    reachable by repetition). This is the only case where
    ``requires_move_stack`` should still be True.
    """
    b = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"):
        b.push_san(san)
    status = evaluate_rule_status(b, history_complete="complete")
    assert status.can_claim_draw is True
    assert "threefold_repetition" in status.claim_reasons_now
    assert status.history_dependent_status is True
    assert status.requires_move_stack is True
    assert status.fen_sufficient_for_status is False


def test_50_move_with_full_history_still_fen_sufficient() -> None:
    """50-move claim + full PGN history still FEN-sufficient (claim is provable from board)."""
    b = chess.Board()
    b.halfmove_clock = 100
    status = evaluate_rule_status(b, history_complete="complete")
    assert "fifty_moves" in status.claim_reasons_now
    assert status.history_dependent_status is False
    assert status.requires_move_stack is False
    assert status.fen_sufficient_for_status is True


def test_terminal_checkmate_still_history_independent() -> None:
    """Terminal positions remain history-independent (control)."""
    b = chess.Board()
    # Fool's mate: 1.f3 e5 2.g4 Qh4#
    for san in ("f3", "e5", "g4", "Qh4#"):
        b.push_san(san)
    assert b.is_checkmate()
    status = evaluate_rule_status(b, history_complete="incomplete")
    assert status.terminal == "checkmate"
    assert status.history_dependent_status is False
    assert status.requires_move_stack is False
    assert status.fen_sufficient_for_status is True
