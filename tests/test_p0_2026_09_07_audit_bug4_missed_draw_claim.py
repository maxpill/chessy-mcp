"""2026-09-07 audit regression: missed_draw_claim uses literal claim-opportunity semantics.

Bug §4 — with FEN ``7k/8/8/8/8/8/R7/K7 w - - 100 1`` and a quiet move::

    best_action = "claim_draw"
    missed_draw_claim = False   # bug — caller did miss the policy-optimal claim

Root cause: ``score_optimal_claim_recommended`` gated ``missed_draw_claim`` on
``is_down_material`` (≥200 cp deficit). For drawn positions the gate blocked
a true positive. Literal definition per the audit::

    missed_draw_claim = (policy-selected legal draw claim existed) AND
                        (caller chose a non-claim action)
"""

from __future__ import annotations

import pytest

from mcp_server.analysis.move_grading.strategies.draw import (
    score_optimal_claim_recommended,
)


@pytest.mark.asyncio
async def test_missed_draw_claim_does_not_require_is_down_material() -> None:
    """§4 invariant: is_down_material must NOT gate the flag.

    When the draw-preserved branch fires with canonical_best_action =
    "claim_draw_with_intended_move" and the player played the engine's exact
    best move (which equals the intended claim move) at zero loss, the move
    is the implicit claim — so missed_draw_claim stays False. But for
    canonical_best_action = "claim_draw" played as play_move, missed_draw_claim
    must be True regardless of is_down_material.
    """

    class _Stub:
        can_claim_draw = True
        can_claim_now = False
        can_claim_with_intended_move = False
        claim_reasons = ["fifty_moves"]
        claim_reasons_now = []
        intended_claim_ucis = []
        intended_claim_sans = []
        intended_claim_reasons_by_uci = {}
        claim_moves: list[str] = []
        claim_move_uci = None
        claim_move_san = None
        claim_move = None
        repetition_status = "none"

    eval_before = type(
        "E",
        (),
        {"decision_value": {"outcome": "draw"}, "claim_reasons": []},
    )()
    eval_after = type(
        "E",
        (),
        {"decision_value": {"outcome": "draw"}, "claim_reasons": []},
    )()

    # Case A: claim_draw best + play_move + is_down_material=False → must flag
    out_a = score_optimal_claim_recommended(
        optimal_claim_recommended=True,
        is_auto_terminal_draw=False,
        eval_before=eval_before,
        eval_after=eval_after,
        rule_before=_Stub(),
        raw_cpl=0,
        raw_board_delta=0,
        is_best_engine_move=True,
        canonical_best_action="claim_draw",
        action_type="play_move",
        is_after_losing=False,
        is_after_winning=False,
        is_down_material=False,
        is_mover_forced_win=False,
        before_mover=0,
        after_mover=0,
        opp_mat=0,
        mover_mat=0,
    )
    assert out_a is not None
    assert out_a.missed_draw_claim is True, (
        f"§4: missed_draw_claim must be True with canonical=claim_draw, "
        f"action=play_move, is_down_material=False; got {out_a.missed_draw_claim}"
    )

    # Case B: claim_draw_with_intended_move + best engine move + zero cpl →
    # the player's move IS the intended claim — missed_draw_claim stays False.
    out_b = score_optimal_claim_recommended(
        optimal_claim_recommended=True,
        is_auto_terminal_draw=False,
        eval_before=eval_before,
        eval_after=eval_after,
        rule_before=_Stub(),
        raw_cpl=0,
        raw_board_delta=0,
        is_best_engine_move=True,
        canonical_best_action="claim_draw_with_intended_move",
        action_type="play_move",
        is_after_losing=False,
        is_after_winning=False,
        is_down_material=False,
        is_mover_forced_win=False,
        before_mover=0,
        after_mover=0,
        opp_mat=0,
        mover_mat=0,
    )
    assert out_b is not None
    assert out_b.missed_draw_claim is False, (
        f"§4 exception: claim_draw_with_intended_move + best engine move + zero cpl "
        f"is the implicit claim; missed_draw_claim must stay False; got {out_b.missed_draw_claim}"
    )
