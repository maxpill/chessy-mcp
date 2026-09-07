"""2026-09-07 audit regression: best_action_obj must mirror best_action string.

Bug §3 — with FEN ``7k/8/8/8/8/8/R7/K7 w - - 100 1`` and ``move="a2b2"``::

    best_action = "claim_draw"
    best_action_obj = {type: "play_move", ...}   # shadowed!

Root cause: ``build_classification`` hardcoded ``recommended_action="play_move"``
when unifying ``best_action_obj`` with the post-state. When the policy's
``score.best_action == "claim_draw"``, ``build_best_action`` short-circuits
and returns the claim dict, restoring the wire-contract invariant::

    canonical_action_key(result.best_action_obj) == result.best_action
"""

from __future__ import annotations

import pytest

import chess
from core.engines.types import Eval, MoveClass
from mcp_server.analysis.classify_helpers import build_classification
from mcp_server.engine.retry import reset_breaker
from mcp_server.models import MCPEval


@pytest.fixture(autouse=True)
async def _cleanup() -> None:
    yield
    reset_breaker()


@pytest.mark.asyncio
async def test_build_classification_preserves_claim_best_action_obj() -> None:
    """§3: when policy best_action=claim_draw, best_action_obj.type == claim_draw."""
    board = chess.Board("7k/8/8/8/8/8/R7/K7 w - - 100 1")
    move = chess.Move.from_uci("a2b2")
    board_after = board.copy(stack=True)
    board_after.push(move)

    eval_before = MCPEval.from_eval(
        Eval(cp=0, best_move="a2b2", pv=["a2b2"], depth=1),
        board.fen(),
        board=board,
        history_complete="complete",
    )
    eval_after = MCPEval.from_eval(
        Eval(cp=0, depth=1),
        board_after.fen(),
        board=board_after,
        history_complete="complete",
    )

    class _FakeScore:
        move_class = MoveClass.BEST
        is_best_engine_move = True
        centipawn_loss = 0
        mate_distance_loss = None
        raw_centipawn_loss = 0
        raw_centipawn_delta = 0
        effective_loss = 0
        loss_kind = "none"
        engine_cp_loss = None
        mate_distance_penalty = None
        outcome_penalty = None
        rule_action_penalty = None
        best_action = "claim_draw"
        is_best_action = False
        action_equivalent = True
        missed_draw_claim = True
        conceded_draw_claim = False
        claim_reason = "fifty_moves"
        claim_move = None
        can_claim_now = True
        can_claim_with_intended_move = False
        claim_moves: list[str] = []

    rule_status = type(
        "RS",
        (),
        {
            "recommended_action": "claim_draw",
            "can_claim_now": True,
            "claim_reasons_now": ["fifty_moves"],
            "can_claim_with_intended_move": False,
            "claim_reasons": ["fifty_moves"],
            "intended_claim_ucis": [],
            "intended_claim_sans": [],
            "intended_claim_reasons_by_uci": {},
            "claim_move": None,
            "claim_move_uci": None,
            "claim_move_san": None,
            "terminal": None,
            "winner": None,
        },
    )()

    res = build_classification(
        played_uci="a2b2",
        played_san="Rb2",
        score=_FakeScore(),
        eval_before=eval_before,
        eval_after=eval_after,
        board=board,
        board_after=board_after,
        rule_status=rule_status,
        best_san="Rb2",
        best_line_san="Rb2",
        played_continuation=None,
        action_type="play_move",
        syntax_warning=None,
    )

    assert res.best_action == "claim_draw"
    assert res.best_action_obj is not None
    assert res.best_action_obj.get("type") == "claim_draw", (
        f"best_action_obj.type must equal best_action={res.best_action!r}; "
        f"got best_action_obj={res.best_action_obj!r}"
    )


@pytest.mark.asyncio
async def test_build_classification_play_move_best_action_obj_unchanged() -> None:
    """§3 regression guard: standard play_move best_action_obj stays as play_move."""
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    move = chess.Move.from_uci("e2e4")
    board_after = board.copy(stack=True)
    board_after.push(move)

    eval_before = MCPEval.from_eval(
        Eval(cp=20, best_move="e2e4", pv=["e2e4"], depth=1),
        board.fen(),
        board=board,
        history_complete="complete",
    )
    eval_after = MCPEval.from_eval(
        Eval(cp=20, depth=1),
        board_after.fen(),
        board=board_after,
        history_complete="complete",
    )

    class _FakeScore:
        move_class = MoveClass.BEST
        is_best_engine_move = True
        centipawn_loss = 0
        mate_distance_loss = None
        raw_centipawn_loss = 0
        raw_centipawn_delta = 0
        effective_loss = 0
        loss_kind = "none"
        engine_cp_loss = None
        mate_distance_penalty = None
        outcome_penalty = None
        rule_action_penalty = None
        best_action = "play_move"
        is_best_action = True
        action_equivalent = True
        missed_draw_claim = False
        conceded_draw_claim = False
        claim_reason = None
        claim_move = None
        can_claim_now = False
        can_claim_with_intended_move = False
        claim_moves: list[str] = []

    rule_status = type(
        "RS",
        (),
        {
            "recommended_action": "play_move",
            "can_claim_now": False,
            "claim_reasons_now": [],
            "can_claim_with_intended_move": False,
            "claim_reasons": [],
            "intended_claim_ucis": [],
            "intended_claim_sans": [],
            "intended_claim_reasons_by_uci": {},
            "claim_move": None,
            "claim_move_uci": None,
            "claim_move_san": None,
            "terminal": None,
            "winner": None,
        },
    )()

    res = build_classification(
        played_uci="e2e4",
        played_san="e4",
        score=_FakeScore(),
        eval_before=eval_before,
        eval_after=eval_after,
        board=board,
        board_after=board_after,
        rule_status=rule_status,
        best_san="e4",
        best_line_san="e4",
        played_continuation=None,
        action_type="play_move",
        syntax_warning=None,
    )

    assert res.best_action == "play_move"
    assert res.best_action_obj is not None
    assert res.best_action_obj.get("type") == "play_move", (
        f"best_action_obj.type must equal best_action={res.best_action!r}; "
        f"got best_action_obj={res.best_action_obj!r}"
    )
