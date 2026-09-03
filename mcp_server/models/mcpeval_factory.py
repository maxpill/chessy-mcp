"""``MCPEval.from_eval`` factory — extracts the construction logic into a
focused, testable unit.

The pre-atomization ``MCPEval.from_eval`` was buried inside ``models.py``
as a 250-line ``@classmethod``. Phase 18 separates it into this module so
``models.mcpeval`` carries only the data shape and the factory's audit
invariants (B-04, C-01, H-03, U-08, L-06) live in one place.

Construction order preserved byte-identically:

  1. Engine-output: ``EvalBlock`` populated from the Stockfish ``Eval``.
  2. Status / rule status: terminal vs active, winner, depth/0 fallback.
  3. PV truncation: drop tail moves past terminal in the root frame.
  4. Decision outcome: White-POV outcome / cp for the response.
  5. WDL tuple + percentage view (when Stockfish surfaced WDL).
  6. Action surface: best_move, executable_move, recommended_action,
     legal_actions, typed best_action_obj.
  7. Post-state cp/mate (audit B-05 wiring).
  8. History & policy blocks (provenance + decision_value).
"""

from __future__ import annotations

from typing import Any

import chess

from core.engines.types import Eval
from mcp_server.actions import build_best_action, build_legal_actions
from mcp_server.models.action import ActionBlock
from mcp_server.models.eval import EvalBlock
from mcp_server.models.history import HistoryBlock
from mcp_server.models.policy import PolicyBlock
from mcp_server.rules import evaluate_rule_status, truncate_pv_at_terminal
from mcp_server.urls import lichess_urls


def build_mcpeval_from_eval(
    cls: type,
    ev: Eval,
    fen: str,
    *,
    status: str | None = None,
    board: chess.Board | None = None,
    requested_depth: int | None = None,
    history_complete: str | bool = "incomplete",
    pv_board: chess.Board | None = None,
    legal_engine_moves: list[Eval] | None = None,
    zeroing_move_best_score: int | None = None,
    zeroing_move_best_mate: int | None = None,
) -> Any:
    """Build an ``MCPEval`` (the model's class) from a Stockfish ``Eval``.

    Returns an instance of ``cls`` (caller is expected to pass
    :class:`MCPEval`). The factory accepts ``cls`` so subclasses can
    share the construction logic.
    """
    url, img = lichess_urls(fen)
    clean_best_move = None if ev.best_move in (None, "(none)", "none") else ev.best_move
    clean_pv = [p for p in (ev.pv or []) if p not in ("(none)", "none")]

    b = board.copy(stack=True) if board is not None else chess.Board(fen)
    sign = 1 if b.turn == chess.WHITE else -1

    rule_status = evaluate_rule_status(
        b,
        sign * ev.cp
        if ev.cp is not None
        else (sign * ev.mate * 1000 if ev.mate is not None else None),
        mate_for_mover=sign * ev.mate if ev.mate is not None else None,
        history_complete=history_complete,
        zeroing_move_best_score=zeroing_move_best_score,
        zeroing_move_best_mate=zeroing_move_best_mate,
    )

    calc_status = status or rule_status.terminal or "active"
    winner = rule_status.winner
    mate_val = ev.mate
    cp_val = ev.cp
    depth_val = ev.depth

    if rule_status.terminal is not None:
        if rule_status.terminal == "checkmate":
            cp_val = None
            depth_val = 0
            if mate_val is None:
                mate_val = 1
        else:
            mate_val = None
            cp_val = 0
            depth_val = 0

    if clean_pv:
        pv_anchor = pv_board if pv_board is not None else b
        clean_pv = truncate_pv_at_terminal(pv_anchor, clean_pv)

    decision_outcome = "active"
    decision_cp = cp_val
    if calc_status in (
        "checkmate",
        "stalemate",
        "insufficient_material",
        "seventyfive_moves",
        "fivefold_repetition",
        "dead_position",
        "game_over",
    ):
        if calc_status == "checkmate":
            decision_outcome = "loss" if b.turn == chess.WHITE else "win"
            decision_cp = None
        else:
            decision_outcome = "draw"
            decision_cp = 0
    elif ev.mate is not None:
        if ev.mate > 0:
            decision_outcome = "win"
            decision_cp = None
        else:
            decision_outcome = "loss"
            decision_cp = None
    elif rule_status.can_claim_now and rule_status.recommended_action == "claim_draw":
        decision_outcome = "draw"
        decision_cp = 0
    elif (
        rule_status.can_claim_with_intended_move
        and rule_status.recommended_action == "claim_draw_with_intended_move"
    ):
        decision_outcome = "draw"
        decision_cp = 0

    wdl_tuple: tuple[int, int, int] | None = ev.wdl
    wdl_pct_dict: dict[str, float] | None = (
        {
            "win": wdl_tuple[0] / 10.0,
            "draw": wdl_tuple[1] / 10.0,
            "loss": wdl_tuple[2] / 10.0,
        }
        if wdl_tuple is not None
        else None
    )

    engine_eval_dict = {
        "cp": ev.cp,
        "mate": ev.mate,
        "best_move": clean_best_move,
        "pv": clean_pv,
        "depth": depth_val,
        "wdl": wdl_tuple,
        "wdl_pct": wdl_pct_dict,
        "build_sha": None,
        "engine_config": {},
        "requested_depth": (requested_depth if requested_depth is not None else depth_val),
        "searched_depth": depth_val,
    }

    decision_val_dict = {
        "outcome": decision_outcome,
        "cp_equivalent": decision_cp,
        "best_action": rule_status.recommended_action,
        "perspective": "white",
    }

    if rule_status.recommended_action in ("claim_draw", "claim_draw_with_intended_move"):
        executable_move: str | None = None
    else:
        executable_move = clean_best_move

    best_action_payload = build_best_action(
        recommended_action=rule_status.recommended_action,
        rule_status=rule_status,
        engine_eval=ev,
        board=b,
        sign=sign,
    )
    legal_actions_payload = build_legal_actions(
        rule_status=rule_status,
        engine_eval=ev,
        board=b,
        legal_engine_moves=legal_engine_moves,
    )
    legal_move_uci_list: list[str] = [m.uci() for m in b.legal_moves]

    eval_block = EvalBlock(
        cp=cp_val,
        mate=mate_val,
        depth=depth_val,
        requested_depth=requested_depth if requested_depth is not None else depth_val,
        searched_depth=depth_val,
        pv=clean_pv,
        wdl=wdl_tuple,
        wdl_pct=wdl_pct_dict,
        root_score_cp=cp_val,
        root_score_mate=mate_val,
        engine_eval=engine_eval_dict,
    )
    action_block = ActionBlock(
        best_move=clean_best_move,
        executable_move=executable_move,
        recommended_action=rule_status.recommended_action,
        best_action=rule_status.recommended_action,
        best_action_type=rule_status.recommended_action,
        best_action_obj=best_action_payload,
        legal_actions=legal_actions_payload,
        legal_rule_actions=legal_actions_payload,
        legal_move_uci=legal_move_uci_list,
        can_claim_draw=rule_status.can_claim_draw,
        claim_reasons=rule_status.claim_reasons,
        claim_move=rule_status.claim_move,
        claim_move_san=rule_status.claim_move_san,
        claim_move_uci=rule_status.claim_move_uci,
        can_claim_now=rule_status.can_claim_now,
        claim_reasons_now=rule_status.claim_reasons_now,
        can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
        claim_moves=rule_status.claim_moves,
    )
    history_block = HistoryBlock(
        canonical_fen=fen,
        fen_was_canonicalized=False,
        post_fen=b.fen() if board is None else board.fen(),
        history_dependent_status=rule_status.history_dependent_status,
        lichess_url_reproduces_history=rule_status.fen_sufficient_for_status,
        requires_move_stack=rule_status.requires_move_stack,
        fen_sufficient_for_status=rule_status.fen_sufficient_for_status,
        history_completeness=rule_status.history_completeness,
        repetition_status=rule_status.repetition_status,
    )
    policy_block = PolicyBlock(
        decision_value=decision_val_dict,
    )

    return cls(
        status=calc_status,
        winner=winner,
        eval=eval_block,
        action=action_block,
        history=history_block,
        policy=policy_block,
        lichess_url=url,
        lichess_image=img,
    )


def attach_factory(cls: type) -> type:
    """Bind :func:`build_mcpeval_from_eval` as ``cls.from_eval``."""
    cls.from_eval = classmethod(  # type: ignore[attr-defined]
        lambda self_cls, ev, fen, **kwargs: build_mcpeval_from_eval(self_cls, ev, fen, **kwargs)
    )
    return cls
