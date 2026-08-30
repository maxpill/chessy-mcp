"""Typed chess action contract for Chess MCP.

This module implements the structured action discriminated union proposed by
the 2026-08-28 ultra-detailed audit:

    {
      "best_action": { "type": "play_move", "move": {...}, "value": {...} }
    }
    {
      "best_action": { "type": "claim_draw", "reason": "..." }
    }
    {
      "best_action": { "type": "claim_draw_with_intended_move", "reason": "...", "intended_move": {...} }
    }
    {
      "best_action": { "type": "game_over", "outcome": "win|loss|draw", "reason": "..." }
    }

A typed action union makes it impossible for a client to mistakenly play a move
that is actually a draw claim (C-01), and makes `claim_draw_with_intended_move`
and `play_move` structurally distinct (audit invariant I-03).
"""

from __future__ import annotations

from typing import Any, Literal

import chess
from pydantic import BaseModel


class MovePayload(BaseModel):
    """A single chess move in both UCI and SAN form."""

    uci: str
    san: str | None = None


class ActionValue(BaseModel):
    """The game-theoretic value of an action from White's POV."""

    cp: int | None = None
    mate: int | None = None
    outcome: str | None = None  # "win" | "loss" | "draw" | "active" | None


class PlayMoveAction(BaseModel):
    """Action: play the given move on the board."""

    type: Literal["play_move"] = "play_move"
    move: MovePayload
    value: ActionValue | None = None


class ClaimDrawAction(BaseModel):
    """Action: claim a draw immediately (50-move / threefold repetitions where claim is already available)."""

    type: Literal["claim_draw"] = "claim_draw"
    reason: str  # "fifty_moves" | "threefold_repetition"


class ClaimDrawWithIntendedMoveAction(BaseModel):
    """Action: declare an intended move and claim a draw with it (50-move / threefold).

    The intended move itself is NOT played - the claim is procedural, declared
    before the move is executed. A client must NOT just play `intended_move`
    and expect a draw.
    """

    type: Literal["claim_draw_with_intended_move"] = "claim_draw_with_intended_move"
    reason: str  # "fifty_moves" | "threefold_repetition"
    intended_move: MovePayload


class GameOverAction(BaseModel):
    """Action: the game is over (no action possible)."""

    type: Literal["game_over"] = "game_over"
    outcome: str  # "win" | "loss" | "draw"
    reason: str  # terminal status, e.g. "checkmate", "stalemate"


GameAction = PlayMoveAction | ClaimDrawAction | ClaimDrawWithIntendedMoveAction | GameOverAction


def _build_play_move_action(
    move_uci: str,
    move_san: str | None,
    cp: int | None = None,
    mate: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "play_move",
        "move": {"uci": move_uci, "san": move_san},
    }
    if cp is not None or mate is not None:
        payload["value"] = {"cp": cp, "mate": mate}
    return payload


def _build_claim_draw_action(reason: str) -> dict[str, Any]:
    return {"type": "claim_draw", "reason": reason}


def _build_claim_with_intended_action(
    reason: str, intended_move_uci: str, intended_move_san: str | None
) -> dict[str, Any]:
    return {
        "type": "claim_draw_with_intended_move",
        "reason": reason,
        "intended_move": {"uci": intended_move_uci, "san": intended_move_san},
    }


def _build_game_over_action(outcome: str, reason: str) -> dict[str, Any]:
    return {"type": "game_over", "outcome": outcome, "reason": reason}


def build_played_action(
    action_type: str,
    *,
    move_uci: str,
    move_san: str | None,
    rule_status: Any,
    cp: int | None = None,
    mate: int | None = None,
) -> dict[str, Any]:
    """Build the typed payload for the action the caller actually requested."""
    if action_type == "play_move":
        return _build_play_move_action(move_uci, move_san, cp=cp, mate=mate)
    if action_type == "claim_draw":
        if not rule_status.can_claim_now:
            raise ValueError("ILLEGAL_ACTION: draw cannot be claimed now")
        reason = rule_status.claim_reasons_now[0] if rule_status.claim_reasons_now else "threefold_repetition"
        return _build_claim_draw_action(reason)
    if action_type == "claim_draw_with_intended_move":
        if move_uci not in (rule_status.intended_claim_ucis or []):
            raise ValueError("ILLEGAL_ACTION: intended move does not create a legal draw claim")
        reason_map = getattr(rule_status, "intended_claim_reasons_by_uci", {}) or {}
        reasons = reason_map.get(move_uci) or rule_status.claim_reasons or ["threefold_repetition"]
        return _build_claim_with_intended_action(reasons[0], move_uci, move_san)
    raise ValueError(f"INVALID_ACTION_TYPE: {action_type}")


def build_best_action(
    recommended_action: str,
    rule_status: Any,
    engine_eval: Any | None = None,
    board: Any | None = None,
    sign: int = 1,
) -> dict[str, Any]:
    if rule_status.terminal is not None:
        if rule_status.terminal == "checkmate":
            outcome = "win" if rule_status.winner == "white" else "loss"
        else:
            outcome = "draw"
        return _build_game_over_action(outcome, rule_status.terminal)

    if recommended_action == "claim_draw" and rule_status.can_claim_now:
        reason = rule_status.claim_reasons_now[0] if rule_status.claim_reasons_now else "threefold_repetition"
        return _build_claim_draw_action(reason)

    if recommended_action == "claim_draw_with_intended_move" and rule_status.can_claim_with_intended_move:
        intended_uci = rule_status.claim_move_uci
        intended_san = rule_status.claim_move_san or rule_status.claim_move
        if intended_uci:
            reason_map = getattr(rule_status, "intended_claim_reasons_by_uci", {}) or {}
            reasons = reason_map.get(intended_uci) or rule_status.claim_reasons or ["threefold_repetition"]
            return _build_claim_with_intended_action(reasons[0], intended_uci, intended_san)

    if engine_eval is not None and engine_eval.best_move:
        bm_uci = engine_eval.best_move
        bm_san: str | None = None
        if board is not None:
            try:
                m = chess.Move.from_uci(bm_uci.lower())
                if m in board.legal_moves:
                    bm_san = board.san(m)
            except Exception:
                pass
        return _build_play_move_action(bm_uci, bm_san, cp=engine_eval.cp, mate=engine_eval.mate)

    return _build_play_move_action("", None)


def build_legal_actions(
    rule_status: Any,
    engine_eval: Any | None,
    board: Any | None,
    legal_engine_moves: list[Any] | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    if rule_status.can_claim_now:
        for reason in rule_status.claim_reasons_now:
            actions.append(_build_claim_draw_action(reason))

    if rule_status.can_claim_with_intended_move:
        intended_ucis = list(rule_status.intended_claim_ucis or [])
        intended_sans = list(rule_status.intended_claim_sans or [])
        if not intended_ucis and rule_status.claim_move_uci:
            intended_ucis = [rule_status.claim_move_uci]
            intended_sans = [rule_status.claim_move_san or rule_status.claim_move]
        reason_map = getattr(rule_status, "intended_claim_reasons_by_uci", {}) or {}
        for i, uci in enumerate(intended_ucis):
            san = intended_sans[i] if i < len(intended_sans) else None
            reasons = reason_map.get(uci) or rule_status.claim_reasons or ["threefold_repetition"]
            for reason in reasons:
                if reason in ("fifty_moves", "threefold_repetition"):
                    actions.append(_build_claim_with_intended_action(reason, uci, san))

    if legal_engine_moves:
        for ev in legal_engine_moves:
            if not ev.best_move:
                continue
            bm_san: str | None = None
            if board is not None:
                try:
                    m = chess.Move.from_uci(ev.best_move.lower())
                    if m in board.legal_moves:
                        bm_san = board.san(m)
                except Exception:
                    pass
            actions.append(_build_play_move_action(ev.best_move, bm_san, cp=ev.cp, mate=ev.mate))

    if rule_status.terminal is not None:
        if rule_status.terminal == "checkmate":
            outcome = "win" if rule_status.winner == "white" else "loss"
        else:
            outcome = "draw"
        actions.append(_build_game_over_action(outcome, rule_status.terminal))

    return actions
