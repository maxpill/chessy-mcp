"""Typed action discriminated union for the MCP API contract.

Audit invariant I-03: a client must never confuse a `claim_draw` with a
`play_move`. The typed union makes that structurally impossible. Pydantic
discriminated unions with a `type` literal field model this contract. The
legacy dict-builder functions at the bottom keep the existing MCPEval payloads
working; new code can construct the typed models directly.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

import chess
from pydantic import BaseModel, Discriminator, TypeAdapter


class TypeOfAction(StrEnum):
    """Discriminator for the action discriminated union."""

    PLAY_MOVE = "play_move"
    CLAIM_DRAW = "claim_draw"
    CLAIM_DRAW_WITH_INTENDED_MOVE = "claim_draw_with_intended_move"
    GAME_OVER = "game_over"


DRAW_REASON_FIFTY_MOVES = "fifty_moves"
DRAW_REASON_THREEFOLD = "threefold_repetition"
DRAW_REASONS_VALID: frozenset[str] = frozenset({DRAW_REASON_FIFTY_MOVES, DRAW_REASON_THREEFOLD})


class Outcome(StrEnum):
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


class MovePayload(BaseModel):
    """A single chess move in both UCI and SAN form."""

    uci: str
    san: str | None = None


class ActionValue(BaseModel):
    """Game-theoretic value of an action from White's POV."""

    cp: int | None = None
    mate: int | None = None
    outcome: str | None = None


class PlayMoveAction(BaseModel):
    """Play the move on the board."""

    type: Literal["play_move"] = "play_move"
    move: MovePayload
    value: ActionValue | None = None


class ClaimDrawAction(BaseModel):
    """Claim a draw now."""

    type: Literal["claim_draw"] = "claim_draw"
    reason: Literal["fifty_moves", "threefold_repetition"]


class ClaimDrawWithIntendedMoveAction(BaseModel):
    """Declare an intended move and claim a draw with it."""

    type: Literal["claim_draw_with_intended_move"] = "claim_draw_with_intended_move"
    reason: Literal["fifty_moves", "threefold_repetition"]
    intended_move: MovePayload


class GameOverAction(BaseModel):
    """Game is over, no action possible."""

    type: Literal["game_over"] = "game_over"
    outcome: Literal["win", "loss", "draw"]
    reason: str


GameAction = PlayMoveAction | ClaimDrawAction | ClaimDrawWithIntendedMoveAction | GameOverAction
Action = GameAction

_action_adapter: TypeAdapter[GameAction] = TypeAdapter(Annotated[GameAction, Discriminator("type")])


def parse_action(payload: dict[str, Any]) -> GameAction:
    """Validate and re-parse a dict-shaped action payload into the typed model."""
    return _action_adapter.validate_python(payload)


def _build_play_move_dict(
    move_uci: str,
    move_san: str | None,
    cp: int | None = None,
    mate: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": TypeOfAction.PLAY_MOVE.value,
        "move": {"uci": move_uci, "san": move_san},
    }
    if cp is not None or mate is not None:
        payload["value"] = {"cp": cp, "mate": mate}
    return payload


def _build_claim_draw_dict(reason: str) -> dict[str, Any]:
    return {"type": TypeOfAction.CLAIM_DRAW.value, "reason": reason}


def _build_claim_with_intended_dict(
    reason: str, intended_move_uci: str, intended_move_san: str | None
) -> dict[str, Any]:
    return {
        "type": TypeOfAction.CLAIM_DRAW_WITH_INTENDED_MOVE.value,
        "reason": reason,
        "intended_move": {"uci": intended_move_uci, "san": intended_move_san},
    }


def _build_game_over_dict(outcome: str, reason: str) -> dict[str, Any]:
    return {"type": TypeOfAction.GAME_OVER.value, "outcome": outcome, "reason": reason}


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
    if action_type == TypeOfAction.PLAY_MOVE.value:
        return _build_play_move_dict(move_uci, move_san, cp=cp, mate=mate)
    if action_type == TypeOfAction.CLAIM_DRAW.value:
        if not rule_status.can_claim_now:
            raise ValueError("ILLEGAL_ACTION: draw cannot be claimed now")
        reason = (
            rule_status.claim_reasons_now[0]
            if rule_status.claim_reasons_now
            else DRAW_REASON_THREEFOLD
        )
        return _build_claim_draw_dict(reason)
    if action_type == TypeOfAction.CLAIM_DRAW_WITH_INTENDED_MOVE.value:
        if move_uci not in (rule_status.intended_claim_ucis or []):
            raise ValueError("ILLEGAL_ACTION: intended move does not create a legal draw claim")
        reason_map = getattr(rule_status, "intended_claim_reasons_by_uci", {}) or {}
        reasons = reason_map.get(move_uci) or rule_status.claim_reasons or [DRAW_REASON_THREEFOLD]
        return _build_claim_with_intended_dict(reasons[0], move_uci, move_san)
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
        return _build_game_over_dict(outcome, rule_status.terminal)

    if recommended_action == TypeOfAction.CLAIM_DRAW.value and rule_status.can_claim_now:
        reason = (
            rule_status.claim_reasons_now[0]
            if rule_status.claim_reasons_now
            else DRAW_REASON_THREEFOLD
        )
        return _build_claim_draw_dict(reason)

    if (
        recommended_action == TypeOfAction.CLAIM_DRAW_WITH_INTENDED_MOVE.value
        and rule_status.can_claim_with_intended_move
    ):
        intended_uci = rule_status.claim_move_uci
        intended_san = rule_status.claim_move_san or rule_status.claim_move
        if intended_uci:
            reason_map = getattr(rule_status, "intended_claim_reasons_by_uci", {}) or {}
            reasons = (
                reason_map.get(intended_uci) or rule_status.claim_reasons or [DRAW_REASON_THREEFOLD]
            )
            return _build_claim_with_intended_dict(reasons[0], intended_uci, intended_san)

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
        return _build_play_move_dict(bm_uci, bm_san, cp=engine_eval.cp, mate=engine_eval.mate)

    return _build_play_move_dict("", None)


def build_legal_actions(
    rule_status: Any,
    engine_eval: Any | None,
    board: Any | None,
    legal_engine_moves: list[Any] | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    if rule_status.can_claim_now:
        for reason in rule_status.claim_reasons_now:
            actions.append(_build_claim_draw_dict(reason))

    if rule_status.can_claim_with_intended_move:
        intended_ucis = list(rule_status.intended_claim_ucis or [])
        intended_sans = list(rule_status.intended_claim_sans or [])
        if not intended_ucis and rule_status.claim_move_uci:
            intended_ucis = [rule_status.claim_move_uci]
            intended_sans = [rule_status.claim_move_san or rule_status.claim_move]
        reason_map = getattr(rule_status, "intended_claim_reasons_by_uci", {}) or {}
        for i, uci in enumerate(intended_ucis):
            san = intended_sans[i] if i < len(intended_sans) else None
            reasons = reason_map.get(uci) or rule_status.claim_reasons or [DRAW_REASON_THREEFOLD]
            for reason in reasons:
                if reason in DRAW_REASONS_VALID:
                    actions.append(_build_claim_with_intended_dict(reason, uci, san))

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
            actions.append(_build_play_move_dict(ev.best_move, bm_san, cp=ev.cp, mate=ev.mate))

    if rule_status.terminal is not None:
        if rule_status.terminal == "checkmate":
            outcome = "win" if rule_status.winner == "white" else "loss"
        else:
            outcome = "draw"
        actions.append(_build_game_over_dict(outcome, rule_status.terminal))

    return actions
