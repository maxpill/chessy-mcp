"""Typed action discriminated union for the MCP API contract.

Audit invariant I-03: a client must never confuse a `claim_draw` with a
`play_move`. The typed union makes that structurally impossible. The legacy
dict-builder functions at the bottom keep existing wire contracts working.
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


def _san_for_uci(board: chess.Board, uci: str | None) -> str | None:
    if not uci:
        return None
    try:
        move = chess.Move.from_uci(uci.lower())
        if move in board.legal_moves:
            return board.san(move)
    except Exception:
        pass
    return None
