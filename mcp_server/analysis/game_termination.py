"""Evidence-bounded PGN termination and resignation analysis.

The goal is to answer a coaching question without inventing facts: was the
recorded ending actually a resignation, and if so was the final position
objectively forced? A PGN result alone is not enough to distinguish resignation
from timeout/adjudication, so non-terminal decisive games without an explicit
Termination header are deliberately reported only as candidates.
"""

from __future__ import annotations

import io
from typing import Literal

import chess
import chess.pgn

from mcp_server.models.game_coaching import (
    FinalPositionAssessment,
    GameTerminationAssessment,
)

Side = Literal["white", "black"]


def _winner_loser(result: str) -> tuple[Side | None, Side | None]:
    if result == "1-0":
        return "white", "black"
    if result == "0-1":
        return "black", "white"
    return None, None


def _final_board(pgn: str) -> tuple[chess.pgn.Game, chess.Board]:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError("INVALID_PGN: no game found")
    board = game.board()
    for move in game.mainline_moves():
        if move not in board.legal_moves:
            raise ValueError(f"INVALID_PGN: illegal mainline move {move.uci()}")
        board.push(move)
    return game, board


def _explicit_resignation(termination_header: str | None) -> bool:
    if not termination_header:
        return False
    normalized = termination_header.strip().lower()
    return "resign" in normalized


def _mate_against_loser(
    *,
    loser: Side | None,
    board: chess.Board,
    final_position: FinalPositionAssessment,
) -> bool | None:
    if loser is None:
        return None
    if board.is_checkmate():
        return True
    mate = final_position.mate_distance
    if mate is None:
        return False
    # MCPEval mate values are White POV in this repository.
    if mate == 0:
        return board.is_checkmate()
    return (loser == "black" and mate > 0) or (loser == "white" and mate < 0)


def build_game_termination_assessment(
    pgn: str,
    *,
    final_position: FinalPositionAssessment,
) -> GameTerminationAssessment:
    """Classify the recorded ending without equating a bad eval with resignation.

    `objectively_forced` is intentionally narrow: for a resignation-style ending
    it becomes true only when the losing side is already in a forced-mate state.
    A position such as -3, -5 or even -10 with no forced mate remains false.
    """
    game, board = _final_board(pgn)
    result = str(game.headers.get("Result", "*") or "*").strip()
    termination_raw = str(game.headers.get("Termination", "") or "").strip()
    termination_header = termination_raw or None
    winner, loser = _winner_loser(result)
    decisive = winner is not None
    explicit_resignation = _explicit_resignation(termination_header)
    terminal = board.is_game_over(claim_draw=False)
    checkmate = board.is_checkmate()

    if checkmate:
        status = "board_checkmate"
        confidence = "high"
    elif terminal:
        status = "other_terminal_result"
        confidence = "high"
    elif explicit_resignation and decisive:
        status = "explicit_resignation"
        confidence = "high"
    elif decisive and termination_header is None:
        status = "candidate_nonterminal_decisive_result"
        confidence = "medium" if loser == final_position.side_to_move else "low"
    else:
        status = "ongoing_or_unknown"
        confidence = "medium" if termination_header else "low"

    forced_mate_against_loser = _mate_against_loser(
        loser=loser,
        board=board,
        final_position=final_position,
    )

    objectively_forced: bool | None
    if loser is None:
        objectively_forced = None
    elif checkmate:
        objectively_forced = True
    else:
        objectively_forced = bool(forced_mate_against_loser)

    eval_for_loser: int | None = None
    if loser is not None:
        eval_for_loser = (
            final_position.effective_cp
            if final_position.perspective == loser
            else -final_position.effective_cp
        )

    loser_to_move = loser is not None and final_position.side_to_move == loser
    return GameTerminationAssessment(
        pgn_result=result,
        termination_header=termination_header,
        decisive_result=decisive,
        winner_side=winner,
        loser_side=loser,
        status=status,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        final_board_terminal=terminal,
        final_board_checkmate=checkmate,
        continued_play_was_legal=not terminal and board.legal_moves.count() > 0,
        forced_mate_against_loser=forced_mate_against_loser,
        objectively_forced=objectively_forced,
        mate_distance_white_pov=final_position.mate_distance,
        eval_for_loser_effective_cp=eval_for_loser,
        best_defensive_move_uci=(final_position.best_move_uci if loser_to_move else None),
        best_defensive_move_san=(final_position.best_move_san if loser_to_move else None),
        legal_resource_count=final_position.legal_move_count if loser_to_move else 0,
        reasonable_resource_count=(
            final_position.reasonable_resource_count if loser_to_move else None
        ),
        defensive_resources_exist=(
            final_position.defensive_resources_exist if loser_to_move else False
        ),
    )
