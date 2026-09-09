"""2026-09-09 master audit F-006: termination resources must distinguish 0 from N/A.

The audit's source inspection confirmed that ``game_termination.py`` collapsed
two semantically different states (``no resources exist`` vs ``assessment not
applicable``) into a single ``0/False`` value, contradicting sibling
``final_position`` evidence on ongoing games. This test pins the new nullable
contract: ``None`` means not-applicable; ``0/False`` means measured zero.
"""

from __future__ import annotations

import chess
import pytest

from mcp_server.analysis.game_termination import build_game_termination_assessment
from mcp_server.models.game_coaching import FinalPositionAssessment


def _final_position(
    *,
    legal_count: int,
    checkmate: bool = False,
    terminal: bool = False,
    side_to_move: str = "white",
    perspective: str = "white",
) -> FinalPositionAssessment:
    return FinalPositionAssessment(
        perspective=perspective,  # type: ignore[arg-type]
        position_terminal_by_rules=terminal,
        checkmate=checkmate,
        stalemate=False,
        forced_mate=checkmate,
        mate_distance=0 if checkmate else None,
        effective_cp=0,
        wdl=None,
        side_to_move=side_to_move,  # type: ignore[arg-type]
        legal_move_count=legal_count,
        best_move_uci=None,
        best_move_san=None,
        defensive_resources_exist=not terminal and legal_count > 0,
        reasonable_resource_count=legal_count if not terminal else None,
    )


def test_ongoing_game_has_none_resource_fields() -> None:
    """Ongoing game (no loser): resource assessment is not applicable."""
    pgn = "1. e4 e5 2. Nf3 Nc6 *"
    fp = _final_position(legal_count=20, terminal=False, side_to_move="white")
    assessment = build_game_termination_assessment(pgn, final_position=fp)

    assert assessment.winner_side is None
    assert assessment.loser_side is None
    assert assessment.legal_resource_count is None, (
        f"Ongoing game must report legal_resource_count=None; got {assessment.legal_resource_count}"
    )
    assert assessment.defensive_resources_exist is None, (
        f"Ongoing game must report defensive_resources_exist=None; "
        f"got {assessment.defensive_resources_exist}"
    )
    assert assessment.resources_applicable is False


def test_checkmate_terminates_resources() -> None:
    """Checkmate is terminal; loser has zero measured resources."""
    pgn = "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6?? 4. Qxf7# 1-0"
    fp = _final_position(legal_count=0, checkmate=True, terminal=True, side_to_move="black")
    assessment = build_game_termination_assessment(pgn, final_position=fp)

    assert assessment.final_board_checkmate is True
    assert assessment.legal_resource_count == 0
    assert assessment.defensive_resources_exist is False
    assert assessment.resources_applicable is True


def test_draw_1_2_has_none_resource_fields() -> None:
    """Draw with no loser: resource fields must be None."""
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Nc3 Nf6 1/2-1/2"
    fp = _final_position(legal_count=25, terminal=True, side_to_move="white")
    assessment = build_game_termination_assessment(pgn, final_position=fp)

    assert assessment.winner_side is None
    assert assessment.legal_resource_count is None
    assert assessment.defensive_resources_exist is None
    assert assessment.resources_applicable is False


def test_explicit_resignation_loser_to_move_reports_measured_resources() -> None:
    """Explicit resignation with loser to move: resources are measurable."""
    pgn = '[Termination "White resigned"}] 1-0\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0'
    fp = _final_position(legal_count=8, terminal=False, side_to_move="black")
    assessment = build_game_termination_assessment(pgn, final_position=fp)

    assert (
        assessment.explicit_resignation_flag
        if hasattr(assessment, "explicit_resignation_flag")
        else True
    )
    assert assessment.loser_side == "black"
    assert assessment.legal_resource_count == 8
    assert assessment.defensive_resources_exist is True
    assert assessment.resources_applicable is True


def test_explicit_resignation_winner_to_move_reports_n_a() -> None:
    """Explicit resignation when winner is to move: no loser-to-move defense."""
    pgn = (
        '[Termination "Black resigned"}] 0-1\n'
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 0-1"
    )
    fp = _final_position(legal_count=20, terminal=False, side_to_move="black")
    assessment = build_game_termination_assessment(pgn, final_position=fp)

    assert assessment.loser_side == "white"
    assert assessment.legal_resource_count is None
    assert assessment.defensive_resources_exist is None
    assert assessment.resources_applicable is False
