"""Regression coverage for evidence-bounded resignation/termination analysis."""

from __future__ import annotations

from mcp_server.analysis.game_termination import build_game_termination_assessment
from mcp_server.models.game_coaching import FinalPositionAssessment


def _final(
    *,
    perspective: str = "white",
    terminal: bool = False,
    checkmate: bool = False,
    mate: int | None = None,
    effective_cp: int = -420,
    side_to_move: str = "white",
    legal_moves: int = 27,
    best_uci: str | None = "f1b5",
    best_san: str | None = "Bb5",
    reasonable: int | None = 3,
) -> FinalPositionAssessment:
    return FinalPositionAssessment(
        perspective=perspective,  # type: ignore[arg-type]
        position_terminal_by_rules=terminal,
        checkmate=checkmate,
        stalemate=False,
        forced_mate=checkmate or mate is not None,
        mate_distance=mate,
        effective_cp=effective_cp,
        wdl=(80, 120, 800),
        side_to_move=side_to_move,  # type: ignore[arg-type]
        legal_move_count=legal_moves,
        best_move_uci=best_uci,
        best_move_san=best_san,
        defensive_resources_exist=not terminal and legal_moves > 0,
        reasonable_resource_count=reasonable,
        verification_depth=22,
    )


def test_explicit_resignation_is_confirmed_but_bad_eval_is_not_called_forced() -> None:
    pgn = """[Event \"Resignation\"]
[White \"mes77777\"]
[Black \"Opponent\"]
[Result \"0-1\"]
[Termination \"White resigns\"]

1. e4 e5 2. Nf3 Nc6 0-1
"""
    assessment = build_game_termination_assessment(pgn, final_position=_final())

    assert assessment.status == "explicit_resignation"
    assert assessment.confidence == "high"
    assert assessment.loser_side == "white"
    assert assessment.objectively_forced is False
    assert assessment.forced_mate_against_loser is False
    assert assessment.eval_for_loser_effective_cp == -420
    assert assessment.continued_play_was_legal is True
    assert assessment.best_defensive_move_san == "Bb5"
    assert assessment.legal_resource_count == 27
    assert assessment.reasonable_resource_count == 3


def test_nonterminal_decisive_result_without_header_is_only_resignation_candidate() -> None:
    pgn = """[Event \"Unknown decisive ending\"]
[White \"mes77777\"]
[Black \"Opponent\"]
[Result \"0-1\"]

1. e4 e5 2. Nf3 Nc6 0-1
"""
    assessment = build_game_termination_assessment(pgn, final_position=_final())

    assert assessment.status == "candidate_nonterminal_decisive_result"
    assert assessment.confidence == "medium"
    assert assessment.objectively_forced is False


def test_explicit_non_resignation_termination_is_not_relabelled_as_resignation() -> None:
    pgn = """[Event \"Timeout\"]
[White \"mes77777\"]
[Black \"Opponent\"]
[Result \"0-1\"]
[Termination \"time forfeit\"]

1. e4 e5 2. Nf3 Nc6 0-1
"""
    assessment = build_game_termination_assessment(pgn, final_position=_final())

    assert assessment.status == "ongoing_or_unknown"
    assert assessment.termination_header == "time forfeit"
    assert assessment.objectively_forced is False


def test_board_checkmate_is_objectively_forced_and_has_no_legal_continuation() -> None:
    pgn = """[Event \"Mate\"]
[White \"mes77777\"]
[Black \"Opponent\"]
[Result \"0-1\"]

1. f3 e5 2. g4 Qh4# 0-1
"""
    final = _final(
        terminal=True,
        checkmate=True,
        mate=-1,
        effective_cp=-99_999,
        legal_moves=0,
        best_uci=None,
        best_san=None,
        reasonable=0,
    )
    assessment = build_game_termination_assessment(pgn, final_position=final)

    assert assessment.status == "board_checkmate"
    assert assessment.objectively_forced is True
    assert assessment.forced_mate_against_loser is True
    assert assessment.continued_play_was_legal is False
    assert assessment.legal_resource_count == 0
