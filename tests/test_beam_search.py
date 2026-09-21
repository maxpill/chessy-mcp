"""Tests for :mod:`mcp_server.ocr.beam_search`."""

from __future__ import annotations

import chess
import pytest

from mcp_server.ocr.beam_search import (
    BeamRerankError,
    CellCandidate,
    NeedsReviewError,
    beam_rescore,
    candidates_from_pairs,
)


def _cands(pairs: list[tuple[int, str, float]]) -> list[CellCandidate]:
    return candidates_from_pairs(pairs)


# --- happy path -------------------------------------------------------------


def test_beam_rescore_happy_path_returns_legal_path() -> None:
    cands = _cands(
        [
            (1, "e4", 0.9),
            (2, "e5", 0.9),
            (3, "Nf3", 0.9),
            (4, "Nc6", 0.9),
            (5, "Bb5", 0.9),
            (6, "a6", 0.9),
        ]
    )

    result = beam_rescore(cands, beam_width=5, resolve="auto")

    assert result.status == "ok"
    assert result.selected_path == ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6")
    assert result.uncertainties == ()


def test_beam_rescore_empty_input() -> None:
    result = beam_rescore([], beam_width=5)
    assert result.selected_path == ()
    assert result.status == "ok"


# --- local illegality filtering ---------------------------------------------


def test_beam_rescore_skips_illegal_candidates() -> None:
    # Ply 3 (white) candidates include Nf3 (legal) and Nc6 (illegal — black's move).
    cands = _cands(
        [
            (1, "e4", 0.9),
            (2, "c5", 0.9),
            (3, "Nf3", 0.9),
            (3, "Nc6", 0.9),
            (4, "d6", 0.9),
            (5, "d4", 0.9),
        ]
    )

    result = beam_rescore(cands, beam_width=5)

    assert "Nf3" in result.selected_path
    assert "Nc6" not in result.selected_path


def test_beam_rescore_downstream_bonus_favours_coherent_prefix() -> None:
    # At ply 3 (white), candidates "exd5" (lower OCR score 0.5) and "Nf3" (0.55).
    # Top-next at ply 4 is "Qxd5" — only legal AFTER exd5 captured the pawn.
    # exd5 wins despite lower OCR score because the downstream bonus rewards it.
    cands = _cands(
        [
            (1, "e4", 0.9),
            (2, "d5", 0.9),
            (3, "exd5", 0.5),
            (3, "Nf3", 0.55),
            (4, "Qxd5", 0.9),
            (4, "Nc6", 0.9),
        ]
    )

    result = beam_rescore(cands, beam_width=5)

    assert result.selected_path[0] == "e4"
    assert result.selected_path[1] == "d5"
    assert result.selected_path[2] == "exd5"


def test_beam_rescore_raises_when_no_continuation_legal() -> None:
    cands = _cands(
        [
            (1, "e4", 0.9),
            (2, "e5", 0.9),
            (3, "Nc6", 0.9),
            (3, "Qxf7", 0.9),
            (4, "???", 0.9),
        ]
    )

    with pytest.raises(BeamRerankError):
        beam_rescore(cands, beam_width=5)


# --- ambiguous plies --------------------------------------------------------


def test_beam_rescore_flags_two_ambiguous_plies() -> None:
    cands = _cands(
        [
            (1, "e4", 0.9),
            (2, "e5", 0.9),
            (3, "Nf3", 0.5),
            (3, "Nc3", 0.5),
            (4, "Nc6", 0.9),
            (5, "Bb5", 0.5),
            (5, "Bc4", 0.5),
            (6, "a6", 0.9),
        ]
    )

    result = beam_rescore(cands, beam_width=5, resolve="auto")

    assert result.status == "needs_review"
    assert len(result.uncertainties) == 2
    by_ply = {u.ply: u for u in result.uncertainties}
    assert set(by_ply) == {3, 5}
    for ply in (3, 5):
        assert by_ply[ply].alternatives  # at least one alternative


def test_beam_rescore_strict_raises_needs_review() -> None:
    cands = _cands(
        [
            (1, "e4", 0.9),
            (2, "e5", 0.9),
            (3, "Nf3", 0.5),
            (3, "Nc3", 0.5),
        ]
    )

    with pytest.raises(NeedsReviewError) as ei:
        beam_rescore(cands, beam_width=5, resolve="strict")
    assert len(ei.value.uncertainties) == 1


def test_beam_rescore_best_effort_returns_path() -> None:
    cands = _cands(
        [
            (1, "e4", 0.9),
            (2, "e5", 0.9),
            (3, "Nf3", 0.5),
            (3, "Nc3", 0.5),
        ]
    )

    result = beam_rescore(cands, beam_width=5, resolve="best_effort")

    assert result.selected_path[2] in {"Nf3", "Nc3"}
    assert any(
        u.reason in {"handwriting_ambiguity", "low_confidence"} for u in result.uncertainties
    )


# --- engine tiebreak --------------------------------------------------------


def test_beam_rescore_engine_tiebreak_demotes_blunder() -> None:
    cands = _cands(
        [
            (1, "e4", 0.9),
            (2, "e5", 0.9),
            (3, "Nf3", 0.55),
            (3, "Nc3", 0.5),
        ]
    )

    def engine_eval(before: chess.Board, after: chess.Board, last_san: str) -> float:
        # cp_before = eval of the before-board; cp_after = eval after the move.
        # Both calls share the same last_san; we distinguish via FEN equality.
        if before.fen() == after.fen():
            return 0.3  # cp_before call
        return -8.0 if last_san == "Nf3" else 0.3  # cp_after call

    result = beam_rescore(
        cands,
        beam_width=5,
        resolve="auto",
        engine_plausibility="tiebreak_only",
        engine_eval=engine_eval,
    )

    assert result.selected_path[2] == "Nc3"
    assert any("engine_tiebreak" in n for n in result.notes)


def test_beam_rescore_engine_plausibility_off_skips_eval() -> None:
    cands = _cands(
        [
            (1, "e4", 0.9),
            (2, "e5", 0.9),
            (3, "Nf3", 0.55),
            (3, "Nc3", 0.5),
        ]
    )

    called: list[int] = []

    def engine_eval(before: chess.Board, after: chess.Board, last_san: str) -> float:
        called.append(1)
        return 0.0

    beam_rescore(
        cands,
        beam_width=5,
        engine_plausibility="off",
        engine_eval=engine_eval,
    )
    assert called == []


# --- input validation -------------------------------------------------------


def test_beam_rescore_rejects_non_positive_beam_width() -> None:
    with pytest.raises(ValueError):
        beam_rescore(_cands([(1, "e4", 0.9)]), beam_width=0)


# --- confidence reporting ---------------------------------------------------


def test_beam_rescore_confidence_drops_on_close_call() -> None:
    cands_close = _cands(
        [
            (1, "e4", 0.9),
            (2, "e5", 0.9),
            (3, "Nf3", 0.51),
            (3, "Nc3", 0.50),
        ]
    )
    cands_clear = _cands(
        [
            (1, "e4", 0.9),
            (2, "e5", 0.9),
            (3, "Nf3", 0.9),
            (3, "Nc3", 0.1),
        ]
    )

    close = beam_rescore(cands_close, beam_width=5, resolve="auto")
    clear = beam_rescore(cands_clear, beam_width=5, resolve="auto")

    assert close.status == "needs_review"
    assert clear.status == "ok"
    assert clear.confidence >= close.confidence


# --- candidates_from_pairs helper ------------------------------------------


def test_candidates_from_pairs_assigns_side() -> None:
    out = candidates_from_pairs([(1, "e4", 0.5), (2, "e5", 0.5), (3, "Nf3", 0.5)])
    assert [c.side for c in out] == ["white", "black", "white"]
    assert [c.ply for c in out] == [1, 2, 3]
