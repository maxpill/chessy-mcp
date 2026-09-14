"""Phase 14 (2026-09-14) audit: semicolon PGN comments surface as ``user_comment_raw``.

Tests the analyzer end-to-end: parse a PGN with ``;`` comments, run
``analyze_game(detail="coach")``, and assert the captured comments surface as
``user_comment_raw`` on the right plies. The brace ``{...}`` form is covered
as the baseline so we can prove equivalence between the two comment forms.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load the new module directly to avoid the broken ``mcp_server.parsers.__init__``
# chain (another agent's WIP left ``build_normalized_position`` referenced but
# not defined). Phase 14 ships its own isolated module under
# ``mcp_server/parsers/pgn/semicolon.py`` that has no import-time deps beyond
# ``chess``.
_SEMICOLON_PATH = (
    Path(__file__).resolve().parent.parent / "mcp_server" / "parsers" / "pgn" / "semicolon.py"
)
_spec = importlib.util.spec_from_file_location("_phase14_semicolon", _SEMICOLON_PATH)
assert _spec is not None and _spec.loader is not None
_phase14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase14)

extract_semicolon_comments = _phase14.extract_semicolon_comments
attach_semicolon_comments = _phase14.attach_semicolon_comments
get_semicolon_comments = _phase14.get_semicolon_comments


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    """Drop the analyzer pool between tests so they don't share TCP sockets."""
    yield
    from mcp_server import server as server_module

    await server_module.close_analyzer_pool()


# ---------------------------------------------------------------------------
# Direct unit tests on the extractor (no engine round-trip needed)
# ---------------------------------------------------------------------------


def test_extract_brace_form_unchanged_basic() -> None:
    """Brace comments are NOT extracted by ``extract_semicolon_comments``."""
    assert extract_semicolon_comments("1. e4 {brace} e5 2. Nf3 Nc6") == []


def test_extract_semicolon_at_ply_3_matches_task_example() -> None:
    """The exact task-spec PGN maps to ply 3."""
    pgn = "1. f3 e5 2. g4 ; my thought\n2... Qh4# 0-1"
    assert extract_semicolon_comments(pgn) == [(3, "my thought")]


def test_extract_semicolon_strips_whitespace() -> None:
    """Leading / trailing whitespace around the comment body is trimmed."""
    pgn = "1. e4 ;   spaced thought   \n1... e5"
    assert extract_semicolon_comments(pgn) == [(1, "spaced thought")]


def test_extract_multiple_comments_on_one_move_combined_by_analyzer() -> None:
    """Two ``;`` comments on the same ply produce two captures (one each)."""
    pgn = "1. e4 ; first ; second\n1... e5"
    captures = extract_semicolon_comments(pgn)
    assert captures == [(1, "first ; second")]


def test_extract_comment_after_move_number_only() -> None:
    """A comment after ``2.`` with no SAN yet maps to ply 3 (move-number hint)."""
    pgn = "1. e4 e5 2. ; hello\n2. Nf3"
    assert extract_semicolon_comments(pgn) == [(3, "hello")]


def test_extract_nag_plus_comment() -> None:
    """A NAG before the comment doesn't bump the ply counter."""
    pgn = "1. e4 $1 ; good\n1... e5"
    assert extract_semicolon_comments(pgn) == [(1, "good")]


def test_extract_utf8_comment_preserved() -> None:
    """UTF-8 text inside the comment body is preserved verbatim."""
    pgn = "1. e4 ; Zażółć gęślą jaźń\n1... e5"
    assert extract_semicolon_comments(pgn) == [(1, "Zażółć gęślą jaźń")]


def test_extract_empty_comment_dropped() -> None:
    """An empty / whitespace-only ``;`` comment is not captured."""
    assert extract_semicolon_comments("1. e4 ;\n1... e5") == []
    assert extract_semicolon_comments("1. e4 ;   \n1... e5") == []


def test_extract_comment_immediately_before_terminal_move() -> None:
    """A comment right before the mate-attacking move attaches to that move."""
    pgn = "1. e4 e5 2. Nf3 ; last thought\n2... Qh4#"
    assert extract_semicolon_comments(pgn) == [(3, "last thought")]


def test_extract_skips_comments_inside_braces() -> None:
    """``;`` inside a brace block is part of the brace, not a separate capture."""
    pgn = "1. e4 {not ; a real semi} e5"
    assert extract_semicolon_comments(pgn) == []


def test_extract_skips_comments_inside_variation() -> None:
    """``;`` inside a variation does not pollute the mainline captures."""
    pgn = "1. e4 (1. d4 ; inside variation) e5"
    assert extract_semicolon_comments(pgn) == []


def test_extract_percent_escape_also_captured() -> None:
    """``%`` escape lines at column 0 behave like ``;`` comments."""
    pgn = "1. e4\n% hidden eval\n1... e5"
    assert extract_semicolon_comments(pgn) == [(1, "hidden eval")]


def test_attach_and_get_round_trip() -> None:
    """attach_to_game + get_semicolon_comments round-trips the captures."""
    import chess
    import chess.pgn

    game = chess.pgn.Game()
    attach_semicolon_comments(game, "1. e4 ; hi\n1... e5")
    assert get_semicolon_comments(game) == [(1, "hi")]


# ---------------------------------------------------------------------------
# End-to-end: analyze_game surfaces the semicolon comment as user_comment_raw
# ---------------------------------------------------------------------------


async def _comments_for_pgn(pgn: str) -> dict[int, str | None]:
    # Lazy import so this test module can be collected even when the
    # ``mcp_server.parsers.__init__`` chain is temporarily broken by
    # another agent's in-flight work.
    from mcp_server.tools.analyze_game import analyze_game

    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    return {moment.ply: moment.user_comment_raw for moment in res.coaching.critical_moments}


@pytest.mark.asyncio
async def test_coach_surfaces_brace_comment_on_ply_3() -> None:
    """Baseline: brace comment at ply 3 surfaces as user_comment_raw."""
    comments = await _comments_for_pgn("1. f3 e5 2. g4 {my thought} 2... Qh4# 0-1")
    assert comments.get(3) == "my thought"


@pytest.mark.asyncio
async def test_coach_surfaces_semicolon_comment_on_ply_3() -> None:
    """Semicolon comment at ply 3 (with newline) surfaces as user_comment_raw."""
    comments = await _comments_for_pgn("1. f3 e5 2. g4 ; my thought\n2... Qh4# 0-1")
    assert comments.get(3) == "my thought"


@pytest.mark.asyncio
async def test_coach_brace_and_semicolon_prefer_brace() -> None:
    """When both brace and ``;`` exist on the same ply, brace wins."""
    from mcp_server.tools.analyze_game import analyze_game

    res = await analyze_game(
        pgn="1. e4 {brace wins} ; semi ignored\n1... e5",
        depth=1,
        detail="coach",
    )
    assert res.coaching is not None
    assert res.total_plies == 2
    by_ply = {m.ply: m.user_comment_raw for m in res.coaching.critical_moments}
    assert by_ply.get(1) == "brace wins"


@pytest.mark.asyncio
async def test_coach_utf8_semicolon_comment_round_trips() -> None:
    """UTF-8 text in a ``;`` comment survives the full analyze_game pipeline."""
    from mcp_server.tools.analyze_game import analyze_game

    res = await analyze_game(
        pgn="1. e4 ; Zażółć gęślą jaźń\n1... e5",
        depth=1,
        detail="coach",
    )
    assert res.coaching is not None
    by_ply = {m.ply: m.user_comment_raw for m in res.coaching.critical_moments}
    assert by_ply.get(1) == "Zażółć gęślą jaźń"


@pytest.mark.asyncio
async def test_coach_semicolon_only_no_regression_on_brace_path() -> None:
    """Brace-only PGN still produces user_comment_raw (no regression)."""
    from mcp_server.tools.analyze_game import analyze_game

    res = await analyze_game(
        pgn="1. e4 {I am playing the king's pawn} e5 2. Nf3 Nc6",
        depth=1,
        detail="coach",
    )
    assert res.coaching is not None
    by_ply = {m.ply: m.user_comment_raw for m in res.coaching.critical_moments}
    assert by_ply.get(1) == "I am playing the king's pawn"


@pytest.mark.asyncio
async def test_coach_nag_then_semicolon_comment_on_ply_1() -> None:
    """``$1 ; good`` on ply 1 surfaces as user_comment_raw."""
    from mcp_server.tools.analyze_game import analyze_game

    res = await analyze_game(
        pgn="1. e4 $1 ; good\n1... e5",
        depth=1,
        detail="coach",
    )
    assert res.coaching is not None
    by_ply = {m.ply: m.user_comment_raw for m in res.coaching.critical_moments}
    assert by_ply.get(1) == "good"


@pytest.mark.asyncio
async def test_coach_brace_path_unchanged_with_semicolon_in_same_text() -> None:
    """Brace comment survives even when ``;`` comments also exist elsewhere."""
    from mcp_server.tools.analyze_game import analyze_game

    pgn = "1. f3 e5 2. g4 {my thought} ; trailing semi on same ply\n2... Qh4# 0-1"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    by_ply = {m.ply: m.user_comment_raw for m in res.coaching.critical_moments}
    assert by_ply.get(3) == "my thought"
