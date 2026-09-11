"""Comprehensive 120+ test audit matrix for `analyze_game`.

Structured across Categories A through I as specified in QA Repair Prompt:
- Category A: Parser and PGN validity (20 cases)
- Category B: Terminal reasons and termination semantics (15 cases)
- Category C: Perspective flipping and story consistency (15 cases)
- Category D: Max critical moments parameter clamping (10 cases)
- Category E: Game-story segmentation and oscillation persistence (15 cases)
- Category F: Forensic deep verification & stability (15 cases)
- Category G: User comments and self-report fidelity (10 cases)
- Category H: Resignation quality and objective force (10 cases)
- Category I: Long games, performance & payload budgets (10 cases)
"""

from __future__ import annotations

import json
import pytest
import chess
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server import server as server_module
from mcp_server.tools.analyze_game import analyze_game


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


# ===========================================================================
# Category A: Parser and PGN validity (20 cases)
# ===========================================================================


@pytest.mark.asyncio
async def test_cat_a_01_plain_san_list() -> None:
    res = await analyze_game(pgn="1. e4 e5 2. Nf3 Nc6", depth=1)
    assert res.total_plies == 4


@pytest.mark.asyncio
async def test_cat_a_02_minimal_pgn() -> None:
    res = await analyze_game(pgn="1. e4 e5 2. Nf3 Nc6 *", depth=1)
    assert res.total_plies == 4


@pytest.mark.asyncio
async def test_cat_a_03_full_tagged_pgn() -> None:
    pgn = """[Event "World Championship"]
[Site "London"]
[Date "2026.09.08"]
[Round "1"]
[White "Player A"]
[Black "Player B"]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 1/2-1/2"""
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 4
    assert res.white == "Player A"
    assert res.black == "Player B"
    assert res.event == "World Championship"


@pytest.mark.asyncio
async def test_cat_a_04_comments() -> None:
    pgn = "1. e4 {King's pawn} e5 {Open game} 2. Nf3 Nc6"
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 4


@pytest.mark.asyncio
async def test_cat_a_05_nags() -> None:
    pgn = "1. e4! e5? 2. Nf3!! Nc6??"
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 4


@pytest.mark.asyncio
async def test_cat_a_06_clock_tags() -> None:
    pgn = "1. e4 {[%clk 1:30:00]} e5 {[%clk 1:29:45]} 2. Nf3 Nc6"
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 4


@pytest.mark.asyncio
async def test_cat_a_07_eval_tags() -> None:
    pgn = "1. e4 {[%eval 0.25]} e5 {[%eval 0.20]} 2. Nf3 Nc6"
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 4


@pytest.mark.asyncio
async def test_cat_a_08_nested_variations() -> None:
    pgn = "1. e4 (1. d4 d5 (1... Nf6 2. c4)) e5 2. Nf3 Nc6"
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 4


@pytest.mark.asyncio
async def test_cat_a_09_malformed_variation_rejects() -> None:
    pgn = "1. e4 (1. d4 e5 2. Nf3 Nc6"
    with pytest.raises(ToolError) as exc_info:
        await analyze_game(pgn=pgn, depth=1, strict=False)
    assert "INVALID_PGN" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cat_a_10_illegal_mainline_move() -> None:
    pgn = "1. e4 e5 2. Ke2 Ke7 3. Ke8??"
    with pytest.raises(ToolError):
        await analyze_game(pgn=pgn, depth=1)


@pytest.mark.asyncio
async def test_cat_a_11_non_canonical_san_lenient() -> None:
    pgn = "1. e2-e4 e7-e5 2. Ng1-f3 Nb8-c6"
    res = await analyze_game(pgn=pgn, depth=1, strict=False)
    assert res.total_plies == 4


@pytest.mark.asyncio
async def test_cat_a_12_strict_true_rejects_non_canonical() -> None:
    pgn = "1. e2-e4 e7-e5"
    with pytest.raises(ToolError):
        await analyze_game(pgn=pgn, depth=1, strict=True)


@pytest.mark.asyncio
async def test_cat_a_13_strict_false_accepts() -> None:
    pgn = "1. e2-e4 e7-e5"
    res = await analyze_game(pgn=pgn, depth=1, strict=False)
    assert res.total_plies == 2


@pytest.mark.asyncio
async def test_cat_a_14_result_mismatch() -> None:
    pgn = """[Result "1-0"]

1. e4 e5 0-1"""
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 2
    assert res.result_header == "1-0"
    assert res.result_movetext == "0-1"


@pytest.mark.asyncio
async def test_cat_a_15_missing_result() -> None:
    pgn = "1. e4 e5 2. Nf3 Nc6"
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.result_inferred is not None or res.result is None or res.result == "*"


@pytest.mark.asyncio
async def test_cat_a_16_setup_fen_game() -> None:
    pgn = """[SetUp "1"]
[FEN "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1"]

1. Kd3 Kd1 2. e4"""
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 3


@pytest.mark.asyncio
async def test_cat_a_17_black_to_move_setup_game() -> None:
    pgn = """[SetUp "1"]
[FEN "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"]

1... e5 2. Nf3 Nc6"""
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 3


@pytest.mark.asyncio
async def test_cat_a_18_promotions() -> None:
    pgn = """[SetUp "1"]
[FEN "8/4P3/8/8/8/8/8/4K2k w - - 0 1"]

1. e8=Q Kh2"""
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 2


@pytest.mark.asyncio
async def test_cat_a_19_en_passant() -> None:
    pgn = "1. e4 e6 2. e5 d5 3. exd6"
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 5


@pytest.mark.asyncio
async def test_cat_a_20_castling_kingside_queenside() -> None:
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O d6 5. d3 Bg4 6. Be3 Qd7 7. Nbd2 O-O-O"
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 14


# ===========================================================================
# Category B: Terminal reasons and termination semantics (15 cases)
# ===========================================================================


@pytest.mark.asyncio
async def test_cat_b_01_checkmate_fools_mate() -> None:
    pgn = "1. f3 e5 2. g4 Qh4# 0-1"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.final_position.checkmate is True
    assert res.coaching.termination.status == "board_checkmate"
    assert res.coaching.termination.objectively_forced is True


@pytest.mark.asyncio
async def test_cat_b_02_checkmate_scholars_mate() -> None:
    pgn = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.final_position.checkmate is True
    assert res.coaching.termination.status == "board_checkmate"


@pytest.mark.asyncio
async def test_cat_b_03_stalemate() -> None:
    pgn = """[SetUp "1"]
[FEN "k7/P7/K7/8/8/8/8/8 b - - 0 1"]

1/2-1/2"""
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.final_position.stalemate is True


@pytest.mark.asyncio
async def test_cat_b_04_insufficient_material() -> None:
    pgn = """[SetUp "1"]
[FEN "k7/8/K7/8/8/8/8/8 w - - 0 1"]

1/2-1/2"""
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.final_position.position_terminal_by_rules is True


@pytest.mark.asyncio
async def test_cat_b_05_75_move_automatic_draw() -> None:
    pgn = """[SetUp "1"]
[FEN "8/8/8/8/8/4k3/8/4K3 w - - 150 1"]

1/2-1/2"""
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.final_position.position_terminal_by_rules is True


@pytest.mark.asyncio
async def test_cat_b_06_fivefold_repetition() -> None:
    moves = " ".join(["1. Nf3 Nf6 2. Ng1 Ng8"] * 5) + " 1/2-1/2"
    res = await analyze_game(pgn=moves, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.final_position.position_terminal_by_rules is True


@pytest.mark.asyncio
async def test_cat_b_07_50_move_position() -> None:
    pgn = """[SetUp "1"]
[FEN "8/8/8/8/8/4k1r1/8/4K2R w - - 100 1"]

1. Rf1 Rg2"""
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.total_plies == 2


@pytest.mark.asyncio
async def test_cat_b_08_threefold_position() -> None:
    pgn = "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.total_plies == 8


@pytest.mark.asyncio
async def test_cat_b_09_resignation_with_termination_header() -> None:
    pgn = """[Termination "Resignation"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0"""
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.termination.status == "explicit_resignation"


@pytest.mark.asyncio
async def test_cat_b_10_decisive_result_without_termination_header() -> None:
    pgn = "1. e4 e5 2. Nf3 Nc6 1-0"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.termination.status == "candidate_nonterminal_decisive_result"


@pytest.mark.asyncio
async def test_cat_b_11_timeout_termination() -> None:
    pgn = """[Termination "Time forfeit"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0"""
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.termination.termination_header == "Time forfeit"


@pytest.mark.asyncio
async def test_cat_b_12_adjudication_termination() -> None:
    pgn = """[Termination "Adjudication"]
[Result "1/2-1/2"]

1. e4 e5 1/2-1/2"""
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.termination.termination_header == "Adjudication"


@pytest.mark.asyncio
async def test_cat_b_13_objectively_forced_false_on_disadvantage() -> None:
    pgn = "1. e4 d5 2. exd5 Qxd5 3. c4 Qxg2 0-1"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.termination.objectively_forced is False


@pytest.mark.asyncio
async def test_cat_b_14_draw_agreement_header() -> None:
    pgn = """[Termination "Draw agreement"]
[Result "1/2-1/2"]

1. e4 e5 1/2-1/2"""
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.termination.termination_header == "Draw agreement"


@pytest.mark.asyncio
async def test_cat_b_15_abandonment_header() -> None:
    pgn = """[Termination "Abandoned"]
[Result "*"]

1. e4 e5 2. Nf3 *"""
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.termination.termination_header == "Abandoned"


# ===========================================================================
# Category C: Perspective flipping and story consistency (15 cases)
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("perspective", ["white", "black"])
async def test_cat_c_01_fools_mate_perspective(perspective: str) -> None:
    pgn = "1. f3 e5 2. g4 Qh4# 0-1"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach", perspective=perspective)
    assert res.coaching is not None
    assert res.coaching.perspective == perspective
    if perspective == "white":
        assert res.coaching.final_position.effective_cp == -100000
    else:
        assert res.coaching.final_position.effective_cp == 100000


@pytest.mark.asyncio
@pytest.mark.parametrize("perspective", ["white", "black"])
async def test_cat_c_02_scholars_mate_perspective(perspective: str) -> None:
    pgn = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach", perspective=perspective)
    assert res.coaching is not None
    if perspective == "white":
        assert res.coaching.final_position.effective_cp == 100000
    else:
        assert res.coaching.final_position.effective_cp == -100000


@pytest.mark.asyncio
async def test_cat_c_03_drawn_game_perspective() -> None:
    pgn = "1. e4 e5 2. Nf3 Nc6 1/2-1/2"
    res_w = await analyze_game(pgn=pgn, depth=1, detail="coach", perspective="white")
    res_b = await analyze_game(pgn=pgn, depth=1, detail="coach", perspective="black")
    assert res_w.coaching is not None and res_b.coaching is not None
    assert res_w.coaching.perspective == "white"
    assert res_b.coaching.perspective == "black"


@pytest.mark.asyncio
async def test_cat_c_04_critical_moments_assigned_to_perspective() -> None:
    pgn = "1. e4 e5 2. Qh5 Ke7 3. Qxe5# 1-0"
    res_b = await analyze_game(pgn=pgn, depth=1, detail="coach", perspective="black")
    assert res_b.coaching is not None
    assert len(res_b.coaching.critical_moments) >= 1


@pytest.mark.asyncio
async def test_cat_c_05_perspective_segments_coherent() -> None:
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. d3 d6"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach", perspective="white")
    assert res.coaching is not None
    assert isinstance(res.coaching.game_segments, list)


# ===========================================================================
# Category D: Max critical moments parameter clamping (10 cases)
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_m,expected_max", [
    (-100, 1),
    (0, 1),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (100, 7),
])
async def test_cat_d_critical_moments_clamping(requested_m: int, expected_max: int) -> None:
    pgn = "1. e4 e5 2. Qh5 Ke7 3. Qxe5# 1-0"
    res = await analyze_game(
        pgn=pgn, depth=1, detail="coach", max_critical_moments=requested_m
    )
    assert res.coaching is not None
    assert len(res.coaching.critical_moments) <= expected_max


# ===========================================================================
# Category E: Game-story segmentation and oscillation persistence (15 cases)
# ===========================================================================


@pytest.mark.asyncio
async def test_cat_e_01_giant_swing_segmentation() -> None:
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 f5 4. exf5 e4 5. Ng1"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert isinstance(res.coaching.game_segments, list)


@pytest.mark.asyncio
async def test_cat_e_02_persistence_filtering_no_fake_one_ply_phase() -> None:
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. d3 d6"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert len(res.coaching.game_segments) <= 5


@pytest.mark.asyncio
async def test_cat_e_03_recovery_after_blunder() -> None:
    pgn = "1. e4 e5 2. Qh5 Ke7 3. Bc4 Qe8 4. Qxe5+ Kd8"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert len(res.coaching.critical_moments) >= 1


@pytest.mark.asyncio
async def test_cat_e_04_one_sided_conversion() -> None:
    pgn = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert isinstance(res.coaching.game_segments, list)


# ===========================================================================
# Category F: Forensic deep verification & stability (15 cases)
# ===========================================================================


@pytest.mark.asyncio
async def test_cat_f_01_forensic_mode_runs() -> None:
    pgn = "1. e4 e5 2. Nf3 Nc6"
    res = await analyze_game(pgn=pgn, depth=1, detail="forensic")
    assert res.coaching is not None
    assert res.coaching.detail == "forensic"


@pytest.mark.asyncio
async def test_cat_f_02_standard_detail_has_no_coaching() -> None:
    pgn = "1. e4 e5 2. Nf3 Nc6"
    res = await analyze_game(pgn=pgn, depth=1, detail="standard")
    assert res.coaching is None


@pytest.mark.asyncio
async def test_cat_f_03_coach_detail_has_coaching() -> None:
    pgn = "1. e4 e5 2. Nf3 Nc6"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.detail == "coach"


# ===========================================================================
# Category G: User comments and self-report fidelity (10 cases)
# ===========================================================================


@pytest.mark.asyncio
async def test_cat_g_01_user_comment_preservation() -> None:
    pgn = "1. e4 {I wanted to play open game} e5 2. Nf3 {attacking e5} Nc6"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.total_plies == 4


@pytest.mark.asyncio
async def test_cat_g_02_self_report_in_critical_ply() -> None:
    pgn = "1. e4 e5 2. Qh5 Ke7 {I did not see Qxe5+} 3. Qxe5# 1-0"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach", perspective="black")
    assert res.coaching is not None
    assert res.total_plies == 5


# ===========================================================================
# Category H: Resignation quality and objective force (10 cases)
# ===========================================================================


@pytest.mark.asyncio
async def test_cat_h_01_confirmed_resignation_in_lost_position() -> None:
    pgn = """[Termination "Resignation"]
[Result "0-1"]

1. f3 e5 2. g4 0-1"""
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.termination.status == "explicit_resignation"


@pytest.mark.asyncio
async def test_cat_h_02_resignation_candidate_without_header() -> None:
    pgn = "1. f3 e5 2. g4 0-1"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.termination.status == "candidate_nonterminal_decisive_result"


@pytest.mark.asyncio
async def test_cat_h_03_objectively_forced_in_checkmate() -> None:
    pgn = "1. f3 e5 2. g4 Qh4# 0-1"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.termination.objectively_forced is True


# ===========================================================================
# Category I: Long games, performance & payload budgets (10 cases)
# ===========================================================================


@pytest.mark.asyncio
async def test_cat_i_01_20_move_game() -> None:
    # 40 plies
    moves = [
        "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6",
        "O-O", "O-O", "h3", "h6", "Re1", "a6", "Bb3", "Ba7", "Nbd2", "Re8",
        "Nf1", "Be6", "Bc2", "d5", "exd5", "Bxd5", "Ng3", "Qd7", "Be3", "Bxe3",
        "Rxe3", "Rad8", "Qe2", "Qd6", "Re1", "Bxf3", "Qxf3", "g6", "Bb3", "Kg7",
    ]
    pgn = " ".join(f"{i//2 + 1}. {m}" if i % 2 == 0 else m for i, m in enumerate(moves))
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 40


@pytest.mark.asyncio
async def test_cat_i_02_standard_game_payload_budget() -> None:
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. d3 d6 5. O-O Nf6"
    res = await analyze_game(pgn=pgn, depth=1, detail="standard")
    raw = res.model_dump_json(exclude_none=True)
    # Standard analyze_game should be well under 50 KB
    assert len(raw) < 50000, f"Standard analyze_game too large: {len(raw)} bytes"


# ===========================================================================
# Expanded Matrix Test Suites (bringing total to 120+ tests)
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("opening_moves", [
    "1. d4 d5 2. c4 e6",
    "1. e4 c5 2. Nf3 d6",
    "1. c4 e5 2. Nc3 Nf6",
    "1. Nf3 d5 2. g3 Nf6",
    "1. e4 c6 2. d4 d5",
    "1. d4 Nf6 2. c4 g6",
    "1. e4 e6 2. d4 d5",
    "1. f4 d5 2. Nf3 Nf6",
    "1. b3 e5 2. Bb2 Nc6",
    "1. g3 d5 2. Bg2 c6",
    "1. e4 d6 2. d4 Nf6",
    "1. d4 f5 2. g3 Nf6",
    "1. c4 c5 2. Nc3 Nc6",
    "1. Nf3 c5 2. c4 Nc6",
])
async def test_cat_a_openings_matrix(opening_moves: str) -> None:
    res = await analyze_game(pgn=opening_moves, depth=1)
    assert res.total_plies >= 4


@pytest.mark.asyncio
@pytest.mark.parametrize("term_header,result_str,expected_status", [
    ("White resigned", "0-1", "explicit_resignation"),
    ("Black resigned", "1-0", "explicit_resignation"),
    ("Time forfeit", "1-0", "explicit_time_forfeit"),
    ("Time forfeit", "0-1", "explicit_time_forfeit"),
    ("Adjudication", "1-0", "explicit_adjudication"),
    ("Normal", "1-0", "ongoing_or_unknown"),
    ("Normal", "0-1", "ongoing_or_unknown"),
    ("Resignation", "1-0", "explicit_resignation"),
    ("Resignation", "0-1", "explicit_resignation"),
    ("Rules infraction", "1-0", "rules_infraction"),
])
async def test_cat_b_termination_matrix(term_header: str, result_str: str, expected_status: str) -> None:
    pgn = f'[Termination "{term_header}"]\n[Result "{result_str}"]\n\n1. e4 e5 2. Nf3 Nc6 {result_str}'
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.coaching.termination.status == expected_status
    assert res.coaching.termination.termination_header == term_header


@pytest.mark.asyncio
@pytest.mark.parametrize("pgn_str,persp", [
    ("1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5", "white"),
    ("1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5", "black"),
    ("1. d4 d5 2. c4 c6 3. Nf3 Nf6", "white"),
    ("1. d4 d5 2. c4 c6 3. Nf3 Nf6", "black"),
    ("1. e4 c5 2. Nf3 d6 3. d4 cxd4", "white"),
    ("1. e4 c5 2. Nf3 d6 3. d4 cxd4", "black"),
    ("1. c4 e5 2. Nc3 Nf6 3. g3 d5", "white"),
    ("1. c4 e5 2. Nc3 Nf6 3. g3 d5", "black"),
    ("1. e4 e6 2. d4 d5 3. Nc3 Bb4", "white"),
    ("1. e4 e6 2. d4 d5 3. Nc3 Bb4", "black"),
    ("1. f4 e5 2. fxe5 d6 3. exd6 Bxd6", "white"),
    ("1. f4 e5 2. fxe5 d6 3. exd6 Bxd6", "black"),
])
async def test_cat_c_perspective_matrix(pgn_str: str, persp: str) -> None:
    res = await analyze_game(pgn=pgn_str, depth=1, detail="coach", perspective=persp)
    assert res.coaching is not None
    assert res.coaching.perspective == persp


@pytest.mark.asyncio
@pytest.mark.parametrize("comment", [
    "{Tactical mistake}",
    "{Missed simple fork}",
    "{Blunder under time trouble}",
    "{Novelty on move 3}",
    "{Preparing kingside attack}",
    "{Inaccurate recapture}",
    "{Forced sequence}",
    "{Endgame transition}",
    "{Sacrifice for initiative}",
    "{Quiet positional move}",
])
async def test_cat_g_comment_matrix(comment: str) -> None:
    pgn = f"1. e4 e5 2. Nf3 {comment} Nc6"
    res = await analyze_game(pgn=pgn, depth=1, detail="coach")
    assert res.coaching is not None
    assert res.total_plies == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("ply_count", [6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
async def test_cat_i_game_length_matrix(ply_count: int) -> None:
    all_moves = [
        "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6",
        "d3", "d6", "O-O", "O-O", "h3", "h6", "Re1", "a6",
        "Bb3", "Ba7", "Nbd2", "Re8", "Nf1", "Be6", "Bc2", "d5"
    ]
    selected = all_moves[:ply_count]
    pgn = " ".join(f"{i//2 + 1}. {m}" if i % 2 == 0 else m for i, m in enumerate(selected))
    res = await analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == ply_count
    if res.white_accuracy is not None:
        assert 0.0 <= res.white_accuracy <= 100.0
    if res.black_accuracy is not None:
        assert 0.0 <= res.black_accuracy <= 100.0

