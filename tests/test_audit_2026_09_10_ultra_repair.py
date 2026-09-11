"""Automated regression suite verifying all 22 ultra-audit findings and repairs.

Validates:
- P0: Finding 1 (History Replay FEN Mismatch)
- P1: Findings 2 & 19 (50-move draw claim on zero-ply & legal action distinction)
- P1: Findings 5, 6, 10 (Tactical terminology, SEE en_prise filtering, loose corner rooks)
- P1: Findings 4 & 16 (Strict SAN annotation acceptance & machine-readable normalization changes)
- P1: Findings 7, 17, 21 (Explicit non-resignation terminations & result/board provenance)
- P2: Findings 8, 9, 13 (Compactness control, recursive verbosity, input_fen preservation)
- P2: Findings 11, 12, 20 (Parser observability, tag malformation, trailing tokens, empty game reasons)
- P2: Findings 3, 14, 15, 18, 22 (MultiPV telemetry, candidate metadata, critical moment clamping)
"""

from __future__ import annotations

import chess
import pytest

from mcp_server.analysis.forensics import build_tactical_snapshot
from mcp_server.analysis.game_termination import build_game_termination_assessment
from mcp_server.analysis.game_validation import extract_game_metadata
from mcp_server.analysis.mainline_parser import parse_mainline
from mcp_server.analysis.move_classifier import validate_classify_input
from mcp_server.models.forensics import MechanismCandidateEvidence
from mcp_server.models.game_coaching import FinalPositionAssessment
from mcp_server.models.mcpeval import MCPEval
from mcp_server.parsers import (
    build_board_from_history,
    parse_move_with_details,
    sanitize_malformed_pgn_header_lines,
    validate_strict_header_syntax,
)
from mcp_server.parsers.pgn.tokens import validate_strict_mainline_surface
from mcp_server import server as server_module
from mcp_server.server import analyze_game, classify_move, evaluate_position, top_moves
from mcp_server.tools._common import _compact_mcpeval, _minimal_mcpeval


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


def _dummy_final(
    *,
    legal_count: int = 20,
    checkmate: bool = False,
    terminal: bool = False,
    side: str = "white",
) -> FinalPositionAssessment:
    return FinalPositionAssessment(
        perspective=side,  # type: ignore[arg-type]
        position_terminal_by_rules=terminal,
        checkmate=checkmate,
        stalemate=False,
        forced_mate=checkmate,
        mate_distance=0 if checkmate else None,
        effective_cp=0,
        wdl=None,
        side_to_move=side,  # type: ignore[arg-type]
        legal_move_count=legal_count,
        best_move_uci=None,
        best_move_san=None,
        defensive_resources_exist=not terminal and legal_count > 0,
        reasonable_resource_count=legal_count if not terminal else None,
    )


# ---------------------------------------------------------------------------
# Finding 1 (P0): History Replay & FEN Mismatch
# ---------------------------------------------------------------------------


def test_01_build_board_from_history_mismatch_and_preservation() -> None:
    # 1. Matching history returns board with move stack intact
    fen_after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    b, _ = build_board_from_history(fen_after_e4, ["e4"])
    assert b.fen() == fen_after_e4
    assert len(b.move_stack) == 1
    assert b.move_stack[0] == chess.Move.from_uci("e2e4")

    # 2. Mismatched history raises POSITION_HISTORY_MISMATCH
    fen_after_d4 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
    with pytest.raises(ValueError, match="POSITION_HISTORY_MISMATCH"):
        build_board_from_history(fen_after_d4, ["e4"])

    # 3. classify_move validate_classify_input uses history without double-pushing
    val = validate_classify_input(
        fen=fen_after_e4, moves=["e4"], move="e5", action_type="play_move", strict=True
    )
    assert val.board.fen() == fen_after_e4
    assert len(val.board.move_stack) == 1


# ---------------------------------------------------------------------------
# Findings 2 & 19 (P1/P2): 50-move draw claim & legal action fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_02_zero_ply_fifty_move_draw_claim_and_action_fields() -> None:
    # Position with halfmove_clock = 100 (50 moves without pawn move or capture) and White down a rook
    fen_50 = "8/8/4k3/8/8/4K3/8/4r3 w - - 100 51"
    res = await analyze_game(f'[SetUp "1"]\n[FEN "{fen_50}"]\n*', detail="coach")
    assert res.coaching is not None
    fp = res.coaching.final_position
    assert fp.can_claim_draw is True
    assert fp.can_claim_now is True
    assert "fifty_moves" in fp.claim_reasons_now
    assert fp.recommended_action == "claim_draw"
    assert fp.board_legal_move_count > 0
    assert fp.continued_play_legal_under_rules is True
    assert fp.position_terminal_by_rules is False


# ---------------------------------------------------------------------------
# Findings 3 & 14 (P1/P2): Top moves MultiPV rank & search_provenance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_03_top_moves_multipv_telemetry() -> None:
    res = await top_moves(chess.STARTING_FEN, n=3, depth=1)
    assert len(res.result) > 0
    for idx, candidate in enumerate(res.result, start=1):
        assert candidate.multipv == idx
        assert candidate.search_provenance is not None
        assert candidate.search_provenance.get("multipv") == idx


# ---------------------------------------------------------------------------
# Findings 4 & 16 (P1/P2): Strict SAN accepts stripped glyphs & change codes
# ---------------------------------------------------------------------------


def test_04_strict_san_annotation_and_normalization_changes() -> None:
    b = chess.Board()
    b.push_san("e4")
    b.push_san("e5")
    b.push_san("Nf3")
    b.push_san("Nc6")

    # 1. Annotation glyphs (!!, ?, !?, etc.) are stripped and recorded in normalization_changes in lenient mode
    res = parse_move_with_details(b, "Bb5!!", strict=False)
    assert res.move == chess.Move.from_uci("f1b5")
    assert res.canonical_san == "Bb5"
    assert "annotation_suffix_removed" in res.normalization_changes

    # Under Option B, strict mode on direct move endpoints rejects annotation glyphs
    with pytest.raises(ValueError, match="STRICT_SAN_ERROR"):
        parse_move_with_details(b, "Bb5!!", strict=True)

    # 2. Syntax changes (e.g. castling zeros, promotion equals) recorded in normalization_changes
    b_castle = chess.Board()
    b_castle.push_san("e4")
    b_castle.push_san("e5")
    b_castle.push_san("Nf3")
    b_castle.push_san("Nc6")
    b_castle.push_san("Bc4")
    b_castle.push_san("Bc5")
    res_ep = parse_move_with_details(b_castle, "0-0", strict=False)
    assert "castling_zeros_normalized" in res_ep.normalization_changes

    b_prom = chess.Board("8/P7/8/8/8/8/k7/4K3 w - - 0 1")
    res_prom = parse_move_with_details(b_prom, "a8Q+", strict=False)
    assert "promotion_equals_inserted" in res_prom.normalization_changes

    # 3. Check / Mate suffix corrected when move claims check/mate incorrectly
    res_chk = parse_move_with_details(chess.Board(), "e4+", strict=False)
    assert "check_suffix_corrected" in res_chk.normalization_changes
    res_mate = parse_move_with_details(chess.Board(), "e4#", strict=False)
    assert "mate_suffix_corrected" in res_mate.normalization_changes


# ---------------------------------------------------------------------------
# Findings 5, 6, 10 (P1/P2): Tactical terminology, SEE en_prise & corner rooks
# ---------------------------------------------------------------------------


def test_05_tactical_terminology_and_corner_rooks() -> None:
    # 1. MechanismCandidateEvidence carries evidence_level
    mech = MechanismCandidateEvidence(
        mechanism="absolute_pin",
        proof_scope="immediate_legality_only",
        evidence_level="legally_actionable",
    )
    assert mech.evidence_level == "legally_actionable"

    # 2. Starting position tactical snapshot excludes corner rooks from tactically_loose_pieces
    b = chess.Board()
    snap = build_tactical_snapshot(b)
    loose_squares = {p.square for p in snap.tactically_loose_pieces}
    for rook_sq in ("a1", "h1", "a8", "h8"):
        assert rook_sq not in loose_squares

    # 3. Defended piece where capture is unprofitable is not marked en_prise
    snap2 = build_tactical_snapshot(chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"))
    en_prise_squares = {p.square for p in snap2.en_prise_pieces}
    assert "e4" not in en_prise_squares


# ---------------------------------------------------------------------------
# Findings 7, 17, 21 (P1/P2/P3): Non-resignation terminations & result provenance
# ---------------------------------------------------------------------------


def test_07_explicit_non_resignation_terminations_and_provenance() -> None:
    # 1. Explicit time forfeit
    pgn_time = '[Event "Blitz"]\n[Result "1-0"]\n[Termination "time forfeit"]\n\n1. e4 e5 1-0'
    term_time = build_game_termination_assessment(pgn_time, final_position=_dummy_final())
    assert term_time.status == "explicit_time_forfeit"
    assert term_time.confidence == "high"

    # 2. Explicit abandoned
    pgn_aband = '[Event "Blitz"]\n[Result "0-1"]\n[Termination "abandoned"]\n\n1. e4 0-1'
    term_aband = build_game_termination_assessment(pgn_aband, final_position=_dummy_final())
    assert term_aband.status == "explicit_abandoned"

    # 3. Explicit adjudication
    pgn_adj = '[Event "Simul"]\n[Result "1/2-1/2"]\n[Termination "adjudication"]\n\n1. e4 e5 *'
    term_adj = build_game_termination_assessment(pgn_adj, final_position=_dummy_final())
    assert term_adj.status == "explicit_adjudication"

    # 4. Rules infraction
    pgn_infr = '[Event "Simul"]\n[Result "0-1"]\n[Termination "rules infraction"]\n\n1. e4 e5 *'
    term_infr = build_game_termination_assessment(pgn_infr, final_position=_dummy_final())
    assert term_infr.status == "rules_infraction"

    # 5. Rules terminal (stalemate)
    pgn_stalemate = '[Event "Draw"]\n[Result "1/2-1/2"]\n[SetUp "1"]\n[FEN "k7/8/1K1Q4/8/8/8/8/8 b - - 0 1"]\n\n*'
    term_stale = build_game_termination_assessment(pgn_stalemate, final_position=_dummy_final(terminal=True))
    assert term_stale.status == "rules_terminal"
    assert term_stale.board_outcome == "stalemate"


# ---------------------------------------------------------------------------
# Findings 8, 9, 13 (P2): Compactness control & recursive serialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_08_analyze_game_verbosity_and_top_moves_recursion() -> None:
    # 1. analyze_game supports verbosity="compact"
    pgn_zero = '[SetUp "1"]\n[FEN "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"]\n*'
    res_compact = await analyze_game(pgn_zero, detail="coach", verbosity="compact")
    assert res_compact.is_compact is True
    assert res_compact.is_minimal is False

    # 2. analyze_game supports verbosity="minimal"
    res_minimal = await analyze_game(pgn_zero, detail="coach", verbosity="minimal")
    assert res_minimal.is_minimal is True
    assert res_minimal.turning_points == []

    # 3. top_moves verbosity="compact" is recursive
    tm = await top_moves(chess.STARTING_FEN, n=2, depth=1, verbosity="compact")
    for c in tm.result:
        assert c.is_compact is True

    # 4. _compact_mcpeval preserves input_fen
    eval_orig = MCPEval(status="active", input_fen="test_fen", canonical_fen="test_fen")
    eval_comp = _compact_mcpeval(eval_orig)
    assert eval_comp.input_fen == "test_fen"
    assert eval_comp.is_compact is True


# ---------------------------------------------------------------------------
# Findings 11, 12, 20 (P2/P3): Parser observability, tag malformation & tokens
# ---------------------------------------------------------------------------


def test_11_parser_observability_and_neutral_empty_warning() -> None:
    # 1. Malformed tag [Event "x] in strict mode raises STRICT_PGN_ERROR
    with pytest.raises(ValueError, match="STRICT_PGN_ERROR"):
        validate_strict_header_syntax('[Event "x]\n\n1. e4 e5 *')

    # In lenient mode, emits warning and sanitizes
    _, warnings = sanitize_malformed_pgn_header_lines('[Event "x]\n\n1. e4 e5 *', strict=False)
    assert any("Malformed PGN header line ignored" in w for w in warnings)

    # 2. Trailing tokens after result in strict mode raises STRICT_PGN_ERROR
    g = chess.pgn.Game()
    with pytest.raises(ValueError, match="STRICT_PGN_ERROR"):
        validate_strict_mainline_surface("1. e4 e5 * 1-0", g)

    # In lenient mode, parse_mainline emits TRAILING_TOKENS_AFTER_RESULT warning
    _, _, main_warnings, _, _ = parse_mainline("1. e4 e5 * 1-0", g, strict=False)
    assert any("TRAILING_TOKENS_AFTER_RESULT" in w for w in main_warnings)

    # 3. Result-only empty game emits neutral warning and sets empty_game_reason
    md = extract_game_metadata("*", g, strict=False, is_comment_only_input=True)
    assert md.empty_game_reason == "result_only"
    assert any("Input PGN contains no mainline moves; returning a zero-ply game." in w for w in md.metadata_warnings)


# ---------------------------------------------------------------------------
# Findings 18 & 22 (P2/P3): Critical moment clamping & depth telemetry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_18_critical_moment_clamping_and_depth_telemetry() -> None:
    # 1. max_critical_moments clamping observability
    res = await analyze_game("1. e4 e5 2. Nf3 Nc6 *", max_critical_moments=10, depth=1)
    assert res.requested_max_critical_moments == 10
    assert res.clamped_max_critical_moments == 7
    assert res.returned_critical_moments is not None

    # 2. classify_move exposes requested_depth and searched_depth
    cl = await classify_move(chess.STARTING_FEN, "e4", depth=1)
    assert cl.requested_depth == 1
    assert cl.searched_depth == 1
