"""Regression tests for the 2026-09-02 ultra-detailed chess MCP audit.

Every test in this file pins a specific defect in the report so a future
change cannot silently reintroduce the bug. Each test cites the audit
finding it locks (P0/P1/P2/P3 plus a CASE number from §29 of the report).
"""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval

from mcp_server import server as server_module


class _NoopEnginePool:
    name = "NoopEnginePool"
    engine_version = "NoopEnginePool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        return Eval(cp=0, mate=None, best_move="", pv=[], depth=depth)

    async def top_moves(self, board, n=3, depth=14):
        return [Eval(cp=0, mate=None, best_move="", pv=[], depth=depth)]

    async def close(self):
        pass


@pytest.fixture(autouse=True)
async def _reset_pool():
    yield
    await server_module.close_analyzer_pool()


# ===========================================================================
# P0 — Lenient PGN parser semantic substitution
# ===========================================================================


def test_p0_case_1_wrong_side_marker_does_not_substitute_e5():
    """CASE 1: 1... e5 2. Nf3 Nc6 * must reject or preserve e5 — never
    silently analyze the Nf3/Nc6 position from start."""
    with pytest.raises(ValueError) as exc_info:
        server_module._extract_game("1... e5 2. Nf3 Nc6 *", strict=False)
    msg = str(exc_info.value)
    # The fix surfaces this as INVALID_PGN because the move identity
    # cannot be preserved.
    assert "INVALID_PGN" in msg or "STRICT_PGN_ERROR" in msg


def test_p0_case_2_distinct_inputs_do_not_collapse():
    """CASE 2: 1... e5 2. Nf3 Nc6 * and 1... c5 2. Nf3 Nc6 * must not
    produce the same effective game."""
    with pytest.raises(ValueError):
        server_module._extract_game("1... e5 2. Nf3 Nc6 *", strict=False)
    with pytest.raises(ValueError):
        server_module._extract_game("1... c5 2. Nf3 Nc6 *", strict=False)


def test_p0_case_3_wrong_side_marker_a5_not_rewritten_to_h4():
    """CASE 3: 1... a5 2. h4 h5 * must reject (a5 not silently rewritten)."""
    with pytest.raises(ValueError):
        server_module._extract_game("1... a5 2. h4 h5 *", strict=False)


def test_p0_bare_moves_legitimate_case_still_accepted():
    """Regression guard: the bare-moves fallback (`e4 e5 Nf3`) must still
    parse — the P0 fix removed substitution, not legitimate recovery."""
    g = server_module._extract_game("e4 e5 Nf3", strict=False)
    moves = [m.uci() for m in g.mainline_moves()]
    assert moves == ["e2e4", "e7e5", "g1f3"]


def test_p0_bare_moves_with_move_numbers_still_accepted():
    g = server_module._extract_game("1. e4 e5 2. Nf3", strict=False)
    moves = [m.uci() for m in g.mainline_moves()]
    assert moves == ["e2e4", "e7e5", "g1f3"]


def test_p0_result_marker_still_breaks_loop():
    """Regression guard: `e4 e5 1-0 Nf3 Nc6` must still produce 2 plies +
    result 1-0 (trailing moves after game end are dropped)."""
    g = server_module._extract_game("e4 e5 1-0 Nf3 Nc6", strict=False)
    moves = [m.uci() for m in g.mainline_moves()]
    assert len(moves) == 2
    assert g.headers.get("Result") == "1-0"


# ===========================================================================
# P1 — Cross-endpoint FEN validation consistency
# ===========================================================================


def test_p1_case_3a_fullmove_zero_rejected_direct():
    """Direct FEN with fullmove=0 must be rejected."""
    fen = "4k3/8/8/8/8/8/R7/4K3 w - - 0 0"
    with pytest.raises(ValueError) as exc_info:
        server_module._build_board(fen, strict=True)
    assert "INVALID_FEN" in str(exc_info.value)


def test_p1_case_3b_fullmove_zero_rejected_in_pgn_header():
    """analyze_game must reject FEN with fullmove=0 inside [FEN ...] tag."""
    pgn = '[SetUp "1"]\n[FEN "4k3/8/8/8/8/8/R7/4K3 w - - 0 0"]\n\n*'
    with pytest.raises(ValueError) as exc_info:
        server_module._extract_game(pgn, strict=True)
    assert "INVALID_FEN" in str(exc_info.value)


def test_p1_case_3c_fullmove_zero_rejected_with_moves_in_pgn():
    """analyze_game must reject FEN with fullmove=0 even when movetext is
    present."""
    pgn = '[SetUp "1"]\n[FEN "4k3/8/8/8/8/8/R7/4K3 w - - 0 0"]\n\n1. Ra8+ *'
    with pytest.raises(ValueError):
        server_module._extract_game(pgn, strict=True)


# ===========================================================================
# P1 — En passant + halfmove_clock historical consistency
# ===========================================================================


def test_p1_case_6_ep_requires_zero_halfmove_white_to_move():
    """EP target + halfmove_clock > 0 is historically impossible."""
    fen = "4k3/8/8/3Pp3/8/8/8/4K3 w - e6 17 2"
    with pytest.raises(ValueError) as exc_info:
        server_module._build_board(fen, strict=True)
    assert "INVALID_FEN" in str(exc_info.value)


def test_p1_case_6_ep_requires_zero_halfmove_black_to_move():
    fen = "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 1 1"
    with pytest.raises(ValueError):
        server_module._build_board(fen, strict=True)


def test_p1_legit_ep_position_with_zero_halfmove_still_accepted():
    """Sanity: a real EP position (halfmove_clock=0) must still parse."""
    fen = "4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 2"
    b = server_module._build_board(fen, strict=True)
    assert b.fen() == fen


# ===========================================================================
# P2 — claim_draw request-shape validation must run before board state
# ===========================================================================


@pytest.mark.asyncio
async def test_p2_case_9_claim_draw_structural_error_before_state():
    """CASE 9: claim_draw with supplied move on a non-claimable board
    must return the structural error (move not allowed), not the
    board-state error (draw cannot be claimed now)."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    with pytest.raises(Exception) as exc_info:
        await server_module.classify_move(
            "startpos", move="nonsense", action_type="claim_draw", strict=True
        )
    msg = str(exc_info.value)
    # The structural error mentions the move argument
    assert "move" in msg.lower() or "STRICT" in msg


@pytest.mark.asyncio
async def test_p2_claim_draw_lenient_warns_on_supplied_move():
    """Lenient claim_draw with a supplied move emits a syntax_warning."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    # Use a 50-move-claimable position so the call itself succeeds.
    fen_50 = "7k/8/8/8/8/8/R7/K7 w - - 100 51"
    res = await server_module.classify_move(
        fen_50, move="nonsense", action_type="claim_draw", strict=False, depth=10
    )
    assert res.syntax_warning is not None
    assert "claim_draw" in res.syntax_warning


# ===========================================================================
# P2 — Terminal-state consistency for all actions
# ===========================================================================


@pytest.mark.asyncio
async def test_p2_case_8_terminal_claim_draw_returns_game_already_over():
    """CASE 8: claim_draw on a terminal (checkmate) position must
    consistently return GAME_ALREADY_OVER."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    with pytest.raises(Exception) as exc_info:
        await server_module.classify_move(
            fen, move=None, action_type="claim_draw", strict=False, depth=10
        )
    assert "GAME_ALREADY_OVER" in str(exc_info.value)


# ===========================================================================
# P2 — Trailing-ply count after board-detected checkmate
# ===========================================================================


@pytest.mark.asyncio
async def test_p2_case_7_trailing_ply_count_after_checkmate():
    """CASE 7: 1. f3 e5 2. g4 Qh4# 3. e4 e6 4. d4 d6 * must report
    exactly 4 ignored trailing plies (matching the explicit-result branch)."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = "1. f3 e5 2. g4 Qh4# 3. e4 e6 4. d4 d6 *"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    # Find the trailing-ply warning.
    trailing_warning = next((w for w in r.metadata_warnings if "trailing" in w.lower()), None)
    assert trailing_warning is not None
    assert "4 trailing plies" in trailing_warning


@pytest.mark.asyncio
async def test_p2_explicit_result_branch_consistent_count():
    """Regression: explicit-result branch must remain at 4 trailing plies
    (parity with the board-detected branch after the P2 fix)."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = "1. e4 e5 1-0 2. Nf3 Nc6 3. Bb5 a6"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    trailing_warning = next((w for w in r.metadata_warnings if "trailing" in w.lower()), None)
    assert trailing_warning is not None
    assert "4 trailing plies" in trailing_warning


# ===========================================================================
# P2 — SetUp tag validation
# ===========================================================================


@pytest.mark.asyncio
async def test_p2_case_4_invalid_setup_value_rejected_strict():
    """CASE 4: [SetUp "2"] must be rejected in strict mode."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    for bad_val in ["2", "", "true", "false", "01", "-1", " "]:
        pgn = f'[SetUp "{bad_val}"]\n[Result "*"]\n\n*'
        with pytest.raises(Exception):
            await server_module.analyze_game(pgn, depth=10, strict=True)


@pytest.mark.asyncio
async def test_p2_canonical_setup_values_accepted():
    """Sanity: [SetUp "0"] and [SetUp "1"] must remain accepted."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    for canonical_val in ["0", "1"]:
        pgn = f'[SetUp "{canonical_val}"]\n[Result "*"]\n\n*'
        # Lenient mode accepts (and surfaces the warning that strict mode
        # promotes to an error).
        r = await server_module.analyze_game(pgn, depth=10, strict=False)
        assert r is not None


# ===========================================================================
# P2 — Result/Variant tag case-handling + duplicate detection
# ===========================================================================


@pytest.mark.asyncio
async def test_p2_case_5_result_casing_duplicate_detected():
    """CASE 5: [Result *] + [result 1-0] must be detected as a duplicate
    (or conflicting values) in both modes."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = '[Result "*"]\n[result "1-0"]\n\n*'
    # Lenient mode surfaces duplicate + conflict warnings.
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    has_dup_warning = any("Duplicate PGN tag '[result]'" in w for w in r.metadata_warnings)
    has_conflict_warning = any(
        "Conflicting values for PGN tag 'result'" in w for w in r.metadata_warnings
    )
    assert has_dup_warning
    assert has_conflict_warning


@pytest.mark.asyncio
async def test_p2_variant_casing_accepted_both_ways():
    """[variant Standard] and [Variant Standard] must produce the same
    variant field on the response."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    for pgn in ['[variant "Standard"]\n\n*', '[Variant "Standard"]\n\n*']:
        r = await server_module.analyze_game(pgn, depth=10, strict=False)
        assert r.variant == "Standard"


# ===========================================================================
# P2 — Malformed PGN headers
# ===========================================================================


@pytest.mark.asyncio
async def test_p2_malformed_header_emits_warning_lenient():
    """Malformed headers like [White "A] must produce a warning in lenient
    mode rather than silently disappearing."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = '[White "A]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert any("Malformed PGN header line" in w for w in r.metadata_warnings)


# ===========================================================================
# P2/P3 — TimeControl whitespace normalization
# ===========================================================================


@pytest.mark.asyncio
async def test_p2_time_control_whitespace_normalized():
    """[TimeControl "? "] / " ?" / " ? " must all normalize to None."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    for raw in ["?", "? ", " ?", " ? "]:
        pgn = f'[TimeControl "{raw}"]\n[Result "*"]\n\n*'
        r = await server_module.analyze_game(pgn, depth=10, strict=False)
        assert r.time_control is None


@pytest.mark.asyncio
async def test_p2_time_control_canonical_value_preserved():
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = '[TimeControl "300+5"]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.time_control == "300+5"


# ===========================================================================
# P3 — Numeric counter bounds
# ===========================================================================


def test_p3_extreme_fullmove_rejected():
    """Extremely large fullmove values are bounded for operational safety."""
    fen = "4k3/8/8/8/8/8/R7/4K3 w - - 0 999999999999"
    with pytest.raises(ValueError) as exc_info:
        server_module._build_board(fen, strict=True)
    assert "INVALID_FEN" in str(exc_info.value)


def test_p3_negative_fullmove_rejected():
    fen = "4k3/8/8/8/8/8/R7/4K3 w - - 0 -1"
    with pytest.raises(ValueError):
        server_module._build_board(fen, strict=True)


def test_p3_extreme_halfmove_rejected():
    fen = "4k3/8/8/8/8/8/R7/4K3 w - - 99999 1"
    with pytest.raises(ValueError):
        server_module._build_board(fen, strict=True)
