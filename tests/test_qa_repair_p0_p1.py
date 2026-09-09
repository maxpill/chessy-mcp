"""P0/P1 regression tests for chess-mcp QA repair prompt.

Covers:
1. `include_moves` MUST-be-included guarantee (result.result, legal_actions, candidate_comparisons).
2. Parameter bounds and clamping (n clamped 1-10, depth clamped 1-30, max 8 include_moves).
3. Action fields parity (root_candidate_action == action_block.root_candidate_action).
4. Semantic SAN normalization and strict mode validation.
5. Search provenance metadata (multipv_root vs candidate_research).
6. Payload byte budgets (minimal < 1.5 KB, compact < 3 KB, top_moves n=3 compact < 8 KB, classify < 8 KB).
"""

from __future__ import annotations

import pytest
import chess
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server.parsers.move_parser import parse_move_with_details
from mcp_server.rules.constants import (
    TOP_MOVES_MIN_N,
    TOP_MOVES_MAX_N,
    DEPTH_MIN,
    DEPTH_MAX,
    MAX_INCLUDE_MOVES,
)
from mcp_server.tools._common import (
    VERBOSITY_MINIMAL,
    VERBOSITY_COMPACT,
    VERBOSITY_FULL,
    _resolve_verbosity,
)
from mcp_server import server as server_module
from mcp_server.tools.evaluate_position import evaluate_position
from mcp_server.tools.top_moves import top_moves
from mcp_server.tools.classify_move import classify_move


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


# ---------------------------------------------------------------------------
# 1. Parameter bounds and clamping
# ---------------------------------------------------------------------------


def test_constants_definitions() -> None:
    # F-002 fix (2026-09-09): TOP_MOVES_MAX_N is the single canonical constant
    # set to 20 (was 10). Update follows the audit's recommendation to align
    # constants with the runtime/cost-estimator behavior already observed in
    # production.
    assert TOP_MOVES_MIN_N == 1
    assert TOP_MOVES_MAX_N == 20
    assert DEPTH_MIN == 1
    assert DEPTH_MAX == 30
    assert MAX_INCLUDE_MOVES == 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_n,expected_clamped",
    [
        (-100, 1),
        (0, 1),
        (1, 1),
        (5, 5),
        (10, 10),
        (11, 11),
        (20, 20),
        (1000, 20),
    ],
)
async def test_top_moves_n_clamping(test_n: int, expected_clamped: int) -> None:
    # F-002 fix (2026-09-09): n is clamped to 1..20 (canonical max). The old
    # parameter table was pinning the broken 10 cap; values above 10 now
    # remain at their caller value up to the 20 ceiling.
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res = await top_moves(fen=fen, n=test_n, depth=1)
    assert res.requested_n == test_n
    assert res.clamped_n == expected_clamped
    assert len(res.result) <= expected_clamped


@pytest.mark.asyncio
async def test_top_moves_include_moves_max_exceeded() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # 9 moves exceeds MAX_INCLUDE_MOVES = 8
    too_many = ["e4", "d4", "Nf3", "Nc3", "c4", "f4", "g3", "b3", "a3"]
    with pytest.raises(ToolError) as exc_info:
        await top_moves(fen=fen, n=3, depth=1, include_moves=too_many)
    assert "at most 8 moves" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. include_moves MUST-be-included guarantee
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_moves_include_moves_guaranteed_in_result() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # Request n=1 with include_moves=['a3', 'h3']
    res = await top_moves(fen=fen, n=1, depth=1, include_moves=["a3", "h3"])

    # Both a3 and h3 must be present in result.result and candidate_comparisons
    result_ucis = {c.best_move.lower() for c in res.result if c.best_move}
    assert "a2a3" in result_ucis
    assert "h2h3" in result_ucis
    assert res.returned_n == len(res.result)
    assert res.returned_n >= 3  # 1 best + 2 included

    # Check legal_actions parity
    action_moves = {
        a.get("move", {}).get("uci")
        for a in res.legal_actions
        if isinstance(a, dict) and isinstance(a.get("move"), dict)
    }
    assert "a2a3" in action_moves
    assert "h2h3" in action_moves

    # Check tracking fields
    assert res.requested_include_moves == ["a3", "h3"]
    assert res.included_move_count == 2

    # Check forensics candidate comparisons
    assert res.forensics is not None
    comp_ucis = {c.uci.lower() for c in res.forensics.candidate_comparisons}
    assert "a2a3" in comp_ucis
    assert "h2h3" in comp_ucis


@pytest.mark.asyncio
async def test_top_moves_include_moves_provenance() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res = await top_moves(fen=fen, n=2, depth=1, include_moves=["Na3"])
    # Explicitly included move Na3 should have candidate_research provenance
    for cand in res.result:
        assert cand.search_provenance is not None
        if cand.best_move == "b1a3":
            assert cand.search_provenance.get("kind") == "candidate_research"


# ---------------------------------------------------------------------------
# 3. Action fields parity and completeness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_position_action_parity() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res = await evaluate_position(fen=fen, depth=1)

    assert res.recommended_action == "play_move"
    assert res.root_candidate_action == "play_move"
    assert res.action_block is not None
    assert res.action_block.root_candidate_action == "play_move"
    assert res.action_block.recommended_action == "play_move"

    # Board legal moves check
    assert len(res.board_legal_move_uci) == 20
    assert len(res.action_block.board_legal_move_uci) == 20


@pytest.mark.asyncio
async def test_top_moves_candidates_action_parity() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res = await top_moves(fen=fen, n=3, depth=1)

    for cand in res.result:
        assert cand.recommended_action == "play_move"
        assert cand.root_candidate_action == "play_move"
        if cand.action_block is not None:
            assert cand.action_block.root_candidate_action == "play_move"
            assert cand.action_block.recommended_action == "play_move"


# ---------------------------------------------------------------------------
# 4. Semantic SAN Normalization and Strict Mode
# ---------------------------------------------------------------------------


def test_san_normalization_categories() -> None:
    board = chess.Board()

    # None: standard canonical SAN
    r_canon = parse_move_with_details(board, "e4")
    assert r_canon.normalization_kind == "none"
    assert r_canon.normalization_changes == []

    # Cosmetic: whitespace
    r_cosmetic = parse_move_with_details(board, "  e4  ")
    assert r_cosmetic.normalization_kind == "cosmetic"
    assert "whitespace_trimmed" in r_cosmetic.normalization_changes

    # Notation variant: long algebraic format
    r_variant = parse_move_with_details(board, "e2-e4")
    assert r_variant.normalization_kind == "notation_variant"
    assert "hyphenated_san_normalized" in r_variant.normalization_changes

    # Semantic: false capture symbol (Nxf3 when f3 is empty)
    r_capture = parse_move_with_details(board, "Nxf3")
    assert r_capture.normalization_kind == "semantic"
    assert "capture_marker_removed" in r_capture.normalization_changes
    assert r_capture.move.uci() == "g1f3"

    # Strict mode rejection on semantic error
    with pytest.raises(ValueError) as exc:
        parse_move_with_details(board, "Nxf3", strict=True)
    assert "STRICT_SAN_ERROR" in str(exc.value)

    # Check symbol errors
    board_check = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/5PP1/8/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    # Qh4# gives mate; check if Qh4 is given
    r_mate_missing = parse_move_with_details(board_check, "Qh4")
    assert r_mate_missing.normalization_kind == "semantic"
    assert "mate_marker_added" in r_mate_missing.normalization_changes


@pytest.mark.asyncio
async def test_classify_move_normalization_metadata() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # Play move with false capture 'Nxf3' in lenient mode
    res = await classify_move(fen=fen, move="Nxf3", depth=1, strict=False)
    assert res.played_san == "Nf3"
    assert res.normalization_kind == "semantic"
    assert "capture_marker_removed" in res.normalization_changes

    # Strict mode must raise STRICT_SAN_ERROR or STRICT_VALIDATION_ERROR
    with pytest.raises(ToolError) as exc_info:
        await classify_move(fen=fen, move="Nxf3", depth=1, strict=True)
    err = str(exc_info.value)
    assert "STRICT_VALIDATION_ERROR" in err or "STRICT_SAN_ERROR" in err


# ---------------------------------------------------------------------------
# 5. Verbosity normalization and Payload Size Budgets
# ---------------------------------------------------------------------------


def test_verbosity_normalization() -> None:
    assert _resolve_verbosity(None) == VERBOSITY_FULL
    assert _resolve_verbosity("min") == VERBOSITY_MINIMAL
    assert _resolve_verbosity("minimal") == VERBOSITY_MINIMAL
    assert _resolve_verbosity("compact") == VERBOSITY_COMPACT
    assert _resolve_verbosity("full") == VERBOSITY_FULL
    assert _resolve_verbosity("standard") == VERBOSITY_FULL
    assert _resolve_verbosity("default") == VERBOSITY_FULL

    with pytest.raises(ValueError) as exc:
        _resolve_verbosity("super_verbose")
    assert "INVALID_VERBOSITY" in str(exc.value)


@pytest.mark.asyncio
async def test_evaluate_position_payload_budgets() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # Minimal budget: < 1.5 KB (1536 bytes)
    res_minimal = await evaluate_position(fen=fen, depth=1, verbosity="minimal")
    raw_min = res_minimal.model_dump_json(exclude_none=True)
    assert len(raw_min) < 1536, f"Minimal evaluate_position too large: {len(raw_min)} bytes"

    # Compact budget: < 3.0 KB (3072 bytes)
    res_compact = await evaluate_position(fen=fen, depth=1, verbosity="compact")
    raw_compact = res_compact.model_dump_json(exclude_none=True)
    assert len(raw_compact) < 3072, f"Compact evaluate_position too large: {len(raw_compact)} bytes"


@pytest.mark.asyncio
async def test_top_moves_payload_budget() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # Compact top_moves n=3 budget: < 8.0 KB (8192 bytes)
    res = await top_moves(fen=fen, n=3, depth=1, verbosity="compact")
    raw = res.model_dump_json(exclude_none=True)
    assert len(raw) < 8192, f"Compact top_moves n=3 too large: {len(raw)} bytes"


@pytest.mark.asyncio
async def test_classify_move_payload_budget() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # Standard classify_move budget: < 8.0 KB (8192 bytes)
    res = await classify_move(fen=fen, move="e4", depth=1, detail="standard")
    raw = res.model_dump_json(exclude_none=True)
    assert len(raw) < 8192, f"Standard classify_move too large: {len(raw)} bytes"
