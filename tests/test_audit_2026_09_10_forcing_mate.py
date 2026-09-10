"""Regression tests for 2026-09-10 ultra audit findings AUDIT-001, AUDIT-002, AUDIT-005, AUDIT-006."""

from __future__ import annotations

import chess
import pytest

from mcp_server.analysis.forensics import (
    build_tactical_snapshot,
)
from mcp_server import server as server_module


def test_audit_001_fools_mate_is_mate_flag() -> None:
    """AUDIT-001: Qh4# in Fool's Mate must have is_mate=True and is_check=True."""
    # 1. f3 e5 2. g4
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    m = chess.Move.from_uci("d8h4")
    assert m in board.legal_moves
    assert board.gives_check(m)
    child = board.copy(stack=False)
    child.push(m)
    assert child.is_checkmate()

    snapshot = build_tactical_snapshot(board)
    checks_by_san = {c.san: c for c in snapshot.checks}
    assert "Qh4#" in checks_by_san
    qh4 = checks_by_san["Qh4#"]
    assert qh4.is_check is True
    assert qh4.is_mate is True, (
        f"AUDIT-001 failed: Qh4# has is_mate={qh4.is_mate}, expected True"
    )


def test_audit_001_kqk_mate_and_rook_mate() -> None:
    """AUDIT-001: ordinary KQK and KRK mates must report is_mate=True."""
    # KQK mate: White king on g6, queen on f7, Black king on h8.
    board_kqk = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    snapshot = build_tactical_snapshot(board_kqk)
    checks = {c.san: c for c in snapshot.checks}
    assert "Qg7#" in checks
    assert checks["Qg7#"].is_mate is True

    # KRK back-rank mate: White rook on a1, Black king on e8 with pawns on d7,e7,f7.
    board_krk = chess.Board("4k3/3ppp2/8/8/8/8/8/R3K3 w - - 0 1")
    snapshot_krk = build_tactical_snapshot(board_krk)
    checks_krk = {c.san: c for c in snapshot_krk.checks}
    assert "Ra8#" in checks_krk
    assert checks_krk["Ra8#"].is_mate is True


def test_audit_001_mating_capture_and_promotion_mate() -> None:
    """AUDIT-001: mating capture and promotion with mate must report is_mate=True."""
    # Scholar's mate: Qxf7# is a mating capture
    board_scholar = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    snapshot = build_tactical_snapshot(board_scholar)
    captures = {c.san: c for c in snapshot.captures}
    assert "Qxf7#" in captures
    assert captures["Qxf7#"].is_mate is True
    assert captures["Qxf7#"].is_check is True
    assert captures["Qxf7#"].is_capture is True

    # Promotion with mate
    board_promo_mate = chess.Board("1k6/7P/1K6/8/8/8/8/8 w - - 0 1")
    snapshot_pm = build_tactical_snapshot(board_promo_mate)
    checks_pm = {c.san: c for c in snapshot_pm.checks}
    assert "h8=Q#" in checks_pm
    assert checks_pm["h8=Q#"].is_mate is True
    assert checks_pm["h8=Q#"].promotion == "queen"


def test_audit_001_ordinary_check_not_mate() -> None:
    """AUDIT-001: Ordinary check must have is_check=True, is_mate=False."""
    board = chess.Board("8/8/4k3/8/8/4K3/3Q4/8 w - - 0 1")
    snapshot = build_tactical_snapshot(board)
    checks = {c.san: c for c in snapshot.checks}
    assert "Qd6+" in checks
    assert checks["Qd6+"].is_check is True
    assert checks["Qd6+"].is_mate is False


@pytest.mark.asyncio
async def test_audit_005_include_moves_deduplication_accounting() -> None:
    """AUDIT-005: included_move_count must count unique included root moves, not raw strings."""
    await server_module._cache.clear()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    res = await server_module.top_moves(
        fen=fen,
        include_moves=["e4", "e4", "e2e4"],
        detail="coach",
        depth=1,
    )
    assert res.included_move_count == 1, (
        f"AUDIT-005 failed: included_move_count was {res.included_move_count}, expected 1"
    )


@pytest.fixture(autouse=True)
async def _cleanup_pool() -> None:
    yield
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_audit_006_minimal_verbosity_rejects_rich_detail() -> None:
    """AUDIT-006: evaluate_position must reject verbosity='minimal' with detail='coach'/'forensic'."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    with pytest.raises(Exception) as exc_info:
        await server_module.evaluate_position(
            fen=fen,
            verbosity="minimal",
            detail="coach",
            depth=1,
        )
    err = str(exc_info.value)
    assert "INVALID_ARGUMENT" in err or "invalid_argument" in err.lower()
