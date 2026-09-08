"""2026-09-08 audit regression: ``top_moves`` forensic enrichment must accept ``strict``.

Bug 1 — ``top_moves(startpos, n=3, depth=2, detail="standard", verbosity="compact",
include_moves=["e2-e4"])`` crashed with::

    [ENGINE_ERROR] enrich_top_moves_result() got an unexpected keyword argument 'strict'

Root cause: ``enrich_top_moves_result`` had no ``strict`` parameter, but
``mcp_server/tools/top_moves.py`` passed ``strict=strict`` into it on every
rich call. The fix threads ``strict`` into the function and forwards it into
``_comparison_requests`` so non-canonical SAN raises
``INVALID_COMPARE_MOVE`` symmetrically with ``classify_move`` strict mode.
"""

from __future__ import annotations

import pytest

from mcp_server.analysis.top_moves_forensics import enrich_top_moves_result
from mcp_server.models.forensics import ForensicTopMovesResult
from mcp_server.models.legacy import TopMovesResult


class _FakePool:
    name = "FakePool"
    engine_version = "FakePool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        return None

    async def top_moves(self, board, n=3, depth=14):
        return []

    async def classify_move(self, board, move, depth=14):
        return None

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_enrich_top_moves_result_accepts_strict_kwarg():
    """Bug 1: pre-fix this raised ``TypeError: unexpected keyword 'strict'``.

    The audit's exact failing payload was::
        top_moves(fen=startpos, n=3, depth=2, strict=False, detail="standard",
                  verbosity="compact", include_moves=["e2-e4"])

    which triggers the rich enrichment path (``include_moves`` non-empty).
    Post-fix the call returns a ``ForensicTopMovesResult`` with a populated
    forensics block.
    """
    import chess

    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    base = TopMovesResult(
        status="active",
        result=[],
        requested_depth=1,
        effective_depth=1,
        searched_depth=1,
        requested_n=1,
        clamped_n=1,
        returned_n=0,
        canonical_fen=board.fen(),
        fen_was_canonicalized=False,
    )
    out = await enrich_top_moves_result(
        base,
        board,
        pool=_FakePool(),
        depth=1,
        detail="forensic",
        include_moves=["e2-e4"],
        proof_mode="none",
        proof_defenses=3,
        strict=False,
    )
    assert isinstance(out, ForensicTopMovesResult)
    assert out.forensics is not None
    assert out.forensics.detail == "forensic"


@pytest.mark.asyncio
async def test_enrich_top_moves_strict_rejects_non_canonical_san():
    """Strict mode must reject non-canonical SAN ("e2-e4") with INVALID_COMPARE_MOVE.

    Mirrors the symmetric behavior the audit verified on ``classify_move``:
    non-canonical text under ``strict=True`` raises the structured
    INVALID_COMPARE_MOVE rather than silently canonicalizing.
    """
    import chess

    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    base = TopMovesResult(
        status="active",
        result=[],
        requested_depth=1,
        effective_depth=1,
        searched_depth=1,
        requested_n=1,
        clamped_n=1,
        returned_n=0,
        canonical_fen=board.fen(),
        fen_was_canonicalized=False,
    )
    with pytest.raises(ValueError, match="INVALID_COMPARE_MOVE"):
        await enrich_top_moves_result(
            base,
            board,
            pool=_FakePool(),
            depth=1,
            detail="forensic",
            include_moves=["e2-e4"],
            proof_mode="none",
            proof_defenses=3,
            strict=True,
        )


@pytest.mark.asyncio
async def test_enrich_top_moves_lenient_accepts_non_canonical_san():
    """Lenient mode must accept non-canonical SAN and canonicalize it.

    Symmetric with the strict test — ``e2-e4`` is well-formed UCI-style
    SAN notation; lenient mode canonicalizes to ``e4`` and proceeds.
    """
    import chess

    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    base = TopMovesResult(
        status="active",
        result=[],
        requested_depth=1,
        effective_depth=1,
        searched_depth=1,
        requested_n=1,
        clamped_n=1,
        returned_n=0,
        canonical_fen=board.fen(),
        fen_was_canonicalized=False,
    )
    out = await enrich_top_moves_result(
        base,
        board,
        pool=_FakePool(),
        depth=1,
        detail="forensic",
        include_moves=["e2-e4"],
        proof_mode="none",
        proof_defenses=3,
        strict=False,
    )
    assert out.forensics is not None
    assert len(out.forensics.candidate_comparisons) >= 1
