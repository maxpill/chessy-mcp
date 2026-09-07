"""2026-09-07 audit regression: top_moves forensic result construction must not crash.

Bug §1 — ``top_moves(startpos, n=1, depth=1, detail="forensic")`` previously
crashed with::

    [ENGINE_ERROR] mcp_server.models.forensics.ForensicTopMovesResult()
    got multiple values for keyword argument 'forensics'

Root cause: ``ForensicTopMovesResult(TopMovesResult)`` carries its own
``forensics: TopMovesForensicEvidence | None = None`` field, and the
top-moves forensic enrichment path previously did::

    ForensicTopMovesResult(**result.model_dump(), forensics=forensic)

When ``result`` is already a ``ForensicTopMovesResult`` (the enrichment path
that runs after the standard construction in ``tools/top_moves.py:110``),
its ``model_dump()`` includes ``forensics=None`` from the parent schema.
Adding ``forensics=forensic`` as an explicit keyword produced the duplicate-
key crash. The fix excludes ``forensics`` from the dump and assigns it once
via the explicit keyword.
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
async def test_enrich_top_moves_result_does_not_double_forensics_keyword():
    """§1: re-enriching an already-ForensicTopMovesResult must not crash.

    The original bug fired when ``enrich_top_moves_result`` was passed an
    existing ``ForensicTopMovesResult`` (the typical case from
    ``tools.top_moves`` after the initial forensic promotion at line 110) and
    the body did ``ForensicTopMovesResult(**result.model_dump(), forensics=...)``
    — the second ``forensics`` kwarg collided with the parent schema's field.
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
    promoted = ForensicTopMovesResult(**base.model_dump())
    # The promotion-without-forensics path is safe (no crash). Verify it first.
    assert promoted.forensics is None

    # Now the audit's actual trap: re-enrich a ForensicTopMovesResult. Pre-fix
    # this crashed; post-fix it must produce a new ForensicTopMovesResult with
    # the freshly populated forensics block.
    pool = _FakePool()
    out = await enrich_top_moves_result(
        promoted,
        board,
        pool=pool,
        depth=1,
        detail="forensic",
        include_moves=None,
        proof_mode="none",
        proof_defenses=3,
    )
    assert isinstance(out, ForensicTopMovesResult)
    # The freshly-built forensics block must round-trip; the previous value
    # (None) must have been replaced — proving the single-assignment path
    # wrote the new value instead of crashing.
    assert out.forensics is not None
    assert out.forensics.detail == "forensic"


def test_forensic_top_moves_result_default_forensics_is_none():
    """The forensic-specific ``forensics`` field has ``None`` as its default
    and is settable via the keyword. Pins the typed-model contract surface
    the enrich_top_moves_result depends on."""
    # A freshly-promoted ForensicTopMovesResult (no enrichment yet) must have
    # forensics=None — the path that constructed it supplied a None value
    # rather than crashing on the duplicate keyword.
    base = TopMovesResult(status="active", result=[])
    promoted = ForensicTopMovesResult(**base.model_dump())
    assert promoted.forensics is None
    assert hasattr(promoted, "forensics"), "forensics field must exist on ForensicTopMovesResult"
