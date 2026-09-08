"""2026-09-08 audit regression: zero-ply rich ``analyze_game`` returns a real coaching block.

Bug 5 (audit C.25) — Live ``analyze_game`` with a header-only zero-ply PGN
(``[Event "ZeroPly"]\\n[Site "?"]\\n[Result "*"]\\n\\n*``) and ``detail="coach"``
or ``detail="forensic"`` returned::

    "searched_depth":0,...,"coaching":null

The pre-fix code at ``mcp_server/analysis/game_analyzer.py:234`` entered
an early ``if not moves:`` return and hardcoded ``coaching=None``. The
normal rich coaching construction ran only after that branch. Rich modes
should still produce a coaching block (initial board, termination,
fingerprint) even when no plies have been played — that's the audit's
fix contract.

Fix: a ``_build_zero_ply_coaching`` helper constructs a minimal
``GameCoachingEvidence`` from the initial board without an engine call;
``searched_depth=0`` is preserved.
"""

from __future__ import annotations

import pytest

from mcp_server.analysis.game_analyzer import (
    GameAnalyzer,
    _build_zero_ply_coaching,
)
from mcp_server.models.game_coaching import GameCoachingEvidence
import chess


_ZERO_PLY_PGN = '[Event "ZeroPly"]\n[Site "?"]\n[Result "*"]\n\n*'


def test_zero_ply_helper_returns_real_coaching() -> None:
    """The helper builds a GameCoachingEvidence without an engine call."""
    board = chess.Board()
    coaching = _build_zero_ply_coaching(
        board,
        detail="coach",
        perspective="white",
        scan_depth=1,
        pgn=_ZERO_PLY_PGN,
    )
    assert isinstance(coaching, GameCoachingEvidence)
    assert coaching.detail == "coach"
    assert coaching.perspective == "white"
    assert coaching.critical_moments == []
    assert coaching.game_segments == []
    assert coaching.advantage_events == []
    assert coaching.final_position.legal_move_count == 20
    assert coaching.final_position.position_terminal_by_rules is False
    assert coaching.scan_depth == 1
    assert coaching.termination is not None


@pytest.mark.asyncio
async def test_analyze_game_zero_ply_coach_returns_real_coaching_block(monkeypatch):
    """The audit's exact zero-ply PGN with ``detail='coach'`` returns a coaching block.

    Pre-fix the response was ``coaching=null`` for rich modes. Post-fix
    the response carries a populated ``coaching`` block and
    ``searched_depth=0`` (no engine call needed for an empty game).
    """
    from mcp_server.analysis.game_analyzer import GameMetrics

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

    async def _fake_get_pool(ctx):
        return _FakePool()

    async def _fake_eval_positions(positions, depth, pool, **kwargs):
        return []

    def _fake_compute(positions, moves, evals):
        return GameMetrics(
            white_accuracy=None,
            black_accuracy=None,
            white_acpl=None,
            black_acpl=None,
            white_raw_acpl=None,
            black_raw_acpl=None,
            white_effective_acpl=None,
            black_effective_acpl=None,
            white_blunders=0,
            white_mistakes=0,
            white_inaccuracies=0,
            black_blunders=0,
            black_mistakes=0,
            black_inaccuracies=0,
            turning_points=[],
        )

    def _fake_identity(pool):
        return {}

    def _fake_engine(pool):
        return "Stockfish 18"

    analyzer = GameAnalyzer(
        get_pool=_fake_get_pool,
        evaluate_positions=_fake_eval_positions,
        compute_metrics=_fake_compute,
        identity=_fake_identity,
        engine_version=_fake_engine,
    )
    result = await analyzer.analyze(
        pgn=_ZERO_PLY_PGN,
        depth=1,
        strict=False,
        ctx=None,
        detail="coach",
        perspective="white",
        max_critical_moments=3,
    )
    assert result.total_plies == 0
    assert result.searched_depth == 0
    assert result.coaching is not None
    assert result.coaching.detail == "coach"
    assert result.coaching.final_position.legal_move_count == 20


@pytest.mark.asyncio
async def test_analyze_game_zero_ply_forensic_returns_real_coaching_block():
    """The forensic zero-ply variant also returns a populated coaching block.

    Same as the coach case but with ``detail='forensic'`` — the
    verification_depth on the coaching block should be the scan depth.
    """
    from mcp_server.analysis.game_analyzer import GameMetrics

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

    async def _fake_get_pool(ctx):
        return _FakePool()

    async def _fake_eval_positions(positions, depth, pool, **kwargs):
        return []

    def _fake_compute(positions, moves, evals):
        return GameMetrics(
            white_accuracy=None,
            black_accuracy=None,
            white_acpl=None,
            black_acpl=None,
            white_raw_acpl=None,
            black_raw_acpl=None,
            white_effective_acpl=None,
            black_effective_acpl=None,
            white_blunders=0,
            white_mistakes=0,
            white_inaccuracies=0,
            black_blunders=0,
            black_mistakes=0,
            black_inaccuracies=0,
            turning_points=[],
        )

    def _fake_identity(pool):
        return {}

    def _fake_engine(pool):
        return "Stockfish 18"

    analyzer = GameAnalyzer(
        get_pool=_fake_get_pool,
        evaluate_positions=_fake_eval_positions,
        compute_metrics=_fake_compute,
        identity=_fake_identity,
        engine_version=_fake_engine,
    )
    result = await analyzer.analyze(
        pgn=_ZERO_PLY_PGN,
        depth=4,
        strict=False,
        ctx=None,
        detail="forensic",
        perspective="black",
        max_critical_moments=5,
    )
    assert result.total_plies == 0
    assert result.coaching is not None
    assert result.coaching.detail == "forensic"
    assert result.coaching.perspective == "black"
    assert result.coaching.scan_depth == 4


@pytest.mark.asyncio
async def test_analyze_game_zero_ply_standard_still_returns_none():
    """Standard detail keeps the cheap ``coaching=None`` shape — control."""
    from mcp_server.analysis.game_analyzer import GameMetrics

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

    async def _fake_get_pool(ctx):
        return _FakePool()

    async def _fake_eval_positions(positions, depth, pool, **kwargs):
        return []

    def _fake_compute(positions, moves, evals):
        return GameMetrics(
            white_accuracy=None,
            black_accuracy=None,
            white_acpl=None,
            black_acpl=None,
            white_raw_acpl=None,
            black_raw_acpl=None,
            white_effective_acpl=None,
            black_effective_acpl=None,
            white_blunders=0,
            white_mistakes=0,
            white_inaccuracies=0,
            black_blunders=0,
            black_mistakes=0,
            black_inaccuracies=0,
            turning_points=[],
        )

    def _fake_identity(pool):
        return {}

    def _fake_engine(pool):
        return "Stockfish 18"

    analyzer = GameAnalyzer(
        get_pool=_fake_get_pool,
        evaluate_positions=_fake_eval_positions,
        compute_metrics=_fake_compute,
        identity=_fake_identity,
        engine_version=_fake_engine,
    )
    result = await analyzer.analyze(
        pgn=_ZERO_PLY_PGN,
        depth=1,
        strict=False,
        ctx=None,
        detail="standard",
        perspective="white",
    )
    assert result.total_plies == 0
    assert result.coaching is None
