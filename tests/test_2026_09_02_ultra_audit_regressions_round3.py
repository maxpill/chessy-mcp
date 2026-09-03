"""Round-3 regressions: partial Date wildcards, strict SAN markers/promotion."""

import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from core.engines.types import Eval


class _NoopPool:
    name = "N"
    engine_version = "N"

    async def evaluate(self, *a, **k):
        return Eval(cp=0, mate=None, best_move="", pv=[], depth=10)

    async def top_moves(self, *a, **k):
        return [Eval(cp=0, mate=None, best_move="", pv=[], depth=10)]

    async def classify(self, *a, **k):
        return None

    async def close(self):
        pass


def _run(coro):
    """Run an async coroutine in a fresh loop."""
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def _install_pool():
    from mcp_server import server as sm

    sm._analyzer_pool = _NoopPool()
    yield


@pytest.mark.parametrize(
    "date_value",
    ["????.??.??", "????.09.02", "2026.09.??", "2026.??.02", "2026.??.??"],
)
def test_partial_date_wildcards_accepted_strict(date_value):
    from mcp_server import server as sm

    pgn = f'[Date "{date_value}"]\n[Result "*"]\n\n*'

    async def _go():
        return await sm.analyze_game(pgn, depth=10, strict=True)

    try:
        _run(_go())
    except Exception as e:
        pytest.fail(f"Date {date_value!r} was rejected: {type(e).__name__}: {e}")


@pytest.mark.parametrize(
    "date_value",
    ["2026.13.01", "2026.04.31", "2023.02.29", "2026.02.31", "hello", "2026-09-02", "abc.def.ghi"],
)
def test_invalid_date_rejected_strict(date_value):
    from mcp_server import server as sm

    pgn = f'[Date "{date_value}"]\n[Result "*"]\n\n*'

    async def _go():
        return await sm.analyze_game(pgn, depth=10, strict=True)

    with pytest.raises(ToolError):
        _run(_go())


@pytest.mark.parametrize("date_value", ["", " ", "  ", "?"])
def test_empty_or_sentinel_date_normalized_to_none(date_value):
    from mcp_server import server as sm

    pgn = f'[Date "{date_value}"]\n[Result "*"]\n\n*'

    async def _go():
        return await sm.analyze_game(pgn, depth=10, strict=True)

    r = _run(_go())
    assert r.date is None, f"[Date {date_value!r}] should normalize to None, got {r.date!r}"


def test_strict_legal_check_marker_accepted():
    """Qh5+ is a legal check from h1; strict must accept."""
    from mcp_server import server as sm

    async def _go():
        return await sm.analyze_game(
            '[SetUp "1"]\n[FEN "4k3/8/8/8/8/8/8/4K2Q w - - 0 1"]\n\n1. Qh5+ *',
            depth=10,
            strict=True,
        )

    _run(_go())


def test_strict_false_check_marker_rejected():
    """e4+ (illegal — pawn move cannot give check) must be rejected."""
    from mcp_server import server as sm

    async def _go():
        return await sm.analyze_game('[Result "*"]\n\n1. e4+ e5 *', depth=10, strict=True)

    with pytest.raises(ToolError):
        _run(_go())


def test_strict_false_mate_marker_rejected():
    """Qh5# (illegal — not mate) must be rejected."""
    from mcp_server import server as sm

    async def _go():
        return await sm.analyze_game('[Result "*"]\n\n1. e4 e5 2. Qh5# *', depth=10, strict=True)

    with pytest.raises(ToolError):
        _run(_go())


def test_strict_promotion_no_eq_accepted():
    """e8Q (no '=') is valid PGN §8.1.4; strict must accept."""
    from mcp_server import server as sm

    async def _go():
        return await sm.analyze_game(
            '[SetUp "1"]\n[FEN "8/4P3/8/8/8/8/8/4K2k w - - 0 1"]\n\n1. e8Q *',
            depth=10,
            strict=True,
        )

    _run(_go())


def test_strict_promotion_lowercase_rejected():
    """e8=q (lowercase) is invalid PGN §8.1.4; strict must reject."""
    from mcp_server import server as sm

    async def _go():
        return await sm.analyze_game(
            '[SetUp "1"]\n[FEN "8/4P3/8/8/8/8/8/4K2k w - - 0 1"]\n\n1. e8=q *',
            depth=10,
            strict=True,
        )

    with pytest.raises(ToolError):
        _run(_go())


def test_strict_canonical_promotion_accepted():
    """e8=Q (canonical) must continue to be accepted."""
    from mcp_server import server as sm

    async def _go():
        return await sm.analyze_game(
            '[SetUp "1"]\n[FEN "8/4P3/8/8/8/8/8/4K2k w - - 0 1"]\n\n1. e8=Q *',
            depth=10,
            strict=True,
        )

    _run(_go())


def test_strict_check_marker_when_canonical_emits_it():
    """Rd8+ after Rd1-d8 with black king on e8 — python-chess returns 'Rd8+' so both forms must accept."""
    from mcp_server import server as sm

    async def _go():
        return await sm.analyze_game(
            '[SetUp "1"]\n[FEN "4k3/8/8/8/8/8/8/3R1K2 w - - 0 1"]\n\n1. Rd8+ *',
            depth=10,
            strict=True,
        )

    _run(_go())
