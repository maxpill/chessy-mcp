"""2026-09-09 master audit F-002: top_moves.n must have one canonical maximum.

The audit showed public schema description and runtime clamp were inconsistent
in source: ``TOP_MOVES_MAX_N = 10`` in ``rules/constants.py`` and the ``Field``
description in ``tools/top_moves.py`` both said 10, while production runtime
clamped at 20 and ``middleware/request_cost.py`` cost-estimated at 20.

This test pins the single canonical contract: 20 is the public maximum, the
runtime clamp, the cost estimator clamp, and the schema description. It
parameterizes boundary inputs to catch any drift in either direction.
"""

from __future__ import annotations

import pytest

from mcp_server import server as server_module
from mcp_server.engine.retry import reset_breaker
from mcp_server.middleware.request_cost import estimate_mcp_request_cost
from mcp_server.rules.constants import TOP_MOVES_MAX_N, TOP_MOVES_MIN_N


EXPECTED_MAX = 20
EXPECTED_MIN = 1


class _EmptyPool:
    """Minimal pool returning no candidates (engine best is none, no list)."""

    name = "EmptyPool"
    engine_version = "EmptyPool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        return None

    async def classify_move(self, board, move, depth=14):
        raise NotImplementedError

    async def top_moves(self, board, *, n=3, depth=14):
        return []

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _cleanup() -> None:
    yield
    reset_breaker()
    await server_module.close_analyzer_pool()


def test_constants_match_canonical_contract() -> None:
    """Single source of truth: TOP_MOVES_MIN_N=1, TOP_MOVES_MAX_N=20."""
    assert TOP_MOVES_MIN_N == EXPECTED_MIN, (
        f"TOP_MOVES_MIN_N must be {EXPECTED_MIN}; got {TOP_MOVES_MIN_N}"
    )
    assert TOP_MOVES_MAX_N == EXPECTED_MAX, (
        f"TOP_MOVES_MAX_N must be {EXPECTED_MAX}; got {TOP_MOVES_MAX_N}"
    )


def test_field_description_says_20() -> None:
    """Public MCP field description must say clamped 1-20."""
    tool = server_module.mcp._tool_manager.get_tool("top_moves")
    description = tool.parameters["properties"]["n"]["description"]
    assert "1-20" in description, (
        f"top_moves.n field description must say '1-20'; got {description!r}"
    )
    assert "clamped 1-10" not in description, (
        f"top_moves.n field description must NOT say '1-10'; got {description!r}"
    )


@pytest.mark.parametrize(
    "raw_n, expected_clamped",
    [
        (-999, EXPECTED_MIN),
        (-1, EXPECTED_MIN),
        (0, EXPECTED_MIN),
        (1, 1),
        (2, 2),
        (3, 3),
        (9, 9),
        (10, 10),
        (11, 11),
        (19, 19),
        (20, EXPECTED_MAX),
        (21, EXPECTED_MAX),
        (999999, EXPECTED_MAX),
    ],
)
@pytest.mark.asyncio
async def test_top_moves_clamps_to_canonical_max(raw_n: int, expected_clamped: int) -> None:
    """Runtime clamp must match the canonical MAX (not 10, not arbitrary)."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _EmptyPool()

    res = await server_module.top_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        n=raw_n,
        depth=8,
    )

    assert res.requested_n == raw_n
    assert res.clamped_n == expected_clamped, (
        f"raw_n={raw_n} must clamp to {expected_clamped}; got {res.clamped_n}"
    )
    assert res.returned_n <= expected_clamped


def test_request_cost_estimator_uses_canonical_max() -> None:
    """Cost estimator must clamp n to the same canonical MAX."""
    import json

    body = json.dumps(
        {
            "params": {
                "name": "top_moves",
                "arguments": {"fen": "startpos", "n": 999, "depth": 12},
            }
        }
    ).encode()
    cost_999 = estimate_mcp_request_cost(body)
    body_20 = json.dumps(
        {
            "params": {
                "name": "top_moves",
                "arguments": {"fen": "startpos", "n": 20, "depth": 12},
            }
        }
    ).encode()
    cost_20 = estimate_mcp_request_cost(body_20)
    assert cost_999 == cost_20, (
        f"Cost estimator must treat n=999 as n={EXPECTED_MAX}; "
        f"got cost_999={cost_999} vs cost_20={cost_20}"
    )
