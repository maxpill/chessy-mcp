"""2026-09-09 master audit F-004: payload byte budgets must be enforced.

Production observed 245 KB responses for ordinary top_moves n=20 calls.
F-004 expanded ``_compact_mcpeval`` to also drop best_action_obj,
post_position, legal_actions, legal_rule_actions, action_policy and the
nested claim_* fields. This test pins representative budgets so future
changes don't regress the wire size.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module
from mcp_server.engine.retry import reset_breaker


class _ReprTopMovesPool:
    """Pool returning realistic top_moves payloads with forensic structures."""

    name = "ReprTopMovesPool"
    engine_version = "ReprTopMovesPool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        legal = list(board.legal_moves)
        best = legal[0].uci() if legal else None
        return Eval(
            cp=30,
            best_move=best,
            pv=[best, "e7e5", "g1f3"] if best else [],
            depth=max(depth, 1),
        )

    async def top_moves(self, board, *, n=3, depth=14):
        legal = list(board.legal_moves)
        out = []
        for i, m in enumerate(legal[:n]):
            out.append(
                Eval(
                    cp=30 - i * 5,
                    best_move=m.uci(),
                    pv=[m.uci(), "e7e5", "g1f3"],
                    depth=max(depth, 1),
                )
            )
        return out

    async def classify_move(self, board, move, depth=14):
        played = move.uci() if move else ""
        return MoveAnalysis(
            played=played,
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(cp=30, best_move=played),
            eval_after=Eval(cp=30),
            best_move_san=board.san(move) if move in board.legal_moves else "",
        )

    async def close(self) -> None:
        pass


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _payload_bytes(result: Any) -> int:
    return len(result.model_dump_json(exclude_none=True).encode("utf-8"))


@pytest.fixture(autouse=True)
async def _cleanup() -> None:
    yield
    reset_breaker()
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_evaluate_position_compact_stays_lean() -> None:
    """evaluate_position(startpos, compact) must stay under budget."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _ReprTopMovesPool()

    res = await server_module.evaluate_position(STARTPOS, depth=8, verbosity="compact")
    size = _payload_bytes(res)
    # Budget is generous to allow real-world content; the audit's median was
    # ~11 KB pre-fix. After F-004 it must be smaller than pre-fix.
    assert size < 12_000, (
        f"evaluate_position compact payload {size} B exceeded 12 KB budget; "
        f"audit median pre-fix was ~11 KB"
    )


@pytest.mark.asyncio
async def test_top_moves_n10_compact_stays_lean() -> None:
    """top_moves(startpos, n=10, compact) must stay under budget."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _ReprTopMovesPool()

    res = await server_module.top_moves(STARTPOS, n=10, depth=8, verbosity="compact")
    size = _payload_bytes(res)
    # Pre-fix audit observed ~209 KB for this call. Post-fix must drop materially.
    assert size < 20_000, (
        f"top_moves n=10 compact payload {size} B exceeded 20 KB budget; "
        f"pre-fix audit median was ~75 KB, max ~209 KB"
    )


@pytest.mark.asyncio
async def test_top_moves_n3_compact_stays_lean() -> None:
    """top_moves(startpos, n=3, compact) must stay under budget."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _ReprTopMovesPool()

    res = await server_module.top_moves(STARTPOS, n=3, depth=8, verbosity="compact")
    size = _payload_bytes(res)
    assert size < 8_000, f"top_moves n=3 compact payload {size} B exceeded 8 KB budget"


@pytest.mark.asyncio
async def test_classify_move_standard_stays_lean() -> None:
    """classify_move(startpos, e4, standard) must stay under budget."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _ReprTopMovesPool()

    res = await server_module.classify_move(STARTPOS, move="e2e4", depth=8)
    size = _payload_bytes(res)
    assert size < 15_000, f"classify_move standard payload {size} B exceeded 15 KB budget"


@pytest.mark.asyncio
async def test_full_payload_remains_larger_than_compact() -> None:
    """Full verbosity must remain larger than compact for the same call."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _ReprTopMovesPool()

    full = await server_module.top_moves(STARTPOS, n=3, depth=8, verbosity="full")
    compact = await server_module.top_moves(STARTPOS, n=3, depth=8, verbosity="compact")

    full_size = _payload_bytes(full)
    compact_size = _payload_bytes(compact)
    assert compact_size < full_size, (
        f"Compact ({compact_size}) must be smaller than full ({full_size})"
    )


@pytest.mark.asyncio
async def test_compact_drops_heavy_nested_fields() -> None:
    """Compact must explicitly null best_action_obj / post_position / legal_actions."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _ReprTopMovesPool()

    res = await server_module.top_moves(STARTPOS, n=3, depth=8, verbosity="compact")
    cand = res.result[0]

    assert cand.is_compact is True
    assert cand.best_action_obj is None
    assert cand.legal_actions == []
    assert cand.legal_rule_actions == []
    assert cand.action_policy is None
    assert cand.lichess_url is None
    assert cand.lichess_image is None
    assert cand.decision_value is None
    assert cand.engine_eval is None
    assert cand.input_fen is None
