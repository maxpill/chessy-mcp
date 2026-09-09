"""2026-09-09 master audit: cross-tool invariant suite.

Per audit master prompt §21, seven invariants must hold across the four
public tools. Each test is self-contained and uses a minimal pool mock so
the suite can run without a real Stockfish.
"""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module
from mcp_server.engine.retry import reset_breaker
from mcp_server.rules.constants import TOP_MOVES_MAX_N, DEPTH_MAX


class _FixedBestPool:
    """Minimal pool returning a single fixed best move; mocks classify + top_moves."""

    name = "FixedBestPool"
    engine_version = "FixedBestPool"

    def __init__(self, *, best_move_uci: str, score_cp: int = 30, mate: int | None = None) -> None:
        self._best = best_move_uci
        self._cp = score_cp
        self._mate = mate

    async def evaluate(self, board, *, depth=14, root_moves=None):
        legal = list(board.legal_moves)
        best = (
            self._best
            if any(m.uci() == self._best for m in legal)
            else (legal[0].uci() if legal else None)
        )
        return Eval(
            cp=self._cp if self._mate is None else None,
            mate=self._mate,
            best_move=best,
            pv=[best] if best else [],
            depth=max(depth, 1),
        )

    async def classify_move(self, board, move, depth=14):
        played = move.uci() if move else ""
        is_best = played == self._best
        return MoveAnalysis(
            played=played,
            move_class=MoveClass.BEST if is_best else MoveClass.GOOD,
            centipawn_loss=0 if is_best else 30,
            eval_before=Eval(
                cp=self._cp if self._mate is None else None,
                mate=self._mate,
                best_move=self._best,
            ),
            eval_after=Eval(
                cp=self._cp if self._mate is None else None,
                mate=self._mate,
            ),
            best_move_san=(board.san(chess.Move.from_uci(self._best)) if self._best else None),
        )

    async def top_moves(self, board, *, n=3, depth=14):
        legal = list(board.legal_moves)
        # Make sure the canonical best move is the first candidate, so
        # invariants checking tm.result[0].best_move agree with classify.
        ordered: list[chess.Move] = []
        for m in legal:
            if m.uci() == self._best:
                ordered.append(m)
                break
        for m in legal:
            if m.uci() != self._best:
                ordered.append(m)
        return [
            Eval(
                cp=self._cp - i,
                best_move=m.uci(),
                pv=[m.uci()],
                depth=max(depth, 1),
            )
            for i, m in enumerate(ordered[:n])
        ]

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _cleanup() -> None:
    yield
    reset_breaker()
    await server_module.close_analyzer_pool()


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.mark.asyncio
async def test_invariant_1_exact_best_board_move_agreement() -> None:
    """evaluate.best_move, top_moves[0].best_move, classify.is_engine_best agree."""
    await server_module._cache.clear()
    pool = _FixedBestPool(best_move_uci="e2e4")
    server_module._analyzer_pool = pool

    ev = await server_module.evaluate_position(STARTPOS, depth=8)
    tm = await server_module.top_moves(STARTPOS, n=1, depth=8)
    cl = await server_module.classify_move(STARTPOS, move="e2e4", depth=8)

    assert ev.best_move == "e2e4"
    assert tm.result[0].best_move == "e2e4"
    assert cl.is_engine_best is True


@pytest.mark.asyncio
async def test_invariant_2_terminal_coherence() -> None:
    """Terminal positions return zero playable actions across tools."""
    await server_module._cache.clear()
    pool = _FixedBestPool(best_move_uci="c8c7")
    server_module._analyzer_pool = pool

    # Stalemate: black king has no legal move but is not in check.
    stalemate_fen = "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"
    tm = await server_module.top_moves(stalemate_fen, n=3, depth=4)
    assert tm.returned_n == 0, (
        f"Stalemate must return zero playable actions; got returned_n={tm.returned_n}"
    )


@pytest.mark.asyncio
async def test_invariant_4_canonical_fen_after_replay() -> None:
    """Same input + moves must yield the same canonical final position across tools."""
    await server_module._cache.clear()
    pool = _FixedBestPool(best_move_uci="e7e5")
    server_module._analyzer_pool = pool

    moves = ["e4"]
    ev = await server_module.evaluate_position(STARTPOS, moves=moves, depth=4)
    tm = await server_module.top_moves(STARTPOS, moves=moves, n=1, depth=4)
    assert ev.canonical_fen is not None
    assert tm.canonical_fen is not None
    # After 1.e4, the white pawn sits on e4 and ranks 2-3 are "PPPP1PPP" / "8".
    assert "PPPP1PPP" in ev.canonical_fen
    assert "PPPP1PPP" in tm.canonical_fen


@pytest.mark.asyncio
async def test_invariant_6_depth_provenance_consistency() -> None:
    """requested_depth and searched_depth must never contradict."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _FixedBestPool(best_move_uci="e2e4")

    res = await server_module.evaluate_position(STARTPOS, depth=7)
    assert res.requested_depth == 7
    assert res.searched_depth == 7
    assert 1 <= res.searched_depth <= DEPTH_MAX


@pytest.mark.asyncio
async def test_invariant_7_exact_best_mate_action_equivalent() -> None:
    """Exact same move + same action + same outcome must NOT be action_equivalent=False."""

    class _MatePool(_FixedBestPool):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._mate = 1

        async def evaluate(self, board, *, depth=14, root_moves=None):  # type: ignore[override]
            legal = list(board.legal_moves)
            best = (
                self._best
                if any(m.uci() == self._best for m in legal)
                else (legal[0].uci() if legal else None)
            )
            return Eval(
                mate=1,
                best_move=best,
                pv=[best] if best else [],
                depth=max(depth, 1),
            )

        async def classify_move(self, board, move, depth=14):  # type: ignore[override]
            played = move.uci() if move else ""
            board_after = board.copy(stack=True)
            board_after.push(move) if move in board.legal_moves else None
            return MoveAnalysis(
                played=played,
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(mate=1, best_move=played),
                eval_after=Eval(mate=0),
                best_move_san=board.san(move) if move in board.legal_moves else "",
            )

    await server_module._cache.clear()
    fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    server_module._analyzer_pool = _MatePool(best_move_uci="f7g7")

    res = await server_module.classify_move(fen, move="Qg7#", depth=8)
    assert res.is_engine_best is True
    assert res.is_best_action is True
    assert res.action_equivalent is True, (
        f"Invariant 7 violated: exact engine-best mating move reports "
        f"action_equivalent={res.action_equivalent}"
    )


def test_top_moves_max_n_is_single_canonical_constant() -> None:
    """Schema description must match TOP_MOVES_MAX_N constant."""
    tool = server_module.mcp._tool_manager.get_tool("top_moves")
    description = tool.parameters["properties"]["n"]["description"]
    assert f"1-{TOP_MOVES_MAX_N}" in description
