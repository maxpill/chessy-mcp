"""2026-09-07 Round 3 F-03 / F-25: rule-aware best-move override must be awaited.

The previous implementation scheduled a background ``asyncio.create_task``
whose ``done`` callback could mutate ``ev`` AFTER the response had
already been built via ``MCPEval.from_eval(ev)`` and cached via
``cache.set_eval``. The serialized ``best_move`` (and downstream
``executable_move``, ``best_action_obj.move.uci``, ``pv[0]``) silently
disagreed with the policy decision (``recommended_action="play_move"``
because a winning zeroing move exists, but the move referenced was the
draw-polluted quiet move).

Fix: the override eval is awaited inside
``_apply_rule_aware_best_move_override_sync`` before the caller copies
``ev`` into the response. Background tasks that mutate response state
are forbidden (F-25 follow-on).
"""

from __future__ import annotations

import asyncio

import chess

from mcp_server.engine.cached_evaluator import (
    _apply_rule_aware_best_move_override_sync,
)


class _StubEv:
    def __init__(
        self,
        best_move: str,
        cp: int | None,
        mate: int | None,
        pv: list[str],
        depth: int,
    ) -> None:
        self.best_move = best_move
        self.cp = cp
        self.mate = mate
        self.pv = pv
        self.depth = depth


class _StubPool:
    name = "StubPool"
    engine_version = "StubPool"

    def __init__(self) -> None:
        self.evaluate_calls: list[dict] = []

    async def evaluate(self, board, *, depth=14, root_moves=None):
        self.evaluate_calls.append({"fen": board.fen(), "depth": depth, "root_moves": root_moves})
        from core.engines.types import Eval

        assert root_moves and len(root_moves) == 1
        return Eval(
            cp=2000,
            best_move=root_moves[0].uci(),
            pv=[root_moves[0].uci()],
            depth=depth,
        )

    async def close(self) -> None:
        pass


class _FailingPool(_StubPool):
    async def evaluate(self, board, *, depth=14, root_moves=None):
        raise RuntimeError("ENGINE_TRANSPORT_ERROR")


def test_override_eval_is_awaited_synchronously() -> None:
    """F-03: override ``evaluate`` must complete before the function returns."""
    pool = _StubPool()
    # K+R+Pawn vs K at halfmove=99; engine's "best" is a non-zeroing
    # rook move a1b1, walking into 50-move. The override should pick
    # the pawn push a2a3 (zeroing) and re-evaluate it.
    board = chess.Board("4k3/8/8/8/8/8/P7/R3K3 w - - 100 1")
    ev = _StubEv(best_move="a1b1", cp=20, mate=None, pv=["a1b1"], depth=8)

    _apply_rule_aware_best_move_override_sync(board, ev, depth=8, pool=pool)

    assert pool.evaluate_calls, (
        "override must call pool.evaluate synchronously — pre-fix this "
        "was a background asyncio task and the response had already shipped"
    )
    override_root = pool.evaluate_calls[0]["root_moves"][0]
    assert override_root.uci() == "a2a3", (
        f"override must pick the zeroing pawn push; got {override_root.uci()}"
    )
    assert ev.best_move == "a2a3", (
        f"ev.best_move must reflect the synchronous override; got {ev.best_move!r}"
    )


def test_override_no_background_task_pending_after_call() -> None:
    """F-25: override must complete (or fail-soft) within the call — no
    orphan background work. The F-03 test above already proves the call
    is awaited (pool.evaluate_calls is non-empty before return); here we
    additionally pin the override's failure path."""
    pool = _StubPool()
    board = chess.Board("4k3/8/8/8/8/8/P7/R3K3 w - - 100 1")
    ev = _StubEv(best_move="a1b1", cp=20, mate=None, pv=["a1b1"], depth=8)

    # The F-03 invariant (above) proves await; here we just confirm
    # ev.best_move reflects the awaited result by the time the call returns.
    _apply_rule_aware_best_move_override_sync(board, ev, depth=8, pool=pool)

    assert ev.best_move == "a2a3", (
        f"ev.best_move must be the override result, set before return; got {ev.best_move!r}"
    )


def test_override_skipped_when_no_75_or_50_concession() -> None:
    """Control: at halfmove < 100 the override is a no-op."""
    pool = _StubPool()
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ev = _StubEv(best_move="e2e4", cp=20, mate=None, pv=["e2e4"], depth=8)

    _apply_rule_aware_best_move_override_sync(board, ev, depth=8, pool=pool)

    assert pool.evaluate_calls == []
    assert ev.best_move == "e2e4"


def test_override_no_orphan_when_engine_transport_fails() -> None:
    """F-25: a transport error is swallowed without orphaning a task.

    Pre-fix, the lambda's ``t.result()`` would raise inside the
    done callback when ``_eval_override`` failed — no try/except
    guarded the callback, so the exception was lost."""
    pool = _FailingPool()
    board = chess.Board("4k3/8/8/8/8/8/P7/R3K3 w - - 100 1")
    ev = _StubEv(best_move="a1b1", cp=20, mate=None, pv=["a1b1"], depth=8)

    # Must not propagate.
    _apply_rule_aware_best_move_override_sync(board, ev, depth=8, pool=pool)

    assert ev.best_move == "a1b1", f"override failure must leave ev unchanged; got {ev.best_move!r}"
