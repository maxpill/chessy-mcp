"""2026-09-08 audit regression: ``classify_move`` cache hits must rebind nested engine_eval depth.

Bug 7 — A live ``classify_move(..., depth=1)`` returned::

    "eval_before":{..."requested_depth":1,"searched_depth":1,...,
                  "engine_eval":{..."requested_depth":0,"searched_depth":1}}
    "eval_after":{..."requested_depth":1,"searched_depth":1,...,
                  "engine_eval":{..."requested_depth":-100,"searched_depth":1}}

The top-level and the nested ``engine_eval.requested_depth`` disagreed
because ``tools/classify_move.py:156-161`` only rebound the top-level
field on cache hits. The cached snapshot retained whatever depth the
original request used (including ``0`` / ``-100`` from the audit's
edge-case depth sweep).

Fix: a small ``_stamp_requested_depth`` helper walks the nested
``engine_eval`` dict and rewrites its ``requested_depth`` to match the
top-level field.
"""

from __future__ import annotations

from mcp_server.models.mcpeval import MCPEval
from mcp_server.tools.classify_move import _stamp_requested_depth


def test_stamp_rebinds_top_level_depth() -> None:
    """Pre-fix the top-level rebind worked; pin the contract."""
    ev = MCPEval(
        cp=20,
        mate=None,
        depth=1,
        requested_depth=1,
        searched_depth=1,
        best_move="e2e4",
        pv=["e2e4"],
        engine_eval={"requested_depth": 1, "searched_depth": 1, "best_move": "e2e4", "cp": 20},
    )
    out = _stamp_requested_depth(ev, 5)
    assert out.requested_depth == 5


def test_stamp_rebinds_nested_engine_eval_depth() -> None:
    """Bug 7 primary regression: nested ``engine_eval.requested_depth`` is rebound."""
    ev = MCPEval(
        cp=20,
        mate=None,
        depth=1,
        requested_depth=1,
        searched_depth=1,
        best_move="e2e4",
        pv=["e2e4"],
        engine_eval={
            "requested_depth": 0,  # stale snapshot
            "searched_depth": 1,
            "best_move": "e2e4",
            "cp": 20,
        },
    )
    out = _stamp_requested_depth(ev, 1)
    assert out.engine_eval is not None
    assert out.engine_eval["requested_depth"] == 1, (
        f"nested engine_eval.requested_depth must be rebound to 1; got "
        f"{out.engine_eval.get('requested_depth')!r}"
    )


def test_stamp_handles_audit_negative_depth_snapshot() -> None:
    """Audit verified a cached snapshot with ``requested_depth=-100`` in nested engine_eval.

    The new request is depth=1 — the helper must overwrite the negative
    snapshot value and not propagate it.
    """
    ev = MCPEval(
        cp=15,
        mate=None,
        depth=1,
        requested_depth=1,
        searched_depth=1,
        best_move="",
        pv=[],
        engine_eval={
            "requested_depth": -100,  # audit-captured edge case
            "searched_depth": 1,
            "best_move": "",
            "cp": 15,
        },
    )
    out = _stamp_requested_depth(ev, 1)
    assert out.engine_eval is not None
    assert out.engine_eval["requested_depth"] == 1


def test_stamp_preserves_other_engine_eval_fields() -> None:
    """The nested ``engine_eval`` dict may carry unrelated fields (cp, best_move, etc.).

    Only ``requested_depth`` is rebound; everything else must pass through.
    """
    ev = MCPEval(
        cp=42,
        mate=None,
        depth=8,
        requested_depth=8,
        searched_depth=8,
        best_move="e2e4",
        pv=["e2e4"],
        engine_eval={
            "requested_depth": 8,
            "searched_depth": 8,
            "best_move": "e2e4",
            "cp": 42,
            "wdl": (10, 5, 0),
        },
    )
    out = _stamp_requested_depth(ev, 12)
    assert out.engine_eval is not None
    assert out.engine_eval["requested_depth"] == 12
    assert out.engine_eval["searched_depth"] == 8
    assert out.engine_eval["best_move"] == "e2e4"
    assert out.engine_eval["cp"] == 42
    assert out.engine_eval["wdl"] == (10, 5, 0)


def test_stamp_handles_missing_engine_eval() -> None:
    """A snapshot without an ``engine_eval`` payload must not crash the helper."""
    ev = MCPEval(
        cp=20,
        mate=None,
        depth=1,
        requested_depth=1,
        searched_depth=1,
        best_move="e2e4",
        pv=["e2e4"],
        engine_eval=None,
    )
    out = _stamp_requested_depth(ev, 7)
    assert out.requested_depth == 7
    assert out.engine_eval is None
