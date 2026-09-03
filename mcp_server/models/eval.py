"""``EvalBlock`` aggregate — engine output surface of :class:`MCPEval`.

Groups the Stockfish engine's raw output (cp / mate / pv / depth) plus
the wire-level aliases (root_score_* / post_state_*) and the WDL per-mille
tuple. The action side of the response (``best_move``, legal moves,
claim-action surface) is intentionally NOT here — see
:class:`mcp_server.models.action.ActionBlock`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalBlock(BaseModel):
    """Engine-side fields of a single position evaluation.

    All fields keep their pre-atomization wire semantics so
    :meth:`MCPEval.from_eval` can compose them unchanged.
    """

    cp: int | None = None
    mate: int | None = None
    depth: int = 0
    requested_depth: int | None = None
    searched_depth: int | None = None
    pv: list[str] = Field(default_factory=list[str])
    wdl: tuple[int, int, int] | None = None
    wdl_pct: dict[str, float] | None = None
    root_score_cp: int | None = None
    root_score_mate: int | None = None
    post_state_cp: int | None = None
    post_state_mate: int | None = None
    engine_eval: dict[str, Any] | None = None
