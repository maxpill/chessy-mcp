"""``mcp_server.models`` — public model package.

Flat-keep facade for the rest of the codebase: every Pydantic model and
the ``score_played_move`` helper stay importable from
``mcp_server.models`` so the existing call sites and the audit-invoked
test fixtures don't need to change their import paths.
"""

from __future__ import annotations

from mcp_server.models.action import ActionBlock
from mcp_server.models.action_policy import ActionPolicyMetadata
from mcp_server.models.eval import EvalBlock
from mcp_server.models.history import HistoryBlock
from mcp_server.models.mcpeval import MCPEval
from mcp_server.models.mcpeval_factory import attach_factory
from mcp_server.models.policy import PolicyBlock

# Bind ``from_eval`` so MCPEval.from_eval(...) keeps working unchanged.
attach_factory(MCPEval)

# ``score_played_move`` lives in :mod:`mcp_server.move_grading` for
# navigability. To break the circular import (``mcp_server.move_grading``
# uses ``MCPEval`` / ``PlayedMoveScore`` from this package), expose it via
# a function that defers the actual import to call time.


def score_played_move(*args: Any, **kwargs: Any) -> Any:
    from mcp_server.move_grading import score_played_move as _impl

    return _impl(*args, **kwargs)


# Other models that lived in the legacy ``mcp_server/models.py`` and
# remained unchanged. Splitting them out is on the Phase 18 follow-up.
from core.engines.types import MoveClass
from mcp_server.models.legacy import (  # noqa: E402
    GameAnalysisResult,
    MCPMoveAnalysis,
    PlayedMoveScore,
    PlyAnalysisItem,
    TopMovesResult,
)

__all__ = [
    "ActionBlock",
    "ActionPolicyMetadata",
    "EvalBlock",
    "GameAnalysisResult",
    "HistoryBlock",
    "MCPEval",
    "MCPMoveAnalysis",
    "MoveClass",
    "PlayedMoveScore",
    "PlyAnalysisItem",
    "PolicyBlock",
    "TopMovesResult",
    "score_played_move",
]
