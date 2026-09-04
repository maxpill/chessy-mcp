"""Engine Protocol — public typing surface.

Defines the structural type both ``core.engines.pool.AnalyzerPool`` and
``mcp_server.tcp_analyzer.TCPAnalyzerPool`` implement, plus narrower
sub-protocols for the operations the eval pipeline actually uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import chess


@runtime_checkable
class EngineLike(Protocol):
    """Minimum eval surface used by ``mcp_server.engine.eval_pipeline``."""

    name: str


@runtime_checkable
class EnginePoolLike(Protocol):
    """Pool used by the eval pipeline — exposes ``.evaluate(board, depth, ...)``.

    Both ``AnalyzerPool`` (local Stockfish subprocesses) and ``TCPAnalyzerPool``
    (remote UCI over TCP) conform.
    """

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int = ...,
        root_moves: Any = ...,
        reuse_tt: bool = ...,
    ) -> Any: ...

    def engine_version(self) -> str: ...
