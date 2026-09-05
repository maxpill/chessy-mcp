"""Engine protocol: the common typing surface for local and TCP pools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import chess


@runtime_checkable
class EngineLike(Protocol):
    """Minimum engine identity surface used by the evaluation pipeline."""

    name: str


@runtime_checkable
class EnginePoolLike(Protocol):
    """Common subset implemented by AnalyzerPool and TCPAnalyzerPool."""

    name: str
    engine_version: str

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int | None = None,
        root_moves: list[chess.Move] | None = None,
    ) -> Any: ...
