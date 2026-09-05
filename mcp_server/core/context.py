"""``ToolContext``: dependency-injected context for MCP tools."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from mcp_server.cache import MultiTierCache, SingleFlight
from mcp_server.config import MCPSettings

__all__ = ["ToolContext"]

EvalPair = tuple[Any, bool]


@dataclass(frozen=True)
class ToolContext:
    """Frozen container for tool-level dependencies built at lifespan start."""

    engine: Any
    cache: MultiTierCache
    semaphore: asyncio.Semaphore
    single_flight: SingleFlight[Any]
    settings: MCPSettings
    identity_fn: Callable[[Any], dict[str, Any]]
    engine_version_fn: Callable[[Any], str]
    evaluate_game_position: Callable[..., Awaitable[EvalPair]]
    gather_positions: Callable[..., Awaitable[list[EvalPair]]]
    extras: dict[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def from_engine_layer(
        cls,
        *,
        engine: Any,
        settings: MCPSettings,
        cache: MultiTierCache | None = None,
        semaphore: asyncio.Semaphore | None = None,
        single_flight: SingleFlight[Any] | None = None,
        identity_fn: Callable[[Any], dict[str, Any]] | None = None,
        engine_version_fn: Callable[[Any], str] | None = None,
        evaluate_game_position: Callable[..., Awaitable[EvalPair]] | None = None,
        gather_positions: Callable[..., Awaitable[list[EvalPair]]] | None = None,
    ) -> ToolContext:
        """Build a context from the live engine layer."""
        from mcp_server.engine import (
            _build_identity,
            _engine_version_str,
            _evaluate_game_position_cached,
            _gather_evaluate_positions_bounded,
            cache as _engine_cache,
            single_flight as _engine_sf,
        )

        return cls(
            engine=engine,
            settings=settings,
            cache=cache if cache is not None else _engine_cache,
            semaphore=semaphore if semaphore is not None else asyncio.Semaphore(1),
            single_flight=single_flight if single_flight is not None else _engine_sf,
            identity_fn=identity_fn if identity_fn is not None else _build_identity,
            engine_version_fn=(
                engine_version_fn if engine_version_fn is not None else _engine_version_str
            ),
            evaluate_game_position=(
                evaluate_game_position
                if evaluate_game_position is not None
                else _evaluate_game_position_cached
            ),
            gather_positions=(
                gather_positions
                if gather_positions is not None
                else _gather_evaluate_positions_bounded
            ),
        )

    @classmethod
    def for_test(
        cls,
        *,
        engine: Any = None,
        cache: MultiTierCache | None = None,
        semaphore: asyncio.Semaphore | None = None,
        single_flight: SingleFlight[Any] | None = None,
        identity_fn: Callable[[Any], dict[str, Any]] | None = None,
        engine_version_fn: Callable[[Any], str] | None = None,
        evaluate_game_position: Callable[..., Awaitable[EvalPair]] | None = None,
        gather_positions: Callable[..., Awaitable[list[EvalPair]]] | None = None,
        settings: MCPSettings | None = None,
    ) -> ToolContext:
        """Build a context with safe async defaults for tests."""
        from mcp_server.config import get_mcp_settings

        return cls(
            engine=engine,
            cache=cache if cache is not None else MultiTierCache(l1_size=128),
            semaphore=semaphore if semaphore is not None else asyncio.Semaphore(1),
            single_flight=single_flight if single_flight is not None else SingleFlight[Any](),
            settings=settings if settings is not None else get_mcp_settings(),
            identity_fn=identity_fn if identity_fn is not None else (lambda pool: {}),
            engine_version_fn=(
                engine_version_fn if engine_version_fn is not None else (lambda pool: "Stockfish")
            ),
            evaluate_game_position=(
                evaluate_game_position
                if evaluate_game_position is not None
                else _empty_evaluate_game_position
            ),
            gather_positions=(
                gather_positions if gather_positions is not None else _empty_gather_positions
            ),
        )


async def _empty_evaluate_game_position(*args: Any, **kwargs: Any) -> EvalPair:
    return _empty_eval(), False


async def _empty_gather_positions(*args: Any, **kwargs: Any) -> list[EvalPair]:
    return []


def _empty_eval() -> Any:
    """Empty MCPEval placeholder for tests that omit an evaluator."""
    from mcp_server.models import MCPEval

    return MCPEval()
