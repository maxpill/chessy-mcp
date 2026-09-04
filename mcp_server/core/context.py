"""``ToolContext`` — connected, dependency-injected context for MCP tools.

Replaces the module-level globals scattered across ``mcp_server.server``
(``_analyzer_pool``, ``_evaluate_semaphore``, ``_cache``, ``_single_flight``,
``_build_identity``, ``_engine_version_str``, ``_evaluate_game_position_cached``)
with a single frozen dataclass that the FastMCP lifespan builds once
and the tools fetch via ``request_context.lifespan_context["ctx"]``.

DI surface:

  * ``engine``        — ``AnalyzerPool | TCPAnalyzerPool`` (engine pool handle)
  * ``cache``         — ``MultiTierCache`` (multi-tier L1/L2 cache)
  * ``semaphore``     — ``asyncio.Semaphore`` (bounded-parallel gate)
  * ``single_flight`` — ``SingleFlight`` (coalesces in-flight cache misses)
  * ``identity_fn``   — ``Callable[[Any], dict]`` (build identity from a pool)
  * ``engine_version_fn`` — ``Callable[[Any], str]`` (engine version fingerprint)
  * ``evaluate_game_position`` — ``Callable[..., Awaitable[tuple[MCPEval, bool]]]``
  * ``gather_positions`` — ``Callable[..., Awaitable[list[tuple[MCPEval, bool]]]]``
  * ``settings`` — ``MCPSettings`` (configuration snapshot)

Tool entry points read ``ctx.engine``, ``ctx.cache`` etc. instead of
reaching back into module globals — tests can construct a
``ToolContext.with_test(...)`` factory with stub callables and pass it
through without touching the FastMCP lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from mcp_server.cache import MultiTierCache, SingleFlight
from mcp_server.config import MCPSettings

__all__ = ["ToolContext"]


@dataclass(frozen=True)
class ToolContext:
    """Frozen, hashable container for tool-level dependencies.

    Built once at lifespan start; passed to every tool entry point via
    ``request_context.lifespan_context["ctx"]``. Fields are typed but
    intentionally not validated at construction time — the lifespan is
    the single source of truth for which fields are required.
    """

    engine: Any
    cache: MultiTierCache
    semaphore: asyncio.Semaphore
    single_flight: SingleFlight
    settings: MCPSettings
    identity_fn: Callable[[Any], dict]
    engine_version_fn: Callable[[Any], str]
    evaluate_game_position: Callable[..., Awaitable[tuple]]
    gather_positions: Callable[..., Awaitable[list[tuple]]]
    extras: dict[str, Any] = field(default_factory=dict[str, Any])

    @classmethod
    def from_engine_layer(
        cls,
        *,
        engine: Any,
        settings: MCPSettings,
        cache: MultiTierCache | None = None,
        semaphore: asyncio.Semaphore | None = None,
        single_flight: SingleFlight | None = None,
        identity_fn: Callable[[Any], dict] | None = None,
        engine_version_fn: Callable[[Any], str] | None = None,
        evaluate_game_position: Callable[..., Awaitable[tuple]] | None = None,
        gather_positions: Callable[..., Awaitable[list[tuple]]] | None = None,
    ) -> ToolContext:
        """Build a context from a live engine + cache setup.

        Defaults pull from :mod:\`mcp_server.engine\` so the lifespan can
        construct one with a single ``engine\` handle. Tests pass the
        field-by-field overrides to swap in stubs.
        """
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
        single_flight: SingleFlight | None = None,
        identity_fn: Callable[[Any], dict] | None = None,
        engine_version_fn: Callable[[Any], str] | None = None,
        evaluate_game_position: Callable[..., Awaitable[tuple]] | None = None,
        gather_positions: Callable[..., Awaitable[list[tuple]]] | None = None,
        settings: MCPSettings | None = None,
    ) -> ToolContext:
        """Build a context with sensible defaults for tests.

        Use :func:\`make_test_context\` in conftest.py for a one-line fixture.
        """
        from mcp_server.config import get_mcp_settings

        return cls(
            engine=engine,
            cache=cache if cache is not None else MultiTierCache(l1_size=128),
            semaphore=semaphore if semaphore is not None else asyncio.Semaphore(1),
            single_flight=single_flight if single_flight is not None else SingleFlight(),
            settings=settings if settings is not None else get_mcp_settings(),
            identity_fn=identity_fn if identity_fn is not None else (lambda pool: {}),
            engine_version_fn=(
                engine_version_fn if engine_version_fn is not None else (lambda pool: "Stockfish")
            ),
            evaluate_game_position=(
                evaluate_game_position
                if evaluate_game_position is not None
                else (lambda *a, **k: (_empty_eval(), False))
            ),
            gather_positions=(
                gather_positions if gather_positions is not None else (lambda *a, **k: [])
            ),
        )


def _empty_eval():
    """Fallback ``MCPEval\` for the default tool-context — empty
    placeholder used when the test forgets to inject a real evaluator."""
    from mcp_server.models import MCPEval

    return MCPEval()
