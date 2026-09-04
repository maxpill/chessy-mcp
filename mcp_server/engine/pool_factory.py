"""Back-compat shim — :mod:`mcp_server.engine` was split into focused
modules in Phase 24.

This module is kept so existing test monkeypatch paths like
`monkeypatch.setattr(\"mcp_server.engine.pool_factory._evaluate_game_position_cached\", ...)`
keep working. New code should import from :mod:`mcp_server.engine`
directly.
"""

from mcp_server.engine.cached_evaluator import (
    _cache,
    _evaluate_game_position_cached,
    _get_evaluate_semaphore,
    _single_flight,
    cache,
    single_flight,
)
from mcp_server.engine.lifespan import (
    _mcp_lifespan,
    _pool_stats_logger,
)
from mcp_server.engine.parallel_gather import (
    _gather_evaluate_positions_bounded,
)
from mcp_server.engine.ponder import (
    _maybe_ponder_warm,
    _ponder_warm_cache,
)
from mcp_server.engine.pool_lifecycle import (
    _create_analyzer_pool,
    _eval_via_analyzer_or_pool,
    _get_analyzer_pool,
    _pool_supports_root_moves,
    close_analyzer_pool,
)


__all__ = [
    "_cache",
    "_create_analyzer_pool",
    "_eval_via_analyzer_or_pool",
    "_evaluate_game_position_cached",
    "_gather_evaluate_positions_bounded",
    "_get_analyzer_pool",
    "_get_evaluate_semaphore",
    "_maybe_ponder_warm",
    "_mcp_lifespan",
    "_ponder_warm_cache",
    "_pool_stats_logger",
    "_pool_supports_root_moves",
    "_single_flight",
    "cache",
    "close_analyzer_pool",
    "single_flight",
]
