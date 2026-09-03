"""Engine pool factory, lifespan wiring, and build-identity helpers.

The two modules split the engine concerns:

- :mod:`mcp_server.engine.pool_factory` — Stockfish pool creation, FastMCP
  lifespan, cached single-position evaluation pipeline, and the bounded
  parallel-position gatherer used by ``analyze_game``.
- :mod:`mcp_server.engine.identity` — version / git SHA / engine
  configuration snapshot via the :class:`BuildIdentity` dataclass, plus
  Stockfish binary path resolution.
"""

from mcp_server.engine.identity import (
    BuildIdentity,
    DEFAULT_STOCKFISH_PATH,
    _build_identity,
    _build_sha,
    _engine_config,
    _package_version,
    _stockfish_path,
    build_identity,
    engine_config,
    git_sha,
    package_version,
    stockfish_path,
)
from mcp_server.engine.pool_factory import (
    _cache,
    _create_analyzer_pool,
    _eval_via_analyzer_or_pool,
    _evaluate_game_position_cached,
    _gather_evaluate_positions_bounded,
    _get_analyzer_pool,
    _get_evaluate_semaphore,
    _maybe_ponder_warm,
    _mcp_lifespan,
    _pool_stats_logger,
    _pool_supports_root_moves,
    _ponder_warm_cache,
    _single_flight,
    cache,
    close_analyzer_pool,
    single_flight,
)

__all__ = [
    "BuildIdentity",
    "DEFAULT_STOCKFISH_PATH",
    "build_identity",
    "cache",
    "close_analyzer_pool",
    "engine_config",
    "git_sha",
    "package_version",
    "single_flight",
    "stockfish_path",
    "_cache",
    "_create_analyzer_pool",
    "_eval_via_analyzer_or_pool",
    "_evaluate_game_position_cached",
    "_gather_evaluate_positions_bounded",
    "_get_analyzer_pool",
    "_get_evaluate_semaphore",
    "_maybe_ponder_warm",
    "_mcp_lifespan",
    "_pool_stats_logger",
    "_pool_supports_root_moves",
    "_ponder_warm_cache",
    "_single_flight",
]
