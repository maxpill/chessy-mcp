"""Engine layer — pool lifecycle, lifespan, cached evaluator, parallel gather, identity.

The split into focused modules (Phase 24):

  * :mod:`mcp_server.engine.identity` — version / git SHA / engine
    configuration snapshot via the :class:`BuildIdentity` dataclass,
    plus Stockfish binary path resolution.
  * :mod:`mcp_server.engine.pool_lifecycle` — pool creation, lifespan-
    aware lookup, capability probes (root_moves), and the analyzer-or-
    pool eval dispatcher.
  * :mod:`mcp_server.engine.lifespan` — FastMCP lifespan handler +
    pool-stats logger.
  * :mod:`mcp_server.engine.cached_evaluator` — single-position eval
    with rule status, terminal short-circuit, multi-tier cache, single-
    flight, U-02 zeroing post-state, P0 rule-aware best-move override,
    identity stamping, ponder warming.
  * :mod:`mcp_server.engine.zeroing_post_state` — U-02 pure helper.
  * :mod:`mcp_server.engine.parallel_gather` — N-position parallel
    gather with TT reuse (audit P1).
  * :mod:`mcp_server.engine.ponder` — background cache warmer.
  * :mod:`mcp_server.engine.eval_pipeline` — terminal / zeroing post
    helpers used by the cached evaluator.
"""

from mcp_server.engine.cached_evaluator import (
    _cache,
    _evaluate_game_position_cached,
    _get_evaluate_semaphore,
    _single_flight,
    cache,
    single_flight,
)
from mcp_server.engine.identity import (  # noqa: F401
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
    "DEFAULT_STOCKFISH_PATH",
    "BuildIdentity",
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
    "build_identity",
    "cache",
    "close_analyzer_pool",
    "engine_config",
    "git_sha",
    "package_version",
    "single_flight",
    "stockfish_path",
]
