"""Typed settings for the chess MCP server.

Centralizes every env read so the server no longer scatters `os.environ.get(...)`
across lifespan, `_get_analyzer_pool`, and transport selection. Backed by
pydantic-settings so values are validated and documented in one place.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPSettings(BaseSettings):
    """Configuration for the chess MCP server."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        # No env_prefix — the existing names (`CHESS_MCP_POOL_SIZE`,
        # `STOCKFISH_HASH_MB`, `CHESS_MCP_TRANSPORT`, etc.) are the contract.
        case_sensitive=False,
    )

    # Engine transport — when both `host` and `port` are set, the MCP server
    # connects to a pre-existing Stockfish over TCP (e.g. behind socat) instead
    # of spawning its own subprocesses.
    pool_size: int | None = Field(default=None, validation_alias="CHESS_MCP_POOL_SIZE")
    host: str | None = Field(default=None, validation_alias="STOCKFISH_HOST")
    port: int = Field(default=0, validation_alias="STOCKFISH_PORT")
    hash_mb: int = Field(default=64, validation_alias="STOCKFISH_HASH_MB")
    # Threads per Stockfish worker. Default 2 pairs with pool_size=4 (4×2=8
    # logical cores = perfect fit on the OVH 4C/8T host). Raise only if the
    # host gains cores AND pool_size is dropped accordingly.
    threads_per_worker: int = Field(default=2, validation_alias="STOCKFISH_THREADS_PER_WORKER")

    # Cap on concurrent `_evaluate_game_position_cached` calls across all
    # tools. analyze_game fans out N positions via asyncio.gather; without
    # a cap, a single mega-burst (depth 30 over 80 plies) can starve every
    # other request for the full 15 s pool timeout. 8 = 2x pool size (4)
    # so a single request never self-throttles while bursts are bounded.
    max_concurrent_evaluates: int = Field(default=8, validation_alias="CHESS_MCP_MAX_CONCURRENT_EVALUATES")

    # Ponder (search opponent's likely reply during idle time). Disabled by
    # default — costs CPU on a small host. Set CHESS_MCP_PONDER_ENABLED=true
    # to enable on a beefier host.
    ponder_enabled: bool = Field(default=False, validation_alias="CHESS_MCP_PONDER_ENABLED")

    # Periodic pool-stats logging interval (seconds). Emits a structured
    # log line with queue depth, alive count, busy count.
    pool_stats_interval_s: float = Field(default=30.0, validation_alias="CHESS_MCP_POOL_STATS_INTERVAL_S")

    # Syzygy tablebase path (5-piece WDL/DTZ). Set CHESS_MCP_SYZYGY_PATH=/syzygy
    # to enable exact endgame lookups; empty disables.
    syzygy_path: str = Field(default="", validation_alias="CHESS_MCP_SYZYGY_PATH")

    # UCI_ShowWDL: include Win/Draw/Loss percentages in responses.
    # Default true (Stockfish 18 supports WDL natively, no engine cost).
    show_wdl: bool = Field(default=True, validation_alias="CHESS_MCP_SHOW_WDL")

    # Multi-tier cache (L1 in-memory + L2 SQLite WAL).
    cache_db: str = Field(default="/tmp/chess_mcp_eval_cache.sqlite3", validation_alias="CHESS_MCP_CACHE_DB")

    # Transport — stdio (ChatGPT) or streamable HTTP (web/agents).
    transport: str = Field(default="stdio", validation_alias="CHESS_MCP_TRANSPORT")
    http_host: str = Field(default="0.0.0.0", validation_alias="CHESS_MCP_HOST")
    http_port: int = Field(default=8000, validation_alias="CHESS_MCP_PORT")

    # Bearer token gating for HTTP transport. Empty = no auth.
    auth_token: str = Field(default="", validation_alias="CHESS_MCP_AUTH_TOKEN")

    # When true, lock the server down to ChatGPT's MCP client only (extra origin
    # check on the streamable HTTP transport).
    lock_chatgpt: bool = Field(default=False, validation_alias="CHESS_MCP_LOCK_CHATGPT")


class BuildMetadata(BaseSettings):
    """Build-time metadata injected by CI / docker. Read from top-level env vars."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    package_version: str = "dev"
    build_sha: str = ""


@lru_cache
def get_mcp_settings() -> MCPSettings:
    return MCPSettings()


@lru_cache
def get_build_metadata() -> BuildMetadata:
    return BuildMetadata()
