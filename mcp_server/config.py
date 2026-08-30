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
        # No env_prefix - the existing names (`CHESS_MCP_POOL_SIZE`,
        # `STOCKFISH_HASH_MB`, `CHESS_MCP_TRANSPORT`, etc.) are the contract.
        case_sensitive=False,
    )

    # Engine transport - when both `host` and `port` are set, the MCP server
    # connects to a pre-existing Stockfish over TCP (e.g. behind socat) instead
    # of spawning its own subprocesses.
    pool_size: int | None = Field(default=None, validation_alias="CHESS_MCP_POOL_SIZE")
    host: str | None = Field(default=None, validation_alias="STOCKFISH_HOST")
    port: int = Field(default=0, validation_alias="STOCKFISH_PORT")
    hash_mb: int = Field(default=64, validation_alias="STOCKFISH_HASH_MB")
    # Threads per Stockfish worker. Default 2 pairs with pool_size=4 (4x2=8
    # logical cores = a good fit on the current production host).
    threads_per_worker: int = Field(default=2, validation_alias="STOCKFISH_THREADS_PER_WORKER")

    # Cap on concurrent `_evaluate_game_position_cached` calls across all
    # tools. The game analyzer fans out positions, so admission is bounded.
    max_concurrent_evaluates: int = Field(default=8, validation_alias="CHESS_MCP_MAX_CONCURRENT_EVALUATES")

    # Ponder (search opponent's likely reply during idle time). Disabled by
    # default because it consumes spare CPU.
    ponder_enabled: bool = Field(default=False, validation_alias="CHESS_MCP_PONDER_ENABLED")

    # Periodic pool-stats logging interval (seconds).
    pool_stats_interval_s: float = Field(default=30.0, validation_alias="CHESS_MCP_POOL_STATS_INTERVAL_S")

    # Syzygy tablebase path. Empty disables tablebases.
    syzygy_path: str = Field(default="", validation_alias="CHESS_MCP_SYZYGY_PATH")

    # UCI_ShowWDL: include White-POV Win/Draw/Loss data in responses.
    show_wdl: bool = Field(default=True, validation_alias="CHESS_MCP_SHOW_WDL")

    # Multi-tier cache (L1 in-memory + L2 SQLite WAL).
    cache_db: str = Field(default="/tmp/chess_mcp_eval_cache.sqlite3", validation_alias="CHESS_MCP_CACHE_DB")

    # Transport - stdio or streamable HTTP.
    transport: str = Field(default="stdio", validation_alias="CHESS_MCP_TRANSPORT")
    http_host: str = Field(default="0.0.0.0", validation_alias="CHESS_MCP_HOST")
    http_port: int = Field(default=8000, validation_alias="CHESS_MCP_PORT")

    # Shared secret for HTTP transport. Empty means no configured credential.
    auth_token: str = Field(default="", validation_alias="CHESS_MCP_AUTH_TOKEN")

    # When true, every non-health HTTP request must present the configured
    # token. User-Agent and Origin strings are never treated as credentials.
    lock_chatgpt: bool = Field(default=False, validation_alias="CHESS_MCP_LOCK_CHATGPT")


class BuildMetadata(BaseSettings):
    """Build-time metadata injected by CI or Docker."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    package_version: str = "dev"
    build_sha: str = ""


@lru_cache
def get_mcp_settings() -> MCPSettings:
    return MCPSettings()


@lru_cache
def get_build_metadata() -> BuildMetadata:
    return BuildMetadata()
