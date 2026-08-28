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
    hash_mb: int = Field(default=256, validation_alias="STOCKFISH_HASH_MB")

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
