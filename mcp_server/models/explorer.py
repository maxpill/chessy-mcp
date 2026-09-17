"""Pydantic models for the Lichess Opening Explorer response.

These models deliberately mirror the official Lichess JSON contract so a schema
drift in the upstream API surfaces as a ``ValidationError`` (which the tool
layer maps to ``INVALID_EXPLORER_RESPONSE``). They expose the fields that
matter for coaching and opening research — per-move W/D/L counts, average
rating, sample games, and the ECO-classified opening name — without
re-publishing every upstream housekeeping field.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ExplorerDb = Literal["lichess", "masters", "player"]
ExplorerMode = Literal["casual", "rated"]
ExplorerSpeed = Literal[
    "ultraBullet",
    "bullet",
    "blitz",
    "rapid",
    "classical",
    "correspondence",
]
ExplorerColor = Literal["white", "black"]


class OpeningPlayer(BaseModel):
    """One side of an Explorer sample game."""

    name: str
    rating: int | None = None


class OpeningGameRef(BaseModel):
    """A reference game embedded in the Explorer response."""

    id: str
    white: OpeningPlayer | None = None
    black: OpeningPlayer | None = None
    winner: ExplorerColor | None = None
    year: str | None = None
    month: str | None = None
    speed: ExplorerSpeed | None = None
    mode: ExplorerMode | None = None
    uci: str | None = None


class ExplorerOpeningMeta(BaseModel):
    eco: str
    name: str


class OpeningMoveStats(BaseModel):
    """One legal reply with aggregated W/D/L outcomes."""

    uci: str
    san: str
    white: int = Field(ge=0)
    draws: int = Field(ge=0)
    black: int = Field(ge=0)
    average_rating: int | None = Field(default=None, alias="averageRating")
    average_opponent_rating: int | None = Field(default=None, alias="averageOpponentRating")
    performance: int | None = None
    game: OpeningGameRef | None = None
    opening: ExplorerOpeningMeta | None = None

    model_config = {"populate_by_name": True}

    @field_validator("uci", "san")
    @classmethod
    def _strip_str(cls, value: str) -> str:
        return value.strip()


class OpeningHistoryPoint(BaseModel):
    """Per-month popularity snapshot used in ``history``."""

    white: int = Field(ge=0)
    draws: int = Field(ge=0)
    black: int = Field(ge=0)

    @property
    def sum(self) -> int:
        return self.white + self.draws + self.black


class ExplorerFilters(BaseModel):
    """Filter set actually applied to the request (post-clamp)."""

    speeds: tuple[str, ...] = ()
    ratings: tuple[int, ...] = ()
    modes: tuple[str, ...] = ()
    since: str | None = None
    until: str | None = None
    player: str | None = None
    color: ExplorerColor | None = None
    top_games: int = Field(ge=0, alias="topGames")
    recent_games: int = Field(ge=0, alias="recentGames")
    moves: int = Field(ge=0, default=12)

    model_config = {"populate_by_name": True}


class OpeningData(BaseModel):
    """Top-level Lichess Explorer payload (the last NDJSON line / single body)."""

    fen: str | None = None
    white: int = Field(ge=0, default=0)
    draws: int = Field(ge=0, default=0)
    black: int = Field(ge=0, default=0)
    moves: list[OpeningMoveStats] = Field(default_factory=list)
    top_games: list[OpeningGameRef] = Field(default_factory=list, alias="topGames")
    recent_games: list[OpeningGameRef] = Field(default_factory=list, alias="recentGames")
    opening: ExplorerOpeningMeta | None = None
    history: list[OpeningHistoryPoint] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @property
    def total_games(self) -> int:
        return self.white + self.draws + self.black


class ExplorerRequestFilters(BaseModel):
    """Filter set the caller asked for (verbatim), echoed back in the response."""

    db: ExplorerDb
    variant: str
    fen: str
    play: tuple[str, ...]
    speeds: tuple[str, ...] | None = None
    ratings: tuple[int, ...] | None = None
    modes: tuple[str, ...] | None = None
    since: str | None = None
    until: str | None = None
    player: str | None = None
    color: ExplorerColor | None = None
    top_games: int | None = Field(default=None, alias="topGames")
    recent_games: int | None = Field(default=None, alias="recentGames")

    model_config = {"populate_by_name": True}


class LichessExplorerResult(BaseModel):
    """Wire format of the ``explore_opening`` MCP tool."""

    source: Literal["lichess-explorer"] = "lichess-explorer"
    db: ExplorerDb
    variant: str
    canonical_fen: str
    requested_filters: ExplorerRequestFilters
    applied_filters: ExplorerFilters
    opening: OpeningData
    cache_hit: bool
    cache_key: str
    fetched_at: float
    request_duration_ms: float
    diagnostics: dict[str, Any] = Field(default_factory=dict)
