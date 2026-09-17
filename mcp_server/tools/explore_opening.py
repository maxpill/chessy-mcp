"""``explore_opening`` MCP tool — Lichess Opening Explorer lookup.

Wraps ``https://explorer.lichess.ovh/{lichess|masters|player}`` and returns
opening statistics (W/D/L counts per legal reply, ECO-classified name, top +
recent sample games) for a given position + play prefix.

OAuth token is read from ``LICHESS_EXPLORER_TOKEN`` (configured via
``mcp_server.config.get_mcp_settings``). Endpoint override is
``LICHESS_EXPLORER_ENDPOINT`` (defaults to the public Lichess host).

Caching:
    L1 in-memory LRU + L2 SQLite WAL via :class:`LichessExplorerCache`,
    TTL 8 min, single-flight coalescing. Every filter value (speeds, ratings,
    modes, since/until, player, color, top_games, recent_games) is part of
    the cache key so different filter sets do not collide.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Annotated, Any

import chess
from pydantic import Field

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from mcp_server._mcp import mcp
from mcp_server.lichess_explorer import (
    LichessExplorerCache,
    LichessExplorerClient,
    explorer_cache_key,
)
from mcp_server.lichess_explorer.client import (
    LICHESS_DBS,
    LICHESS_VARIANTS,
    LichessExplorerAuthError,
    LichessExplorerNotFound,
    LichessExplorerRateLimited,
    LichessExplorerResponseError,
    LichessExplorerTimeout,
    LichessExplorerUnavailable,
    LichessExplorerUnreachable,
)
from mcp_server.metrics import metrics
from mcp_server.models import (
    ExplorerFilters,
    ExplorerRequestFilters,
    LichessExplorerResult,
    OpeningData,
)
from mcp_server.parsers import build_normalized_position
from mcp_server.tools._common import _tool_error

log = logging.getLogger("chessy_mcp.explore_opening")


_VALID_SPEEDS: frozenset[str] = frozenset(
    {"ultraBullet", "bullet", "blitz", "rapid", "classical", "correspondence"}
)
_VALID_RATINGS: frozenset[int] = frozenset({400, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500})
_VALID_MODES: frozenset[str] = frozenset({"casual", "rated"})
_VALID_COLORS: frozenset[str] = frozenset({"white", "black"})

_MONTH_RE = re.compile(r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])$")

_DEFAULT_LICHESS_SPEEDS: tuple[str, ...] = (
    "bullet",
    "blitz",
    "rapid",
    "classical",
    "correspondence",
)
_DEFAULT_LICHESS_RATINGS: tuple[int, ...] = (
    1000,
    1200,
    1400,
    1600,
    1800,
    2000,
    2200,
    2500,
)
_DEFAULT_MODES: tuple[str, ...] = ("casual", "rated")

_TOP_GAMES_DEFAULT = 5
_RECENT_GAMES_DEFAULT = 5
_TOP_GAMES_MAX = 20
_RECENT_GAMES_MAX = 20


_CLIENT_SINGLETON: LichessExplorerClient | None = None
_CACHE_SINGLETON: LichessExplorerCache | None = None


def _client() -> LichessExplorerClient:
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is None:
        _CLIENT_SINGLETON = LichessExplorerClient()
    return _CLIENT_SINGLETON


def _cache() -> LichessExplorerCache:
    global _CACHE_SINGLETON
    if _CACHE_SINGLETON is None:
        _CACHE_SINGLETON = LichessExplorerCache()
    return _CACHE_SINGLETON


def reset_singletons_for_tests(
    client: LichessExplorerClient | None = None,
    cache: LichessExplorerCache | None = None,
) -> None:
    """Test hook — replaces module-level singletons and resets global state."""
    global _CLIENT_SINGLETON, _CACHE_SINGLETON
    _CLIENT_SINGLETON = client
    _CACHE_SINGLETON = cache


def _validate_month(value: str | None, field_name: str) -> str | None:
    if value is None or value == "":
        return None
    if not _MONTH_RE.match(value):
        raise ValueError(
            f"INVALID_ARGUMENT: {field_name} must be YYYY-MM (e.g. 2024-01), got {value!r}"
        )
    return value


def _validate_play_uci(play: list[str], board: chess.Board) -> list[str]:
    """Push each UCI onto a stack-preserving copy; reject illegal moves.

    The caller-supplied ``play`` list is deduped via ``tuple(sorted(set(play)))``
    after validation so the cache key is stable across permutations and duplicates.
    """
    walker = board.copy(stack=True)
    out: list[str] = []
    for idx, raw in enumerate(play):
        move = raw.strip()
        if not move:
            raise ValueError(
                f"INVALID_MOVE_SYNTAX: play[{idx}] is empty; pass UCI moves like 'e2e4'."
            )
        try:
            uci_move = chess.Move.from_uci(move)
        except (ValueError, chess.InvalidMoveError, chess.AmbiguousMoveError) as exc:
            raise ValueError(
                f"INVALID_MOVE_SYNTAX: play[{idx}]={move!r} is not a valid UCI move: {exc}"
            ) from exc
        if uci_move not in walker.legal_moves:
            san_attempt = walker.san(uci_move) if uci_move in walker.pseudo_legal_moves else "?"
            raise ValueError(
                f"INVALID_MOVE_SYNTAX: play[{idx}]={move!r} is not legal at ply {idx} "
                f"(side_to_move={(walker.turn and 'white') or 'black'}); pseudo-legal SAN would be {san_attempt}."
            )
        walker.push(uci_move)
        out.append(uci_move.uci())
    return out


def _build_query_params(
    fen: str,
    play_uci: tuple[str, ...],
    variant: str,
    db: str,
    speeds: tuple[str, ...],
    ratings: tuple[int, ...],
    modes: tuple[str, ...],
    since: str | None,
    until: str | None,
    player: str | None,
    color: str | None,
    top_games: int,
    recent_games: int,
    moves: int,
) -> dict[str, str]:
    params: dict[str, str] = {
        "variant": variant,
        "fen": fen,
        "play": ",".join(play_uci),
        "source": "chessy_mcp",
        "moves": str(moves),
        "topGames": str(top_games),
        "recentGames": str(recent_games),
    }
    if db != "player":
        params["speeds"] = ",".join(speeds)
    if db == "lichess":
        params["ratings"] = ",".join(str(r) for r in ratings)
    if db == "masters":
        # masters DB does not take speeds/ratings; year filters are different.
        # We accept since/until as years when db == masters per lila config.
        if since:
            params["since"] = since.split("-", 1)[0]
        if until:
            params["until"] = until.split("-", 1)[0]
    else:
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if db == "lichess":
            params["modes"] = ",".join(modes)
    if db == "player":
        params["player"] = player or ""
        if color:
            params["color"] = color
    return params


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def explore_opening(
    fen: Annotated[
        str,
        Field(
            description=(
                "Root position FEN (with fullmove/halfmove counters OK to omit — "
                "they are completed automatically). Accepts 'startpos' as an alias "
                "for the standard initial position."
            )
        ),
    ],
    play: Annotated[
        list[str] | None,
        Field(
            description=(
                "Optional list of UCI moves applied on top of ``fen``. "
                "Each move must be legal at the resulting position; an illegal "
                "entry raises INVALID_MOVE_SYNTAX. Empty list is the root position."
            )
        ),
    ] = None,
    db: Annotated[
        str,
        Field(
            description=(
                "Explorer dataset: 'lichess' (all rated online games), 'masters' "
                "(OTB master games, supports year-based since/until), or 'player' "
                "(one player's online games — requires the ``player`` argument)."
            ),
            json_schema_extra={"enum": ["lichess", "masters", "player"]},
        ),
    ] = "lichess",
    variant: Annotated[
        str,
        Field(
            description=(
                "Chess variant. One of: standard, chess960, fromPosition, antichess, "
                "atomic, crazyhouse, horde, kingOfTheHill, racingKings, threeCheck."
            ),
            json_schema_extra={
                "enum": [
                    "standard",
                    "chess960",
                    "fromPosition",
                    "antichess",
                    "atomic",
                    "crazyhouse",
                    "horde",
                    "kingOfTheHill",
                    "racingKings",
                    "threeCheck",
                ]
            },
        ),
    ] = "standard",
    speeds: Annotated[
        list[str] | None,
        Field(
            description=(
                "Speed filters for ``db='lichess'``: any of ultraBullet, bullet, "
                "blitz, rapid, classical, correspondence. Ignored for ``db='masters'``."
            ),
            json_schema_extra={
                "items": {
                    "enum": [
                        "ultraBullet",
                        "bullet",
                        "blitz",
                        "rapid",
                        "classical",
                        "correspondence",
                    ]
                }
            },
        ),
    ] = None,
    ratings: Annotated[
        list[int] | None,
        Field(
            description=(
                "Rating bucket filters for ``db='lichess'``: any of "
                "400, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500."
            )
        ),
    ] = None,
    modes: Annotated[
        list[str] | None,
        Field(
            description=("Game-mode filters for ``db='lichess'``: 'casual' and/or 'rated'."),
            json_schema_extra={"items": {"enum": ["casual", "rated"]}},
        ),
    ] = None,
    since: Annotated[
        str | None,
        Field(description="Lower bound. YYYY-MM (e.g. '2017-01'). Empty disables."),
    ] = None,
    until: Annotated[
        str | None,
        Field(description="Upper bound. YYYY-MM (e.g. '2026-09'). Empty disables."),
    ] = None,
    player: Annotated[
        str | None,
        Field(
            description=(
                "Required when ``db='player'``. Lichess username (lowercase, no "
                "@). Lookups for non-existent accounts raise MISSING_PLAYER."
            )
        ),
    ] = None,
    color: Annotated[
        str | None,
        Field(
            description=(
                "Required when ``db='player'``. Restricts to games where the "
                "player had White ('white') or Black ('black')."
            ),
            json_schema_extra={"enum": ["white", "black"]},
        ),
    ] = None,
    top_games: Annotated[
        int,
        Field(
            description=(
                "Number of highest-rated sample games to embed (0..20). "
                "Default 5; pass 0 to suppress the heavy payload."
            )
        ),
    ] = _TOP_GAMES_DEFAULT,
    recent_games: Annotated[
        int,
        Field(
            description=(
                "Number of most-recent sample games to embed (0..20). "
                "Default 5; pass 0 to suppress."
            )
        ),
    ] = _RECENT_GAMES_DEFAULT,
    moves: Annotated[
        int,
        Field(description="How many replies to fetch in the move list (1..40). Default 12."),
    ] = 12,
) -> LichessExplorerResult:
    """Look up opening statistics for a position from the Lichess Opening Explorer.

    Returns per-reply win/draw/loss counts, average rating, sample games and the
    ECO-classified opening name (when Lichess has one). Cached for 8 min;
    concurrent identical calls coalesce into one HTTP roundtrip via
    ``SingleFlight``.

    Args:
        fen: Root position FEN (or 'startpos').
        play: UCI moves applied after ``fen`` (legal moves only).
        db: Dataset — 'lichess' (default), 'masters', or 'player'.
        variant: Chess variant. Defaults to 'standard'.
        speeds: Speed filter (lichess DB only). Defaults to bullet..classical.
        ratings: Rating-bucket filter (lichess DB only). Defaults to 1000..2500.
        modes: Game-mode filter (lichess DB only). Defaults to both.
        since: YYYY-MM lower bound on game month (lichess/player) or year (masters).
        until: YYYY-MM upper bound.
        player: Required when ``db='player'``.
        color: Required when ``db='player'`` (white or black).
        top_games: Highest-rated sample games to embed (0..20, default 5).
        recent_games: Most-recent sample games to embed (0..20, default 5).
        moves: Number of replies to include in the move list (1..40, default 12).

    Returns:
        ``LichessExplorerResult`` with cache key + duration + applied filters.
    """
    t0 = time.time()
    tool_name = "explore_opening"

    try:
        if db not in LICHESS_DBS:
            raise ValueError(
                f"INVALID_ARGUMENT: db must be one of {sorted(LICHESS_DBS)}, got {db!r}."
            )
        if variant not in LICHESS_VARIANTS:
            raise ValueError(
                f"INVALID_ARGUMENT: variant must be one of {sorted(LICHESS_VARIANTS)}, "
                f"got {variant!r}."
            )
        if db == "player" and not (player and player.strip()):
            raise ValueError(
                "MISSING_PLAYER: db='player' requires the `player` argument (Lichess username)."
            )

        since_clean = _validate_month(since, "since")
        until_clean = _validate_month(until, "until")
        if since_clean and until_clean and since_clean > until_clean:
            raise ValueError(
                f"INVALID_ARGUMENT: since ({since_clean!r}) must be <= until ({until_clean!r})."
            )

        # FEN parse via existing parser (raises INVALID_FEN on bad input).
        normalized = build_normalized_position(fen, [], strict=False)
        if normalized.canonical_fen is None:
            raise ValueError(f"INVALID_FEN: {fen!r} did not yield a canonical FEN.")
        root_board = chess.Board(normalized.canonical_fen)
        canonical_root = root_board.fen()
        input_fen = (
            " ".join(normalized.raw_fen_fields) if normalized.raw_fen_fields else canonical_root
        )

        play_list = list(play or [])
        # Reject empty/whitespace-only strings early — Lichess treats them as
        # "no play", but a caller that explicitly sends [\"\"] almost certainly
        # meant to send a UCI move and accidentally left the entry blank.
        for idx, raw in enumerate(play_list):
            if not raw or not raw.strip():
                raise ValueError(
                    f"INVALID_MOVE_SYNTAX: play[{idx}] is empty; pass UCI moves like 'e2e4'."
                )
        # Dedupe by FIRST occurrence so the validator walks the position in
        # the caller's intent order (alphabetical sorting would re-order
        # moves and silently make white's moves fail because they would be
        # applied on black's turns). Duplicates collapse to a no-op.
        seen: set[str] = set()
        play_in_order: list[str] = []
        for raw in play_list:
            token = raw.strip()
            if token not in seen:
                seen.add(token)
                play_in_order.append(token)
        play_uci = tuple(_validate_play_uci(play_in_order, root_board))
        # Stable cache key: canonical order = position after applying all
        # moves, which is invariant to caller order.
        play_key = tuple(sorted(play_uci))
        walker = root_board.copy(stack=True)
        for uci in play_uci:
            walker.push_uci(uci)
        canonical_fen = walker.fen()

        if db == "player" and color is None:
            raise ValueError(
                "MISSING_PLAYER: db='player' requires the `color` argument ('white' or 'black')."
            )
        if color is not None and color not in _VALID_COLORS:
            raise ValueError(f"INVALID_ARGUMENT: color must be 'white' or 'black', got {color!r}.")

        effective_speeds: tuple[str, ...]
        effective_ratings: tuple[int, ...]
        effective_modes: tuple[str, ...]

        if db == "lichess":
            speeds_in = list(speeds) if speeds else list(_DEFAULT_LICHESS_SPEEDS)
            ratings_in = list(ratings) if ratings else list(_DEFAULT_LICHESS_RATINGS)
            modes_in = list(modes) if modes else list(_DEFAULT_MODES)
            for s in speeds_in:
                if s not in _VALID_SPEEDS:
                    raise ValueError(
                        f"INVALID_ARGUMENT: speed {s!r} not in {sorted(_VALID_SPEEDS)}."
                    )
            for r in ratings_in:
                if r not in _VALID_RATINGS:
                    raise ValueError(
                        f"INVALID_ARGUMENT: rating {r} not in {sorted(_VALID_RATINGS)}."
                    )
            for m in modes_in:
                if m not in _VALID_MODES:
                    raise ValueError(f"INVALID_ARGUMENT: mode {m!r} not in {sorted(_VALID_MODES)}.")
            effective_speeds = tuple(sorted(set(speeds_in)))
            effective_ratings = tuple(sorted(set(ratings_in)))
            effective_modes = tuple(sorted(set(modes_in)))
        else:
            effective_speeds = ()
            effective_ratings = ()
            effective_modes = ()

        clamped_top = max(0, min(int(top_games), _TOP_GAMES_MAX))
        clamped_recent = max(0, min(int(recent_games), _RECENT_GAMES_MAX))
        clamped_moves = max(1, min(int(moves), 40))

        params = _build_query_params(
            fen=canonical_root,
            play_uci=play_uci,
            variant=variant,
            db=db,
            speeds=effective_speeds,
            ratings=effective_ratings,
            modes=effective_modes,
            since=since_clean,
            until=until_clean,
            player=(player.strip() if player else None),
            color=color,
            top_games=clamped_top,
            recent_games=clamped_recent,
            moves=clamped_moves,
        )

        applied_filters = ExplorerFilters(
            speeds=effective_speeds,
            ratings=effective_ratings,
            modes=effective_modes,
            since=since_clean,
            until=until_clean,
            player=(player.strip() if player else None),
            color=color,
            topGames=clamped_top,
            recentGames=clamped_recent,
            moves=clamped_moves,
        )
        requested_filters = ExplorerRequestFilters(
            db=db,
            variant=variant,
            fen=input_fen,
            play=play_uci,
            speeds=tuple(speeds) if speeds else None,
            ratings=tuple(ratings) if ratings else None,
            modes=tuple(modes) if modes else None,
            since=since_clean,
            until=until_clean,
            player=(player.strip() if player else None),
            color=color,
            topGames=int(top_games),
            recentGames=int(recent_games),
        )

        key = explorer_cache_key(
            db=db,
            variant=variant,
            fen=canonical_fen,
            play=play_key,
            speeds=frozenset(effective_speeds),
            ratings=frozenset(effective_ratings),
            modes=frozenset(effective_modes),
            since=since_clean or "",
            until=until_clean or "",
            player=(player.strip() if player else "") or "",
            color=color or "",
            top_games=clamped_top,
            recent_games=clamped_recent,
        )

        client = _client()
        cache = _cache()

        async def _fetch() -> dict[str, Any]:
            return await client.query(db, params)

        try:
            payload, cache_hit = await cache.get_or_fetch(key, _fetch)
        except LichessExplorerAuthError as exc:
            raise _tool_error(
                "missing_token" if "MISSING_TOKEN" in str(exc) else "lichess_auth_failed",
                str(exc),
                tool_name,
                db=db,
            ) from exc
        except LichessExplorerRateLimited as exc:
            raise _tool_error("lichess_rate_limited", str(exc), tool_name, db=db) from exc
        except LichessExplorerUnavailable as exc:
            raise _tool_error("explorer_unavailable", str(exc), tool_name, db=db) from exc
        except LichessExplorerTimeout as exc:
            raise _tool_error("explorer_timeout", str(exc), tool_name, db=db) from exc
        except LichessExplorerUnreachable as exc:
            raise _tool_error("explorer_unreachable", str(exc), tool_name, db=db) from exc
        except LichessExplorerNotFound as exc:
            raise _tool_error("missing_player", str(exc), tool_name, player=player) from exc
        except LichessExplorerResponseError as exc:
            raise _tool_error("invalid_explorer_response", str(exc), tool_name, db=db) from exc

        try:
            opening = OpeningData.model_validate(payload)
        except Exception as exc:
            raise _tool_error(
                "invalid_explorer_response",
                f"Explorer payload failed schema validation: {exc}",
                tool_name,
                db=db,
            ) from exc

        result = LichessExplorerResult(
            db=db,
            variant=variant,
            canonical_fen=canonical_fen,
            requested_filters=requested_filters,
            applied_filters=applied_filters,
            opening=opening,
            cache_hit=cache_hit,
            cache_key=key,
            fetched_at=time.time(),
            request_duration_ms=(time.time() - t0) * 1000.0,
        )
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, cache_hit=cache_hit)
        return result
    except ToolError:
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, is_error=True)
        msg = str(exc)
        code = "invalid_input"
        for marker, mapped in (
            ("INVALID_ARGUMENT", "invalid_argument"),
            ("INVALID_MOVE_SYNTAX", "invalid_move_syntax"),
            ("INVALID_FEN", "invalid_fen"),
            ("MISSING_PLAYER", "missing_player"),
        ):
            if marker in msg:
                code = mapped
                break
        raise _tool_error(code, msg, tool_name, db=db, fen=fen[:80]) from exc
    except Exception as exc:
        await metrics.record(tool_name, (time.time() - t0) * 1000.0, is_error=True)
        raise _tool_error("explorer_error", str(exc), tool_name) from exc
