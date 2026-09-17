"""Comprehensive test suite for the ``explore_opening`` MCP tool.

Layers:
    6.1 HTTP layer (respx)
    6.2 Validation
    6.3 Cache + single-flight
    6.4 Model serialization
    6.5 MCP transport (in-process)
    6.6 Live smoke (gated by LICHESS_LIVE=1, marked ``live``)

Tests are independent — every test resets the module-level singletons via
``reset_singletons_for_tests``.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import respx

from mcp import ClientSession
from mcp.client._memory import create_client_server_memory_streams

from mcp_server import server as server_module
from mcp_server.lichess_explorer import (
    LichessExplorerCache,
    LichessExplorerClient,
    explorer_cache_key,
)
from mcp_server.lichess_explorer.client import (
    LICHESS_EXPLORER_TIMEOUT_S,
)
from mcp_server.models import LichessExplorerResult, OpeningData
from mcp_server.tools import explore_opening as explore_module
from mcp_server.tools.explore_opening import explore_opening

FIXTURES = Path(__file__).parent / "fixtures"

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
DEFAULT_ENDPOINT = "https://explorer.lichess.ovh"
TEST_TOKEN = "lip_test_token_12345"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _install_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path | None = None
) -> respx.MockRouter:
    """Reset module singletons, return a respx router bound to a fresh client.

    ``tmp_path`` is wired through ``LICHESS_EXPLORER_TOKEN``-less settings by
    monkeypatching ``settings.cache_db`` to a per-test file so the L2 SQLite
    cache never leaks across tests.
    """
    base = (
        tmp_path
        if tmp_path is not None
        else Path(f"/tmp/lichess_explorer_{os.getpid()}_{uuid.uuid4().hex}")
    )
    base.mkdir(parents=True, exist_ok=True)
    db_path = str(base.resolve() / "lichess_explorer_test.sqlite3")
    # Touch the file so SQLiteDiskCache's _migrate_legacy_cache skips the
    # copy-from-default step. Without this touch, the per-test DB inherits
    # every row from any prior test session that wrote to the default path.
    Path(db_path).touch()
    # Pydantic-settings gives ``validation_alias`` priority over explicit
    # kwargs, so we set the env vars directly and rebuild settings. This is
    # the only reliable way to point the cache module at a per-test DB.
    monkeypatch.setenv("CHESS_MCP_CACHE_DB", db_path)
    monkeypatch.setenv("LICHESS_EXPLORER_TOKEN", TEST_TOKEN)
    import mcp_server.config as config_mod
    import mcp_server.cache.disk as disk_mod
    import mcp_server.lichess_explorer.cache as explorer_cache_mod

    config_mod.get_mcp_settings.cache_clear()
    fresh = config_mod.get_mcp_settings()
    monkeypatch.setattr(config_mod, "get_mcp_settings", lambda: fresh)
    monkeypatch.setattr(disk_mod, "get_mcp_settings", lambda: fresh)
    monkeypatch.setattr(explorer_cache_mod, "get_mcp_settings", lambda: fresh)

    client = LichessExplorerClient(endpoint=DEFAULT_ENDPOINT, token=TEST_TOKEN)
    cache = LichessExplorerCache(l1_size=128, ttl_s=480)
    explore_module.reset_singletons_for_tests(client=client, cache=cache)
    router = respx.mock(assert_all_called=False)
    return router


# ---------------------------------------------------------------------------
# 6.1 HTTP layer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_lichess_db(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    import mcp_server.tools.explore_opening as em

    cache = em._CACHE_SINGLETON
    print("DEBUG: l2 db_path:", cache._l2.db_path)
    print("DEBUG: l1 size:", await cache._l1.size())
    # Try all plausible keys
    from mcp_server.lichess_explorer import explorer_cache_key

    for k in [
        explorer_cache_key(
            db="lichess",
            variant="standard",
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            play=("e2e4",),
            speeds=frozenset({"blitz", "bullet", "rapid", "classical", "correspondence"})
            if False
            else frozenset(),
            ratings=frozenset({1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500}),
            modes=frozenset({"casual", "rated"}),
            since="",
            until="",
            player="",
            color="",
            top_games=5,
            recent_games=5,
        ),
    ]:
        print("DEBUG: L2.get returned:", await cache._l2.get(k))
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(200, json=body)
        result = await explore_opening(fen="startpos", play=["e2e4"], db="lichess")
    print("DEBUG after: l1 size:", await cache._l1.size())
    print("DEBUG: result.cache_hit:", result.cache_hit, "call_count:", route.call_count)
    print("DEBUG: result.cache_key:", result.cache_key)
    assert isinstance(result, LichessExplorerResult)
    assert result.opening.total_games > 1_000_000
    assert len(result.opening.moves) == 5
    assert result.opening.moves[0].san == "e4"
    assert result.cache_hit is False
    assert route.call_count == 1
    assert result.request_duration_ms >= 0


@pytest.mark.asyncio
async def test_happy_path_masters_db(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_masters_e4e5.json")
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/masters").respond(200, json=body)
        result = await explore_opening(fen=STARTPOS, play=["e2e4", "e7e5"], db="masters")
    assert isinstance(result, LichessExplorerResult)
    assert result.opening.opening is not None
    assert result.opening.opening.eco == "C20"
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_happy_path_player_db(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_player_erigunal_e4.json")
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/player").respond(200, json=body)
        result = await explore_opening(
            fen="startpos", play=["e2e4"], db="player", player="erigunal", color="white"
        )
    assert result.opening.total_games == 4321 + 876 + 3987
    assert result.opening.top_games and result.opening.top_games[0].id == "ZxCvBnMa"
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_filters_propagated_to_query_string(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(200, json=body)
        await explore_opening(
            fen="startpos",
            play=["e2e4"],
            db="lichess",
            speeds=["blitz", "rapid"],
            ratings=[2000, 2200],
            modes=["rated"],
            since="2024-01",
            until="2025-12",
            top_games=3,
            recent_games=2,
        )
    assert route.call_count == 1
    url = route.calls.last.request.url
    params = url.params
    assert params["variant"] == "standard"
    assert params["fen"].startswith("rnbqkbnr/pppppppp")
    assert params["play"] == "e2e4"
    assert params["speeds"] == "blitz,rapid"
    assert params["ratings"] == "2000,2200"
    assert params["modes"] == "rated"
    assert params["since"] == "2024-01"
    assert params["until"] == "2025-12"
    assert params["topGames"] == "3"
    assert params["recentGames"] == "2"
    assert params["source"] == "chessy_mcp"


@pytest.mark.asyncio
async def test_bearer_header_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(200, json=body)
        await explore_opening(fen="startpos", play=["e2e4"])
    auth = route.calls.last.request.headers.get("authorization")
    assert auth == f"Bearer {TEST_TOKEN}"


@pytest.mark.asyncio
async def test_ndjson_streaming_uses_last_complete_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _install_client(monkeypatch)
    ndjson_text = (FIXTURES / "lichess_explorer_ndjson_streaming.json").read_text()
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(
            200,
            text=ndjson_text,
            headers={"content-type": "application/x-ndjson"},
        )
        result = await explore_opening(fen="startpos", play=["e2e4"])
    assert route.call_count == 1
    assert result.opening.total_games == 110 + 55 + 85
    assert result.opening.moves[0].average_rating == 1820


@pytest.mark.asyncio
async def test_401_raises_lichess_auth_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    from mcp_server.tools._common import tool_error as _te  # noqa: F401

    router = _install_client(monkeypatch)
    with router:
        router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(401, text="unauthorized")
        with pytest.raises(Exception) as excinfo:
            await explore_opening(fen="startpos", play=["e2e4"])
    msg = str(excinfo.value)
    assert "LICHESS_AUTH_FAILED" in msg.upper() or "lichess_auth_failed" in msg.lower()


@pytest.mark.asyncio
async def test_429_raises_lichess_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    with router:
        router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(
            429, text="slow down", headers={"Retry-After": "5"}
        )
        with pytest.raises(Exception) as excinfo:
            await explore_opening(fen="startpos", play=["e2e4"])
    msg = str(excinfo.value).lower()
    assert "lichess_rate_limited" in msg or "rate" in msg


@pytest.mark.asyncio
async def test_5xx_retries_then_raises_explorer_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _install_client(monkeypatch)
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(503, text="boom")
        with pytest.raises(Exception) as excinfo:
            await explore_opening(fen="startpos", play=["e2e4"])
    msg = str(excinfo.value).lower()
    assert "explorer_unavailable" in msg
    assert route.call_count == 2  # one retry


@pytest.mark.asyncio
async def test_timeout_raises_explorer_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    with router:
        router.get(f"{DEFAULT_ENDPOINT}/lichess").mock(
            side_effect=httpx.TimeoutException("timeout", request=None)
        )
        with pytest.raises(Exception) as excinfo:
            await explore_opening(fen="startpos", play=["e2e4"])
    msg = str(excinfo.value).lower()
    assert "explorer_timeout" in msg


@pytest.mark.asyncio
async def test_network_error_raises_explorer_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _install_client(monkeypatch)
    with router:
        router.get(f"{DEFAULT_ENDPOINT}/lichess").mock(
            side_effect=httpx.ConnectError("dns failed", request=None)
        )
        with pytest.raises(Exception) as excinfo:
            await explore_opening(fen="startpos", play=["e2e4"])
    msg = str(excinfo.value).lower()
    assert "explorer_unreachable" in msg


@pytest.mark.asyncio
async def test_malformed_json_raises_invalid_explorer_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _install_client(monkeypatch)
    with router:
        router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(200, text="{not-json}")
        with pytest.raises(Exception) as excinfo:
            await explore_opening(fen="startpos", play=["e2e4"])
    msg = str(excinfo.value).lower()
    assert "invalid_explorer_response" in msg or "explorer_response" in msg


# ---------------------------------------------------------------------------
# 6.2 Validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_token_raises_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LichessExplorerClient(endpoint=DEFAULT_ENDPOINT, token="")
    cache = LichessExplorerCache(l1_size=16, ttl_s=60)
    explore_module.reset_singletons_for_tests(client=client, cache=cache)
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="startpos", play=["e2e4"])
    assert "missing_token" in str(excinfo.value).lower() or "MISSING_TOKEN" in str(excinfo.value)


@pytest.mark.asyncio
async def test_invalid_fen_raises_invalid_fen(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="not-a-fen", play=["e2e4"])
    assert "invalid_fen" in str(excinfo.value).lower() or "INVALID_FEN" in str(excinfo.value)


@pytest.mark.asyncio
async def test_illegal_first_move_raises_invalid_move_syntax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_client(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="startpos", play=["e2e5"])
    msg = str(excinfo.value)
    assert "INVALID_MOVE_SYNTAX" in msg.upper() or "invalid_move_syntax" in msg.lower()


@pytest.mark.asyncio
async def test_illegal_intermediate_move_raises_invalid_move_syntax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_client(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        # After 1.e4 e5 it is WHITE to move. Trying to play BLACK's knight
        # (b8c6) on white's turn is illegal — the side-to-move check fails.
        await explore_opening(fen="startpos", play=["e2e4", "e7e5", "b8c6"])
    msg = str(excinfo.value)
    assert (
        "INVALID_MOVE_SYNTAX" in msg.upper()
        or "invalid_move_syntax" in msg.lower()
        or "invalid_input" in msg.lower()
    )


@pytest.mark.asyncio
async def test_bad_db_raises_invalid_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="startpos", play=["e2e4"], db="bogus")  # type: ignore[arg-type]
    assert "invalid_argument" in str(excinfo.value).lower() or "INVALID_ARGUMENT" in str(
        excinfo.value
    )


@pytest.mark.asyncio
async def test_bad_variant_raises_invalid_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="startpos", play=["e2e4"], variant="standard999")
    assert "invalid_argument" in str(excinfo.value).lower() or "INVALID_ARGUMENT" in str(
        excinfo.value
    )


@pytest.mark.asyncio
async def test_player_db_requires_player_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="startpos", play=["e2e4"], db="player")
    assert "missing_player" in str(excinfo.value).lower() or "MISSING_PLAYER" in str(excinfo.value)


@pytest.mark.asyncio
async def test_player_db_requires_color_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="startpos", play=["e2e4"], db="player", player="erigunal")
    assert "missing_player" in str(excinfo.value).lower() or "MISSING_PLAYER" in str(excinfo.value)


@pytest.mark.asyncio
async def test_bad_since_format_raises_invalid_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_client(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="startpos", play=["e2e4"], since="2025/01")
    assert "invalid_argument" in str(excinfo.value).lower() or "INVALID_ARGUMENT" in str(
        excinfo.value
    )


@pytest.mark.asyncio
async def test_since_after_until_raises_invalid_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_client(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="startpos", play=["e2e4"], since="2025-06", until="2025-01")
    assert "invalid_argument" in str(excinfo.value).lower() or "INVALID_ARGUMENT" in str(
        excinfo.value
    )


@pytest.mark.asyncio
async def test_bad_rating_value_raises_invalid_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_client(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="startpos", play=["e2e4"], ratings=[1500])
    assert "invalid_argument" in str(excinfo.value).lower() or "INVALID_ARGUMENT" in str(
        excinfo.value
    )


@pytest.mark.asyncio
async def test_bad_speed_value_raises_invalid_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_client(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="startpos", play=["e2e4"], speeds=["atomic_blitz"])
    assert "invalid_argument" in str(excinfo.value).lower() or "INVALID_ARGUMENT" in str(
        excinfo.value
    )


@pytest.mark.asyncio
async def test_bad_mode_value_raises_invalid_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_client(monkeypatch)
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="startpos", play=["e2e4"], modes=["blitz"])
    assert "invalid_argument" in str(excinfo.value).lower() or "INVALID_ARGUMENT" in str(
        excinfo.value
    )


@pytest.mark.asyncio
async def test_play_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate UCI moves in ``play`` collapse to a single canonical key."""
    router = _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(200, json=body)
        result = await explore_opening(fen="startpos", play=["e2e4", "e2e4"])
    assert result.requested_filters.play == ("e2e4",)
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_empty_play_uses_root_position(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(200, json=body)
        result = await explore_opening(fen="startpos")
    assert result.requested_filters.play == ()
    assert route.call_count == 1
    params = route.calls.last.request.url.params
    assert params["play"] == ""


@pytest.mark.asyncio
async def test_top_games_clamped_to_max(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(200, json=body)
        result = await explore_opening(
            fen="startpos", play=["e2e4"], top_games=999, recent_games=-5
        )
    assert result.applied_filters.top_games == 20
    assert result.applied_filters.recent_games == 0
    params = route.calls.last.request.url.params
    assert params["topGames"] == "20"
    assert params["recentGames"] == "0"


# ---------------------------------------------------------------------------
# 6.3 Cache + single-flight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_on_second_call(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(200, json=body)
        a = await explore_opening(fen="startpos", play=["e2e4"])
        b = await explore_opening(fen="startpos", play=["e2e4"])
    assert a.cache_hit is False
    assert b.cache_hit is True
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_cache_miss_on_different_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(200, json=body)
        await explore_opening(fen="startpos", play=["e2e4"], ratings=[2000])
        await explore_opening(fen="startpos", play=["e2e4"], ratings=[2200])
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_cache_miss_on_different_fen(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(200, json=body)
        await explore_opening(fen="startpos", play=["e2e4"])
        await explore_opening(fen="startpos", play=["d2d4"])
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_cache_miss_on_different_play(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    with router:
        route = router.get(f"{DEFAULT_ENDPOINT}/lichess").respond(200, json=body)
        await explore_opening(fen="startpos", play=["e2e4"])
        await explore_opening(fen="startpos", play=["e2e4", "e7e5", "g1f3"])
    assert route.call_count == 2


@pytest.mark.asyncio
async def test_singleflight_coalesces_concurrent(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch)
    body = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    fetch_calls = 0

    async def wrapped_query(db, params):
        nonlocal fetch_calls
        fetch_calls += 1
        await asyncio.sleep(0.03)
        return body

    explore_module._CLIENT_SINGLETON.query = wrapped_query  # type: ignore[assignment]

    try:
        results = await asyncio.gather(
            *(explore_opening(fen="startpos", play=["e2e4"]) for _ in range(5))
        )
    finally:
        explore_module._CLIENT_SINGLETON = None  # reset for subsequent tests
        explore_module.reset_singletons_for_tests()
    assert all(isinstance(r, LichessExplorerResult) for r in results)
    assert fetch_calls == 1


def test_cache_key_deterministic_and_filter_sensitive() -> None:
    k1 = explorer_cache_key(
        db="lichess",
        variant="standard",
        fen=STARTPOS,
        play=("e2e4",),
        speeds=frozenset({"blitz"}),
        ratings=frozenset({2000}),
        modes=frozenset({"rated"}),
        since="",
        until="",
        player="",
        color="",
        top_games=5,
        recent_games=5,
    )
    k2 = explorer_cache_key(
        db="lichess",
        variant="standard",
        fen=STARTPOS,
        play=("e2e4",),
        speeds=frozenset({"blitz"}),
        ratings=frozenset({2000}),
        modes=frozenset({"rated"}),
        since="",
        until="",
        player="",
        color="",
        top_games=5,
        recent_games=5,
    )
    assert k1 == k2
    k_other = explorer_cache_key(
        db="lichess",
        variant="standard",
        fen=STARTPOS,
        play=("e2e4",),
        speeds=frozenset({"blitz"}),
        ratings=frozenset({2200}),  # different
        modes=frozenset({"rated"}),
        since="",
        until="",
        player="",
        color="",
        top_games=5,
        recent_games=5,
    )
    assert k_other != k1


# ---------------------------------------------------------------------------
# 6.4 Model
# ---------------------------------------------------------------------------


def test_opening_data_pydantic_validation() -> None:
    payload = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    od = OpeningData.model_validate(payload)
    assert od.total_games > 0
    assert od.moves[0].san == "e4"
    assert od.moves[0].uci == "e2e4"

    bad = dict(payload)
    bad["moves"] = [{"uci": "x", "san": "x", "white": -1, "draws": 0, "black": 0}]
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OpeningData.model_validate(bad)


def test_result_round_trips_through_model_dump() -> None:
    fixture = _load_fixture("lichess_explorer_lichess_startpos_e2e4.json")
    od = OpeningData.model_validate(fixture)
    sample = {
        "source": "lichess-explorer",
        "db": "lichess",
        "variant": "standard",
        "canonical_fen": STARTPOS,
        "requested_filters": {
            "db": "lichess",
            "variant": "standard",
            "fen": STARTPOS,
            "play": ["e2e4"],
        },
        "applied_filters": {
            "speeds": ["blitz"],
            "ratings": [2000],
            "modes": ["rated"],
            "topGames": 5,
            "recentGames": 5,
            "moves": 12,
        },
        "opening": od.model_dump(by_alias=True),
        "cache_hit": False,
        "cache_key": "lichess_explorer:v1:test",
        "fetched_at": 1.0,
        "request_duration_ms": 0.5,
    }
    result = LichessExplorerResult.model_validate(sample)
    assert result.cache_hit is False
    assert result.cache_key.startswith("lichess_explorer:")
    json.dumps(result.model_dump(by_alias=True))


# ---------------------------------------------------------------------------
# 6.5 MCP transport
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _mcp_session() -> AsyncIterator[ClientSession]:
    async with create_client_server_memory_streams() as ((c_read, c_write), (s_read, s_write)):
        server_task = asyncio.create_task(
            server_module.mcp._lowlevel_server.run(
                s_read, s_write, server_module.mcp._lowlevel_server.create_initialization_options()
            )
        )
        try:
            async with ClientSession(c_read, c_write) as session:
                await session.initialize()
                yield session
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_explore_opening_in_tool_list() -> None:
    async with _mcp_session() as session:
        tools = await session.list_tools()
        assert "explore_opening" in {t.name for t in tools.tools}


@pytest.mark.asyncio
async def test_explore_opening_annotations_read_only_and_idempotent() -> None:
    async with _mcp_session() as session:
        tools = await session.list_tools()
        tool = next(t for t in tools.tools if t.name == "explore_opening")
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.idempotent_hint is True


@pytest.mark.asyncio
async def test_explore_opening_input_schema_enums() -> None:
    async with _mcp_session() as session:
        tools = await session.list_tools()
        schema = next(t for t in tools.tools if t.name == "explore_opening").input_schema
        db_prop = schema["properties"]["db"]
        db_enum = db_prop.get("enum")
        if db_enum is None:
            db_enum = next(
                (item["enum"] for item in db_prop.get("anyOf", []) if "enum" in item),
                None,
            )
        assert db_enum == ["lichess", "masters", "player"]
        variant_prop = schema["properties"]["variant"]
        variant_enum = variant_prop.get("enum")
        if variant_enum is None:
            variant_enum = next(
                (item["enum"] for item in variant_prop.get("anyOf", []) if "enum" in item),
                None,
            )
        assert variant_enum is not None
        assert "standard" in variant_enum
        assert "fromPosition" in variant_enum
        mode_items = schema["properties"]["modes"]["anyOf"][0]
        mode_enum = mode_items["items"]["enum"]
        assert mode_enum == ["casual", "rated"]


@pytest.mark.asyncio
async def test_call_over_transport_propagates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_client(monkeypatch)
    async with _mcp_session() as session:
        result = await session.call_tool("explore_opening", {"fen": "not-a-fen", "play": ["e2e4"]})
        assert result.is_error is True
        assert "INVALID_FEN" in result.content[0].text or "invalid_fen" in result.content[0].text


# ---------------------------------------------------------------------------
# 6.6 Live smoke (gated)
# ---------------------------------------------------------------------------


LIVE = pytest.mark.skipif(
    os.environ.get("LICHESS_LIVE", "").lower() not in {"1", "true", "yes"},
    reason="Set LICHESS_LIVE=1 to hit the real Lichess explorer.",
)


@pytest.mark.asyncio
@LIVE
async def test_lichess_explorer_live_smoke() -> None:
    explore_module.reset_singletons_for_tests()  # use real config + singletons
    result = await explore_opening(
        fen="startpos",
        play=["e2e4"],
        speeds=["blitz", "rapid"],
        ratings=[2000, 2200],
        since="2024-01",
    )
    assert isinstance(result, LichessExplorerResult)
    assert result.opening.total_games > 1000
    assert any(m.uci == "e2e4" for m in result.opening.moves) is False  # we asked after e4
    assert any(m.uci for m in result.opening.moves)
    assert result.request_duration_ms >= 0


@pytest.mark.asyncio
@LIVE
async def test_lichess_explorer_live_smoke_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LICHESS_EXPLORER_TOKEN", "")
    explore_module.reset_singletons_for_tests()
    with pytest.raises(Exception) as excinfo:
        await explore_opening(fen="startpos", play=["e2e4"])
    assert "missing_token" in str(excinfo.value).lower()


def test_timeout_constant_matches_lila() -> None:
    assert LICHESS_EXPLORER_TIMEOUT_S == 4.0
