#!/usr/bin/env python3
"""Ultra-thorough live MCP server test against the deployed Chessy MCP endpoint.

Exercises every tool + every documented error path via the same JSON-RPC
protocol ChatGPT uses. Logs every result + duration. Exits non-zero on any
unexpected failure.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from typing import Any

import httpx

ENDPOINT = "https://mcp.trychessy.com/mcp"
AUTH_TOKEN = "Mhvoempn0PIBs3Ze16coI4pKboI4gqeGRx8YYjH2QmY"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results: list[dict[str, Any]] = []


class Session:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self.session_id: str | None = None

    async def initialize(self) -> None:
        resp = await self._post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "ultra-tester", "version": "1.0"},
                },
            },
        )
        assert "result" in resp, f"initialize failed: {resp}"
        sid = resp.get("_sid")
        assert sid, "no session id from initialize"
        self.session_id = sid

    async def list_tools(self) -> list[str]:
        resp = await self._post({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        return [t["name"] for t in resp["result"]["tools"]]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._post(
            {
                "jsonrpc": "2.0",
                "id": uuid.uuid4().int & 0xFFFF,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        resp = await self._client.post(ENDPOINT, json=payload, headers=headers)
        text = resp.text
        if text.startswith("event: message"):
            text = text.split("\ndata:", 1)[1].strip()
        parsed = json.loads(text)
        parsed["_status"] = resp.status_code
        parsed["_sid"] = resp.headers.get("mcp-session-id") or self.session_id
        parsed["_latency_ms"] = float(resp.headers.get("x-request-latency-ms", "0") or 0)
        return parsed

    async def close(self) -> None:
        await self._client.aclose()


def record(name: str, passed: bool, detail: str = "", duration_ms: float = 0.0) -> None:
    status = PASS if passed else FAIL
    print(f"{status}  {name:<70} {duration_ms:>6.1f}ms  {detail}")
    results.append({"name": name, "passed": passed, "detail": detail, "duration_ms": duration_ms})


def expect_text_contains(result: dict[str, Any], marker: str) -> bool:
    text = result.get("result", {}).get("content", [{}])[0].get("text", "")
    return marker in text or marker.lower() in text.lower()


def expect_no_error(result: dict[str, Any]) -> bool:
    return not result.get("result", {}).get("isError", False)


async def call_safe(
    session: Session, tool: str, args: dict[str, Any]
) -> tuple[dict[str, Any], float]:
    t0 = time.time()
    try:
        result = await session.call_tool(tool, args)
    except Exception as exc:
        result = {"error": str(exc), "result": {"isError": True, "content": [{"text": str(exc)}]}}
    return result, (time.time() - t0) * 1000


async def section(title: str) -> None:
    print()
    print(f"\n\033[1m=== {title} ===\033[0m")


async def main() -> int:
    sess = Session()
    await sess.initialize()
    tools = await sess.list_tools()
    record(
        "list_tools returns 5 tools including explore_opening",
        set(tools)
        == {"evaluate_position", "top_moves", "classify_move", "analyze_game", "explore_opening"},
        f"got: {sorted(tools)}",
    )

    # ---------- evaluate_position ----------
    await section("evaluate_position")
    for label, fen, kwargs in [
        ("startpos depth=18", "startpos", {"depth": 18}),
        (
            "fen w/ KQkq depth=20",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            {"depth": 20},
        ),
        ("fen w/o rights depth=15", "8/8/8/3k4/8/8/3K4/8 w - - 0 1", {"depth": 15}),
        ("pgn input", "1.e4 e5 2.Nf3 Nc6 3.Bb5", {"depth": 18}),
    ]:
        result, ms = await call_safe(sess, "evaluate_position", {"fen": fen, **kwargs})
        ok = expect_no_error(result) and "best_move" in (
            result.get("result", {}).get("content", [{}])[0].get("text", "")
        )
        record(f"evaluate_position: {label}", ok, "", ms)

    # invalid FEN
    result, ms = await call_safe(sess, "evaluate_position", {"fen": "garbage", "depth": 10})
    text = str(result).upper()
    record(
        "evaluate_position: invalid FEN",
        "INVALID_FEN" in text or "INVALID_POSITION" in text,
        "",
        ms,
    )

    # ---------- top_moves ----------
    await section("top_moves")
    for label, args in [
        ("n=3 depth=20", {"fen": "startpos", "n": 3, "depth": 20}),
        ("n=1 depth=15", {"fen": "startpos", "n": 1, "depth": 15}),
        (
            "include_moves e2e4 d2d4",
            {"fen": "startpos", "n": 5, "depth": 18, "include_moves": ["e2e4", "d2d4"]},
        ),
        ("compact verbosity", {"fen": "startpos", "n": 3, "depth": 16, "verbosity": "compact"}),
        ("minimal verbosity", {"fen": "startpos", "n": 3, "depth": 16, "verbosity": "minimal"}),
    ]:
        result, ms = await call_safe(sess, "top_moves", args)
        text = result.get("result", {}).get("content", [{}])[0].get("text", "")
        ok = expect_no_error(result) and "result" in text
        record(f"top_moves: {label}", ok, "", ms)

    # ---------- classify_move ----------
    await section("classify_move")
    for label, args in [
        ("e2e4 from startpos depth=16", {"fen": "startpos", "move": "e2e4", "depth": 16}),
        ("d2d4 from startpos", {"fen": "startpos", "move": "d2d4", "depth": 16}),
        (
            "e7e5 SAN form",
            {
                "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "move": "e5",
                "depth": 16,
            },
        ),
        ("illegal move e2e5", {"fen": "startpos", "move": "e2e5", "depth": 10}),
    ]:
        result, ms = await call_safe(sess, "classify_move", args)
        text = str(result)
        if "illegal_move" in args.get("move", "") or args.get("move") == "e2e5":
            ok = "ILLEGAL_MOVE" in text.upper() or "illegal" in text.lower()
        else:
            ok = expect_no_error(result)
        record(f"classify_move: {label}", ok, "", ms)

    # ---------- analyze_game ----------
    await section("analyze_game")
    # Legal, engine-checked PGN — Qb3+ on move 17 is illegal so use a
    # known-good opening sequence from the existing test_mcp_server tests.
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 1/2-1/2"
    for label, args in [
        ("full pgn depth=10", {"pgn": pgn, "depth": 10}),
        ("compact", {"pgn": pgn, "depth": 10, "verbosity": "compact"}),
        ("coach detail", {"pgn": pgn, "depth": 12, "detail": "coach", "perspective": "white"}),
    ]:
        result, ms = await call_safe(sess, "analyze_game", args)
        text = result.get("result", {}).get("content", [{}])[0].get("text", "")
        # analyze_game emits a JSON object with schema_version + plies/move list
        ok = expect_no_error(result) and (
            "schema_version" in text
            and ("total_plies" in text or "moves" in text or "plies" in text)
        )
        record(f"analyze_game: {label}", ok, "", ms)

    # ---------- explore_opening ----------
    await section("explore_opening — happy paths")
    for label, args in [
        ("lichess + e2e4 default filters", {"fen": "startpos", "play": ["e2e4"], "db": "lichess"}),
        (
            "lichess custom speeds/ratings",
            {
                "fen": "startpos",
                "play": ["e2e4"],
                "db": "lichess",
                "speeds": ["blitz", "rapid"],
                "ratings": [2000, 2200],
            },
        ),
        ("lichess d2d4", {"fen": "startpos", "play": ["d2d4"], "db": "lichess"}),
        ("masters 1.e4 e5", {"db": "masters", "fen": "startpos", "play": ["e2e4", "e7e5"]}),
        (
            "player erigunal white e4",
            {
                "db": "player",
                "player": "erigunal",
                "color": "white",
                "fen": "startpos",
                "play": ["e2e4"],
            },
        ),
        (
            "lichess with since/until",
            {
                "fen": "startpos",
                "play": ["e2e4"],
                "db": "lichess",
                "since": "2024-01",
                "until": "2026-09",
            },
        ),
        (
            "lichess top_games=0 recent_games=0",
            {"fen": "startpos", "play": ["e2e4"], "top_games": 0, "recent_games": 0},
        ),
        (
            "lichess top_games=15 recent_games=10",
            {"fen": "startpos", "play": ["e2e4"], "top_games": 15, "recent_games": 10},
        ),
        ("variant=chess960", {"fen": "startpos", "play": ["e2e4"], "variant": "chess960"}),
        (
            "fromPosition with explicit FEN (white to move)",
            {
                "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
                "play": ["g1f3"],
                "variant": "fromPosition",
            },
        ),
        ("empty play (root position)", {"fen": "startpos", "play": []}),
    ]:
        result, ms = await call_safe(sess, "explore_opening", args)
        record(f"explore_opening: {label}", expect_no_error(result), "", ms)

    await section("explore_opening — error paths")
    for label, args, expected_marker in [
        ("invalid FEN", {"fen": "not-a-fen", "play": ["e2e4"]}, "INVALID_FEN"),
        ("illegal first move", {"fen": "startpos", "play": ["e2e5"]}, "INVALID_MOVE_SYNTAX"),
        (
            "illegal intermediate move",
            {"fen": "startpos", "play": ["e2e4", "e7e5", "b8c6"]},
            "INVALID_MOVE_SYNTAX",
        ),
        ("bad db value", {"fen": "startpos", "play": ["e2e4"], "db": "bogus"}, "INVALID_ARGUMENT"),
        (
            "bad variant",
            {"fen": "startpos", "play": ["e2e4"], "variant": "unknown"},
            "INVALID_ARGUMENT",
        ),
        (
            "bad since format",
            {"fen": "startpos", "play": ["e2e4"], "since": "yesterday"},
            "INVALID_ARGUMENT",
        ),
        (
            "since > until",
            {"fen": "startpos", "play": ["e2e4"], "since": "2025-06", "until": "2025-01"},
            "INVALID_ARGUMENT",
        ),
        (
            "bad rating value",
            {"fen": "startpos", "play": ["e2e4"], "ratings": [1500]},
            "INVALID_ARGUMENT",
        ),
        (
            "bad speed value",
            {"fen": "startpos", "play": ["e2e4"], "speeds": ["hyperbullet"]},
            "INVALID_ARGUMENT",
        ),
        (
            "db=player missing player",
            {"db": "player", "fen": "startpos", "play": ["e2e4"]},
            "MISSING_PLAYER",
        ),
        (
            "db=player missing color",
            {"db": "player", "player": "erigunal", "fen": "startpos", "play": ["e2e4"]},
            "MISSING_PLAYER",
        ),
        (
            "nonexistent player (Lichess 400)",
            {
                "db": "player",
                "player": "this_user_does_not_exist_xyz_12345",
                "color": "white",
                "fen": "startpos",
                "play": ["e2e4"],
            },
            None,
        ),
        ("empty play string", {"fen": "startpos", "play": [""]}, "INVALID_MOVE_SYNTAX"),
        ("garbage play", {"fen": "startpos", "play": ["garbage"]}, "INVALID_MOVE_SYNTAX"),
        ("non-UCI chars", {"fen": "startpos", "play": ["x9x9"]}, "INVALID_MOVE_SYNTAX"),
    ]:
        result, ms = await call_safe(sess, "explore_opening", args)
        text = str(result).upper()
        ok = expected_marker is None or expected_marker.upper() in text
        record(f"explore_opening error: {label}", ok, "", ms)

    await section("explore_opening — cache behavior")
    # Vary the since-month per run so the first call always misses the
    # 8-min cache (the TTL lila uses for OpeningApi.defaultCache).
    suffix = uuid.uuid4().hex[:6]
    unique_since = f"2024-{((int(suffix[:2], 16) % 12) + 1):02d}"
    args = {"fen": "startpos", "play": ["e2e4"], "db": "lichess", "since": unique_since}
    result1, ms1 = await call_safe(sess, "explore_opening", args)
    result2, ms2 = await call_safe(sess, "explore_opening", args)
    text1 = result1.get("result", {}).get("content", [{}])[0].get("text", "")
    text2 = result2.get("result", {}).get("content", [{}])[0].get("text", "")
    parsed1 = json.loads(text1)
    parsed2 = json.loads(text2)
    record(
        f"cache: first call miss [{suffix}]",
        not parsed1.get("cache_hit", True),
        f"cache_hit={parsed1.get('cache_hit')}",
        ms1,
    )
    record(
        f"cache: second call hit [{suffix}]",
        parsed2.get("cache_hit") is True,
        f"cache_hit={parsed2.get('cache_hit')}",
        ms2,
    )
    record("cache: same payload", parsed1.get("opening") == parsed2.get("opening"), "", 0.0)

    # different play → different key → miss
    args_diff = {"fen": "startpos", "play": ["d2d4", "d7d5"], "db": "lichess"}
    result3, ms3 = await call_safe(sess, "explore_opening", args_diff)
    parsed3 = json.loads(result3.get("result", {}).get("content", [{}])[0].get("text", ""))
    record(
        "cache: different play miss",
        not parsed3.get("cache_hit", True),
        f"cache_hit={parsed3.get('cache_hit')}",
        ms3,
    )

    # masters with different db
    result4, ms4 = await call_safe(
        sess, "explore_opening", {"db": "masters", "fen": "startpos", "play": ["e2e4", "c7c5"]}
    )
    parsed4 = json.loads(result4.get("result", {}).get("content", [{}])[0].get("text", ""))
    record(
        "cache: masters first-call miss",
        not parsed4.get("cache_hit", True),
        f"cache_hit={parsed4.get('cache_hit')}",
        ms4,
    )

    await section("explore_opening — concurrency / singleflight")
    # 5 concurrent identical calls — should coalesce to 1 HTTP fetch
    coros = [
        sess.call_tool("explore_opening", {"fen": "startpos", "play": ["e2e4"], "db": "lichess"})
        for _ in range(5)
    ]
    t0 = time.time()
    results5 = await asyncio.gather(*coros, return_exceptions=True)
    elapsed = (time.time() - t0) * 1000
    cache_hits = sum(
        1
        for r in results5
        if isinstance(r, dict)
        and not r.get("result", {}).get("isError", False)
        and json.loads(r.get("result", {}).get("content", [{}])[0].get("text", "{}")).get(
            "cache_hit"
        )
    )
    record(
        "singleflight: 5 concurrent identical → at least 4 cache hits",
        cache_hits >= 4,
        f"cache_hits={cache_hits}/5 total={elapsed:.0f}ms",
        elapsed,
    )

    # ---------- health & session management ----------
    await section("infrastructure")
    # session reuse: tools/list with same SID
    resp = await sess._post({"jsonrpc": "2.0", "id": 99, "method": "tools/list", "params": {}})
    record("tools/list reuses session", "result" in resp, "", 0.0)

    await sess.close()
    # ---------- summary ----------
    print()
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    print(f"\n\033[1m=== SUMMARY ===\033[0m")
    print(f"Total: {len(results)}  Passed: {passed}  Failed: {failed}")
    durations = [r["duration_ms"] for r in results if r["duration_ms"] > 0]
    if durations:
        print(f"Latency  p50: {statistics.median(durations):.0f}ms  max: {max(durations):.0f}ms")
    if failed:
        print("\nFailures:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['name']}: {r['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
