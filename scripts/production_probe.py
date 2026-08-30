from __future__ import annotations

import argparse
import asyncio
import json
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

EXPECTED_TOOLS = {
    "evaluate_position",
    "top_moves",
    "classify_move",
    "analyze_game",
}


def _print(status: str, message: str) -> None:
    print(f"[{status}] {message}", flush=True)


def _dump_model(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json", by_alias=True)
        except TypeError:
            return model_dump()
    return value


def _find_values(value: Any, key: str) -> list[Any]:
    value = _dump_model(value)
    found: list[Any] = []
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if item_key == key:
                found.append(item_value)
            found.extend(_find_values(item_value, key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_find_values(item, key))
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                pass
            else:
                found.extend(_find_values(parsed, key))
    return found


def _result_is_error(value: Any) -> bool:
    dumped = _dump_model(value)
    if not isinstance(dumped, dict):
        return False
    return bool(dumped.get("isError") or dumped.get("is_error"))


def _short(value: Any, limit: int = 300) -> str:
    dumped = _dump_model(value)
    try:
        text = json.dumps(dumped, ensure_ascii=True, separators=(",", ":"), default=str)
    except TypeError:
        text = repr(dumped)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _sha_matches(expected: str, actual: str) -> bool:
    expected = expected.strip().lower()
    actual = actual.strip().lower()
    if not expected or not actual:
        return False
    return expected.startswith(actual) or actual.startswith(expected)


def _probe_dns_tls(base_url: str) -> list[str]:
    errors: list[str] = []
    parsed = urlparse(base_url)
    host = parsed.hostname
    if not host:
        return [f"Invalid target URL: {base_url}"]
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        addresses = sorted(
            {
                sockaddr[0]
                for _family, _socktype, _proto, _canonname, sockaddr in socket.getaddrinfo(
                    host, port, type=socket.SOCK_STREAM
                )
            }
        )
        _print("OK", f"DNS {host} -> {', '.join(addresses)}")
    except OSError as exc:
        errors.append(f"DNS resolution failed for {host}: {exc}")
        _print("FAIL", errors[-1])
        return errors

    if parsed.scheme != "https":
        _print("WARN", "TLS probe skipped because target is not HTTPS")
        return errors

    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=10) as raw_sock:
            with context.wrap_socket(raw_sock, server_hostname=host) as tls_sock:
                cert = tls_sock.getpeercert()
                cipher = tls_sock.cipher()
                expires = cert.get("notAfter", "unknown") if isinstance(cert, dict) else "unknown"
                cipher_name = cipher[0] if cipher else "unknown"
                _print(
                    "OK",
                    f"TLS {tls_sock.version()} cipher={cipher_name} certificate_not_after={expires}",
                )
    except (OSError, ssl.SSLError) as exc:
        errors.append(f"TLS handshake failed for {host}:{port}: {exc}")
        _print("FAIL", errors[-1])
    return errors


def _probe_health(base_url: str) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    health_url = base_url.rstrip("/") + "/health"
    last_error = ""
    for attempt in range(1, 4):
        started = time.monotonic()
        try:
            req = urllib.request.Request(
                health_url,
                headers={"User-Agent": "chessy-mcp-github-diagnostics/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                body = response.read().decode("utf-8", "replace")
                elapsed_ms = (time.monotonic() - started) * 1000
                headers = {key.lower(): value for key, value in response.headers.items()}
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    errors.append(f"Health returned non-JSON body: {body[:300]!r}")
                    _print("FAIL", errors[-1])
                    return errors, None
                _print(
                    "OK",
                    "Health HTTP "
                    f"{response.status} in {elapsed_ms:.0f}ms "
                    f"server={headers.get('server', '?')} via={headers.get('via', '?')} "
                    f"cf_ray={headers.get('cf-ray', '?')} body={_short(payload)}",
                )
                if response.status != 200:
                    errors.append(f"Health returned HTTP {response.status}")
                if not isinstance(payload, dict) or payload.get("status") != "ok":
                    errors.append(f"Health contract is not OK: {_short(payload)}")
                return errors, payload if isinstance(payload, dict) else None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            last_error = f"HTTP {exc.code}: {body[:300]}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        _print("WARN", f"Health attempt {attempt}/3 failed: {last_error}")
        if attempt < 3:
            time.sleep(attempt * 2)

    errors.append(f"Health endpoint unreachable after 3 attempts: {last_error}")
    _print("FAIL", errors[-1])
    return errors, None


async def _call_tool(session: ClientSession, name: str, arguments: dict[str, Any]) -> Any:
    started = time.monotonic()
    result = await session.call_tool(name, arguments=arguments)
    elapsed_ms = (time.monotonic() - started) * 1000
    if _result_is_error(result):
        raise RuntimeError(f"{name} returned MCP tool error: {_short(result, 700)}")
    _print("OK", f"{name} completed in {elapsed_ms:.0f}ms")
    return result


async def _probe_mcp(base_url: str, expected_sha: str, depth: int) -> list[str]:
    errors: list[str] = []
    mcp_url = base_url.rstrip("/") + "/mcp"
    try:
        async with streamable_http_client(mcp_url) as streams:
            read_stream, write_stream, *_ = streams
            async with ClientSession(read_stream, write_stream) as session:
                started = time.monotonic()
                init = await session.initialize()
                _print(
                    "OK",
                    f"MCP initialize completed in {(time.monotonic() - started) * 1000:.0f}ms: {_short(init)}",
                )

                tools_result = await session.list_tools()
                tool_names = {tool.name for tool in tools_result.tools}
                if tool_names != EXPECTED_TOOLS:
                    errors.append(
                        f"Tool surface mismatch: expected {sorted(EXPECTED_TOOLS)}, got {sorted(tool_names)}"
                    )
                    _print("FAIL", errors[-1])
                else:
                    _print("OK", f"MCP tools/list exposes {sorted(tool_names)}")

                evaluate = await _call_tool(
                    session,
                    "evaluate_position",
                    {"fen": "startpos", "depth": depth, "verbosity": "compact"},
                )
                await _call_tool(
                    session,
                    "top_moves",
                    {"fen": "startpos", "n": 2, "depth": depth, "verbosity": "compact"},
                )
                await _call_tool(
                    session,
                    "classify_move",
                    {"fen": "startpos", "move": "e2e4", "depth": depth},
                )
                await _call_tool(
                    session,
                    "analyze_game",
                    {"pgn": "1. e4 e5 2. Nf3 Nc6 *", "depth": depth},
                )

                build_values = [
                    str(value)
                    for value in _find_values(evaluate, "build_sha")
                    if value not in (None, "")
                ]
                actual_sha = next((value for value in build_values if value != "unknown"), "")
                if not actual_sha:
                    errors.append(
                        "Deployment identity missing: evaluate_position did not expose a usable build_sha"
                    )
                    _print("FAIL", errors[-1])
                elif expected_sha and not _sha_matches(expected_sha, actual_sha):
                    errors.append(
                        f"Stale/wrong deployment: expected SHA {expected_sha}, production reports {actual_sha}"
                    )
                    _print("FAIL", errors[-1])
                else:
                    _print("OK", f"Deployment identity build_sha={actual_sha}")
    except Exception as exc:
        message = str(exc)
        lower = message.lower()
        if "403" in lower or "forbidden" in lower or "auth token" in lower:
            diagnosis = "public edge reached backend but MCP authentication/gateway injection failed"
        elif "502" in lower or "bad gateway" in lower:
            diagnosis = "edge/proxy could not reach a healthy MCP upstream"
        elif "503" in lower or "service unavailable" in lower:
            diagnosis = "MCP upstream is unavailable or still starting"
        elif "timeout" in lower or "timed out" in lower:
            diagnosis = "MCP transport or Stockfish backend timed out"
        else:
            diagnosis = "MCP initialize/tool path failed"
        errors.append(f"{diagnosis}: {message}")
        _print("FAIL", errors[-1])
    return errors


async def _run(args: argparse.Namespace) -> int:
    errors: list[str] = []
    target = args.target.rstrip("/")
    _print("INFO", f"Target={target} expected_sha={args.expected_sha or '(not checked)'} depth={args.depth}")

    errors.extend(_probe_dns_tls(target))
    health_errors, _health_payload = _probe_health(target)
    errors.extend(health_errors)
    errors.extend(await _probe_mcp(target, args.expected_sha, args.depth))

    if errors:
        _print("FAIL", f"Production diagnostics found {len(errors)} problem(s)")
        for index, error in enumerate(errors, 1):
            _print("DIAG", f"{index}. {error}")
        return 1

    _print("OK", "Production deployment, MCP transport and all four engine tools passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the deployed Chess MCP end to end")
    parser.add_argument("--target", default="https://mcp.trychessy.com")
    parser.add_argument("--expected-sha", default="")
    parser.add_argument("--depth", type=int, default=6, choices=range(4, 13))
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
