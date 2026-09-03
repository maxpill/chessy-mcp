"""ASGI request-logging + auth + rate-limit middleware and app composition.

Owns:

- :class:`ASGIRequestLoggerMiddleware` — request logging, weighted admission
  control, token authentication, and 32 MiB POST body cap.
- :func:`_effective_client_ip` / :func:`_is_trusted_proxy_peer` /
  :func:`_estimate_mcp_request_cost` — small helpers used by the middleware.
- :func:`_build_app` — composes the FastMCP streamable-HTTP app with
  :class:`GZipMiddleware` (innermost) and the request logger (outermost).
- :func:`main` — CLI entry point; picks stdio vs streamable-http based on
  ``MCPSettings.transport``.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
from typing import Any, cast

from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mcp_server.middleware.rate_limit import TokenBucketRateLimiter

__all__ = [
    "ASGIRequestLoggerMiddleware",
    "TokenBucketRateLimiter",
    "_build_app",
    "_effective_client_ip",
    "_estimate_mcp_request_cost",
    "_is_trusted_proxy_peer",
    "main",
]


log = logging.getLogger("chessy_mcp.middleware")


def _is_trusted_proxy_peer(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.strip().strip("[]"))
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private


def _effective_client_ip(peer_ip: str, forwarded_for: str) -> str:
    peer = peer_ip.strip().strip("[]")
    if forwarded_for and _is_trusted_proxy_peer(peer):
        candidate = forwarded_for.split(",", 1)[0].strip().strip("[]")
        return candidate or peer
    return peer


def _estimate_mcp_request_cost(body: bytes) -> float:
    """Approximate CPU admission cost from tool, depth, MultiPV and PGN size."""
    try:
        payload_any: Any = json.loads(body.decode("utf-8"))
        payload = cast(dict[str, Any], payload_any) if isinstance(payload_any, dict) else {}
        params_any: Any = payload.get("params")
        params = cast(dict[str, Any], params_any) if isinstance(params_any, dict) else {}
        tool_name = str(params.get("name") or params.get("tool") or "")
        args_any: Any = params.get("arguments")
        args = cast(dict[str, Any], args_any) if isinstance(args_any, dict) else {}
        depth = max(1, min(int(args.get("depth", 18)), 30))
        if tool_name == "evaluate_position":
            return 1.0 + depth / 14.0
        if tool_name == "top_moves":
            n = max(1, min(int(args.get("n", 3)), 20))
            return 1.0 + (depth * n) / 14.0
        if tool_name == "classify_move":
            return 2.0 + depth / 10.0
        if tool_name == "analyze_game":
            pgn = str(args.get("pgn", ""))
            estimated_plies = max(1.0, min(200.0, len(pgn) / 24.0))
            return 5.0 + (depth * estimated_plies) / 28.0
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    return 1.0


class ASGIRequestLoggerMiddleware:
    """Request logging, weighted admission control and token authentication."""

    MAX_BUFFERED_BODY = 32 * 1024 * 1024

    def __init__(
        self,
        app: ASGIApp,
        restrict_to_chatgpt: bool = False,
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        self.app = app
        self.restrict_to_chatgpt = restrict_to_chatgpt
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_headers = cast(list[tuple[bytes, bytes]], scope.get("headers", []))
        headers_dict: dict[bytes, bytes] = dict(raw_headers)
        ua = headers_dict.get(b"user-agent", b"").decode("utf-8", "ignore")
        origin = headers_dict.get(b"origin", b"").decode("utf-8", "ignore")
        host = headers_dict.get(b"host", b"").decode("utf-8", "ignore")
        path = str(scope.get("path", ""))
        method = str(scope.get("method", "")).upper()
        session_id = headers_dict.get(b"mcp-session-id", b"").decode("utf-8", "ignore")

        client_tuple = cast(tuple[str, int] | None, scope.get("client"))
        peer_ip = client_tuple[0] if client_tuple else "127.0.0.1"
        forwarded_for = headers_dict.get(b"x-forwarded-for", b"").decode("utf-8", "ignore")
        client_ip = _effective_client_ip(peer_ip, forwarded_for)

        if method == "OPTIONS":
            await send(
                cast(
                    Message,
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"access-control-allow-origin", b"*"),
                            (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
                            (b"access-control-allow-headers", b"*"),
                            (b"access-control-max-age", b"86400"),
                            (b"content-length", b"0"),
                        ],
                    },
                )
            )
            await send(cast(Message, {"type": "http.response.body", "body": b""}))
            return

        if path != "/health" and self.restrict_to_chatgpt:
            from mcp_server.config import get_mcp_settings

            auth_token = get_mcp_settings().auth_token
            provided = headers_dict.get(b"x-chessy-auth", b"").decode("utf-8", "ignore").strip()
            authorization = (
                headers_dict.get(b"authorization", b"").decode("utf-8", "ignore").strip()
            )
            bearer = (
                authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
            )

            def token_matches(candidate: str) -> bool:
                return bool(auth_token) and hmac.compare_digest(candidate, auth_token)

            if not (token_matches(provided) or token_matches(bearer)):
                log.warning(
                    "Blocked unauthenticated client ip=%s ua=%r origin=%r", client_ip, ua, origin
                )
                response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Forbidden: valid MCP auth token required"}}\n'
                await send(
                    cast(
                        Message,
                        {
                            "type": "http.response.start",
                            "status": 403,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(response_body)).encode("ascii")),
                            ],
                        },
                    )
                )
                await send(cast(Message, {"type": "http.response.body", "body": response_body}))
                return

        request_cost = 1.0
        if method == "POST":
            raw_content_length = (
                headers_dict.get(b"content-length", b"").decode("ascii", "ignore").strip()
            )
            if raw_content_length:
                try:
                    declared_length = int(raw_content_length)
                except ValueError:
                    declared_length = -1
                if declared_length > self.MAX_BUFFERED_BODY:
                    response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Request body too large"}}\n'
                    await send(
                        cast(
                            Message,
                            {
                                "type": "http.response.start",
                                "status": 413,
                                "headers": [
                                    (b"content-type", b"application/json"),
                                    (b"content-length", str(len(response_body)).encode("ascii")),
                                ],
                            },
                        )
                    )
                    await send(cast(Message, {"type": "http.response.body", "body": response_body}))
                    return
                if declared_length > 8 * 1024:
                    request_cost = 5.0 + min(40.0, declared_length / 1024.0)
                elif declared_length > 1024:
                    request_cost = 2.0

        if not await self.rate_limiter.is_allowed(client_ip, cost=request_cost):
            response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Rate limit exceeded. Please slow down."}}\n'
            await send(
                cast(
                    Message,
                    {
                        "type": "http.response.start",
                        "status": 429,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(response_body)).encode("ascii")),
                            (b"retry-after", b"2"),
                        ],
                    },
                )
            )
            await send(cast(Message, {"type": "http.response.body", "body": response_body}))
            return

        log.info(
            "%s %s host=%s ip=%s session=%s origin=%s ua=%s cost=%.2f",
            method,
            path,
            host,
            client_ip,
            session_id,
            origin,
            ua,
            request_cost,
        )

        if path == "/health":
            await self.app(scope, receive, send)
            return

        await self.app(scope, receive, send)


def _build_app(restrict_chatgpt: bool) -> ASGIApp:
    """Compose the ASGI middleware stack around the FastMCP streamable-HTTP app.

    Order matters: outermost runs first on the way in, last on the way out.
    """
    from starlette.middleware.gzip import GZipMiddleware

    from mcp_server.server import mcp

    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    base = mcp.streamable_http_app(transport_security=security)
    gzipped = GZipMiddleware(base, minimum_size=1024, compresslevel=5)
    return ASGIRequestLoggerMiddleware(gzipped, restrict_to_chatgpt=restrict_chatgpt)


def main() -> None:
    from mcp_server.config import get_mcp_settings
    from mcp_server.server import mcp

    mcp_cfg = get_mcp_settings()
    transport = mcp_cfg.transport
    if transport == "streamable-http":
        host = mcp_cfg.http_host
        port = mcp_cfg.http_port
        restrict_chatgpt = mcp_cfg.lock_chatgpt
        wrapped_app = _build_app(restrict_chatgpt)
        import uvicorn

        uvicorn.run(
            wrapped_app,
            host=host,
            port=port,
            log_level="info",
            loop="uvloop",
            http="httptools",
            access_log=False,
            timeout_keep_alive=75,
            h11_max_incomplete_event_size=32 * 1024 * 1024,
        )
    else:
        mcp.run()
