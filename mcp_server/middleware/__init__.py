"""ASGI middleware: rate limiting, structured request logging, IP / cost helpers.

Modules split the middleware concerns (Phase 28):

  * :mod:`mcp_server.middleware.rate_limit` — weighted token-bucket rate
    limiter.
  * :mod:`mcp_server.middleware.request_logger` — request logging,
    weighted admission control, token authentication, ASGI app
    composition, and the CLI ``main()` entry point.
  * :mod:`mcp_server.middleware.client_ip` — :func:`effective_client_ip`
    + :func:`is_trusted_proxy_peer` (X-Forwarded-For unwrap).
  * :mod:`mcp_server.middleware.request_cost` — per-tool admission cost
    estimator.
"""

from mcp_server.middleware.client_ip import (
    _effective_client_ip,
    _is_trusted_proxy_peer,
)
from mcp_server.middleware.rate_limit import TokenBucketRateLimiter
from mcp_server.middleware.request_cost import _estimate_mcp_request_cost
from mcp_server.middleware.request_logger import (
    ASGIRequestLoggerMiddleware,
    _build_app,
    main,
)


__all__ = [
    "ASGIRequestLoggerMiddleware",
    "TokenBucketRateLimiter",
    "_build_app",
    "_effective_client_ip",
    "_estimate_mcp_request_cost",
    "_is_trusted_proxy_peer",
    "main",
]
