"""ASGI middleware: rate limiting and structured request logging.

Two modules split the middleware concerns:

- :mod:`mcp_server.middleware.rate_limit` — weighted token-bucket rate limiter.
- :mod:`mcp_server.middleware.request_logger` — request logging, weighted
  admission control, token authentication, ASGI app composition, and the
  CLI ``main()`` entry point.
"""

from mcp_server.middleware.rate_limit import TokenBucketRateLimiter
from mcp_server.middleware.request_logger import (
    ASGIRequestLoggerMiddleware,
    _build_app,
    _effective_client_ip,
    _estimate_mcp_request_cost,
    _is_trusted_proxy_peer,
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
