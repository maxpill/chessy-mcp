"""ASGI middleware: rate limiting and structured request logging.

Split from ``mcp_server.server``. ``main()`` and ``_build_app`` also live
here because the middleware stack IS the application composition.
"""
