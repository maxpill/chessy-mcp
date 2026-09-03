"""PGN / FEN / SAN parsing pipeline.

Split from ``mcp_server.server`` for navigability. Each stage of the parsing
pipeline (canonicalization, sanitization, validation, board construction, move
parsing) lives in its own module under :mod:`mcp_server.parsers`.
"""
