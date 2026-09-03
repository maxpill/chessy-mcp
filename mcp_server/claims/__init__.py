"""Draw-claim projection helpers.

Split from :mod:`mcp_server.server`; the implementation lives in
:mod:`mcp_server.claims.draw_projection`.
"""

from mcp_server.claims.draw_projection import (
    _force_draw_outcome,
    force_draw_outcome,
)

__all__ = ["force_draw_outcome", "_force_draw_outcome"]
