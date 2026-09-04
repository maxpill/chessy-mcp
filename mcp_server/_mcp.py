"""The shared :class:`MCPServer` instance.

Lives in its own module so tool modules can ``from mcp_server._mcp import mcp``
without triggering the full ``mcp_server.server`` import chain — which in turn
imports the tool modules to register their ``@mcp.tool(...)`` decorators.

A previous layout imported ``mcp`` from ``mcp_server.server`` directly inside
each tool module. That worked under ``from mcp_server.server import mcp``
because server.py builds ``mcp`` before importing the tools, but it was
fragile: ``python -m mcp_server.server`` re-enters the tool import during
``mcp_server.server``'s own top-level execution, hitting a partially
initialized module and crashing on boot with::

    ImportError: cannot import name 'evaluate_position' from partially
    initialized module 'mcp_server.tools.evaluate_position'

Pulling the binding out into a leaf module breaks the cycle.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from mcp_server.engine.lifespan import _mcp_lifespan


mcp = MCPServer(
    "chess-analysis",
    description="Streamable Stockfish chess analysis and move grading MCP server",
    lifespan=_mcp_lifespan,
)


__all__ = ["mcp"]
