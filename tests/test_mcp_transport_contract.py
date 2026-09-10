"""FastMCP in-process transport and JSON-RPC contract tests.

Verifies the actual MCP wire transport via ClientSession:
  - Server initialization and tool discovery
  - Tool parameter JSON schemas and enums
  - Calling all four tools over the MCP transport
  - Verbosity alias acceptance across 'min', 'minimal', 'compact', 'full', 'standard', 'default'
  - Rejection of invalid parameter combinations (e.g. minimal + coach detail)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession
from mcp.client._memory import create_client_server_memory_streams

from mcp_server.rules.constants import TOP_MOVES_MAX_N
from mcp_server.server import mcp
from mcp_server.tools._common import SUPPORTED_VERBOSITY_INPUTS


@asynccontextmanager
async def _mcp_session() -> AsyncIterator[ClientSession]:
    """Yield a connected ClientSession talking to the in-process MCPServer instance."""
    async with create_client_server_memory_streams() as ((c_read, c_write), (s_read, s_write)):
        server_task = asyncio.create_task(
            mcp._lowlevel_server.run(
                s_read, s_write, mcp._lowlevel_server.create_initialization_options()
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
async def test_mcp_tools_list_schema_and_enums() -> None:
    """AUDIT-002 / AUDIT-003: tools/list must expose exact schemas, canonical constants, and all verbosity aliases."""
    async with _mcp_session() as session:
        tools_result = await session.list_tools()
        tools_by_name = {tool.name: tool for tool in tools_result.tools}

        assert {"evaluate_position", "top_moves", "classify_move", "analyze_game"} <= set(
            tools_by_name.keys()
        )

        # Check evaluate_position verbosity enum
        eval_schema = tools_by_name["evaluate_position"].input_schema
        verbosity_prop = eval_schema["properties"]["verbosity"]
        enum_values = None
        for item in verbosity_prop.get("anyOf", []):
            if "enum" in item:
                enum_values = set(item["enum"])
                break
        assert enum_values is not None, "verbosity must expose enum options"
        assert enum_values == set(SUPPORTED_VERBOSITY_INPUTS)
        assert {"min", "standard", "default", "minimal", "compact", "full"} <= enum_values

        # Check top_moves verbosity enum and n max description
        top_schema = tools_by_name["top_moves"].input_schema
        top_verbosity = top_schema["properties"]["verbosity"]
        top_enum = None
        for item in top_verbosity.get("anyOf", []):
            if "enum" in item:
                top_enum = set(item["enum"])
                break
        assert top_enum == set(SUPPORTED_VERBOSITY_INPUTS)

        n_desc = top_schema["properties"]["n"]["description"]
        assert f"1-{TOP_MOVES_MAX_N}" in n_desc
        assert "1-10" not in n_desc

        # Check classify_move default depth is 16
        classify_schema = tools_by_name["classify_move"].input_schema
        depth_prop = classify_schema["properties"]["depth"]
        assert depth_prop.get("default") == 16


@pytest.mark.asyncio
async def test_mcp_call_all_four_tools_over_transport() -> None:
    """Test calling all four tools over the MCP transport."""
    async with _mcp_session() as session:
        # evaluate_position
        res_eval = await session.call_tool(
            "evaluate_position", {"fen": "startpos", "depth": 1, "verbosity": "compact"}
        )
        assert res_eval.is_error is False or not res_eval.is_error
        eval_data = json.loads(res_eval.content[0].text)
        assert "best_move" in eval_data

        # top_moves
        res_top = await session.call_tool(
            "top_moves", {"fen": "startpos", "n": 2, "depth": 1, "verbosity": "compact"}
        )
        assert res_top.is_error is False or not res_top.is_error
        top_data = json.loads(res_top.content[0].text)
        assert len(top_data["result"]) == 2

        # classify_move
        res_classify = await session.call_tool(
            "classify_move", {"fen": "startpos", "move": "e2e4", "depth": 1}
        )
        assert res_classify.is_error is False or not res_classify.is_error
        classify_data = json.loads(res_classify.content[0].text)
        assert "move_class" in classify_data

        # analyze_game
        res_analyze = await session.call_tool(
            "analyze_game", {"pgn": "1. e4 e5 2. Nf3 Nc6 *", "depth": 1}
        )
        assert res_analyze.is_error is False or not res_analyze.is_error
        analyze_data = json.loads(res_analyze.content[0].text)
        assert analyze_data["total_plies"] == 4


@pytest.mark.asyncio
async def test_mcp_all_verbosity_aliases_accepted_over_transport() -> None:
    """AUDIT-002: Test that all six verbosity spellings are accepted without transport error."""
    async with _mcp_session() as session:
        for alias in ("min", "minimal", "compact", "full", "standard", "default"):
            res = await session.call_tool(
                "evaluate_position",
                {"fen": "startpos", "depth": 1, "verbosity": alias},
            )
            assert res.is_error is False, f"Alias {alias} failed over transport: {res}"


@pytest.mark.asyncio
async def test_mcp_minimal_with_coach_detail_rejected_over_transport() -> None:
    """AUDIT-006: verbosity='minimal' with detail='coach' must return a tool error over transport."""
    async with _mcp_session() as session:
        res = await session.call_tool(
            "evaluate_position",
            {"fen": "startpos", "depth": 1, "verbosity": "minimal", "detail": "coach"},
        )
        assert res.is_error is True
        text = res.content[0].text
        assert "INVALID_ARGUMENT" in text
