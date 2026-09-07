"""2026-09-07 audit regression: ToolError must surface the failing tool and parameter.

Bug §8 — every tool entry point passed ``tool="..."`` and ``input="..."`` to
``_tool_error``, but the helper accepted those kwargs and silently dropped them
on the floor. The resulting ToolError carried only ``[CODE] message`` with no
way for an LLM caller to identify which tool or argument tripped the error.

Fix: ``_tool_error`` now appends structured ``(tool=... input=...)`` context
in parentheses — only when kwargs are present — preserving the existing
``[CODE] message`` prefix that the prior regression suite asserts on.
"""

from __future__ import annotations

import pytest

from mcp_server.tools._common import _tool_error, error_code_for


def test_tool_error_appends_tool_name() -> None:
    """The failing tool name must appear in the message."""
    err = _tool_error("invalid_input", "missing move", tool="classify_move")
    msg = str(err)
    assert msg.startswith("[INVALID_INPUT]"), f"prefix broken: {msg!r}"
    assert "tool=classify_move" in msg, f"tool context missing: {msg!r}"
    assert "missing move" in msg, f"message body missing: {msg!r}"


def test_tool_error_appends_input_kwarg() -> None:
    """Supplied ``input`` kwarg must be quoted and present."""
    err = _tool_error(
        "invalid_fen", "truncated FEN", tool="evaluate_position", input="bad-fen-string"
    )
    msg = str(err)
    assert "tool=evaluate_position" in msg
    assert "input='bad-fen-string'" in msg


def test_tool_error_truncates_large_inputs() -> None:
    """Inputs >120 chars must be truncated so errors stay short."""
    long_pgn = "x" * 500
    err = _tool_error("invalid_pgn", "could not parse", tool="analyze_game", input=long_pgn)
    msg = str(err)
    assert "input=" in msg
    assert "x" * 500 not in msg, "raw 500-char input leaked into error message"


def test_tool_error_with_no_kwargs_drops_parentheses() -> None:
    """When no kwargs are present the message must NOT have parentheses appended."""
    err = _tool_error("invalid_input", "depth must be int", tool="")
    msg = str(err)
    assert msg == "[INVALID_INPUT] depth must be int", f"unexpected: {msg!r}"


def test_tool_error_skips_none_kwargs() -> None:
    """None kwargs are filtered out so the context stays clean."""
    err = _tool_error("invalid_input", "missing move", tool="classify_move", input=None, extra="hi")
    msg = str(err)
    assert "tool=classify_move" in msg
    assert "extra='hi'" in msg
    assert "input=" not in msg, f"None kwarg leaked: {msg!r}"


def test_tool_error_handles_base_exception_message() -> None:
    """Passing a BaseException must render its __str__ cleanly."""
    err = _tool_error("engine_error", ValueError("bad value"), tool="top_moves")
    msg = str(err)
    assert msg.startswith("[ENGINE_ERROR]")
    assert "tool=top_moves" in msg
    assert "bad value" in msg


def test_error_code_for_preserves_prefix_mapping() -> None:
    """The inference map must keep working — no regression in existing tests."""
    assert error_code_for("INVALID_FEN: bad") == "invalid_fen"
    assert error_code_for("MISSING_MOVE: foo") == "invalid_input"
    assert error_code_for("nothing recognizable") == "invalid_input"


def test_tool_error_message_starts_with_bracketed_code() -> None:
    """Existing tests assert ``[CODE]`` prefix; verify the new format keeps it."""
    for code in ("invalid_input", "invalid_fen", "engine_error"):
        err = _tool_error(code, "boom", tool="x")
        assert str(err).startswith(f"[{code.upper()}] "), (
            f"broken prefix for code={code}: {str(err)!r}"
        )


def test_tool_error_propagates_baseexception_group() -> None:
    """Exception groups must still be flattened in the body."""
    eg = ValueError("a")
    err = _tool_error("engine_error", eg, tool="evaluate_position")
    assert "tool=evaluate_position" in str(err)
