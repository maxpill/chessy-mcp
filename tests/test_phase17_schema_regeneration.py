"""Phase 17 (2026-09-14): schema regeneration + descriptions.

Locks the public MCP schema in place so future drift in tool descriptions,
enum values, or numeric bounds is detected by a fast test.

Tests:
- :func:`test_top_moves_n_description_pin` — top_moves.n says "1-20".
- :func:`test_proof_defenses_description_pin` — proof_defenses says "1..8" / "rejected".
- :func:`test_top_moves_include_moves_cap_pin` — include_moves says "unique canonical moves".
- :func:`test_classify_move_compare_moves_cap_pin` — same wording.
- :func:`test_position_hash_deprecation_in_docs` — README mentions ``fen_hash`` / ``repetition_key``.
- :func:`test_schema_no_dual_messaging` — verbose schema never says "max 10" anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path


from mcp_server import server as server_module

SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "schema_phase17_main.json"


def _tool_schemas() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tool_name in ("evaluate_position", "top_moves", "classify_move", "analyze_game"):
        tool = server_module.mcp._tool_manager.get_tool(tool_name)
        params = tool.parameters
        if isinstance(params, dict):
            out[tool_name] = {k: v for k, v in params.items() if k != "$schema"}
        else:
            out[tool_name] = params
    return out


def test_top_moves_n_description_pin() -> None:
    schemas = _tool_schemas()
    desc = schemas["top_moves"]["properties"]["n"]["description"]
    assert "1-20" in desc, f"top_moves.n description must reference '1-20'; got {desc!r}"


def test_proof_defenses_description_pin() -> None:
    schemas = _tool_schemas()
    desc = schemas["top_moves"]["properties"]["proof_defenses"]["description"]
    assert "1" in desc and "8" in desc, (
        f"proof_defenses description must reference the 1..8 range; got {desc!r}"
    )
    assert "rejected" in desc.lower() or "invalid" in desc.lower(), (
        f"proof_defenses description must mention strict rejection; got {desc!r}"
    )


def test_top_moves_include_moves_cap_pin() -> None:
    schemas = _tool_schemas()
    desc = schemas["top_moves"]["properties"]["include_moves"]["description"].lower()
    assert "8" in desc and ("unique" in desc or "canonical" in desc), (
        f"include_moves description must mention '8 unique canonical'; got {desc!r}"
    )


def test_classify_move_compare_moves_cap_pin() -> None:
    schemas = _tool_schemas()
    desc = schemas["classify_move"]["properties"]["compare_moves"]["description"].lower()
    assert "8" in desc and ("canonical" in desc or "unique" in desc), (
        f"compare_moves description must mention the canonical-unique cap; got {desc!r}"
    )


def test_schema_no_dual_messaging() -> None:
    """Schema must not advertise 1..10 anywhere; 1..20 is the canonical cap."""
    schemas = _tool_schemas()
    payload = json.dumps(schemas, sort_keys=True)
    # Allow the literal "1-10" only if it's clearly describing something other
    # than the public n-cap. We forbid it explicitly to lock the F-002 fix.
    assert "1-10" not in payload, (
        "Schema advertises 1-10 somewhere. Phase 1 locked the canonical n-cap "
        "to 1-20; revert any 'max 10' / '1-10' mentions in tool descriptions."
    )


def test_top_moves_depth_pin() -> None:
    """top_moves depth description must reference 1..30 clamp."""
    schemas = _tool_schemas()
    desc = schemas["top_moves"]["properties"]["depth"]["description"]
    assert "1-30" in desc or "1..30" in desc, (
        f"top_moves.depth description must mention 1-30; got {desc!r}"
    )


def test_classify_move_depth_pin() -> None:
    schemas = _tool_schemas()
    desc = schemas["classify_move"]["properties"]["depth"]["description"]
    assert "1-30" in desc or "1..30" in desc


def test_evaluate_position_verbosity_alias_pin() -> None:
    schemas = _tool_schemas()
    desc = schemas["evaluate_position"]["properties"]["verbosity"]["description"].lower()
    assert "min" in desc and "standard" in desc and "default" in desc, (
        f"verbosity description must document the three aliases; got {desc!r}"
    )


def test_classify_move_no_verbosity_regression() -> None:
    """Phase 10 fix: classify_move does not need a verbosity parameter to
    remain a low-cost path, but the schema should not regress and accidentally
    expose verbosity to callers that don't expect it."""
    schemas = _tool_schemas()
    assert "verbosity" not in schemas["classify_move"]["properties"], (
        "Phase 10 deliberately did NOT add verbosity to classify_move — the "
        "audit decided to keep classify_move's default response lean. If you "
        "want to add it, do so in a separate change."
    )


def test_position_hash_deprecation_in_readme() -> None:
    readme = Path(__file__).resolve().parent.parent / "README.md"
    text = readme.read_text().lower()
    assert "fen_hash" in text, "README must mention the new fen_hash field"
    assert "repetition_key" in text, "README must mention the new repetition_key field"
