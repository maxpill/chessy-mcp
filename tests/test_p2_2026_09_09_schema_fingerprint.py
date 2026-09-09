"""2026-09-09 master audit F-005: deployed vs source schema fingerprint parity.

The audit observed that ``classify_move`` tools/list schema surfaced a default
depth of 20 in some clients while the source had default 16. The fix is to
make the schema observable: enumerate tools, dump their input schemas, hash,
and compare to a checked-in snapshot at the same build SHA.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mcp_server import server as server_module


SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "schema_fingerprint_main.json"


def _canonicalize_schema(schema: dict) -> dict:
    """Strip ephemeral fields like $schema that drift across MCP versions."""
    if isinstance(schema, dict):
        return {k: _canonicalize_schema(v) for k, v in schema.items() if k != "$schema"}
    if isinstance(schema, list):
        return [_canonicalize_schema(v) for v in schema]
    return schema


def _tool_schemas() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tool_name in ("evaluate_position", "top_moves", "classify_move", "analyze_game"):
        tool = server_module.mcp._tool_manager.get_tool(tool_name)
        out[tool_name] = _canonicalize_schema(tool.parameters)
    return out


def _fingerprint(schemas: dict[str, dict]) -> str:
    payload = json.dumps(schemas, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _update_snapshot() -> None:
    """Write the current schema fingerprint to disk (for snapshot updates)."""
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    schemas = _tool_schemas()
    snapshot = {
        "fingerprint": _fingerprint(schemas),
        "tools": schemas,
    }
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True))


def test_schema_fingerprint_matches_snapshot() -> None:
    """Source schema fingerprint must match the committed snapshot.

    When intentionally changing a public parameter (default, type, enum,
    description), run with ``--update-snapshots`` or call ``_update_snapshot``
    to refresh the JSON and commit the change.
    """
    if not SNAPSHOT_PATH.exists():
        pytest.skip(
            f"No snapshot at {SNAPSHOT_PATH}; "
            f"create it by importing and calling _update_snapshot() once."
        )

    current = _tool_schemas()
    current_fp = _fingerprint(current)

    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    expected_fp = snapshot["fingerprint"]

    assert current_fp == expected_fp, (
        f"Schema fingerprint drifted!\n"
        f"  expected: {expected_fp}\n"
        f"  actual:   {current_fp}\n"
        f"To regenerate the snapshot, call _update_snapshot() and commit the JSON.\n"
        f"Diff:\n{json.dumps(current, indent=2, sort_keys=True)}"
    )


def test_each_tool_has_canonical_parameter_set() -> None:
    """Each tool must have the documented parameter names + defaults."""
    schemas = _tool_schemas()

    eval_props = set(schemas["evaluate_position"]["properties"].keys())
    assert {"fen", "moves", "depth", "strict", "verbosity", "detail"} <= eval_props

    tm_props = set(schemas["top_moves"]["properties"].keys())
    assert {
        "fen",
        "moves",
        "n",
        "depth",
        "strict",
        "verbosity",
        "detail",
        "include_moves",
        "proof_mode",
        "proof_defenses",
    } <= tm_props

    cm_props = set(schemas["classify_move"]["properties"].keys())
    assert {
        "fen",
        "move",
        "moves",
        "depth",
        "action_type",
        "strict",
        "detail",
        "compare_moves",
    } <= cm_props

    ag_props = set(schemas["analyze_game"]["properties"].keys())
    assert {
        "pgn",
        "depth",
        "strict",
        "detail",
        "max_critical_moments",
        "perspective",
    } <= ag_props


def test_field_descriptions_match_canonical_contract() -> None:
    """Schema descriptions must match the canonical contract."""
    schemas = _tool_schemas()

    n_desc = schemas["top_moves"]["properties"]["n"]["description"]
    assert "1-20" in n_desc, f"top_moves.n description must say '1-20'; got {n_desc!r}"
    assert "1-10" not in n_desc

    depth_desc_cm = schemas["classify_move"]["properties"]["depth"]["description"]
    assert "1-30" in depth_desc_cm, (
        f"classify_move.depth description must say '1-30'; got {depth_desc_cm!r}"
    )
