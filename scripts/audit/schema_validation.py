"""Schema validation and effective argument normalization for audit cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.audit.case_spec import CaseSpec

FINGERPRINT_PATH = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "schema_fingerprint_main.json"

_VERBOSITY_ALIASES: dict[str, str] = {
    "min": "minimal",
    "standard": "full",
    "default": "full",
}


def load_canonical_schemas() -> dict[str, dict[str, Any]]:
    if FINGERPRINT_PATH.exists():
        data = json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))
        return data.get("tools", {})
    return {}


def normalize_case_arguments(tool: str, raw_args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Derive canonical normalized arguments with default canonicalization and alias resolution."""
    properties = schema.get("properties", {})
    normalized: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        if prop_name in raw_args:
            val = raw_args[prop_name]
            if val is not None:
                if prop_name == "verbosity" and isinstance(val, str):
                    val = _VERBOSITY_ALIASES.get(val.lower(), val.lower())
                elif prop_name == "detail" and isinstance(val, str):
                    val = val.lower()
                elif prop_name == "action_type" and isinstance(val, str):
                    val = val.lower()
                elif prop_name == "proof_mode" and isinstance(val, str):
                    val = val.lower()
                elif prop_name == "perspective" and isinstance(val, str):
                    val = val.lower()
                elif prop_name in ("moves", "include_moves", "compare_moves") and isinstance(val, list):
                    val = [str(x) for x in val]
                normalized[prop_name] = val
            elif "default" in prop_schema and prop_schema["default"] is not None:
                normalized[prop_name] = prop_schema["default"]
        elif "default" in prop_schema and prop_schema["default"] is not None:
            normalized[prop_name] = prop_schema["default"]

    for k, v in raw_args.items():
        if k not in properties:
            normalized[k] = v

    return normalized


def validate_case_arguments(spec: CaseSpec, schema: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate a single CaseSpec against the tool schema.

    Returns:
        (is_valid, error_list, normalized_arguments)
    """
    errors: list[str] = []
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    raw_args = spec.arguments

    # Check required fields
    for req in required:
        if req not in raw_args:
            errors.append(f"Missing required argument '{req}' for tool '{spec.tool}' in case '{spec.case_id}'")

    # For success cases, reject undeclared properties (R2-003, R2-017)
    if spec.expected_kind == "success":
        for k in raw_args:
            if k not in properties:
                errors.append(
                    f"Undeclared argument '{k}' in success case '{spec.case_id}' for tool '{spec.tool}'. "
                    f"Allowed properties: {sorted(properties.keys())}"
                )

    # Validate enums and basic types
    for k, val in raw_args.items():
        if k in properties and val is not None:
            prop_def = properties[k]
            allowed_enums = prop_def.get("enum")
            if not allowed_enums and "anyOf" in prop_def:
                for option in prop_def["anyOf"]:
                    if "enum" in option:
                        allowed_enums = option["enum"]
                        break
            if allowed_enums and val not in allowed_enums:
                if k == "verbosity" and val in _VERBOSITY_ALIASES:
                    pass
                else:
                    errors.append(f"Invalid enum value '{val}' for argument '{k}' in case '{spec.case_id}'. Allowed: {allowed_enums}")

    normalized = normalize_case_arguments(spec.tool, raw_args, schema)
    return len(errors) == 0, errors, normalized


def preflight_validate_cases(
    specs: list[CaseSpec],
    schemas: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Validate all CaseSpecs, assert uniqueness of effective hashes.

    Returns:
        (validation_errors, dict of case_id -> effective_argument_hash)
    """
    if schemas is None:
        schemas = load_canonical_schemas()

    all_errors: list[str] = []
    effective_hashes: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}

    for spec in specs:
        schema = schemas.get(spec.tool, {})
        is_valid, errors, normalized = validate_case_arguments(spec, schema)
        if not is_valid and spec.expected_kind == "success":
            all_errors.extend(errors)

        eff_hash = spec.compute_effective_argument_hash(normalized)
        effective_hashes[spec.case_id] = eff_hash

        if spec.expected_kind == "success":
            if eff_hash in seen_hashes:
                dup_case_id = seen_hashes[eff_hash]
                all_errors.append(
                    f"Duplicate effective argument hash '{eff_hash[:12]}' between cases '{dup_case_id}' and '{spec.case_id}' for tool '{spec.tool}'"
                )
            else:
                seen_hashes[eff_hash] = spec.case_id

    return all_errors, effective_hashes
