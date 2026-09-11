"""Response extraction and normalization for MCP tool call results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class NormalizedResponse:
    is_error: bool
    text_content: str
    parsed_payload: dict[str, Any] | None
    wire_bytes: int
    text_bytes: int
    structured_payload_bytes: int
    structured_error_code: str | None
    build_sha: str


def deep_find_key(value: Any, key: str) -> str:
    """Recursively find first string matching key in nested dict/list structures."""
    if isinstance(value, dict):
        if key in value and isinstance(value[key], str):
            return value[key]
        for v in value.values():
            found = deep_find_key(v, key)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = deep_find_key(item, key)
            if found:
                return found
    return ""


def extract_structured_error_code(text: str, payload: dict[str, Any] | None) -> str | None:
    """Extract structured error code from payload or brackets [CODE]."""
    if payload:
        code = payload.get("code") or payload.get("error_code") or payload.get("status")
        if isinstance(code, str) and code.strip():
            return code.strip()

    if "[" in text and "]" in text:
        open_b = text.find("[")
        close_b = text.find("]", open_b)
        if close_b > open_b:
            code = text[open_b + 1 : close_b].strip()
            if code and " " not in code:
                return code

    return None


def normalize_call_tool_result(result: Any) -> NormalizedResponse:
    """Normalize CallToolResult or dictionary output across MCP versions and formats."""
    is_error = False
    if isinstance(result, Exception):
        is_error = True
    elif hasattr(result, "isError"):
        is_error = bool(result.isError)
    elif hasattr(result, "is_error"):
        is_error = bool(result.is_error)
    elif isinstance(result, dict):
        is_error = bool(result.get("isError") or result.get("is_error"))

    parsed_payload: dict[str, Any] | None = None
    structured_bytes = 0

    # Handle Pydantic models directly (in-process mode)
    if not isinstance(result, Exception):
        dump_fn = getattr(result, "model_dump", None)
        if callable(dump_fn):
            try:
                dumped = dump_fn(mode="json")
                if isinstance(dumped, dict):
                    parsed_payload = dumped
                    structured_bytes = len(json.dumps(dumped).encode("utf-8"))
            except Exception:
                pass

    # 1. Check structuredContent / structured_content
    struct_attr = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if struct_attr is not None:
        if isinstance(struct_attr, dict):
            parsed_payload = struct_attr
            structured_bytes = len(json.dumps(struct_attr).encode("utf-8"))
        elif hasattr(struct_attr, "model_dump"):
            try:
                dumped = struct_attr.model_dump(mode="json")
                if isinstance(dumped, dict):
                    parsed_payload = dumped
                    structured_bytes = len(json.dumps(dumped).encode("utf-8"))
            except Exception:
                pass

    # 2. Extract and concatenate text across all content blocks
    text_blocks: list[str] = []
    if isinstance(result, Exception):
        text_blocks.append(str(result))

    content_list = getattr(result, "content", None)
    if content_list is None and isinstance(result, dict):
        content_list = result.get("content")

    if isinstance(content_list, (list, tuple)):
        for item in content_list:
            if hasattr(item, "text") and isinstance(item.text, str):
                text_blocks.append(item.text)
            elif isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                text_blocks.append(item["text"])

    full_text = "\n".join(text_blocks).strip()
    text_bytes = len(full_text.encode("utf-8"))

    # 3. If no structured payload yet, attempt JSON parse on text blocks
    if parsed_payload is None and full_text:
        # Try full text
        try:
            val = json.loads(full_text)
            if isinstance(val, dict):
                parsed_payload = val
                structured_bytes = text_bytes
        except Exception:
            # Try parsing first text block
            if text_blocks:
                try:
                    val = json.loads(text_blocks[0])
                    if isinstance(val, dict):
                        parsed_payload = val
                        structured_bytes = len(text_blocks[0].encode("utf-8"))
                except Exception:
                    pass

    # 4. If result is directly a dict with payload fields
    if parsed_payload is None and isinstance(result, dict):
        if any(k in result for k in ("status", "best_move", "result", "total_plies", "move_class")):
            parsed_payload = result
            structured_bytes = len(json.dumps(result).encode("utf-8"))

    # Estimate total wire bytes
    wire_bytes = max(text_bytes, structured_bytes)
    if not isinstance(result, dict):
        dump_fn = getattr(result, "model_dump", None)
        if callable(dump_fn):
            try:
                dumped = dump_fn(mode="json")
                wire_bytes = len(json.dumps(dumped).encode("utf-8"))
            except Exception:
                pass

    # Extract build_sha
    build_sha = deep_find_key(parsed_payload, "build_sha") if parsed_payload else ""
    if not build_sha and not isinstance(result, dict):
        dump_fn = getattr(result, "model_dump", None)
        if callable(dump_fn):
            try:
                dumped = dump_fn(mode="json")
                build_sha = deep_find_key(dumped, "build_sha")
            except Exception:
                pass

    error_code = extract_structured_error_code(full_text, parsed_payload)

    return NormalizedResponse(
        is_error=is_error,
        text_content=full_text,
        parsed_payload=parsed_payload,
        wire_bytes=wire_bytes,
        text_bytes=text_bytes,
        structured_payload_bytes=structured_bytes,
        structured_error_code=error_code,
        build_sha=build_sha,
    )
