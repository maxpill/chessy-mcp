"""Case specification and identity models for Chess MCP stress audits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

ExpectedKind = Literal["success", "tool_error", "schema_error"]
DepthTier = Literal["low", "medium", "hard"]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    tool: str
    arguments: dict[str, Any]
    expected_kind: ExpectedKind
    case_ordinal: int = 0
    expected_error_code: str | None = None
    semantic_profile: str | None = None
    tags: list[str] = field(default_factory=list)
    depth_tier: DepthTier = "low"

    @property
    def arguments_sha256(self) -> str:
        """Raw argument sha256 hash."""
        payload = json.dumps(self.arguments, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{self.tool}\0{payload}".encode()).hexdigest()

    @property
    def raw_argument_hash(self) -> str:
        return self.arguments_sha256

    def compute_effective_argument_hash(self, normalized_arguments: dict[str, Any]) -> str:
        """Compute sha256 derived from schema-normalized, alias-resolved, default-canonicalized inputs."""
        payload = json.dumps(normalized_arguments, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{self.tool}\0{payload}".encode()).hexdigest()
