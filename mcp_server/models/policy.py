"""``PolicyBlock`` aggregate — decision + policy metadata of :class:`MCPEval`.

Groups the decision-value output (outcome / cp_equivalent / perspective),
the action policy version + thresholds (audit M-04 surface), and any
nested policy descriptors tied to the response.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mcp_server.models.action_policy import ActionPolicyMetadata


class PolicyBlock(BaseModel):
    """Decision and policy metadata for a single position evaluation."""

    decision_value: dict[str, Any] | None = None
    action_policy: ActionPolicyMetadata = Field(default_factory=ActionPolicyMetadata)
