"""``ActionPolicyMetadata`` — explicit policy versioning for the
play_move vs claim-draw decision (audit M-04).

Lives in its own module so :class:`MCPEval` (which embeds it in
``PolicyBlock``) can reference it without re-defining it inline.
"""

from __future__ import annotations

from pydantic import BaseModel

from mcp_server.rules.constants import (
    ACTION_EQUIVALENCE_THRESHOLD_CP,
    ACTION_MATERIAL_DOWN_THRESHOLD_CP,
    ACTION_POLICY_NAME,
    ACTION_POLICY_VERSION,
)


class ActionPolicyMetadata(BaseModel):
    """Explicit metadata about the policy used to select between
    ``play_move`` and draw-claim actions (audit M-04)."""

    name: str = ACTION_POLICY_NAME
    version: str = ACTION_POLICY_VERSION
    equivalence_threshold_cp: int = ACTION_EQUIVALENCE_THRESHOLD_CP
    material_down_threshold_cp: int = ACTION_MATERIAL_DOWN_THRESHOLD_CP
    forced_win_overrides_claim: bool = True
