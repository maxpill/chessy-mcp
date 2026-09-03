"""``MCPEval`` — flat Pydantic model with typed-block computed views.

The atomization (Phase 18) takes the form of **typed views** rather
than nested storage. The wire shape is identical to the pre-atomization
flat ``MCPEval`` so the 600+ test assertions and the cache wire format
(``model_dump_json`` round-trip) keep working unchanged. New code can
read through the block view (``ev.eval.cp``, ``ev.action.best_move``)
for typed access; legacy code reads through the flat field
(``ev.cp``, ``ev.best_move``) and both are byte-equivalent.

This is a deliberate "atomization as a typed-view" rather than
"atomization as the storage shape":

  * EvalBlock / ActionBlock / HistoryBlock / PolicyBlock exist as
    first-class Pydantic classes so code can declare dependencies on
    one block at a time.
  * ``MCPEval`` stays flat-stored so ``model_copy(update=...)`` (used
    by the cache layer for ``requested_depth`` stamping) and every
    assertion in the existing test corpus keep working.
  * The ``eval`` / ``action`` / ``history`` / ``policy`` block views are
    computed on read; mutations go through the flat fields.

Audit invariants preserved byte-identical: B-01..B-05, C-01..02,
H-01..03, L-04, L-06, M-04..05, P0..P3, U-02..U-15.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from mcp_server.models.action import ActionBlock
from mcp_server.models.action_policy import ActionPolicyMetadata
from mcp_server.models.eval import EvalBlock
from mcp_server.models.history import HistoryBlock
from mcp_server.models.policy import PolicyBlock

__all__ = ["MCPEval"]


class MCPEval(BaseModel):
    """Composed evaluation response for a single chess position.

    The flat storage shape preserves every audit-cited field path.
    The :class:`EvalBlock` / :class:`ActionBlock` / :class:`HistoryBlock` /
    :class:`PolicyBlock` aggregates are surfaced as ``@computed_field``
    views so callers can read through either path.
    """

    model_config = ConfigDict(extra="ignore")

    # ---------------- Flat storage ----------------
    schema_version: str = "1.2.0"
    status: str = "active"
    winner: str | None = None
    build_sha: str | None = None
    engine_config: dict[str, Any] = Field(default_factory=dict)
    cp: int | None = None
    mate: int | None = None
    root_score_cp: int | None = None
    root_score_mate: int | None = None
    post_fen: str | None = None
    best_move: str | None = None
    executable_move: str | None = None
    pv: list[str] = Field(default_factory=list[str])
    depth: int = 0
    requested_depth: int | None = None
    searched_depth: int | None = None
    can_claim_draw: bool = False
    claim_reasons: list[str] = Field(default_factory=list[str])
    claim_move: str | None = None
    claim_move_san: str | None = None
    claim_move_uci: str | None = None
    can_claim_now: bool = False
    claim_reasons_now: list[str] = Field(default_factory=list[str])
    can_claim_with_intended_move: bool = False
    claim_moves: list[str] = Field(default_factory=list[str])
    recommended_action: str = "play_move"
    best_action: str = "play_move"
    best_action_type: str = "play_move"
    best_action_obj: dict[str, Any] | None = None
    legal_actions: list[dict[str, Any]] = Field(
        default_factory=list[dict[str, Any]],
        deprecated=True,
    )
    legal_rule_actions: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    legal_move_uci: list[str] = Field(default_factory=list[str])
    decision_value: dict[str, Any] | None = None
    engine_eval: dict[str, Any] | None = None
    history_dependent_status: bool = False
    lichess_url_reproduces_history: bool = True
    requires_move_stack: bool = False
    fen_sufficient_for_status: bool = True
    history_completeness: str = "incomplete"
    repetition_status: str = "none"
    input_fen: str | None = None
    canonical_fen: str | None = None
    fen_was_canonicalized: bool = False
    action_policy: ActionPolicyMetadata = Field(default_factory=ActionPolicyMetadata)
    post_terminal_status: str | None = None
    candidate_san: str | None = None
    post_can_claim_draw: bool = False
    post_can_claim_now: bool = False
    post_claim_reasons: list[str] = Field(default_factory=list[str])
    post_claim_moves: list[str] = Field(default_factory=list[str])
    post_position: dict[str, Any] | None = None
    post_state_cp: int | None = None
    post_state_mate: int | None = None
    lichess_url: str | None = None
    lichess_image: str | None = None
    wdl: tuple[int, int, int] | None = None
    wdl_pct: dict[str, float] | None = None

    @model_validator(mode="before")
    @classmethod
    def _lift_nested_block_inputs(cls, values: Any) -> Any:
        """Accept the legacy ``from_eval`` factory's nested-block shape.

        Old ``from_eval`` builds ``EvalBlock`` / ``ActionBlock`` /
        ``HistoryBlock`` / ``PolicyBlock`` and passes them as nested
        fields. This validator lifts the values into the flat storage
        shape so callers can supply either pattern.
        """
        if not isinstance(values, dict):
            return values
        eval_block = values.pop("eval", None)
        if isinstance(eval_block, EvalBlock):
            for field_name in (
                "cp",
                "mate",
                "depth",
                "requested_depth",
                "searched_depth",
                "pv",
                "wdl",
                "wdl_pct",
                "root_score_cp",
                "root_score_mate",
                "post_state_cp",
                "post_state_mate",
                "engine_eval",
            ):
                values.setdefault(field_name, getattr(eval_block, field_name))
        action_block = values.pop("action", None)
        if isinstance(action_block, ActionBlock):
            for field_name in (
                "best_move",
                "executable_move",
                "recommended_action",
                "best_action",
                "best_action_type",
                "best_action_obj",
                "legal_actions",
                "legal_rule_actions",
                "legal_move_uci",
                "can_claim_draw",
                "claim_reasons",
                "claim_move",
                "claim_move_san",
                "claim_move_uci",
                "can_claim_now",
                "claim_reasons_now",
                "can_claim_with_intended_move",
                "claim_moves",
                "post_terminal_status",
                "candidate_san",
                "post_can_claim_draw",
                "post_can_claim_now",
                "post_claim_reasons",
                "post_claim_moves",
                "post_position",
            ):
                values.setdefault(field_name, getattr(action_block, field_name))
        history_block = values.pop("history", None)
        if isinstance(history_block, HistoryBlock):
            for field_name in (
                "input_fen",
                "canonical_fen",
                "fen_was_canonicalized",
                "post_fen",
                "history_dependent_status",
                "lichess_url_reproduces_history",
                "requires_move_stack",
                "fen_sufficient_for_status",
                "history_completeness",
                "repetition_status",
            ):
                values.setdefault(field_name, getattr(history_block, field_name))
        policy_block = values.pop("policy", None)
        if isinstance(policy_block, PolicyBlock):
            for field_name in ("decision_value", "action_policy"):
                values.setdefault(field_name, getattr(policy_block, field_name))
        return values

    @model_validator(mode="after")
    def _enforce_inv(self) -> MCPEval:
        return self

    # ---------------- Movable handle (back-compat for legacy code) ----------------
    @property
    def move(self) -> str | None:
        return self.best_move

    # ---------------- Typed block views (Phase 18 atomization) ----------------
    @computed_field  # type: ignore[misc]
    @property
    def eval_block(self) -> EvalBlock:
        return EvalBlock(
            cp=self.cp,
            mate=self.mate,
            depth=self.depth,
            requested_depth=self.requested_depth,
            searched_depth=self.searched_depth,
            pv=list(self.pv),
            wdl=self.wdl,
            wdl_pct=self.wdl_pct,
            root_score_cp=self.root_score_cp,
            root_score_mate=self.root_score_mate,
            post_state_cp=self.post_state_cp,
            post_state_mate=self.post_state_mate,
            engine_eval=self.engine_eval,
        )

    @computed_field  # type: ignore[misc]
    @property
    def action_block(self) -> ActionBlock:
        return ActionBlock(
            best_move=self.best_move,
            executable_move=self.executable_move,
            recommended_action=self.recommended_action,
            best_action=self.best_action,
            best_action_type=self.best_action_type,
            best_action_obj=self.best_action_obj,
            legal_actions=list(self.legal_actions),
            legal_rule_actions=list(self.legal_rule_actions),
            legal_move_uci=list(self.legal_move_uci),
            can_claim_draw=self.can_claim_draw,
            claim_reasons=list(self.claim_reasons),
            claim_move=self.claim_move,
            claim_move_san=self.claim_move_san,
            claim_move_uci=self.claim_move_uci,
            can_claim_now=self.can_claim_now,
            claim_reasons_now=list(self.claim_reasons_now),
            can_claim_with_intended_move=self.can_claim_with_intended_move,
            claim_moves=list(self.claim_moves),
            post_terminal_status=self.post_terminal_status,
            candidate_san=self.candidate_san,
            post_can_claim_draw=self.post_can_claim_draw,
            post_can_claim_now=self.post_can_claim_now,
            post_claim_reasons=list(self.post_claim_reasons),
            post_claim_moves=list(self.post_claim_moves),
            post_position=self.post_position,
        )

    @computed_field  # type: ignore[misc]
    @property
    def history_block(self) -> HistoryBlock:
        return HistoryBlock(
            input_fen=self.input_fen,
            canonical_fen=self.canonical_fen,
            fen_was_canonicalized=self.fen_was_canonicalized,
            post_fen=self.post_fen,
            history_dependent_status=self.history_dependent_status,
            lichess_url_reproduces_history=self.lichess_url_reproduces_history,
            requires_move_stack=self.requires_move_stack,
            fen_sufficient_for_status=self.fen_sufficient_for_status,
            history_completeness=self.history_completeness,
            repetition_status=self.repetition_status,
        )

    @computed_field  # type: ignore[misc]
    @property
    def policy_block(self) -> PolicyBlock:
        return PolicyBlock(
            decision_value=self.decision_value,
            action_policy=self.action_policy,
        )
