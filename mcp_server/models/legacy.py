"""Legacy models — ``GameAnalysisResult``, ``MCPMoveAnalysis``,
``PlyAnalysisItem``, ``TopMovesResult``, ``PlayedMoveScore``.

These four classes were preserved as flat Pydantic models during the
Phase 18 atomization because their audit contracts are documented as
flat-shape (audit P0..P3, U-02..U-15). Splitting them out lets
``models/__init__.py`` import them lazily and keeps the fac-ade module
under 50 lines.

A future phase (Phase 18 follow-up) can atomize these further if the
audit surface shifts; until then they live here untouched.
"""

from __future__ import annotations

from typing import Any, Literal

import chess
from pydantic import BaseModel, Field, computed_field, model_validator

from core.engines.types import MoveAnalysis, MoveClass
from mcp_server.actions import build_played_action
from mcp_server.models.action_policy import ActionPolicyMetadata
from mcp_server.models.mcpeval import MCPEval
from mcp_server.rules import evaluate_rule_status


def _score_played_move(*args: Any, **kwargs: Any) -> Any:
    """Lazy import of ``score_played_move`` to break the cycle with
    :mod:`mcp_server.move_grading` which imports ``PlayedMoveScore``."""
    from mcp_server.move_grading import score_played_move as _impl

    return _impl(*args, **kwargs)


class PlayedMoveScore(BaseModel):
    """Per-move score returned from ``score_played_move``."""

    move_class: MoveClass
    centipawn_loss: int | None = None
    raw_centipawn_loss: int | None = None
    raw_centipawn_delta: int | None = None
    mate_distance_loss: int | None = None
    effective_loss: int | None = None
    loss_kind: str | None = None
    engine_cp_loss: int | None = None
    mate_distance_penalty: int | None = None
    outcome_penalty: int | None = None
    rule_action_penalty: int | None = None
    is_best_engine_move: bool = False
    win_loss: float = 0.0
    best_action: str = "play_move"
    is_best_action: bool = True
    action_equivalent: bool = False
    action_type: str = "play_move"
    missed_draw_claim: bool = False
    conceded_draw_claim: bool = False
    claim_reason: str | None = None
    claim_move: str | None = None
    can_claim_now: bool = False
    can_claim_with_intended_move: bool = False
    claim_moves: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def same_action_type(self) -> bool:
        return self.action_type == self.best_action

    @computed_field  # type: ignore[misc]
    @property
    def same_outcome(self) -> bool:
        return self.is_best_action

    @computed_field  # type: ignore[misc]
    @property
    def within_cp_threshold(self) -> bool:
        if self.action_type != "play_move" or self.best_action != "play_move":
            return self.is_best_action
        if self.centipawn_loss is None:
            return self.is_best_action
        return self.centipawn_loss <= 50  # ACTION_EQUIVALENCE_THRESHOLD_CP


class MCPMoveAnalysis(BaseModel):
    model_config = {"populate_by_name": True}
    schema_version: str = "1.2.0"
    played: str
    played_san: str | None = None
    move_class: MoveClass
    is_engine_best: bool = False
    is_best_engine_move: bool = False
    centipawn_loss: int | None = None
    mate_distance_loss: int | None = None
    raw_centipawn_loss: int | None = None
    raw_centipawn_delta: int | None = None
    effective_loss: int | None = None
    loss_kind: str | None = None
    engine_cp_loss: int | None = None
    mate_distance_penalty: int | None = None
    outcome_penalty: int | None = None
    rule_action_penalty: int | None = None
    eval_before: MCPEval
    eval_after: MCPEval
    best_move_san: str | None = None
    best_line_san: str | None = None
    best_line_san_truncated: bool = False
    played_line_san: str | None = None
    played_continuation_san: str | None = None
    syntax_warning: str | None = None
    action_type: Literal["play_move", "claim_draw", "claim_draw_with_intended_move"] = "play_move"
    best_action: str = "play_move"
    is_best_action: bool = True
    action_equivalent: bool = False
    played_action_obj: dict[str, Any] | None = None
    best_action_obj: dict[str, Any] | None = None
    action_policy: ActionPolicyMetadata = Field(default_factory=ActionPolicyMetadata)
    missed_draw_claim: bool = False
    conceded_draw_claim: bool = False
    claim_reason: str | None = None
    claim_move: str | None = None
    can_claim_now: bool = False
    can_claim_with_intended_move: bool = False
    claim_moves: list[str] = Field(default_factory=list)
    classification_verified: bool = False

    @computed_field  # type: ignore[misc]
    @property
    def same_action_type(self) -> bool:
        return self.action_type == self.best_action

    @computed_field  # type: ignore[misc]
    @property
    def same_outcome(self) -> bool:
        return self.is_best_action

    @computed_field  # type: ignore[misc]
    @property
    def within_cp_threshold(self) -> bool:
        if self.action_type != "play_move" or self.best_action != "play_move":
            return self.is_best_action
        if self.centipawn_loss is None:
            return self.is_best_action
        return self.centipawn_loss <= 50

    @model_validator(mode="after")
    def _enforce_action_invariants(self) -> MCPMoveAnalysis:
        engine_best = bool(self.is_engine_best or self.is_best_engine_move)
        self.is_engine_best = engine_best
        self.is_best_engine_move = engine_best
        if self.played_action_obj is not None:
            obj_type = self.played_action_obj.get("type")
            if obj_type != self.action_type:
                raise ValueError(
                    f"played_action_obj.type={obj_type!r} does not match action_type={self.action_type!r}"
                )
        if (
            self.is_best_action
            and self.action_type != self.best_action
            and not self.action_equivalent
        ):
            self.is_best_action = False
        return self

    @classmethod
    def from_analysis(
        cls,
        ma: MoveAnalysis,
        fen_before: str,
        fen_after: str,
        played_san: str | None = None,
        played_continuation_san: str | None = None,
        raw_centipawn_loss: int | None = None,
        board_before: chess.Board | None = None,
        board_after: chess.Board | None = None,
        syntax_warning: str | None = None,
        action_type: Literal[
            "play_move", "claim_draw", "claim_draw_with_intended_move"
        ] = "play_move",
        history_complete: str | bool = "incomplete",
    ) -> MCPMoveAnalysis:
        eval_bef = MCPEval.from_eval(
            ma.eval_before, fen_before, board=board_before, history_complete=history_complete
        )
        eval_aft = MCPEval.from_eval(
            ma.eval_after, fen_after, board=board_after, history_complete=history_complete
        )
        b_bef = board_before or chess.Board(fen_before)
        m = chess.Move.from_uci(ma.played)
        b_aft = board_after or (b_bef.copy(stack=True) if board_before else chess.Board(fen_after))
        if not board_after and m in b_bef.legal_moves:
            b_aft.push(m)

        score = _score_played_move(b_bef, m, eval_bef, eval_aft, b_aft, action_type=action_type)

        best_san = ma.best_move_san
        if not best_san and eval_bef.best_move:
            try:
                bm = chess.Move.from_uci(eval_bef.best_move.lower())
                if bm in b_bef.legal_moves:
                    best_san = b_bef.san(bm)
            except Exception:
                pass

        best_line_san = ma.best_line_san or best_san

        verified = True
        if (
            action_type == "play_move"
            and score.best_action != "play_move"
            and score.is_best_action
            and not score.action_equivalent
        ):
            verified = False
        if (
            score.effective_loss
            and score.effective_loss > 0
            and (not score.loss_kind or score.loss_kind == "none")
        ):
            verified = False
        if score.is_best_engine_move and score.effective_loss and score.effective_loss > 0:
            verified = False

        rule_before = evaluate_rule_status(b_bef, history_complete=history_complete)
        played_action_obj = build_played_action(
            action_type,
            move_uci=ma.played,
            move_san=played_san,
            rule_status=rule_before,
            cp=eval_aft.cp,
            mate=eval_aft.mate,
        )
        best_action_payload = eval_bef.best_action_obj

        return cls(
            played=ma.played,
            played_san=played_san,
            move_class=score.move_class,
            is_engine_best=score.is_best_engine_move,
            is_best_engine_move=score.is_best_engine_move,
            centipawn_loss=score.centipawn_loss,
            mate_distance_loss=score.mate_distance_loss,
            raw_centipawn_loss=score.raw_centipawn_loss,
            raw_centipawn_delta=score.raw_centipawn_delta,
            effective_loss=score.effective_loss,
            loss_kind=score.loss_kind,
            engine_cp_loss=score.engine_cp_loss,
            mate_distance_penalty=score.mate_distance_penalty,
            outcome_penalty=score.outcome_penalty,
            rule_action_penalty=score.rule_action_penalty,
            eval_before=eval_bef,
            eval_after=eval_aft,
            best_move_san=best_san,
            best_line_san=best_line_san,
            best_line_san_truncated=bool(eval_bef.pv and len(eval_bef.pv) > 6),
            played_line_san=ma.played_line_san or played_san,
            played_continuation_san=played_continuation_san,
            syntax_warning=syntax_warning,
            action_type=action_type,
            best_action=score.best_action,
            is_best_action=score.is_best_action,
            action_equivalent=score.action_equivalent,
            played_action_obj=played_action_obj,
            best_action_obj=best_action_payload,
            missed_draw_claim=score.missed_draw_claim,
            conceded_draw_claim=score.conceded_draw_claim,
            claim_reason=score.claim_reason,
            claim_move=score.claim_move,
            can_claim_now=score.can_claim_now,
            can_claim_with_intended_move=score.can_claim_with_intended_move,
            claim_moves=score.claim_moves,
            classification_verified=verified,
        )


class PlyAnalysisItem(BaseModel):
    ply: int
    san: str
    uci: str
    move_class: str
    centipawn_loss: int | None = None
    effective_loss: int | None = None
    loss_kind: str | None = None
    engine_cp_loss: int | None = None
    mate_distance_penalty: int | None = None
    outcome_penalty: int | None = None
    rule_action_penalty: int | None = None
    best_move_san: str | None = None
    best_action: str = "play_move"
    missed_draw_claim: bool = False
    conceded_draw_claim: bool = False
    claim_reason: str | None = None
    claim_move: str | None = None


class GameAnalysisResult(BaseModel):
    schema_version: str = "1.2.0"
    total_plies: int
    white_accuracy: float | None = None
    black_accuracy: float | None = None
    white_acpl: float | None = Field(default=None)
    black_acpl: float | None = Field(default=None)
    white_raw_acpl: float | None = Field(default=None)
    black_raw_acpl: float | None = Field(default=None)
    white_effective_acpl: float | None = Field(default=None)
    black_effective_acpl: float | None = Field(default=None)
    white_average_effective_loss: float | None = None
    black_average_effective_loss: float | None = None
    white_blunders: int = 0
    white_mistakes: int = 0
    white_inaccuracies: int = 0
    black_blunders: int = 0
    black_mistakes: int = 0
    black_inaccuracies: int = 0
    turning_points: list[PlyAnalysisItem] = Field(default_factory=list[PlyAnalysisItem])
    white: str | None = None
    black: str | None = None
    event: str | None = None
    site: str | None = None
    date: str | None = None
    round: str | None = None
    result: str | None = None
    result_header: str | None = None
    result_header_raw: str | None = None
    result_movetext: str | None = None
    result_inferred: str | None = None
    white_elo: str | None = None
    black_elo: str | None = None
    time_control: str | None = None
    variant: str | None = None
    eco: str | None = None
    opening: str | None = None
    opening_header: str | None = None
    eco_header: str | None = None
    metadata_warnings: list[str] = Field(default_factory=list)
    syntax_warnings: list[str] = Field(default_factory=list)
    termination: str | None = None
    termination_header: str | None = None
    requested_depth: int | None = None
    searched_depth: int | None = None
    engine: str = "Stockfish"
    engine_version: str | None = None
    service_version: str = "0.1.0"
    build_sha: str | None = None
    engine_config: dict[str, Any] = Field(default_factory=dict)
    accuracy_method: str = "win_probability_logistic"
    mate_penalty_policy: str = "1000_cp_mate_transition"


class TopMovesResult(BaseModel):
    schema_version: str = "1.2.0"
    status: str = "active"
    winner: str | None = None
    recommended_action: str = "play_move"
    can_claim_draw: bool = False
    claim_reasons: list[str] = Field(default_factory=list)
    claim_move: str | None = None
    can_claim_now: bool = False
    claim_reasons_now: list[str] = Field(default_factory=list)
    can_claim_with_intended_move: bool = False
    claim_moves: list[str] = Field(default_factory=list)
    best_action_obj: dict[str, Any] | None = None
    legal_actions: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    legal_rule_actions: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    legal_move_uci: list[str] = Field(default_factory=list[str])
    history_completeness: str = "complete"
    repetition_status: str = "none"
    requested_depth: int | None = None
    searched_depth: int | None = None
    requested_n: int | None = None
    clamped_n: int | None = None
    returned_n: int | None = None
    legal_move_count: int | None = None
    engine: str = "Stockfish"
    engine_version: str | None = None
    service_version: str = "0.1.0"
    build_sha: str | None = None
    engine_config: dict[str, Any] = Field(default_factory=dict)
    action_policy: ActionPolicyMetadata = Field(default_factory=ActionPolicyMetadata)
    canonical_fen: str | None = None
    fen_was_canonicalized: bool = False
    result: list[MCPEval] = Field(default_factory=list[MCPEval])

    def __iter__(self) -> Any:
        return iter(self.result)

    def __len__(self) -> int:
        return len(self.result)

    def __getitem__(self, item: int | slice) -> Any:
        return self.result[item]

    def __bool__(self) -> bool:
        return bool(self.result)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, list):
            return self.result == other
        if isinstance(other, TopMovesResult):
            return self.result == other.result
        return False
