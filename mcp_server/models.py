from __future__ import annotations

from typing import Any, Literal

import chess
from pydantic import BaseModel, Field, model_validator

from core.engines.grading import classify_centipawn_loss
from core.engines.types import Eval, MoveAnalysis, MoveClass
from core.winprob import win_prob as _win_pct
from mcp_server.actions import (
    build_best_action,
    build_legal_actions,
    build_played_action,
)
from mcp_server.rules import (
    ChessActionType,
    evaluate_rule_status,
    truncate_pv_at_terminal,
)
from mcp_server.urls import lichess_urls

# Audit M-04: explicit, versioned action policy. Single source of truth for
# when a non-move draw claim is preferred over play_move.
ACTION_POLICY_NAME = "risk_adjusted_draw_claim"
ACTION_POLICY_VERSION = "1.0.0"
# Claim is preferred when mover-POV cp is at or below this threshold AND the
# mover is materially behind OR the score is unambiguously losing. Mate wins
# always override. See mcp_server.rules.evaluate_rule_status for the live
# implementation; this constant exists for observability only.
ACTION_EQUIVALENCE_THRESHOLD_CP = 50
ACTION_MATERIAL_DOWN_THRESHOLD_CP = 200


class ActionPolicyMetadata(BaseModel):
    """Explicit metadata about the policy used to select between
    `play_move` and draw-claim actions (audit M-04)."""

    name: str = ACTION_POLICY_NAME
    version: str = ACTION_POLICY_VERSION
    equivalence_threshold_cp: int = ACTION_EQUIVALENCE_THRESHOLD_CP
    material_down_threshold_cp: int = ACTION_MATERIAL_DOWN_THRESHOLD_CP
    forced_win_overrides_claim: bool = True


class MCPEval(BaseModel):
    schema_version: str = "1.2.0"
    build_sha: str | None = None
    engine_config: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    winner: str | None = None
    cp: int | None = None
    mate: int | None = None
    # `best_move` is the engine's recommended play_move — populated whenever
    # the engine reports a best move, even if the best legal game action is a
    # claim. This is intentional: coaches and classifiers need the engine's
    # play_move reference. Safe-to-execute pointer is `executable_move`.
    best_move: str | None = None
    # AUDIT C-01: `executable_move` is null whenever the best legal action
    # is a claim_draw / claim_draw_with_intended_move. Clients that want to
    # auto-play the recommendation MUST check this field (or
    # `best_action_obj.type`) — playing `best_move` blindly after a claim
    # scenario is unsound.
    executable_move: str | None = None
    pv: list[str] = Field(default_factory=list)
    depth: int = 0
    requested_depth: int | None = None
    searched_depth: int | None = None
    can_claim_draw: bool = False
    claim_reasons: list[str] = Field(default_factory=list)
    claim_move: str | None = None
    claim_move_san: str | None = None
    claim_move_uci: str | None = None
    can_claim_now: bool = False
    claim_reasons_now: list[str] = Field(default_factory=list)
    can_claim_with_intended_move: bool = False
    claim_moves: list[str] = Field(default_factory=list)
    recommended_action: str = "play_move"
    best_action: str = "play_move"
    best_action_type: str = "play_move"
    # New: typed best_action payload (audit 10.2)
    best_action_obj: dict[str, Any] | None = None
    # New: typed legal actions list (audit 10.1)
    legal_actions: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    decision_value: dict[str, Any] | None = None
    engine_eval: dict[str, Any] | None = None
    # History completeness (audit H-01 / 10.5)
    history_dependent_status: bool = False
    lichess_url_reproduces_history: bool = True
    requires_move_stack: bool = False
    fen_sufficient_for_status: bool = True
    history_completeness: str = "incomplete"  # complete | partial | incomplete | not_required
    repetition_status: str = "none"  # "unknown" | "none" | "threefold_claimable" | "fivefold"
    # FEN canonicalization (audit L-06)
    input_fen: str | None = None
    canonical_fen: str | None = None
    fen_was_canonicalized: bool = False
    # Action policy (audit M-04)
    action_policy: ActionPolicyMetadata = Field(default_factory=ActionPolicyMetadata)
    # Legacy post-position fields (kept for callers; the new structured
    # engine_analysis/post_position objects supersede them in top_moves).
    post_terminal_status: str | None = None
    candidate_san: str | None = None
    post_can_claim_draw: bool = False
    post_can_claim_now: bool = False
    post_claim_reasons: list[str] = Field(default_factory=list)
    post_claim_moves: list[str] = Field(default_factory=list)
    # Structured post-position state (audit H-03) — for top_moves candidates,
    # the post-position that resulted from playing this candidate move.
    post_position: dict[str, Any] | None = None
    # Post-state evaluation (audit B-04/B-05) — for zeroing moves at
    # halfmove=100, the multipv root cp can be "polluted" by the engine
    # seeing the draw on the table, so the candidate is re-evaluated
    # after the move is played and the resulting cp/mate are surfaced
    # here. Mover-POV. None when no re-evaluation was performed.
    post_state_cp: int | None = None
    post_state_mate: int | None = None
    lichess_url: str | None = None
    lichess_image: str | None = None
    # Win/Draw/Loss percentages from Stockfish UCI_ShowWDL — White-POV
    # (per-mille integers 0..1000). `None` when the engine did not surface
    # WDL (e.g. UCI_ShowWDL=false or a mate/terminal path). Surfaced as
    # both a 3-tuple and a structured dict for client convenience.
    wdl: tuple[int, int, int] | None = None
    wdl_pct: dict[str, float] | None = None

    @property
    def move(self) -> str | None:
        return self.best_move

    @classmethod
    def from_eval(
        cls,
        ev: Eval,
        fen: str,
        status: str | None = None,
        board: chess.Board | None = None,
        requested_depth: int | None = None,
        history_complete: str | bool = "incomplete",
        pv_board: chess.Board | None = None,
        legal_engine_moves: list[Eval] | None = None,
    ) -> MCPEval:
        """Build an MCPEval from a Stockfish Eval.

        Args:
            ev: The engine evaluation.
            fen: The position FEN (post-candidate for top_moves, root for
                evaluate_position).
            status: Override the computed status (used for terminal).
            board: Board whose turn+mover-perspective drive rule evaluation.
                For top_moves candidates this is the POST-candidate board
                (the side-to-move is the opponent, not the candidate mover).
                When None, a board is constructed from `fen`.
            requested_depth: Echo back into the response.
            history_complete: True when the caller had the full move stack.
                False for naked FEN (audit H-01).
            pv_board: Optional board used as the starting frame for PV
                truncation. For top_moves candidates, the Stockfish PV was
                computed FROM the root, but `board` is the post-candidate
                position — the first PV move is already played. Pass the
                ROOT board here to get a non-empty PV (audit H-02).
            legal_engine_moves: Engine candidates used to populate
                `legal_actions`. Only set by callers that have them.
        """
        url, img = lichess_urls(fen)
        clean_best_move = None if ev.best_move in (None, "(none)", "none") else ev.best_move
        clean_pv = [p for p in (ev.pv or []) if p not in ("(none)", "none")]

        b = board.copy(stack=True) if board is not None else chess.Board(fen)
        sign = 1 if b.turn == chess.WHITE else -1
        mover_score = (
            sign * ev.cp
            if ev.cp is not None
            else (sign * ev.mate * 1000 if ev.mate is not None else None)
        )
        rule_status = evaluate_rule_status(
            b,
            mover_score,
            mate_for_mover=sign * ev.mate if ev.mate is not None else None,
            history_complete=history_complete,
        )

        calc_status = status or rule_status.terminal or "active"
        winner = rule_status.winner
        mate_val = ev.mate
        cp_val = ev.cp
        depth_val = ev.depth

        if rule_status.terminal is not None:
            # For a terminal ROOT position, callers build MCPEval themselves with
            # best_move=None (see _evaluate_game_position_cached). For a candidate
            # eval (top_moves), we want to preserve the move that LED to the
            # terminal — clearing it here would lose information the user wants
            # (e.g. "this candidate is a7a8b promoting to bishop → insufficient
            # material"). Truncate the PV past the terminal instead.
            #
            # mate semantics: mate_val=1 means "mate is delivered by the move
            # recorded in best_move" (Stockfish-style). mate_val=0 (or None) means
            # "this position is itself a terminal" — used by ROOT, never by a
            # candidate eval that just played the mating move.
            if rule_status.terminal == "checkmate":
                cp_val = None
                depth_val = 0
                # mate_val stays as ev.mate (Stockfish-style distance to mate from the
                # pre-move board; typically +1 for a candidate that mates this turn).
                # If ev.mate is missing for some reason, fall back to 1.
                if mate_val is None:
                    mate_val = 1
            else:
                mate_val = None
                cp_val = 0
                depth_val = 0

        # PV truncation must happen in the ROOT frame for top_moves candidates
        # — PV[0] is the candidate move (legal at root, NOT at b_cand). If a
        # pv_board is provided, use it; otherwise use b. (audit H-02)
        if clean_pv:
            pv_anchor = pv_board if pv_board is not None else b
            clean_pv = truncate_pv_at_terminal(pv_anchor, clean_pv)

        # decision_value.outcome is reported from WHITE's perspective — same convention
        # as the `cp` field (which is documented as White-POV). For a checkmate position,
        # the side-to-move is the LOSER, so:
        #   - board.turn == WHITE  → White is checkmated → outcome "loss" (for White)
        #   - board.turn == BLACK  → Black is checkmated → outcome "win"  (for White)
        # For an active position with a mate distance, sign * ev.mate < 0 means mate
        # is against the side-to-move, so White-POV outcome is determined by whose
        # turn it is.
        decision_outcome = "active"
        decision_cp = cp_val
        if calc_status in (
            "checkmate",
            "stalemate",
            "insufficient_material",
            "seventyfive_moves",
            "fivefold_repetition",
            "dead_position",
            "game_over",
        ):
            if calc_status == "checkmate":
                decision_outcome = "loss" if b.turn == chess.WHITE else "win"
                decision_cp = None
            else:
                decision_outcome = "draw"
                decision_cp = 0
        elif ev.mate is not None:
            # ev.mate is White-POV (Analyzer applies sign-flipped conversion).
            # Positive mate = White mates = "win"; negative = Black mates = "loss".
            if ev.mate > 0:
                decision_outcome = "win"
                decision_cp = None
            else:  # ev.mate < 0
                decision_outcome = "loss"
                decision_cp = None
        elif rule_status.can_claim_now and rule_status.recommended_action == "claim_draw":
            decision_outcome = "draw"
            decision_cp = 0
        elif (
            rule_status.can_claim_with_intended_move
            and rule_status.recommended_action == "claim_draw_with_intended_move"
        ):
            decision_outcome = "draw"
            decision_cp = 0

        # Win/Draw/Loss in per-mille. None when Stockfish didn't surface
        # WDL (UCI_ShowWDL=false or mate-distance output only). Tuple is
        # White-POV (W D L); wdl_pct is a client-friendly dict with the
        # same values scaled to 0-100.
        wdl_tuple: tuple[int, int, int] | None = ev.wdl
        wdl_pct_dict: dict[str, float] | None = (
            {
                "win": wdl_tuple[0] / 10.0,
                "draw": wdl_tuple[1] / 10.0,
                "loss": wdl_tuple[2] / 10.0,
            }
            if wdl_tuple is not None
            else None
        )

        engine_eval_dict = {
            "cp": ev.cp,
            "mate": ev.mate,
            "best_move": clean_best_move,
            "pv": clean_pv,
            "depth": depth_val,
            "wdl": wdl_tuple,
            "wdl_pct": wdl_pct_dict,
        }

        decision_val_dict = {
            "outcome": decision_outcome,
            "cp_equivalent": decision_cp,
            "best_action": rule_status.recommended_action,
            "perspective": "white",
        }

        # AUDIT C-01: a client must NEVER execute `best_move` when the best
        # legal action is a claim. We surface that fact two ways:
        #   1. `executable_move` is null when the best action is a claim —
        #      this is the safe-to-execute pointer.
        #   2. `best_action_obj.type` is the typed discriminator.
        # `best_move` is still populated for backward compat AND because the
        # classifier / coach tooling needs the engine's play_move reference.
        if rule_status.recommended_action in ("claim_draw", "claim_draw_with_intended_move"):
            executable_move: str | None = None
        else:
            executable_move = clean_best_move

        best_action_payload = build_best_action(
            recommended_action=rule_status.recommended_action,
            rule_status=rule_status,
            engine_eval=ev,
            board=b,
            sign=sign,
        )
        legal_actions_payload = build_legal_actions(
            rule_status=rule_status,
            engine_eval=ev,
            board=b,
            legal_engine_moves=legal_engine_moves,
        )

        return cls(
            status=calc_status,
            winner=winner,
            cp=cp_val,
            mate=mate_val,
            best_move=clean_best_move,
            executable_move=executable_move,
            pv=clean_pv,
            depth=depth_val,
            requested_depth=requested_depth if requested_depth is not None else depth_val,
            searched_depth=depth_val,
            can_claim_draw=rule_status.can_claim_draw,
            claim_reasons=rule_status.claim_reasons,
            claim_move=rule_status.claim_move,
            claim_move_san=rule_status.claim_move_san,
            claim_move_uci=rule_status.claim_move_uci,
            can_claim_now=rule_status.can_claim_now,
            claim_reasons_now=rule_status.claim_reasons_now,
            can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
            claim_moves=rule_status.claim_moves,
            post_can_claim_draw=rule_status.can_claim_draw,
            post_can_claim_now=rule_status.can_claim_now,
            post_claim_reasons=rule_status.claim_reasons,
            post_claim_moves=rule_status.claim_moves,
            recommended_action=rule_status.recommended_action,
            wdl=wdl_tuple,
            wdl_pct=wdl_pct_dict,
            best_action=rule_status.recommended_action,
            best_action_type=rule_status.recommended_action,
            best_action_obj=best_action_payload,
            legal_actions=legal_actions_payload,
            decision_value=decision_val_dict,
            engine_eval=engine_eval_dict,
            history_dependent_status=rule_status.history_dependent_status,
            lichess_url_reproduces_history=rule_status.fen_sufficient_for_status,
            requires_move_stack=rule_status.requires_move_stack,
            fen_sufficient_for_status=rule_status.fen_sufficient_for_status,
            history_completeness=rule_status.history_completeness,
            repetition_status=rule_status.repetition_status,
            canonical_fen=fen,
            fen_was_canonicalized=False,
            lichess_url=url,
            lichess_image=img,
        )


class PlayedMoveScore(BaseModel):
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
    missed_draw_claim: bool = False
    conceded_draw_claim: bool = False
    claim_reason: str | None = None
    claim_move: str | None = None
    can_claim_now: bool = False
    can_claim_with_intended_move: bool = False
    claim_moves: list[str] = Field(default_factory=list)


def score_played_move(
    board_before: chess.Board,
    move: chess.Move,
    eval_before: MCPEval,
    eval_after: MCPEval,
    board_after: chess.Board | None = None,
    eval_played: MCPEval | None = None,
    action_type: Literal["play_move", "claim_draw", "claim_draw_with_intended_move"] = "play_move",
) -> PlayedMoveScore:
    """Unified, rule-aware single source of truth for move grading and loss across all tools."""
    if action_type not in {"play_move", "claim_draw", "claim_draw_with_intended_move"}:
        raise ValueError(f"INVALID_ACTION_TYPE: {action_type}")
    if board_after is None:
        board_after = board_before.copy(stack=True)
        board_after.push(move)

    is_white = board_before.turn == chess.WHITE
    sign = 1 if is_white else -1

    # Audit P0/P1: for draw-claim actions, the action IS the claim, not the
    # `move` argument. `claim_draw` arrives here with a placeholder legal move
    # (server substitutes when no move is parsed), and `claim_draw_with_intended_move`
    # arrives with a non-resetting claim move. Neither is the engine's "best
    # legal attempt" in the play_move sense — the player chose the claim, not
    # the move. Force `is_best_engine_move=False` so callers never see a claim
    # action masquerading as the engine's best play.
    is_best_engine_move = bool(
        action_type == "play_move"
        and eval_before.best_move
        and move.uci().lower() == eval_before.best_move.lower()
    )

    eval_move_eval = eval_played if eval_played is not None else eval_after

    before_mover = sign * (eval_before.cp if eval_before.cp is not None else 0)
    if eval_before.cp is None and eval_before.mate is not None:
        before_mover = 10000 if (sign * eval_before.mate > 0) else -10000

    after_mover = sign * (eval_move_eval.cp if eval_move_eval.cp is not None else 0)
    if eval_move_eval.cp is None and eval_move_eval.mate is not None:
        after_mover = 10000 if (sign * eval_move_eval.mate > 0) else -10000

    mover_mate_before = sign * eval_before.mate if eval_before.mate is not None else None
    mover_mate_after = sign * eval_move_eval.mate if eval_move_eval.mate is not None else None

    before_mover_score = (
        before_mover
        if eval_before.mate is None
        else (mover_mate_before * 1000 if mover_mate_before is not None else 0)
    )
    history_state = eval_before.history_completeness
    rule_before = evaluate_rule_status(
        board_before,
        mover_score=before_mover_score,
        mate_for_mover=mover_mate_before,
        history_complete=history_state,
    )
    rule_after = evaluate_rule_status(board_after, history_complete=history_state)

    canonical_best_action = eval_before.best_action or rule_before.recommended_action

    # 1. Delivered Checkmate (mover wins). Only a play_move action actually
    # executes ``move``. Draw claims are procedural actions made instead of
    # playing the supplied move, so a mating placeholder/intended move must
    # not cause a claim to be misclassified as a mating play.
    if (
        action_type == "play_move"
        and board_after.is_checkmate()
        and board_after.turn != board_before.turn
    ):
        return PlayedMoveScore(
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            raw_centipawn_loss=0,
            raw_centipawn_delta=0,
            mate_distance_loss=0,
            effective_loss=0,
            loss_kind="none",
            is_best_engine_move=is_best_engine_move,
            win_loss=0.0,
            best_action=canonical_best_action,
            is_best_action=True,
            action_equivalent=False,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=None,
            claim_move=None,
            can_claim_now=rule_before.can_claim_now,
            can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
            claim_moves=rule_before.claim_moves,
        )

    # 2. Checkmate delivered against mover. As above, post-move terminal
    # consequences apply only when the requested action is play_move.
    if (
        action_type == "play_move"
        and board_after.is_checkmate()
        and board_after.turn == board_before.turn
    ):
        return PlayedMoveScore(
            move_class=MoveClass.BLUNDER,
            centipawn_loss=1000,
            raw_centipawn_loss=1000,
            raw_centipawn_delta=1000,
            mate_distance_loss=None,
            effective_loss=1000,
            loss_kind="mate_transition",
            outcome_penalty=1000,
            is_best_engine_move=False,
            win_loss=100.0,
            best_action=canonical_best_action,
            is_best_action=False,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=None,
            claim_move=None,
            can_claim_now=rule_before.can_claim_now,
            can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
            claim_moves=rule_before.claim_moves,
        )

    # Check terminal draw reached by move
    is_auto_terminal_draw = bool(
        rule_after.terminal
        in (
            "stalemate",
            "insufficient_material",
            "seventyfive_moves",
            "fivefold_repetition",
            "dead_position",
        )
    )

    # Baseline mover score with draw claim capability: if mover could have claimed draw, baseline is at least 0
    can_claim_before = bool(rule_before.can_claim_draw or eval_before.can_claim_draw)
    baseline_mover = max(before_mover, 0) if can_claim_before else before_mover

    raw_board_delta = before_mover - after_mover
    raw_cpl = 0 if is_best_engine_move else max(0, raw_board_delta)

    # Procedural draw claim action — but ONLY honor it when the claim is legally
    # available AND forfeiting it is not a worse-than-claim outcome. A "claim" while
    # the mover has a forced mate (e.g. Qg7# at halfmove 99) is a blundered win, not
    # a claim. Likewise a claim with no immediate draw on the board is not a claim
    # at all — it's a move that loses the option.
    if action_type in ("claim_draw", "claim_draw_with_intended_move"):
        is_claim_now_action = action_type in ("claim_draw", ChessActionType.CLAIM_DRAW_NOW.value)
        played_uci = move.uci().lower()

        claim_legal = (is_claim_now_action and rule_before.can_claim_now) or (
            not is_claim_now_action
            and rule_before.can_claim_with_intended_move
            and played_uci in [u.lower() for u in rule_before.intended_claim_ucis]
        )
        if not claim_legal:
            raise ValueError("ILLEGAL_ACTION: requested draw claim is not legally available")

        if is_claim_now_action:
            reasons = rule_before.claim_reasons_now
        else:
            reason_map = rule_before.intended_claim_reasons_by_uci
            reasons = reason_map.get(move.uci(), [])
        claim_r = reasons[0] if reasons else None

        is_mover_forced_win = (
            mover_mate_before is not None and mover_mate_before > 0
        ) or before_mover >= 200
        if is_mover_forced_win:
            return PlayedMoveScore(
                move_class=MoveClass.BLUNDER,
                centipawn_loss=None,
                raw_centipawn_loss=None,
                raw_centipawn_delta=0,
                mate_distance_loss=None,
                effective_loss=1000,
                loss_kind="outcome_penalty",
                outcome_penalty=1000,
                is_best_engine_move=is_best_engine_move,
                win_loss=50.0,
                best_action=canonical_best_action,
                is_best_action=False,
                action_equivalent=False,
                missed_draw_claim=False,
                conceded_draw_claim=False,
                claim_reason=claim_r,
                claim_move=rule_before.claim_move,
                can_claim_now=rule_before.can_claim_now,
                can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                claim_moves=rule_before.claim_moves,
            )

        return PlayedMoveScore(
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            raw_centipawn_loss=0,
            raw_centipawn_delta=0,
            mate_distance_loss=0,
            effective_loss=0,
            loss_kind="none",
            is_best_engine_move=is_best_engine_move,
            win_loss=0.0,
            best_action=canonical_best_action,
            is_best_action=canonical_best_action == action_type,
            action_equivalent=canonical_best_action != action_type,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=claim_r,
            claim_move=rule_before.claim_move,
            can_claim_now=rule_before.can_claim_now,
            can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
            claim_moves=rule_before.claim_moves,
        )

    # If position before was winning and move blundered into an automatic draw
    is_before_winning = (mover_mate_before is not None and mover_mate_before > 0) or (
        before_mover >= 200
    )
    if is_auto_terminal_draw:
        if is_before_winning:
            eff_loss = max(300, min(1000, before_mover if before_mover > 0 else 1000))
            return PlayedMoveScore(
                move_class=MoveClass.BLUNDER,
                centipawn_loss=None if mover_mate_before is not None else raw_cpl,
                raw_centipawn_loss=None if mover_mate_before is not None else raw_cpl,
                raw_centipawn_delta=raw_board_delta,
                mate_distance_loss=None,
                effective_loss=eff_loss,
                loss_kind="blundered_draw",
                outcome_penalty=eff_loss,
                is_best_engine_move=False,
                win_loss=50.0,
                best_action=canonical_best_action,
                is_best_action=False,
                action_equivalent=False,
                missed_draw_claim=False,
                conceded_draw_claim=False,
                claim_reason=None,
                claim_move=None,
                can_claim_now=rule_before.can_claim_now,
                can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                claim_moves=rule_before.claim_moves,
            )
        else:
            return PlayedMoveScore(
                move_class=MoveClass.BEST if is_best_engine_move else MoveClass.GOOD,
                centipawn_loss=0,
                raw_centipawn_loss=raw_cpl,
                raw_centipawn_delta=raw_board_delta,
                mate_distance_loss=None,
                effective_loss=0,
                loss_kind="none",
                is_best_engine_move=is_best_engine_move,
                win_loss=0.0,
                best_action=canonical_best_action,
                is_best_action=True,
                action_equivalent=canonical_best_action
                in ("claim_draw", "claim_draw_with_intended_move"),
                missed_draw_claim=False,
                conceded_draw_claim=False,
                claim_reason=None,
                claim_move=None,
                can_claim_now=rule_before.can_claim_now,
                can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                claim_moves=rule_before.claim_moves,
            )

    # Opponent in board_after can claim immediate draw
    opponent_will_claim = bool(
        (rule_after.can_claim_now or eval_after.can_claim_now or eval_after.can_claim_draw)
        and (is_before_winning or before_mover >= 200)
    )
    if opponent_will_claim:
        eff_loss = max(500, min(1000, before_mover if before_mover > 0 else 1000))
        claim_r = (
            rule_after.claim_reasons_now[0]
            if rule_after.claim_reasons_now
            else (
                eval_after.claim_reasons[0] if eval_after.claim_reasons else "threefold_repetition"
            )
        )
        return PlayedMoveScore(
            move_class=MoveClass.BLUNDER,
            centipawn_loss=raw_cpl,
            raw_centipawn_loss=raw_cpl,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=None,
            effective_loss=eff_loss,
            loss_kind="conceded_draw",
            rule_action_penalty=eff_loss,
            is_best_engine_move=False,
            win_loss=50.0,
            best_action=canonical_best_action,
            is_best_action=False,
            action_equivalent=False,
            missed_draw_claim=False,
            conceded_draw_claim=True,
            claim_reason=claim_r,
            claim_move=None,
            can_claim_now=rule_before.can_claim_now,
            can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
            claim_moves=rule_before.claim_moves,
        )

    # Material balance for mover
    piece_vals = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
    }
    mover_mat = sum(
        len(board_before.pieces(pt, board_before.turn)) * val for pt, val in piece_vals.items()
    )
    opp_mat = sum(
        len(board_before.pieces(pt, not board_before.turn)) * val for pt, val in piece_vals.items()
    )
    is_down_material = opp_mat - mover_mat >= 200

    # Draw claim preservation check:
    is_after_winning = (mover_mate_after is not None and mover_mate_after > 0) or (
        after_mover >= 100
    )
    is_after_losing = (
        after_mover <= -100 or (mover_mate_after is not None and mover_mate_after < 0)
    ) or is_down_material

    # Optimal draw claim forfeiture check:
    # Applies when mover had a draw claim opportunity (current or intended) from a losing position/claim recommendation,
    # but played a move leaving them in a lost position.
    is_mover_forced_win = (mover_mate_before is not None and mover_mate_before > 0) or (
        before_mover >= 100
    )
    optimal_claim_recommended = (
        not is_mover_forced_win
        and not is_after_winning
        and not is_auto_terminal_draw
        and (
            canonical_best_action in ("claim_draw", "claim_draw_with_intended_move")
            or (
                can_claim_before
                and (
                    before_mover <= -100
                    or is_down_material
                    or (mover_mate_before is not None and mover_mate_before < 0)
                )
            )
        )
    )

    if optimal_claim_recommended and not is_auto_terminal_draw:
        decision_before_draw = bool(
            (eval_before.decision_value and eval_before.decision_value.get("outcome") == "draw")
            or (rule_before.can_claim_draw and before_mover <= 100)
        )
        decision_after_draw = bool(
            eval_after.decision_value and eval_after.decision_value.get("outcome") == "draw"
        )
        draw_preserved = bool(
            decision_before_draw and decision_after_draw and is_best_engine_move and raw_cpl == 0
        )

        if draw_preserved:
            claim_r = (
                rule_before.claim_reasons[0]
                if rule_before.claim_reasons
                else (eval_before.claim_reasons[0] if eval_before.claim_reasons else None)
            )
            return PlayedMoveScore(
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                raw_centipawn_loss=0,
                raw_centipawn_delta=raw_board_delta,
                mate_distance_loss=None,
                effective_loss=0,
                loss_kind="none",
                is_best_engine_move=True,
                win_loss=0.0,
                best_action=canonical_best_action,
                is_best_action=canonical_best_action == "play_move",
                action_equivalent=True,
                missed_draw_claim=bool(
                    is_down_material
                    and canonical_best_action in ("claim_draw", "claim_draw_with_intended_move")
                    and action_type == "play_move"
                ),
                conceded_draw_claim=False,
                claim_reason=claim_r,
                claim_move=rule_before.claim_move,
                can_claim_now=rule_before.can_claim_now,
                can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                claim_moves=rule_before.claim_moves,
            )

        if is_after_losing and not is_after_winning:
            loss_val = max(abs(before_mover), abs(after_mover))
            eff_loss = max(
                300,
                min(
                    1000,
                    loss_val
                    if loss_val > 0
                    else (opp_mat - mover_mat if is_down_material else 500),
                ),
            )
            final_class = MoveClass.BLUNDER if eff_loss >= 300 else MoveClass.MISTAKE
            claim_r = (
                rule_before.claim_reasons[0]
                if rule_before.claim_reasons
                else (eval_before.claim_reasons[0] if eval_before.claim_reasons else None)
            )
            return PlayedMoveScore(
                move_class=final_class,
                centipawn_loss=raw_cpl,
                raw_centipawn_loss=raw_cpl,
                raw_centipawn_delta=raw_board_delta,
                mate_distance_loss=None,
                effective_loss=eff_loss,
                loss_kind="draw_claim_forfeit",
                rule_action_penalty=eff_loss,
                is_best_engine_move=is_best_engine_move,
                win_loss=50.0,
                best_action=canonical_best_action,
                is_best_action=False,
                action_equivalent=False,
                missed_draw_claim=True,
                conceded_draw_claim=False,
                claim_reason=claim_r,
                claim_move=rule_before.claim_move,
                can_claim_now=rule_before.can_claim_now,
                can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                claim_moves=rule_before.claim_moves,
            )

    # Mate evaluation logic
    if mover_mate_before is not None and mover_mate_after is not None:
        if mover_mate_before > 0:
            if mover_mate_after > 0:
                if board_after.is_checkmate():
                    mate_dist_loss = 0
                else:
                    mate_dist_loss = max(0, (abs(mover_mate_after) + 1) - abs(mover_mate_before))

                if is_best_engine_move or mate_dist_loss == 0:
                    final_class = MoveClass.BEST
                    eff_loss = 0
                    w_loss = 0.0
                elif mate_dist_loss <= 1:
                    final_class = MoveClass.GOOD
                    eff_loss = 50
                    w_loss = 0.5
                elif mate_dist_loss <= 2:
                    final_class = MoveClass.INACCURACY
                    eff_loss = 150
                    w_loss = 2.0
                else:
                    final_class = MoveClass.MISTAKE
                    eff_loss = 300
                    w_loss = min(20.0, float(mate_dist_loss * 2.0))

                return PlayedMoveScore(
                    move_class=final_class,
                    centipawn_loss=0,
                    raw_centipawn_loss=0,
                    raw_centipawn_delta=0,
                    mate_distance_loss=mate_dist_loss,
                    effective_loss=eff_loss,
                    loss_kind="mate_distance" if mate_dist_loss > 0 else "none",
                    mate_distance_penalty=eff_loss if mate_dist_loss > 0 else None,
                    is_best_engine_move=is_best_engine_move,
                    win_loss=w_loss,
                    best_action=canonical_best_action,
                    is_best_action=is_best_engine_move
                    or (canonical_best_action == "play_move" and mate_dist_loss == 0),
                    missed_draw_claim=False,
                    conceded_draw_claim=False,
                    claim_reason=None,
                    can_claim_now=rule_before.can_claim_now,
                    can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                    claim_moves=rule_before.claim_moves,
                )
            elif mover_mate_after == 0:
                # Mate delivered by mover
                return PlayedMoveScore(
                    move_class=MoveClass.BEST,
                    centipawn_loss=0,
                    raw_centipawn_loss=0,
                    raw_centipawn_delta=0,
                    mate_distance_loss=0,
                    effective_loss=0,
                    loss_kind="none",
                    is_best_engine_move=is_best_engine_move,
                    win_loss=0.0,
                    best_action=canonical_best_action,
                    is_best_action=True,
                    missed_draw_claim=False,
                    conceded_draw_claim=False,
                    claim_reason=None,
                    claim_move=None,
                    can_claim_now=rule_before.can_claim_now,
                    can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                    claim_moves=rule_before.claim_moves,
                )
            else:
                return PlayedMoveScore(
                    move_class=MoveClass.BLUNDER,
                    centipawn_loss=None,
                    raw_centipawn_loss=None,
                    raw_centipawn_delta=None,
                    mate_distance_loss=None,
                    effective_loss=1000,
                    loss_kind="mate_transition",
                    outcome_penalty=1000,
                    is_best_engine_move=False,
                    win_loss=100.0,
                    best_action=canonical_best_action,
                    is_best_action=False,
                    missed_draw_claim=False,
                    conceded_draw_claim=False,
                    claim_reason=None,
                    can_claim_now=rule_before.can_claim_now,
                    can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                    claim_moves=rule_before.claim_moves,
                )
        elif mover_mate_before < 0 and mover_mate_after < 0:
            # Defender perspective: allowing faster mate against self is a loss of resistance
            defender_resistance_loss = max(0, abs(mover_mate_before) - abs(mover_mate_after))
            if is_best_engine_move or defender_resistance_loss == 0:
                final_class = MoveClass.BEST
                eff_loss = 0
                w_loss = 0.0
            elif defender_resistance_loss == 1:
                final_class = MoveClass.INACCURACY
                eff_loss = 150
                w_loss = 2.0
            else:
                final_class = (
                    MoveClass.BLUNDER if defender_resistance_loss >= 3 else MoveClass.MISTAKE
                )
                eff_loss = 500 if defender_resistance_loss >= 3 else 300
                w_loss = min(20.0, float(defender_resistance_loss * 5.0))

            return PlayedMoveScore(
                move_class=final_class,
                centipawn_loss=0,
                raw_centipawn_loss=0,
                raw_centipawn_delta=0,
                mate_distance_loss=defender_resistance_loss,
                effective_loss=eff_loss,
                loss_kind="mate_distance" if defender_resistance_loss > 0 else "none",
                mate_distance_penalty=eff_loss if defender_resistance_loss > 0 else None,
                is_best_engine_move=is_best_engine_move,
                win_loss=w_loss,
                best_action=canonical_best_action,
                is_best_action=is_best_engine_move and canonical_best_action == "play_move",
                missed_draw_claim=False,
                conceded_draw_claim=False,
                claim_reason=None,
                claim_move=rule_before.claim_move,
                can_claim_now=rule_before.can_claim_now,
                can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                claim_moves=rule_before.claim_moves,
            )

    # Mate to CP horizon (e.g. Qh4+)
    if mover_mate_before is not None and mover_mate_before > 0 and eval_move_eval.mate is None:
        if after_mover >= 400:
            final_class = MoveClass.BEST if is_best_engine_move else MoveClass.INACCURACY
            eff_loss = 0 if is_best_engine_move else 150
            w_loss = 0.0 if is_best_engine_move else 2.0
            return PlayedMoveScore(
                move_class=final_class,
                centipawn_loss=0,
                raw_centipawn_loss=0,
                raw_centipawn_delta=0,
                mate_distance_loss=None,
                effective_loss=eff_loss,
                loss_kind="mate_distance" if not is_best_engine_move else "none",
                mate_distance_penalty=eff_loss if not is_best_engine_move else None,
                is_best_engine_move=is_best_engine_move,
                win_loss=w_loss,
                best_action=canonical_best_action,
                is_best_action=is_best_engine_move,
                missed_draw_claim=False,
                conceded_draw_claim=False,
                claim_reason=None,
                can_claim_now=rule_before.can_claim_now,
                can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                claim_moves=rule_before.claim_moves,
            )
        else:
            wp_after = _win_pct(after_mover)
            win_loss = max(0.0, 100.0 - wp_after)
            final_class = MoveClass.BLUNDER if win_loss >= 20.0 else MoveClass.MISTAKE
            eff_loss = 1000 if final_class == MoveClass.BLUNDER else 300
            return PlayedMoveScore(
                move_class=final_class,
                centipawn_loss=raw_cpl,
                raw_centipawn_loss=raw_cpl,
                raw_centipawn_delta=raw_board_delta,
                mate_distance_loss=None,
                effective_loss=eff_loss,
                loss_kind="outcome_penalty",
                outcome_penalty=eff_loss,
                is_best_engine_move=False,
                win_loss=win_loss,
                best_action=canonical_best_action,
                is_best_action=False,
                missed_draw_claim=False,
                conceded_draw_claim=False,
                claim_reason=None,
                can_claim_now=rule_before.can_claim_now,
                can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                claim_moves=rule_before.claim_moves,
            )

    # CP to Mate
    if eval_before.mate is None and mover_mate_after is not None:
        if mover_mate_after < 0:
            wp_before = _win_pct(baseline_mover)
            win_loss = max(0.0, wp_before - 0.0)
            return PlayedMoveScore(
                move_class=MoveClass.BLUNDER,
                centipawn_loss=None,
                raw_centipawn_loss=None,
                raw_centipawn_delta=None,
                mate_distance_loss=None,
                effective_loss=1000,
                loss_kind="mate_transition",
                outcome_penalty=1000,
                is_best_engine_move=False,
                win_loss=win_loss,
                best_action=canonical_best_action,
                is_best_action=False,
                missed_draw_claim=optimal_claim_recommended,
                conceded_draw_claim=False,
                claim_reason=rule_before.claim_reasons[0]
                if (optimal_claim_recommended and rule_before.claim_reasons)
                else None,
                can_claim_now=rule_before.can_claim_now,
                can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                claim_moves=rule_before.claim_moves,
            )
        else:
            return PlayedMoveScore(
                move_class=MoveClass.BEST if is_best_engine_move else MoveClass.GOOD,
                centipawn_loss=0,
                raw_centipawn_loss=0,
                raw_centipawn_delta=0,
                mate_distance_loss=0,
                effective_loss=0,
                loss_kind="none",
                is_best_engine_move=is_best_engine_move,
                win_loss=0.0,
                best_action=canonical_best_action,
                is_best_action=is_best_engine_move,
                action_equivalent=is_best_engine_move
                and canonical_best_action in ("claim_draw", "claim_draw_with_intended_move"),
                missed_draw_claim=False,
                conceded_draw_claim=False,
                claim_reason=None,
                can_claim_now=rule_before.can_claim_now,
                can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                claim_moves=rule_before.claim_moves,
            )

    # Standard CP evaluations
    if is_best_engine_move:
        is_claim_best = canonical_best_action in ("claim_draw", "claim_draw_with_intended_move")
        claim_r = (
            rule_before.claim_reasons[0]
            if rule_before.claim_reasons
            else (eval_before.claim_reasons[0] if eval_before.claim_reasons else None)
        )
        return PlayedMoveScore(
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            raw_centipawn_loss=0,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=None,
            effective_loss=0,
            loss_kind="none",
            is_best_engine_move=is_best_engine_move,
            win_loss=0.0,
            best_action=canonical_best_action,
            is_best_action=not is_claim_best,
            action_equivalent=is_claim_best,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=claim_r,
            claim_move=rule_before.claim_move,
            can_claim_now=rule_before.can_claim_now,
            can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
            claim_moves=rule_before.claim_moves,
        )

    eff_delta = baseline_mover - after_mover
    eff_loss = max(0, eff_delta)
    wp_before = _win_pct(baseline_mover)
    wp_after = _win_pct(after_mover)
    win_loss = max(0.0, wp_before - wp_after)

    # Decisive position saturation
    if (wp_before >= 95.0 and wp_after >= 95.0) or (baseline_mover >= 400 and after_mover >= 400):
        if win_loss < 4.0:
            final_class = MoveClass.BEST if is_best_engine_move else MoveClass.GOOD
            final_cpl = min(eff_loss, 15)
        elif win_loss < 8.0:
            final_class = MoveClass.GOOD
            final_cpl = min(eff_loss, 45)
        else:
            classified = classify_centipawn_loss(eff_loss)
            final_class = (
                MoveClass.GOOD
                if classified == MoveClass.BEST and not is_best_engine_move
                else classified
            )
            final_cpl = min(eff_loss, 1000)
    elif (wp_before >= 90.0 and wp_after >= 90.0) or (baseline_mover >= 300 and after_mover >= 300):
        if win_loss < 4.0:
            final_class = MoveClass.BEST if is_best_engine_move else MoveClass.GOOD
            final_cpl = min(eff_loss, 20)
        elif win_loss < 8.0:
            final_class = MoveClass.GOOD
            final_cpl = min(eff_loss, 50)
        else:
            classified = classify_centipawn_loss(eff_loss)
            final_class = (
                MoveClass.GOOD
                if classified == MoveClass.BEST and not is_best_engine_move
                else classified
            )
            final_cpl = min(eff_loss, 1000)
    else:
        classified = classify_centipawn_loss(eff_loss)
        final_class = (
            MoveClass.GOOD
            if classified == MoveClass.BEST and not is_best_engine_move
            else classified
        )
        final_cpl = min(eff_loss, 1000)

    # Hard invariant safeguard: played move matching engine best move can NEVER be mistake or blunder
    if is_best_engine_move and final_class in (MoveClass.MISTAKE, MoveClass.BLUNDER):
        final_class = MoveClass.BEST
        final_cpl = 0
        win_loss = 0.0

    return PlayedMoveScore(
        move_class=final_class,
        centipawn_loss=raw_cpl,
        raw_centipawn_loss=raw_cpl,
        raw_centipawn_delta=raw_board_delta,
        mate_distance_loss=None,
        effective_loss=final_cpl,
        loss_kind="engine_cp" if final_cpl > 0 else "none",
        engine_cp_loss=final_cpl if final_cpl > 0 else None,
        is_best_engine_move=is_best_engine_move,
        win_loss=win_loss,
        best_action=canonical_best_action,
        is_best_action=is_best_engine_move and canonical_best_action == "play_move",
        action_equivalent=is_best_engine_move
        and canonical_best_action in ("claim_draw", "claim_draw_with_intended_move"),
        missed_draw_claim=False,
        conceded_draw_claim=False,
        claim_reason=rule_before.claim_reasons[0]
        if rule_before.claim_reasons
        else (eval_before.claim_reasons[0] if eval_before.claim_reasons else None),
        claim_move=rule_before.claim_move,
        can_claim_now=rule_before.can_claim_now,
        can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
        claim_moves=rule_before.claim_moves,
    )


class MCPMoveAnalysis(BaseModel):
    model_config = {"populate_by_name": True}
    schema_version: str = "1.2.0"
    played: str
    played_san: str | None = None
    move_class: MoveClass
    # Renamed for clarity (audit M-02): the engine's best move is a separate
    # concept from the best legal game action. `is_engine_best` is kept as the
    # field name on the wire for backward compatibility; new code should read
    # `is_best_engine_move`.
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
    # Renamed for clarity (audit M-02): best legal game action, distinct from
    # the engine's best move (which is just the play_move scoring reference).
    is_best_action: bool = True
    action_equivalent: bool = False
    # Typed action payloads (audit 10.2)
    played_action_obj: dict[str, Any] | None = None
    best_action_obj: dict[str, Any] | None = None
    # Audit M-04: explicit policy metadata
    action_policy: ActionPolicyMetadata = Field(default_factory=ActionPolicyMetadata)
    missed_draw_claim: bool = False
    conceded_draw_claim: bool = False
    claim_reason: str | None = None
    claim_move: str | None = None
    can_claim_now: bool = False
    can_claim_with_intended_move: bool = False
    claim_moves: list[str] = Field(default_factory=list)
    classification_verified: bool = False

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

        score = score_played_move(b_bef, m, eval_bef, eval_aft, b_aft, action_type=action_type)

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
            # Backward compat + new field (audit M-02)
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
    white_acpl: float | None = Field(
        default=None,
        description="Average effective centipawn loss across all White plies (including mate transitions and draw claim forfeitures).",
    )
    black_acpl: float | None = Field(
        default=None,
        description="Average effective centipawn loss across all Black plies (including mate transitions and draw claim forfeitures).",
    )
    white_raw_acpl: float | None = Field(
        default=None, description="Average pure centipawn loss excluding mate transitions."
    )
    black_raw_acpl: float | None = Field(
        default=None, description="Average pure centipawn loss excluding mate transitions."
    )
    white_effective_acpl: float | None = Field(
        default=None, description="Average effective centipawn loss across all White plies."
    )
    black_effective_acpl: float | None = Field(
        default=None, description="Average effective centipawn loss across all Black plies."
    )
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
    # Build identity for observability / staleness debugging. service_version
    # alone ("0.1.0") does not identify a deployment; build_sha does.
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
    # NEW: typed best_action and legal_actions (audit 10.1, 10.2)
    best_action_obj: dict[str, Any] | None = None
    legal_actions: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    # History completeness (audit H-01)
    history_completeness: str = "complete"
    repetition_status: str = "none"
    # Counts (audit L-03)
    requested_depth: int | None = None
    searched_depth: int | None = None
    requested_n: int | None = None
    clamped_n: int | None = None
    returned_n: int | None = None
    legal_move_count: int | None = None
    # Build identity / observability
    engine: str = "Stockfish"
    engine_version: str | None = None
    service_version: str = "0.1.0"
    build_sha: str | None = None
    engine_config: dict[str, Any] = Field(default_factory=dict)
    action_policy: ActionPolicyMetadata = Field(default_factory=ActionPolicyMetadata)
    canonical_fen: str | None = None
    fen_was_canonicalized: bool = False
    result: list[MCPEval] = Field(
        default_factory=list[MCPEval],
        description=(
            "Ranked play_move candidates (best first). Candidate best_move/pv and "
            "engine cp/mate retain the root MultiPV action frame (PV[0] is the "
            "candidate; mating moves may retain root mate distance), while "
            "canonical_fen, terminal/rule fields and post_position describe the "
            "resulting board. Claim actions are reported separately at the outer "
            "level (best_action_obj / legal_actions)."
        ),
    )

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
