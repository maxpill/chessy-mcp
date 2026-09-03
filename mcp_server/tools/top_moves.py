"""``top_moves`` MCP tool.

Extracted from ``mcp_server.server``. Top-N MultiPV candidate moves with
draw-claim projection per candidate, post-position evaluation, and rule-
aware legal action surface.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal, cast

import chess

from core.engines.types import Eval
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from mcp_server.actions import build_best_action, build_legal_actions
from mcp_server.cache import top_moves_cache_key
from mcp_server.engine import (
    _build_identity,
    _cache,
    _evaluate_game_position_cached,
    _get_analyzer_pool,
    _single_flight,
)
from mcp_server.metrics import metrics
from mcp_server.models import MCPEval, TopMovesResult
from mcp_server.parsers import _build_board_with_metadata, _history_provenance_for_input
from mcp_server.rules import (
    choose_recommended_action,
    evaluate_rule_status,
    is_locked_dead_position,
    is_terminal_position,
    truncate_pv_at_terminal,
    validate_mating_possibility,
)
from mcp_server.server import mcp
from mcp_server.tools._common import (
    VERBOSITY_COMPACT,
    VERBOSITY_FULL,
    _compact_mcpeval,
    _format_exception,
    _resolve_verbosity,
    _tool_error,
    _validate_requested_depth,
    error_code_for,
)

log = logging.getLogger("chessy_mcp.top_moves")


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def top_moves(
    fen: str,
    moves: list[str] | None = None,
    n: int = 3,
    depth: int = 14,
    strict: bool = False,
    verbosity: str | None = None,
    ctx: Context | None = None,
) -> TopMovesResult:
    """Get the top N candidate moves for a position, ranked best first.

    Args:
        fen: FEN or PGN string for the position.
        moves: Optional UCI or SAN moves to replay onto the position first.
        n: Number of candidates to return (default 3, clamped 1-20).
        depth: Stockfish search depth (default 14, clamped 1-30).
        strict: When True, reject non-canonical SAN syntax or move numbers (default False).

    Returns:
        TopMovesResult object with `status`, `winner`, `recommended_action`,
        `best_action_obj` (typed discriminated union per audit 10.2),
        `legal_actions` (typed list of legal actions), and a `result` array
        of candidate MCPEval objects ranked best first.

        IMPORTANT (audit C-02 / H-03):
          Each candidate in `result` represents a `play_move` action. Its
          `best_move`, `pv`, and engine `cp`/`mate` retain the root MultiPV
          action value and notation frame, so PV[0] is the candidate move and
          a mating candidate may retain Stockfish's root mate distance (e.g. 1).
          The candidate `canonical_fen`, terminal status, winner, rule fields,
          and `post_position` describe the board AFTER that candidate is played.
          Automatic terminal draws normalize candidate `cp` to 0. Draw-claim
          actions are reported separately via outer `best_action_obj` and
          `legal_actions`; they are not mixed into the MultiPV candidate list.

        For terminal positions (checkmate, stalemate, insufficient material,
        repetition, 75-move rule), returns TopMovesResult with status and
        empty `result: []`.
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="top_moves")
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    raw_requested_n = n
    clamped_n = max(1, min(n, 20))
    n = clamped_n
    try:
        verbosity_mode = _resolve_verbosity(verbosity)
        board, _input_fen, canonical_fen, fen_was_canonicalized = _build_board_with_metadata(
            fen, moves or [], strict=strict
        )
        # evaluate_position with explicit moves has full history; naked FEN doesn't.
        history_complete = _history_provenance_for_input(fen, moves)
        rule_status = evaluate_rule_status(board, history_complete=history_complete)
        pool = await _get_analyzer_pool(ctx)
        engine_name_str = getattr(pool, "engine_version", getattr(pool, "name", "Stockfish"))
        legal_move_count = board.legal_moves.count()

        if rule_status.terminal is not None:
            await metrics.record("top_moves", (time.time() - t0) * 1000, cache_hit=True)
            # Build a typed game_over best_action
            from mcp_server.actions import build_best_action, build_legal_actions

            best_action_obj = build_best_action(
                recommended_action=rule_status.recommended_action,
                rule_status=rule_status,
                engine_eval=None,
                board=board,
                sign=1 if board.turn == chess.WHITE else -1,
            )
            legal_actions = build_legal_actions(
                rule_status=rule_status,
                engine_eval=None,
                board=board,
                legal_engine_moves=None,
            )
            return TopMovesResult(
                status=rule_status.terminal,
                winner=rule_status.winner,
                recommended_action="game_over",
                can_claim_draw=False,
                claim_reasons=[],
                can_claim_now=False,
                claim_reasons_now=[],
                can_claim_with_intended_move=False,
                claim_moves=[],
                best_action_obj=best_action_obj,
                legal_actions=legal_actions,
                history_completeness=rule_status.history_completeness,
                repetition_status=rule_status.repetition_status,
                requested_depth=raw_requested_depth,
                searched_depth=0,
                requested_n=raw_requested_n,
                clamped_n=clamped_n,
                returned_n=0,
                legal_move_count=legal_move_count,
                canonical_fen=canonical_fen,
                fen_was_canonicalized=fen_was_canonicalized,
                engine="Stockfish",
                engine_version=engine_name_str,
                **_build_identity(pool),
                result=[],
            )

        cache_key = top_moves_cache_key(
            board,
            depth,
            n=n,
            engine_version=getattr(pool, "engine_version", None),
            history_completeness=history_complete,
        )

        # sign = mover's perspective sign (White=+1, Black=-1). Used below for
        # both the cache-hit and the freshly-computed paths to decide whether a
        # candidate is winning FOR the side-to-move (cp is White-POV).
        sign = 1 if board.turn == chess.WHITE else -1

        from mcp_server.actions import build_best_action, build_legal_actions

        def _pick_root_recommended_action(items: list[MCPEval]) -> str:
            if not items:
                return rule_status.recommended_action
            best = items[0]
            # U-01 (2026-09-01): mate must take precedence over cp. When
            # Stockfish finds a forced mate it sometimes still emits a
            # saturated cp=±20000; per chess convention, mate wins.
            # Use post_state_* when available (audit B-04 / B-05).
            eff_mate = best.post_state_mate if best.post_state_mate is not None else best.mate
            eff_cp = best.post_state_cp if best.post_state_cp is not None else best.cp
            if eff_mate is not None:
                mover_score: int | None = sign * eff_mate * 1000
            elif eff_cp is not None:
                mover_score = sign * eff_cp
            else:
                mover_score = None
            mate_for_mover = sign * eff_mate if eff_mate is not None else None
            # AUDIT B-04: also surface the best post-state value across all
            # zeroing candidates (capture or pawn move) so the policy can
            # prefer play_move over claim_draw when a zeroing move wins.
            # The post-state cp/mate is attached to each item by the fresh
            # path (audit B-05); the cache-hit path inherits the same data
            # because items are persisted with their post_state_* fields.
            zeroing_best_cp: int | None = None
            zeroing_best_mate: int | None = None
            for item in items:
                if not item.best_move:
                    continue
                try:
                    bm = chess.Move.from_uci(item.best_move)
                except Exception:
                    continue
                if not (board.is_capture(bm) or board.piece_type_at(bm.from_square) == chess.PAWN):
                    continue
                # Prefer the re-evaluated post-state value when present
                # (draw-pollution guard, audit B-04); fall back to the
                # multipv value otherwise.
                eff_cp = item.post_state_cp if item.post_state_cp is not None else item.cp
                eff_mate = item.post_state_mate if item.post_state_mate is not None else item.mate
                if eff_mate is not None:
                    mover_mate = sign * eff_mate
                    if mover_mate > 0 and (
                        zeroing_best_mate is None or mover_mate > zeroing_best_mate
                    ):
                        zeroing_best_mate = mover_mate
                elif eff_cp is not None:
                    mover_cp = sign * eff_cp
                    if zeroing_best_cp is None or mover_cp > zeroing_best_cp:
                        zeroing_best_cp = mover_cp
            return choose_recommended_action(
                board,
                can_claim_now=rule_status.can_claim_now,
                can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
                mover_score=mover_score,
                mate_for_mover=mate_for_mover,
                zeroing_move_best_score=zeroing_best_cp,
                zeroing_move_best_mate=zeroing_best_mate,
            )

        cached = await _cache.get_top_moves(cache_key)
        if cached is not None and len(cached) >= n:
            await metrics.record("top_moves", (time.time() - t0) * 1000, cache_hit=True)
            items = [
                c.model_copy(update={"requested_depth": raw_requested_depth}) for c in cached[:n]
            ]
            # Apply compact verbosity to cached candidates too (audit M-05)
            if verbosity_mode == VERBOSITY_COMPACT:
                items = [_compact_mcpeval(c) for c in items]
            root_rec_action = _pick_root_recommended_action(items)
            best_action_obj = build_best_action(
                recommended_action=root_rec_action,
                rule_status=rule_status,
                engine_eval=items[0] if items else None,
                board=board,
                sign=sign,
            )
            legal_actions = build_legal_actions(
                rule_status=rule_status,
                engine_eval=items[0] if items else None,
                board=board,
                legal_engine_moves=list(items),
            )
            return TopMovesResult(
                status="active",
                winner=None,
                recommended_action=root_rec_action,
                can_claim_draw=rule_status.can_claim_draw,
                claim_reasons=rule_status.claim_reasons,
                claim_move=rule_status.claim_move,
                can_claim_now=rule_status.can_claim_now,
                claim_reasons_now=rule_status.claim_reasons_now,
                can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
                claim_moves=rule_status.claim_moves,
                best_action_obj=best_action_obj,
                legal_actions=legal_actions,
                history_completeness=rule_status.history_completeness,
                repetition_status=rule_status.repetition_status,
                requested_depth=raw_requested_depth,
                searched_depth=depth,
                requested_n=raw_requested_n,
                clamped_n=clamped_n,
                returned_n=len(items),
                legal_move_count=legal_move_count,
                canonical_fen=canonical_fen,
                fen_was_canonicalized=fen_was_canonicalized,
                engine="Stockfish",
                engine_version=engine_name_str,
                **_build_identity(pool),
                result=items,
            )

        async def _compute() -> list[MCPEval]:
            # MultiPV search. Stockfish returns the top-N lines with multipv=N;
            # line 1 (multipv=1) is by definition the engine's canonical best,
            # same as a standalone `evaluate_position` would return. No need
            # for a redundant single-PV pre-search (was costing ~25% of
            # `top_moves` wall time at depth 14).
            res_list: list[MCPEval] = []
            results = await pool.top_moves(board, n=n, depth=depth)
            # AUDIT B-04: when a draw claim is available (immediately or with
            # an intended zeroing move), the root MultiPV cp/mate of a zeroing
            # move can be "polluted" by the engine seeing the draw on the
            # table (e.g. K+R vs K at halfmove=100 reports a tiny cp). We
            # re-evaluate zeroing moves' post-state ONLY when the multipv
            # output looks suspect — i.e. no explicit mate AND a non-positive
            # cp for the mover. Stockfish multipv is authoritative in every
            # other case (no draw on the table, or the engine already gave a
            # clearly winning cp/mate); re-evaluating otherwise just costs an
            # extra engine call without changing the answer. The candidate's
            # reported cp/mate remains the multipv value so ranking and
            # back-compat consumers see the same numbers they did before.
            needs_post_eval = bool(
                rule_status.can_claim_now or rule_status.can_claim_with_intended_move
            )
            zeroing_best_cp: int | None = None
            zeroing_best_mate: int | None = None
            for r in results:
                b_cand = board.copy(stack=True)
                cand_san_val: str | None = None
                cand_post_terminal: str | None = None
                cand_winner: str | None = None
                cand_can_claim_now = False
                cand_can_claim_draw = False
                cand_claim_reasons: list[str] = []
                cand_claim_reasons_now: list[str] = []
                cand_claim_moves: list[str] = []
                # Default to the root rule_status; the post-state branch below
                # refines it. Used by the best_action_obj build below as a
                # fallback when `r.best_move` is missing or fails to parse.
                cand_rule = rule_status
                # Track the post-state cp/mate for the action policy without
                # mutating the candidate's reported values.
                post_state_cp: int | None = None
                post_state_mate: int | None = None

                if r.best_move:
                    try:
                        bm_obj = chess.Move.from_uci(r.best_move.lower())
                        if bm_obj in board.legal_moves:
                            cand_san_val = board.san(bm_obj)
                            is_zeroing = board.is_capture(bm_obj) or (
                                board.piece_type_at(bm_obj.from_square) == chess.PAWN
                            )
                            b_cand.push(bm_obj)
                            # AUDIT B-04: re-evaluate zeroing-move post-state
                            # when the multipv output looks draw-polluted. We
                            # only do this when there's no explicit mate AND
                            # the multipv cp is non-positive for the mover (a
                            # winning move at halfmove=100 should at least
                            # show cp>0; if it doesn't, the engine is treating
                            # the draw as the value of the move and the
                            # post-state is what really matters). The
                            # post-state values feed the action policy
                            # decision; they DO NOT overwrite the candidate's
                            # reported cp/mate (B-05 / C-02 contract).
                            # The post-state re-eval is a draw-pollution guard
                            # (audit B-04 / U-08): when the multipv says the
                            # zeroing move is no better than the draw
                            # (cp<=0 or None), the post-state is what really
                            # matters — the engine is treating the draw as
                            # the value of the move. We do NOT re-evaluate
                            # for strongly positive multipv (the engine has
                            # a clear opinion and a re-eval would only add
                            # cost). The post_state_cp/mate are surfaced on
                            # the wire for client inspection (U-08) — they
                            # are None when no re-eval happened, which is
                            # the honest contract: "no refined post-state
                            # value" rather than fabricating one.
                            multipv_suspect = r.mate is None and (r.cp is None or r.cp <= 0)
                            if (
                                needs_post_eval
                                and is_zeroing
                                and not b_cand.is_game_over(claim_draw=False)
                                and multipv_suspect
                            ):
                                try:
                                    post_ev = await pool.evaluate(b_cand, depth=depth)
                                    if post_ev.mate is not None:
                                        post_state_mate = post_ev.mate
                                    elif post_ev.cp is not None:
                                        post_state_cp = post_ev.cp
                                except Exception:
                                    pass
                            cand_sign = 1 if b_cand.turn == chess.WHITE else -1
                            cand_mover_score: int | None
                            if r.mate is not None:
                                cand_mover_score = cand_sign * r.mate * 1000
                            elif r.cp is not None:
                                cand_mover_score = cand_sign * r.cp
                            else:
                                cand_mover_score = None
                            cand_mate_for_mover = cand_sign * r.mate if r.mate is not None else None
                            cand_rule = evaluate_rule_status(
                                b_cand,
                                mover_score=cand_mover_score,
                                mate_for_mover=cand_mate_for_mover,
                                history_complete=history_complete,
                            )
                            cand_post_terminal = cand_rule.terminal
                            cand_winner = cand_rule.winner
                            cand_can_claim_now = cand_rule.can_claim_now
                            cand_can_claim_draw = cand_rule.can_claim_draw
                            cand_claim_reasons = cand_rule.claim_reasons
                            cand_claim_reasons_now = cand_rule.claim_reasons_now
                            cand_claim_moves = cand_rule.claim_moves
                            # Track best zeroing post-state value for the
                            # action policy below. Sign is mover-POV so we
                            # compare apples to apples. Use the re-evaluated
                            # post-state values when available; fall back to
                            # multipv otherwise (audit B-04 guard).
                            eff_cp = post_state_cp if post_state_cp is not None else r.cp
                            eff_mate = post_state_mate if post_state_mate is not None else r.mate
                            if (
                                needs_post_eval
                                and is_zeroing
                                and (eff_mate is not None or eff_cp is not None)
                            ):
                                mover_sign = 1 if board.turn == chess.WHITE else -1
                                if eff_mate is not None:
                                    mover_mate = mover_sign * eff_mate
                                    if mover_mate > 0 and (
                                        zeroing_best_mate is None or mover_mate > zeroing_best_mate
                                    ):
                                        zeroing_best_mate = mover_mate
                                else:
                                    mover_cp = mover_sign * (eff_cp or 0)
                                    if zeroing_best_cp is None or mover_cp > zeroing_best_cp:
                                        zeroing_best_cp = mover_cp
                    except Exception:
                        pass

                identity = _build_identity(pool)
                # Candidate's reported cp/mate stays at the multipv value so
                # ranking and back-compat callers see the same numbers they
                # did before. Re-evaluated post-state values feed only the
                # action policy decision (audit B-04 / B-05 separation).
                post_eval_for_candidate = Eval(
                    cp=r.cp,
                    mate=r.mate,
                    best_move=r.best_move,
                    pv=r.pv,
                    depth=r.depth,
                )
                # Audit C-03 (2026-09-01 adversarial probe): the candidate's
                # outer action type is the type of move it represents —
                # `play_move` (a candidate IS a play_move action) or
                # `game_over` (the post-state is terminal). The post-state's
                # `rule_status.recommended_action` can be a claim (e.g. after
                # Qb1 the opponent can claim draw) but that is the OPPONENT's
                # perspective, not the candidate's. Reassign `best_action` /
                # `best_action_type` / `best_action_obj` to the candidate's
                # own action type so each candidate reads as a self-consistent
                # play_move or game_over unit. The post-state's recommendation
                # is preserved in `post_position.recommended_action`.
                cand_recommended_action = (
                    "game_over" if cand_post_terminal is not None else "play_move"
                )
                from mcp_server.actions import build_best_action as _build_ba

                if cand_post_terminal is not None:
                    outcome = (
                        "draw"
                        if cand_post_terminal != "checkmate"
                        else ("win" if cand_winner == "white" else "loss")
                    )
                    cand_best_action_obj: dict[str, Any] = {
                        "type": "game_over",
                        "outcome": outcome,
                        "reason": cand_post_terminal,
                    }
                else:
                    # Use the root `board` (not b_cand) for SAN lookup: the
                    # candidate's `best_move` is a legal move AT THE ROOT, not
                    # after it has been played. Passing b_cand would make
                    # `bm in board.legal_moves` False and silently drop SAN.
                    cand_best_action_obj = _build_ba(
                        recommended_action="play_move",
                        rule_status=cand_rule,
                        engine_eval=r,
                        board=board,
                        sign=sign,
                    )
                mcp_eval = MCPEval.from_eval(
                    post_eval_for_candidate,
                    b_cand.fen(),
                    board=b_cand,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                    pv_board=board,
                ).model_copy(
                    update={
                        "build_sha": identity["build_sha"],
                        "engine_config": identity["engine_config"],
                        "post_terminal_status": cand_post_terminal,
                        "candidate_san": cand_san_val,
                        "post_can_claim_draw": cand_can_claim_draw,
                        "post_can_claim_now": cand_can_claim_now,
                        "post_claim_reasons": cand_claim_reasons,
                        "post_claim_moves": cand_claim_moves,
                        "recommended_action": cand_recommended_action,
                        "best_action": cand_recommended_action,
                        "best_action_type": cand_recommended_action,
                        "best_action_obj": cand_best_action_obj,
                        "post_state_cp": post_state_cp,
                        "post_state_mate": post_state_mate,
                        "post_position": {
                            "status": cand_post_terminal or "active",
                            "winner": cand_winner if cand_post_terminal == "checkmate" else None,
                            "can_claim_now": cand_can_claim_now,
                            "can_claim_draw": cand_can_claim_draw,
                            "claim_reasons": cand_claim_reasons_now or cand_claim_reasons,
                            "recommended_action": getattr(
                                cand_rule, "recommended_action", "play_move"
                            ),
                        },
                    }
                )
                res_list.append(mcp_eval)

            def _candidate_rank_key(eval_item: MCPEval) -> float:
                # U-01 (2026-09-01): mate for the mover must outrank any
                # finite-cp win, and the ordering must NOT depend on `n`
                # (the number of candidates requested). The previous rank
                # key had two failure modes:
                #   1. cp was returned unclamped, so a saturated cp=+20000
                #      candidate outranked a mate-in-1 candidate (9999).
                #   2. The rank key preferred the multipv cp of zeroing
                #      moves over the mate branch, so a non-mating capture
                #      could rank above a mating move.
                # Chess-correct total order for the side-to-move is:
                #   delivered mate (terminal) > forced mate for mover
                #     > finite-cp win (clamped to mate ceiling)
                #     > draw  > finite-cp loss > forced mate against mover.
                # We clamp cp to ±MATE_RANK_CEILING so any saturated
                # sentinel (cp=±20000, syzygy fallback, depth=0 win) cannot
                # outrank a forced mate. We always sort (the previous gate
                # `halfmove>=100 or has_terminal_cand` let Stockfish's
                # raw MultiPV order leak through for the >99% case, where
                # a forced mate could still be in slot 2+ at shallow depth).
                MATE_RANK_CEILING = 9999.0
                MATE_VALUE = 10000.0

                # Terminal checks first — these are the strongest signals
                # regardless of cp/mate.
                if eval_item.post_terminal_status == "checkmate":
                    # Candidate delivered mate. Always ranks above any
                    # non-mate candidate (mate=1 is the canonical best).
                    return MATE_VALUE
                if eval_item.post_terminal_status in (
                    "stalemate",
                    "insufficient_material",
                    "seventyfive_moves",
                    "fivefold_repetition",
                    "dead_position",
                ):
                    return 0.0

                # Mate branch BEFORE cp branch (U-01): a mate-in-1 must
                # outrank any finite-cp win. Use the post-state mate when
                # available (audit B-05 — re-eval can refine the multipv
                # mate; falls back to multipv when no re-eval happened).
                eff_mate = (
                    eval_item.post_state_mate
                    if eval_item.post_state_mate is not None
                    else eval_item.mate
                )
                if eff_mate is not None:
                    mover_mate = sign * eff_mate
                    if mover_mate > 0:
                        # Forced mate for mover: shorter is better.
                        return MATE_VALUE - abs(mover_mate)
                    # Forced mate against mover: longer is "less bad",
                    # but always below the floor for any finite cp.
                    return -MATE_VALUE + abs(mover_mate)

                # Cp branch: clamped to the mate ceiling so a saturated
                # cp=±20000 cannot outrank a forced mate. Use post-state
                # cp when available for zeroing moves that were re-eval'd
                # (audit B-04 draw-pollution guard); otherwise use the
                # multipv cp.
                eff_cp = (
                    eval_item.post_state_cp if eval_item.post_state_cp is not None else eval_item.cp
                )
                if eff_cp is not None:
                    mover_cp = sign * eff_cp
                    # Clamp so finite-cp wins never exceed the mate ceiling.
                    if mover_cp > MATE_RANK_CEILING:
                        return MATE_RANK_CEILING
                    if mover_cp < -MATE_RANK_CEILING:
                        return -MATE_RANK_CEILING
                    return float(mover_cp)

                return 0.0

            # Always sort (U-01): n-invariance requires a stable chess-correct
            # ordering regardless of halfmove / terminal state. Removing the
            # gate does not change behavior for positions where Stockfish's
            # raw order already matches the chess-correct order; it just
            # fixes the cases where it doesn't.
            res_list.sort(key=_candidate_rank_key, reverse=True)

            # Persist zeroing-move findings on the cache so the cache-hit path
            # below reuses the same policy decision without re-searching.
            await _cache.set_top_moves(cache_key, res_list)
            return res_list

        sf_key = f"{cache_key}:n={n}"
        res = cast(list[MCPEval], await _single_flight.do(sf_key, _compute))
        await metrics.record("top_moves", (time.time() - t0) * 1000, cache_hit=False)
        items = [c.model_copy(update={"requested_depth": raw_requested_depth}) for c in res[:n]]
        if verbosity_mode == VERBOSITY_COMPACT:
            items = [_compact_mcpeval(c) for c in items]
        root_rec_action = _pick_root_recommended_action(items)
        best_action_obj = build_best_action(
            recommended_action=root_rec_action,
            rule_status=rule_status,
            engine_eval=items[0] if items else None,
            board=board,
            sign=sign,
        )
        legal_actions = build_legal_actions(
            rule_status=rule_status,
            engine_eval=items[0] if items else None,
            board=board,
            legal_engine_moves=list(items),
        )
        return TopMovesResult(
            status="active",
            winner=None,
            recommended_action=root_rec_action,
            can_claim_draw=rule_status.can_claim_draw,
            claim_reasons=rule_status.claim_reasons,
            claim_move=rule_status.claim_move,
            can_claim_now=rule_status.can_claim_now,
            claim_reasons_now=rule_status.claim_reasons_now,
            can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
            claim_moves=rule_status.claim_moves,
            best_action_obj=best_action_obj,
            legal_actions=legal_actions,
            history_completeness=rule_status.history_completeness,
            repetition_status=rule_status.repetition_status,
            requested_depth=raw_requested_depth,
            searched_depth=depth,
            requested_n=raw_requested_n,
            clamped_n=clamped_n,
            returned_n=len(items),
            legal_move_count=legal_move_count,
            canonical_fen=canonical_fen,
            fen_was_canonicalized=fen_was_canonicalized,
            engine="Stockfish",
            engine_version=engine_name_str,
            **_build_identity(pool),
            result=items,
        )
    except ToolError:
        await metrics.record("top_moves", (time.time() - t0) * 1000, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record("top_moves", (time.time() - t0) * 1000, is_error=True)
        msg = str(exc)
        code = error_code_for(msg)
        raise _tool_error(code=code, message=msg, tool="top_moves", input=fen) from exc
    except Exception as exc:
        await metrics.record("top_moves", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="top_moves") from exc
