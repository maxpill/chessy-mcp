"""``classify_move`` MCP tool.

Extracted from ``mcp_server.server``. Grades a played move against
Stockfish's best alternative, with draw-claim classification and rule-aware
loss attribution.
"""

from __future__ import annotations

import logging
import time
from typing import Literal, cast

import chess

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from core.engines.analyzer import pv_to_san
from core.engines.pool import AnalyzerPool
from core.engines.types import MoveClass
from mcp_server.actions import build_played_action
from mcp_server.cache import classify_cache_key
from mcp_server.claims.draw_projection import _force_draw_outcome
from mcp_server.engine import (
    _cache,
    _evaluate_game_position_cached,
    _get_analyzer_pool,
    _single_flight,
)
from mcp_server.metrics import metrics
from mcp_server.models import MCPMoveAnalysis, score_played_move
from mcp_server.parsers import (
    _build_board,
    _history_provenance_for_input,
    _parse_move_on_board_with_warning,
)
from mcp_server.rules import (
    evaluate_rule_status,
    is_terminal_position,
)
from mcp_server.server import mcp
from mcp_server.tcp_analyzer import TCPAnalyzerPool
from mcp_server.tools._common import (
    _tool_error,
    _validate_requested_depth,
    error_code_for,
)

log = logging.getLogger("chessy_mcp.classify_move")


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def classify_move(
    fen: str,
    move: str | None = None,
    moves: list[str] | None = None,
    depth: int = 20,
    action_type: Literal["play_move", "claim_draw", "claim_draw_with_intended_move"] = "play_move",
    strict: bool = False,
    ctx: Context | None = None,
) -> MCPMoveAnalysis:
    """Grade a played move against Stockfish's best alternative.

    Grades: 'best', 'good', 'inaccuracy', 'mistake', 'blunder'. Note that `move_class`
    is derived from `effective_loss` (win probability impact & position context, e.g.
    decisive advantage saturation), NOT directly from raw `centipawn_loss`. Also returns
    centipawn loss, mate distance loss, and evals before/after the move.

    Args:
        fen: FEN or PGN string for the position BEFORE `move`.
        move: The move to grade in UCI (e.g. "e2e4") or SAN (e.g. "e4", "Bxf3", "O-O").
            Required for `play_move` and `claim_draw_with_intended_move`; optional for
            `claim_draw` (the claim outcome does not depend on any specific move).
        moves: Optional UCI or SAN moves to replay onto the position first.
        depth: Stockfish search depth (default 20, clamped 1-30). Sweet spot for
            per-move classification: d20 ≈ 782k nodes, d22 ≈ 1.29M, d24 ≈ 2.06M.
            Lower depths (16-18) ship a faster batch mode; d24-26 only when
            borderline moves need extra sharpening. ``classify_move`` is called
            once per game move, so the default is intentionally high enough to
            distinguish mistakes/blunders cleanly.
        action_type: Intended chess action ('play_move', 'claim_draw', 'claim_draw_with_intended_move').
        strict: When True, reject non-canonical SAN syntax or move numbers (default False).

    Returns:
        MoveAnalysis with move_class, centipawn_loss, effective_loss, eval_before, eval_after,
        best_move_san, best_line_san, and played_line_san.
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="classify_move")
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    try:
        if action_type not in {"play_move", "claim_draw", "claim_draw_with_intended_move"}:
            raise ValueError(f"INVALID_ACTION_TYPE: {action_type}")
        # P2 (2026-09-02 ultra audit): request-shape validation must run BEFORE
        # board-state validation. The audit found that `claim_draw` with a
        # supplied `move` argument on a non-claimable board returned
        # "draw cannot be claimed now" — a board-state error — instead of
        # the structural "claim_draw must not include a move" error. The
        # structural error is consistent regardless of position and lets
        # callers distinguish bad input from bad state. Same applies to
        # `play_move` / `claim_draw_with_intended_move` missing a move.
        # R5 (2026-09-02 round-5 super-deep audit): the move parameter
        # must be a string. A non-string (int, list, None-as-empty) used
        # to fall through to `move.strip()` and produce a confusing
        # AttributeError. Validate type first so the rejection is a clean
        # INVALID_INPUT message.
        if move is not None and not isinstance(move, str):
            raise ValueError(f"INVALID_INPUT: 'move' must be a string, got {type(move).__name__}.")
        if action_type == "claim_draw":
            if move is not None and move.strip() and move.strip() != "(none)":
                if strict:
                    raise ValueError(
                        f"STRICT_SAN_ERROR: action_type='claim_draw' must not "
                        f"include a `move` argument; got {move!r}. Pass move=None "
                        f"or omit the parameter."
                    )
                # Lenient mode: still record that the caller passed a
                # meaningless argument (per U-12 invariant — B-02 audit).
                # We surface this as a syntax_warning later via the response.
        else:
            if move is None or not move.strip():
                raise ValueError(
                    "MISSING_MOVE: 'move' is required for action_type='play_move' "
                    "and action_type='claim_draw_with_intended_move'"
                )
        board = _build_board(fen, moves or [], strict=strict)
        history_complete = _history_provenance_for_input(fen, moves)
        rule_before = evaluate_rule_status(board, history_complete=history_complete)
        # AUDIT B-01/B-02/B-03: for `claim_draw`, the dummy `move` argument must
        # not be parsed/executed; the claim outcome is purely procedural. Accept
        # `move=None` (or any string) but never push the move onto the board
        # when classifying a draw claim. `claim_draw_with_intended_move` still
        # requires a real intended move because the move IS the claim.
        if action_type == "claim_draw":
            chess_move: chess.Move | None = None
            syntax_warn: str | None = None
            if move is not None and move.strip() and move.strip() != "(none)":
                # P2 (2026-09-02 ultra audit): lenient mode still warns when
                # the caller passes a meaningless `move` argument to
                # `claim_draw`. Strict mode rejects outright (above). The
                # warning makes the structural mismatch observable without
                # breaking the claim.
                syntax_warn = (
                    f"action_type='claim_draw' ignores supplied move argument "
                    f"{move!r} (the claim outcome is purely procedural)."
                )
            # P2 (2026-09-02 ultra audit): terminal-state handling must
            # happen before action-specific claim validation so every
            # action on a finished board returns the same GAME_ALREADY_OVER
            # error, not a position-dependent ILLEGAL_ACTION variant.
            if is_terminal_position(board):
                raise ValueError(
                    f"GAME_ALREADY_OVER: Position '{board.fen()}' is already game over; "
                    f"no further actions can be taken on a finished game."
                )
            if not rule_before.can_claim_now:
                raise ValueError("ILLEGAL_ACTION: draw cannot be claimed now")
        else:
            assert move is not None and move.strip()  # shape validated above
            chess_move, syntax_warn = _parse_move_on_board_with_warning(board, move, strict=strict)
            if (
                action_type == "claim_draw_with_intended_move"
                and chess_move.uci() not in rule_before.intended_claim_ucis
            ):
                raise ValueError("ILLEGAL_ACTION: intended move does not create a legal draw claim")
        pool = await _get_analyzer_pool(ctx)

        # Cache key uses an empty/dummy move for claim_draw so the same
        # underlying position/action always maps to one cache entry, regardless
        # of the dummy `move` the caller passed (audit B-02 invariant).
        cache_move_uci = chess_move.uci() if chess_move is not None else ""
        cache_key = classify_cache_key(
            board,
            cache_move_uci,
            depth,
            action_type=action_type,
            engine_version=getattr(pool, "engine_version", None),
            history_completeness=history_complete,
        )

        cached = await _cache.get_classify(cache_key)
        if cached is not None:
            await metrics.record("classify_move", (time.time() - t0) * 1000, cache_hit=True)
            eval_bef = cached.eval_before.model_copy(
                update={"requested_depth": raw_requested_depth}
            )
            eval_aft = cached.eval_after.model_copy(update={"requested_depth": raw_requested_depth})
            return cached.model_copy(
                update={
                    "eval_before": eval_bef,
                    "eval_after": eval_aft,
                    "syntax_warning": syntax_warn,
                }
            )

        # Build played_san / board_after defensively: for claim_draw they are
        # NOT derived from any chess move because the claim is procedural.
        if chess_move is not None:
            played_san = board.san(chess_move)
            board_after = board.copy(stack=True)
            board_after.push(chess_move)
        else:
            played_san = None
            board_after = board.copy(stack=True)

        async def _compute() -> MCPMoveAnalysis:
            pool = await _get_analyzer_pool(ctx)

            if (
                chess_move is not None
                and hasattr(pool, "classify_move")
                and type(pool)
                not in (
                    AnalyzerPool,
                    TCPAnalyzerPool,
                )
            ):
                result = await pool.classify_move(board, chess_move, depth=depth)
                return MCPMoveAnalysis.from_analysis(
                    result,
                    fen_before=board.fen(),
                    fen_after=board_after.fen(),
                    played_san=played_san,
                    board_before=board,
                    board_after=board_after,
                    syntax_warning=None,
                    action_type=action_type,
                    history_complete=history_complete,
                )

            eval_before, _ = await _evaluate_game_position_cached(
                board,
                depth,
                pool,
                requested_depth=raw_requested_depth,
                history_complete=history_complete,
            )

            # AUDIT B-02/B-03: for draw-claim actions, the post-state is the
            # position AFTER the claim is granted, not after the supplied
            # (irrelevant) move is played. Re-evaluate the same root board so
            # the resulting `eval_after` reflects the draw outcome (cp=0,
            # outcome=draw) regardless of any dummy move the caller passed.
            if action_type in ("claim_draw", "claim_draw_with_intended_move"):
                eval_after, _ = await _evaluate_game_position_cached(
                    board,
                    depth,
                    pool,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                )
                # The claim outcome is a draw; force cp=0 and outcome=draw so
                # every downstream caller sees a consistent post-claim state
                # independent of the dummy move.
                eval_after = _force_draw_outcome(eval_after)
            else:
                # Correctness first: eval_after must describe the immediate
                # post-move position. Reusing the root PV tail or root score
                # can misstate finite-depth CP and mate distance. Engine/cache
                # layers remain responsible for performance reuse.
                eval_after, _ = await _evaluate_game_position_cached(
                    board_after,
                    depth,
                    pool,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                )

            if chess_move is not None:
                score = score_played_move(
                    board,
                    chess_move,
                    eval_before,
                    eval_after,
                    board_after,
                    action_type=action_type,
                )
            else:
                # claim_draw without a move: pass a placeholder Move and the
                # post-claim board (= root board). score_played_move still
                # consults rule_before.can_claim_now and the post-claim eval,
                # so the dummy Move here is purely structural and never
                # affects the score.
                placeholder = next(iter(board.legal_moves), None)
                if placeholder is None:
                    raise ValueError("ILLEGAL_ACTION: no legal moves; cannot evaluate claim")
                score = score_played_move(
                    board,
                    placeholder,
                    eval_before,
                    eval_after,
                    board_after,
                    action_type=action_type,
                )

            # Candidate Verification Search (Opera Morphy invariant enforcement):
            # If played move matched eval_before.best_move, but grading would produce mistake/blunder,
            # run a deeper verification search so eval_before is updated with the true best candidate.
            #
            # P1 audit fix: this branch used to be FAIL-OPEN — when the deeper
            # search threw any exception, the code silently flipped move_class to
            # BEST, effective_loss=0. That makes a buggy engine produce honest
            # answers and a buggy harness produce lies. The fixed behavior:
            #   - if verification succeeds and finds a better move, regrade.
            #   - if verification succeeds and confirms our move, lock to BEST.
            #   - if verification FAILS, do NOT silently overwrite grading;
            #     mark classification_verified=False so callers see the
            #     unverified result instead of a fabricated "best".
            verification_attempted = False
            # Audit P0/P1 (2026-09-01 adversarial probe): the verification
            # block is for `play_move` only. Draw-claim actions classify the
            # CLAIM, not the supplied move; the move may coincidentally match
            # `eval_before.best_move` (e.g. `claim_draw + Qc8#` where the
            # engine's best IS the mating move the player is refusing to play).
            # In that case the depth+4 verification correctly confirms the
            # move is the engine's best legal attempt — but that's irrelevant
            # to grading the CLAIM. Allowing the "else" branch to overwrite
            # `move_class=BEST, effective_loss=0` here violates the invariant
            # `is_best_action==False AND best outcome==win AND played
            # outcome==draw ⇒ effective_loss > 0`. Skip the whole block for
            # claim actions; the score from `score_played_move` is final.
            if (
                action_type == "play_move"
                and chess_move is not None
                and (
                    chess_move.uci().lower() == (eval_before.best_move or "").lower()
                    and score.move_class in (MoveClass.MISTAKE, MoveClass.BLUNDER)
                    and not score.missed_draw_claim
                    and not score.conceded_draw_claim
                )
            ):
                try:
                    # Cache the depth+4 verification result via the same
                    # L1/L2 path as any other eval. Previously this went
                    # straight to pool.evaluate, bypassing the cache — every
                    # classify_move that hit this verification path paid the
                    # full uncached depth+4 cost. Now the depth+4 result is
                    # cached like any other eval.
                    verify_eval_result, _verify_hit = await _evaluate_game_position_cached(
                        board,
                        depth + 4,
                        pool,
                        requested_depth=raw_requested_depth + 4,
                        history_complete=history_complete,
                    )
                    verify_ev: Eval = Eval(
                        cp=verify_eval_result.cp,
                        mate=verify_eval_result.mate,
                        best_move=verify_eval_result.best_move,
                        pv=verify_eval_result.pv,
                        depth=verify_eval_result.searched_depth or (depth + 4),
                    )
                    verification_attempted = True
                    if (
                        verify_ev.best_move
                        and verify_ev.best_move.lower() != chess_move.uci().lower()
                    ):
                        # Verification discovered a better move! Update eval_before
                        eval_before = MCPEval.from_eval(
                            verify_ev,
                            board.fen(),
                            board=board,
                            requested_depth=raw_requested_depth,
                            history_complete=history_complete,
                        )
                        score = score_played_move(
                            board,
                            chess_move,
                            eval_before,
                            eval_after,
                            board_after,
                            action_type=action_type,
                        )
                    else:
                        # Played move is confirmed as the best legal attempt.
                        score.move_class = MoveClass.BEST
                        score.effective_loss = 0
                        score.is_best_engine_move = True
                except Exception:
                    # Verification FAILED — leave the original grading intact
                    # and mark the response unverified rather than fabricating
                    # a BEST verdict we cannot prove (audit P1 fix).
                    verification_attempted = True

            best_san: str | None = None
            if score.is_best_engine_move and chess_move is not None:
                best_san = played_san
            elif eval_before.best_move:
                try:
                    bm = chess.Move.from_uci(eval_before.best_move.lower())
                    if bm in board.legal_moves:
                        best_san = board.san(bm)
                except Exception:
                    pass

            best_line_san = pv_to_san(board, eval_before.pv) if eval_before.pv else best_san
            played_continuation: str | None = None
            if eval_after.pv and not board_after.is_game_over() and chess_move is not None:
                played_continuation = pv_to_san(board_after, eval_after.pv)

            played_line_san = played_san
            if played_continuation and played_san is not None:
                played_line_san = f"{played_san} {played_continuation}"

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
            # P1 audit fix: verification failure must NOT silently downgrade
            # grading. If we tried to verify but couldn't reach a conclusion,
            # the response must be marked unverified.
            if verification_attempted and score.move_class in (
                MoveClass.MISTAKE,
                MoveClass.BLUNDER,
            ):
                verified = False

            played_uci = chess_move.uci() if chess_move is not None else ""
            mcp_analysis = MCPMoveAnalysis(
                played=played_uci,
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
                eval_before=eval_before,
                eval_after=eval_after,
                best_move_san=best_san,
                best_line_san=best_line_san,
                best_line_san_truncated=bool(eval_before.pv and len(eval_before.pv) > 6),
                played_line_san=played_line_san,
                played_continuation_san=played_continuation,
                syntax_warning=None,
                action_type=action_type,
                best_action=score.best_action,
                is_best_action=score.is_best_action,
                action_equivalent=score.action_equivalent,
                played_action_obj=build_played_action(
                    action_type,
                    move_uci=played_uci,
                    move_san=played_san,
                    rule_status=rule_before,
                    cp=eval_after.cp,
                    mate=eval_after.mate,
                ),
                best_action_obj=eval_before.best_action_obj,
                missed_draw_claim=score.missed_draw_claim,
                conceded_draw_claim=score.conceded_draw_claim,
                claim_reason=score.claim_reason,
                claim_move=score.claim_move,
                can_claim_now=score.can_claim_now,
                can_claim_with_intended_move=score.can_claim_with_intended_move,
                claim_moves=score.claim_moves,
                classification_verified=verified,
            )
            await _cache.set_classify(cache_key, mcp_analysis)
            return mcp_analysis

        res = cast(MCPMoveAnalysis, await _single_flight.do(cache_key, _compute))
        await metrics.record("classify_move", (time.time() - t0) * 1000, cache_hit=False)
        return res.model_copy(update={"syntax_warning": syntax_warn})
    except ToolError:
        await metrics.record("classify_move", (time.time() - t0) * 1000, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record("classify_move", (time.time() - t0) * 1000, is_error=True)
        msg = str(exc)
        code = error_code_for(msg)
        raise _tool_error(code=code, message=msg, tool="classify_move", input=move) from exc
    except Exception as exc:
        await metrics.record("classify_move", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="classify_move") from exc
