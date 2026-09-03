"""``TopMovesFinder`` service class — orchestrates ``top_moves``.

Encapsulates the end-to-end flow:

  1. Build the board from FEN / PGN + optional movetext.
  2. Short-circuit on terminal positions (game_over action).
  3. Cache lookup; if hit, rehydrate items and pick root action.
  4. Otherwise: acquire pool, do a MultiPV search, evaluate each
     candidate's post-state for draw-pollution, build MCPEval entries,
     sort by chess-correct rank, persist to cache, pick root action.

Dependencies (injected via constructor):

  * ``get_pool``  — async callable returning the live analyzer pool.
  * ``cache_set_top_moves`` / ``cache_get_top_moves`` — cache adapters.
  * ``single_flight`` — coalescer for in-flight computations.

The orchestrator returns a populated :class:`TopMovesResult` directly,
hiding the cache/single-flight mechanics from the tool entry point.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import chess
from mcp.server.mcpserver import Context
from core.engines.types import Eval

from mcp_server.actions import build_best_action, build_legal_actions
from mcp_server.analysis.multi_pv import rank_candidate, select_root_recommended_action
from mcp_server.cache import top_moves_cache_key
from mcp_server.engine import (
    _build_identity,
    _cache,
    _get_analyzer_pool,
    _single_flight,
)
from mcp_server.engine.identity import _engine_version_str
from mcp_server.metrics import metrics
from mcp_server.models import MCPEval, TopMovesResult
from mcp_server.parsers import _build_board_with_metadata, _history_provenance_for_input
from mcp_server.rules import evaluate_rule_status
from mcp_server.tools._common import VERBOSITY_COMPACT, _compact_mcpeval


EnginePool = Any


@dataclass
class TopMovesOutput:
    """Result wrapper for ``TopMovesFinder.run``."""

    result: TopMovesResult
    cache_hit: bool


class TopMovesFinder:
    def __init__(
        self,
        get_pool: Callable[[Context | None], Awaitable[EnginePool]],
        cache_get_top_moves: Callable[..., Awaitable[list[MCPEval] | None]],
        cache_set_top_moves: Callable[..., Awaitable[None]],
        single_flight: Any,
    ) -> None:
        self._get_pool = get_pool
        self._cache_get_top_moves = cache_get_top_moves
        self._cache_set_top_moves = cache_set_top_moves
        self._single_flight = single_flight

    @classmethod
    def with_defaults(cls) -> TopMovesFinder:
        return cls(
            get_pool=_get_analyzer_pool,
            cache_get_top_moves=_cache.get_top_moves,
            cache_set_top_moves=_cache.set_top_moves,
            single_flight=_single_flight,
        )

    async def run(
        self,
        *,
        fen: str,
        moves: list[str] | None,
        n: int,
        depth: int,
        raw_requested_depth: int,
        raw_requested_n: int,
        clamped_n: int,
        strict: bool,
        verbosity_mode: str,
        ctx: Context | None,
    ) -> TopMovesOutput:
        board, _input_fen, canonical_fen, fen_was_canonicalized = _build_board_with_metadata(
            fen, moves or [], strict=strict
        )
        history_complete = _history_provenance_for_input(fen, moves)
        rule_status = evaluate_rule_status(board, history_complete=history_complete)
        pool = await self._get_pool(ctx)
        engine_name_str = _engine_version_str(pool)
        legal_move_count = board.legal_moves.count()
        sign = 1 if board.turn == chess.WHITE else -1

        if rule_status.terminal is not None:
            best_action_obj = build_best_action(
                recommended_action=rule_status.recommended_action,
                rule_status=rule_status,
                engine_eval=None,
                board=board,
                sign=sign,
            )
            legal_actions = build_legal_actions(
                rule_status=rule_status,
                engine_eval=None,
                board=board,
                legal_engine_moves=None,
            )
            return TopMovesOutput(
                cache_hit=True,
                result=TopMovesResult(
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
                ),
            )

        cache_key = top_moves_cache_key(
            board,
            depth,
            n=n,
            engine_version=getattr(pool, "engine_version", None),
            history_completeness=history_complete,
        )

        cached = await self._cache_get_top_moves(cache_key)
        if cached is not None and len(cached) >= n:
            items = [
                c.model_copy(update={"requested_depth": raw_requested_depth}) for c in cached[:n]
            ]
            if verbosity_mode == VERBOSITY_COMPACT:
                items = [_compact_mcpeval(c) for c in items]
            root_rec = select_root_recommended_action(
                items, board=board, rule_status=rule_status, sign=sign
            )
            best_action_obj = build_best_action(
                recommended_action=root_rec,
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
            return TopMovesOutput(
                cache_hit=True,
                result=_build_top_moves_response(
                    items=items,
                    pool=pool,
                    board=board,
                    rule_status=rule_status,
                    sign=sign,
                    cached_items=items,
                    root_rec_action=root_rec,
                    best_action_obj=best_action_obj,
                    legal_actions=legal_actions,
                    engine_name_str=engine_name_str,
                    canonical_fen=canonical_fen,
                    fen_was_canonicalized=fen_was_canonicalized,
                    raw_requested_depth=raw_requested_depth,
                    searched_depth=depth,
                    raw_requested_n=raw_requested_n,
                    clamped_n=clamped_n,
                    legal_move_count=legal_move_count,
                ),
            )

        async def _compute() -> list[MCPEval]:
            res_list: list[MCPEval] = []
            results = await pool.top_moves(board, n=n, depth=depth)
            needs_post_eval = bool(
                rule_status.can_claim_now or rule_status.can_claim_with_intended_move
            )
            for r in results:
                mcp_eval = await _eval_one_candidate(
                    board=board,
                    r=r,
                    pool=pool,
                    rule_status=rule_status,
                    sign=sign,
                    history_complete=history_complete,
                    raw_requested_depth=raw_requested_depth,
                    depth=depth,
                    needs_post_eval=needs_post_eval,
                )
                res_list.append(mcp_eval)
            res_list.sort(key=lambda item: rank_candidate(item, sign=sign), reverse=True)
            await self._cache_set_top_moves(cache_key, res_list)
            return res_list

        sf_key = f"{cache_key}:n={n}"
        res = await self._single_flight.do(sf_key, _compute)
        items = [c.model_copy(update={"requested_depth": raw_requested_depth}) for c in res[:n]]
        if verbosity_mode == VERBOSITY_COMPACT:
            items = [_compact_mcpeval(c) for c in items]
        root_rec = select_root_recommended_action(
            items, board=board, rule_status=rule_status, sign=sign
        )
        best_action_obj = build_best_action(
            recommended_action=root_rec,
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
        return TopMovesOutput(
            cache_hit=False,
            result=_build_top_moves_response(
                items=items,
                pool=pool,
                board=board,
                rule_status=rule_status,
                sign=sign,
                cached_items=res,
                root_rec_action=root_rec,
                best_action_obj=best_action_obj,
                legal_actions=legal_actions,
                engine_name_str=engine_name_str,
                canonical_fen=canonical_fen,
                fen_was_canonicalized=fen_was_canonicalized,
                raw_requested_depth=raw_requested_depth,
                searched_depth=depth,
                raw_requested_n=raw_requested_n,
                clamped_n=clamped_n,
                legal_move_count=legal_move_count,
            ),
        )


async def _eval_one_candidate(
    *,
    board: chess.Board,
    r: Eval,
    pool: Any,
    rule_status: Any,
    sign: int,
    history_complete: str,
    raw_requested_depth: int,
    depth: int,
    needs_post_eval: bool,
) -> MCPEval:
    """Build the MCPEval entry for one MultiPV candidate.

    Audit B-04/B-05/C-03 invariants: zeroing-move post-state is re-eval'd
    when multipv looks draw-polluted, but the candidate's reported cp/mate
    stays at multipv values so ranking and back-compat consumers see the
    same numbers they did before. Best action is recomputed as a
    play_move / game_over discriminator independent of the OPPONENT's
    post-state claim options (audit C-03).
    """
    b_cand = board.copy(stack=True)
    cand_san_val: str | None = None
    cand_post_terminal: str | None = None
    cand_winner: str | None = None
    cand_can_claim_now = False
    cand_can_claim_draw = False
    cand_claim_reasons: list[str] = []
    cand_claim_reasons_now: list[str] = []
    cand_claim_moves: list[str] = []
    cand_rule = rule_status
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
        except Exception:
            pass

    post_eval_for_candidate = Eval(
        cp=r.cp,
        mate=r.mate,
        best_move=r.best_move,
        pv=r.pv,
        depth=r.depth,
    )
    cand_recommended_action = "game_over" if cand_post_terminal is not None else "play_move"

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
        cand_best_action_obj = build_best_action(
            recommended_action="play_move",
            rule_status=cand_rule,
            engine_eval=r,
            board=board,
            sign=sign,
        )

    identity = _build_identity(pool)
    return MCPEval.from_eval(
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
                "recommended_action": getattr(cand_rule, "recommended_action", "play_move"),
            },
        }
    )


def _build_top_moves_response(
    *,
    items: list[MCPEval],
    pool: Any,
    board: chess.Board,
    rule_status: Any,
    sign: int,
    cached_items: list[MCPEval],
    root_rec_action: str,
    best_action_obj: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    engine_name_str: str,
    canonical_fen: str,
    fen_was_canonicalized: bool,
    raw_requested_depth: int,
    searched_depth: int,
    raw_requested_n: int,
    clamped_n: int,
    legal_move_count: int,
) -> TopMovesResult:
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
        searched_depth=searched_depth,
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
