"""``TopMovesFinder`` service class — orchestrates ``top_moves``.

Pipeline:

  1. Build the board from FEN / PGN + optional movetext.
  2. Short-circuit on terminal positions (game_over action).
  3. Cache lookup; if hit, rehydrate items and pick root action.
  4. Otherwise: acquire pool, do a MultiPV search, evaluate each
     candidate's post-state for draw-pollution, build MCPEval entries,
     sort by chess-correct rank, persist to cache, pick root action.

Per-candidate evaluation lives in :mod:`mcp_server.analysis.candidate_evaluator`.
Response building lives in :mod:`mcp_server.analysis.top_moves_response`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Awaitable, Callable

import chess
from mcp.server.mcpserver import Context

from mcp_server.actions import build_best_action, build_legal_actions
from mcp_server.analysis.candidate_evaluator import evaluate_candidate
from mcp_server.analysis.multi_pv import rank_candidate, select_root_recommended_action
from mcp_server.analysis.top_moves_response import build_top_moves_response
from mcp_server.cache import top_moves_cache_key
from mcp_server.engine import (
    _build_identity,
    _cache,
    _get_analyzer_pool,
    _single_flight,
)
from mcp_server.engine.identity import _engine_version_str
from mcp_server.models import MCPEval, TopMovesResult
from mcp_server.parsers import _build_board_with_metadata, _history_provenance_for_input
from mcp_server.rules import evaluate_rule_status
from mcp_server.tools._common import VERBOSITY_COMPACT, _compact_mcpeval


@dataclass
class TopMovesOutput:
    """Result wrapper for :meth:`TopMovesFinder.run`."""

    result: TopMovesResult
    cache_hit: bool


class TopMovesFinder:
    def __init__(
        self,
        get_pool: Callable[[Context | None], Awaitable[Any]],
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
            return TopMovesOutput(
                cache_hit=True,
                result=_terminal_response(
                    board=board,
                    rule_status=rule_status,
                    pool=pool,
                    engine_name_str=engine_name_str,
                    canonical_fen=canonical_fen,
                    fen_was_canonicalized=fen_was_canonicalized,
                    raw_requested_depth=raw_requested_depth,
                    raw_requested_n=raw_requested_n,
                    clamped_n=clamped_n,
                    legal_move_count=legal_move_count,
                    sign=sign,
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
            return TopMovesOutput(
                cache_hit=True,
                result=_assemble_response(
                    items=items,
                    pool=pool,
                    board=board,
                    rule_status=rule_status,
                    sign=sign,
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
                mcp_eval = await evaluate_candidate(
                    board=board,
                    candidate=r,
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
        return TopMovesOutput(
            cache_hit=False,
            result=_assemble_response(
                items=items,
                pool=pool,
                board=board,
                rule_status=rule_status,
                sign=sign,
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


def _terminal_response(
    *,
    board: chess.Board,
    rule_status: Any,
    pool: Any,
    engine_name_str: str,
    canonical_fen: str,
    fen_was_canonicalized: bool,
    raw_requested_depth: int,
    raw_requested_n: int,
    clamped_n: int,
    legal_move_count: int,
    sign: int,
) -> TopMovesResult:
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


def _assemble_response(
    *,
    items: list[MCPEval],
    pool: Any,
    board: chess.Board,
    rule_status: Any,
    sign: int,
    engine_name_str: str,
    canonical_fen: str,
    fen_was_canonicalized: bool,
    raw_requested_depth: int,
    searched_depth: int,
    raw_requested_n: int,
    clamped_n: int,
    legal_move_count: int,
) -> TopMovesResult:
    """Pick the root recommended action + build action/objects + return response."""
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
    return build_top_moves_response(
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
        searched_depth=searched_depth,
        raw_requested_n=raw_requested_n,
        clamped_n=clamped_n,
        legal_move_count=legal_move_count,
    )
