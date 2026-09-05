"""Opt-in forensic enrichment for ``top_moves``.

The normal top-moves path remains the cheap engine ranking API. Rich modes
reuse the parsed position and add explicit resulting-position evidence for
requested candidates plus a bounded tactical defense proof. The proof surface
never calls a sampled line exhaustive.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import chess

from mcp_server.analysis.forensics import (
    _candidate_evidence,
    _captured_piece,
    _piece_label,
    _principal_line,
    build_position_fingerprint,
    build_tactical_snapshot,
    parse_candidate_move,
)
from mcp_server.models.forensics import (
    CandidateEvidence,
    DefenseEvidence,
    ForensicTopMovesResult,
    TacticalProofEvidence,
    TopMovesForensicEvidence,
)
from mcp_server.models.legacy import TopMovesResult


MATE_VALUE = 100_000
MAX_COMPARE_MOVES = 8
MAX_EXHAUSTIVE_DEFENSES = 8
MAX_SAMPLED_DEFENSES = 8


def _white_value(eval_obj: Any) -> int:
    mate = getattr(eval_obj, "mate", None)
    if mate is not None:
        if mate == 0:
            return MATE_VALUE
        return (MATE_VALUE - min(abs(int(mate)), MATE_VALUE - 1)) * (1 if mate > 0 else -1)
    cp = getattr(eval_obj, "cp", None)
    return int(cp) if cp is not None else 0


def _root_margin(result: TopMovesResult, board: chess.Board) -> int | None:
    if len(result.result) < 2:
        return None
    sign = 1 if board.turn == chess.WHITE else -1
    best = sign * _white_value(result.result[0])
    second = sign * _white_value(result.result[1])
    return max(0, best - second)


def _root_move(result: TopMovesResult, board: chess.Board) -> chess.Move | None:
    if not result.result:
        return None
    raw = result.result[0].best_move
    if not raw:
        return None
    try:
        move = chess.Move.from_uci(raw.lower())
    except (ValueError, chess.InvalidMoveError):
        return None
    return move if move in board.legal_moves else None


async def _evaluate_defense(
    root_post: chess.Board,
    move: chess.Move,
    *,
    pool: Any,
    depth: int,
    rank: int,
) -> DefenseEvidence:
    san = root_post.san(move)
    captured = _captured_piece(root_post, move)
    is_check = root_post.gives_check(move)
    is_capture = root_post.is_capture(move)
    post = root_post.copy(stack=True)
    post.push(move)

    ev: Any | None = None
    if not post.is_game_over(claim_draw=False):
        ev = await pool.evaluate(post, depth=depth)

    return DefenseEvidence(
        rank=rank,
        uci=move.uci(),
        san=san,
        is_check=is_check,
        is_capture=is_capture,
        captured_piece=_piece_label(captured),
        resulting_fen=post.fen(),
        eval_cp=getattr(ev, "cp", None),
        eval_mate=getattr(ev, "mate", None),
        searched_depth=getattr(ev, "depth", None),
        continuation=_principal_line(post, getattr(ev, "pv", None) if ev is not None else None),
    )


def _rank_exhaustive_defenses(
    defenses: list[DefenseEvidence],
    *,
    defender: chess.Color,
) -> list[DefenseEvidence]:
    def score(item: DefenseEvidence) -> int:
        if item.eval_mate is not None:
            mate = item.eval_mate
            return (MATE_VALUE - min(abs(mate), MATE_VALUE - 1)) * (1 if mate > 0 else -1)
        return item.eval_cp or 0

    ordered = sorted(defenses, key=score, reverse=defender == chess.WHITE)
    return [item.model_copy(update={"rank": idx}) for idx, item in enumerate(ordered, start=1)]


async def build_tactical_proof(
    result: TopMovesResult,
    board: chess.Board,
    *,
    pool: Any,
    depth: int,
    proof_defenses: int,
) -> TacticalProofEvidence | None:
    """Evaluate the best move's reply tree with an explicit completeness label."""
    root_move = _root_move(result, board)
    if root_move is None:
        return None

    root_san = board.san(root_move)
    root_post = board.copy(stack=True)
    root_post.push(root_move)
    legal_defenses = list(root_post.legal_moves)

    if root_post.is_game_over(claim_draw=False) or not legal_defenses:
        return TacticalProofEvidence(
            root_move_uci=root_move.uci(),
            root_move_san=root_san,
            root_resulting_fen=root_post.fen(),
            root_margin_effective_cp=_root_margin(result, board),
            proof_status="terminal_after_root",
            legal_defense_count=len(legal_defenses),
            analyzed_defense_count=0,
            defenses=[],
        )

    requested = max(1, min(int(proof_defenses), MAX_SAMPLED_DEFENSES))
    exhaustive = len(legal_defenses) <= MAX_EXHAUSTIVE_DEFENSES

    if exhaustive:
        evaluated = await asyncio.gather(
            *[
                _evaluate_defense(root_post, move, pool=pool, depth=depth, rank=0)
                for move in legal_defenses
            ]
        )
        defenses = _rank_exhaustive_defenses(evaluated, defender=root_post.turn)
        status: Literal["exhaustive", "sampled_top_defenses"] = "exhaustive"
    else:
        top = await pool.top_moves(root_post, n=min(requested, len(legal_defenses)), depth=depth)
        sampled: list[chess.Move] = []
        for item in top:
            raw = getattr(item, "best_move", None)
            if not raw:
                continue
            try:
                move = chess.Move.from_uci(str(raw).lower())
            except (ValueError, chess.InvalidMoveError):
                continue
            if move in root_post.legal_moves and move not in sampled:
                sampled.append(move)
        evaluated = await asyncio.gather(
            *[
                _evaluate_defense(root_post, move, pool=pool, depth=depth, rank=idx)
                for idx, move in enumerate(sampled, start=1)
            ]
        )
        defenses = list(evaluated)
        status = "sampled_top_defenses"

    return TacticalProofEvidence(
        root_move_uci=root_move.uci(),
        root_move_san=root_san,
        root_resulting_fen=root_post.fen(),
        root_margin_effective_cp=_root_margin(result, board),
        proof_status=status,
        legal_defense_count=len(legal_defenses),
        analyzed_defense_count=len(defenses),
        defenses=defenses,
    )


async def enrich_top_moves_result(
    result: TopMovesResult,
    board: chess.Board,
    *,
    pool: Any,
    depth: int,
    detail: Literal["coach", "forensic"],
    include_moves: list[str] | None,
    proof_mode: Literal["none", "tactical"],
    proof_defenses: int,
) -> ForensicTopMovesResult:
    """Attach position evidence, explicit candidates, and optional proof data."""
    requested: list[str] = []
    if detail == "forensic":
        for item in result.result:
            if item.best_move:
                try:
                    move = parse_candidate_move(board, item.best_move)
                    san = board.san(move)
                except ValueError:
                    continue
                if san not in requested:
                    requested.append(san)
    for text in include_moves or []:
        if text not in requested:
            requested.append(text)
    requested = requested[:MAX_COMPARE_MOVES]

    comparisons: list[CandidateEvidence] = []
    for text in requested:
        comparisons.append(await _candidate_evidence(board, text, pool=pool, depth=depth))

    proof = None
    if proof_mode == "tactical":
        proof = await build_tactical_proof(
            result,
            board,
            pool=pool,
            depth=depth,
            proof_defenses=proof_defenses,
        )

    forensic = TopMovesForensicEvidence(
        detail=detail,
        position=build_position_fingerprint(board),
        tactical_snapshot=build_tactical_snapshot(board),
        candidate_comparisons=comparisons,
        proof=proof,
    )
    return ForensicTopMovesResult(**result.model_dump(), forensics=forensic)
