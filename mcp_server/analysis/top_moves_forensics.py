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
    parse_candidate_move,
)
from mcp_server.analysis.position_integrity import build_rich_tactical_snapshot
from mcp_server.models.forensics import (
    CandidateEvidence,
    DefenseEvidence,
    ForensicTopMovesResult,
    TacticalProofEvidence,
    TopMovesForensicEvidence,
)
from mcp_server.models import MCPEval
from mcp_server.models.legacy import TopMovesResult


MATE_VALUE = 100_000
MAX_AUTO_COMPARE_MOVES = 8
MAX_EXPLICIT_COMPARE_MOVES = 8
MAX_TOTAL_COMPARE_MOVES = MAX_EXPLICIT_COMPARE_MOVES + 1
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


def _canonical_candidate_san(board: chess.Board, text: str, *, strict: bool = False) -> str:
    move = parse_candidate_move(board, text, strict=strict)
    return board.san(move)


def _engine_candidate_sans(result: TopMovesResult, board: chess.Board) -> list[str]:
    sans: list[str] = []
    for item in result.result:
        if not item.best_move:
            continue
        try:
            san = _canonical_candidate_san(board, item.best_move)
        except ValueError:
            continue
        if san not in sans:
            sans.append(san)
    return sans


def _comparison_requests(
    result: TopMovesResult,
    board: chess.Board,
    *,
    detail: Literal["coach", "forensic"],
    include_moves: list[str] | None,
    strict: bool = False,
) -> list[tuple[str, str]]:
    """Return comparison candidates as ``(canonical_san, original_text)`` pairs.

    The canonical SAN drives dedupe (so ``include_moves=['e2-e4', 'e4']``
    counts once). The original text is what the caller typed — preserved
    so the candidate's ``requested`` field round-trips the user's exact
    spelling (2026-09-08 ultra-hard test notes §4) instead of silently
    canonicalizing to ``e4``.

    ``coach`` preserves the original contract: explicit ``include_moves`` are
    analyzed exactly as requested and no engine reference is injected.

    ``forensic`` reserves one extra slot for the engine-best reference so the
    caller can compare resulting positions against an explicit baseline. Up to
    eight caller-supplied alternatives remain guaranteed and additional engine
    candidates are appended only while space remains.

    2026-09-08 ultra-hard test notes §3: when ``strict=True`` is threaded
    from the top-level tool, non-canonical SAN (e.g. ``"e2-e4"``) raises
    ``INVALID_COMPARE_MOVE`` symmetrically with how ``classify_move``
    validates the played move under strict mode.
    """
    engine_sans = _engine_candidate_sans(result, board)
    explicit_pairs: list[tuple[str, str]] = []
    for text in include_moves or []:
        san = _canonical_candidate_san(board, text, strict=strict)
        if san not in {s for s, _ in explicit_pairs}:
            explicit_pairs.append((san, text))
    explicit_pairs = explicit_pairs[:MAX_EXPLICIT_COMPARE_MOVES]

    if explicit_pairs and detail == "coach":
        return explicit_pairs

    if explicit_pairs:
        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        if engine_sans:
            out.append((engine_sans[0], engine_sans[0]))
            seen.add(engine_sans[0])
        for san, original in explicit_pairs:
            if san in seen:
                continue
            out.append((san, original))
            seen.add(san)
        for san in engine_sans[1:]:
            if len(out) >= MAX_TOTAL_COMPARE_MOVES:
                break
            if san in seen:
                continue
            out.append((san, san))
            seen.add(san)
        return out[:MAX_TOTAL_COMPARE_MOVES]

    if detail == "forensic":
        return [(san, san) for san in engine_sans[:MAX_AUTO_COMPARE_MOVES]]
    return []


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
    strict: bool = False,
) -> ForensicTopMovesResult:
    """Attach position evidence, explicit candidates, and optional proof data.

    ``strict`` is threaded into :func:`_comparison_requests` so non-canonical
    SAN (e.g. ``"e2-e4"``) raises ``INVALID_COMPARE_MOVE`` symmetrically with
    how ``classify_move`` validates the played move under strict mode
    (2026-09-08 ultra-hard test notes §1).
    """
    requested = _comparison_requests(
        result,
        board,
        detail=detail,
        include_moves=include_moves,
        strict=strict,
    )

    comparisons: list[CandidateEvidence] = []
    for _canonical_san, original_text in requested:
        # 2026-09-08 ultra-hard test notes §4: pass the original user
        # text (e.g. 'e2-e4') to _candidate_evidence so the response
        # payload's `requested` field round-trips the user's spelling.
        # Engine-derived requests fall back to canonical SAN.
        comparisons.append(await _candidate_evidence(board, original_text, pool=pool, depth=depth))

    proof = None
    if proof_mode == "tactical":
        proof = await build_tactical_proof(
            result,
            board,
            pool=pool,
            depth=depth,
            proof_defenses=proof_defenses,
        )

    new_items = list(result.result)
    existing_ucis = {c.best_move.lower() for c in new_items if c.best_move}

    if include_moves:
        from core.engines.types import Eval
        from mcp_server.analysis.candidate_evaluator import evaluate_candidate
        from mcp_server.rules import evaluate_rule_status

        history_complete = (
            "complete" if result.history_completeness == "complete" else "incomplete"
        )
        rule_status = evaluate_rule_status(board, history_complete=history_complete)
        sign = 1 if board.turn == chess.WHITE else -1

        for comp in comparisons:
            comp_uci = comp.uci.lower()
            if comp_uci not in existing_ucis:
                cand_eval = Eval(
                    cp=comp.eval_cp,
                    mate=comp.eval_mate,
                    best_move=comp.uci,
                    pv=[comp.uci, *list(comp.continuation_uci)],
                    depth=comp.searched_depth or depth,
                )
                cand_mcpeval = await evaluate_candidate(
                    board=board,
                    candidate=cand_eval,
                    pool=pool,
                    rule_status=rule_status,
                    sign=sign,
                    history_complete=history_complete,
                    raw_requested_depth=result.requested_depth or depth,
                    depth=depth,
                    needs_post_eval=bool(
                        rule_status.can_claim_now or rule_status.can_claim_with_intended_move
                    ),
                )
                cand_mcpeval = cand_mcpeval.model_copy(
                    update={
                        "search_provenance": {
                            "kind": "candidate_research",
                            "depth": comp.searched_depth or depth,
                            "sources": ["include_moves"],
                        }
                    }
                )
                new_items.append(cand_mcpeval)
                existing_ucis.add(comp_uci)
            else:
                for idx, item in enumerate(new_items):
                    if (item.best_move or "").lower() == comp_uci:
                        prov = dict(item.search_provenance or {})
                        sources = set(prov.get("sources") or [prov.get("kind", "multipv_root")])
                        sources.add("include_moves")
                        prov["sources"] = sorted(sources)
                        new_items[idx] = item.model_copy(update={"search_provenance": prov})
                        break

    def _action_for_candidate(c: MCPEval) -> dict[str, Any]:
        if c.best_action_obj is not None:
            return c.best_action_obj
        uci = c.best_move or ""
        san = c.candidate_san
        if san is None and uci and board is not None:
            try:
                m = chess.Move.from_uci(uci.lower())
                if m in board.legal_moves:
                    san = board.san(m)
            except Exception:
                pass
        payload: dict[str, Any] = {
            "type": "play_move",
            "move": {"uci": uci, "san": san},
        }
        if c.cp is not None or c.mate is not None:
            payload["value"] = {"cp": c.cp, "mate": c.mate}
        return payload

    rule_actions = list(
        result.legal_rule_actions
        or [
            a
            for a in result.legal_actions
            if a.get("type") in ("claim_draw", "claim_draw_with_intended_move")
        ]
    )
    new_legal_actions = [*rule_actions, *[_action_for_candidate(c) for c in new_items]]
    result = result.model_copy(
        update={
            "result": new_items,
            "returned_n": len(new_items),
            "legal_actions": new_legal_actions,
            "requested_include_moves": list(include_moves or []),
            "included_move_count": len([m for m in include_moves or [] if m]),
        }
    )


    forensic = TopMovesForensicEvidence(
        detail=detail,
        position=build_position_fingerprint(board),
        tactical_snapshot=build_rich_tactical_snapshot(board),
        candidate_comparisons=comparisons,
        proof=proof,
    )
    return ForensicTopMovesResult(**result.model_dump(exclude={"forensics"}), forensics=forensic)
