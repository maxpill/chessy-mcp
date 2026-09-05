"""Evidence-first forensic helpers for coaching-oriented move analysis.

The functions in this module are intentionally deterministic where possible.
They report board facts, forcing moves and engine continuations, but avoid
claiming why a human chose a move.  That separation lets the coaching layer
map evidence such as a missed forcing reply to a process hypothesis only when
it has enough context (for example, a player's own post-game comment).
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import chess

from mcp_server.models.forensics import (
    CandidateEvidence,
    ForcedLineEvidence,
    ForensicEvidence,
    ForensicMoveAnalysis,
    ForcingMoveEvidence,
    PieceEvidence,
    PositionDelta,
    PositionFingerprint,
    StrongestReplyEvidence,
    TacticalSnapshot,
)
from mcp_server.models.legacy import MCPMoveAnalysis


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def _color_name(color: chess.Color) -> Literal["white", "black"]:
    return "white" if color == chess.WHITE else "black"


def _piece_label(piece: chess.Piece | None) -> str | None:
    if piece is None:
        return None
    return f"{_color_name(piece.color)}_{PIECE_NAMES[piece.piece_type]}"


def _captured_piece(board: chess.Board, move: chess.Move) -> chess.Piece | None:
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        return board.piece_at(move.to_square + offset)
    return board.piece_at(move.to_square)


def _move_evidence(board: chess.Board, move: chess.Move) -> ForcingMoveEvidence:
    captured = _captured_piece(board, move)
    promotion = PIECE_NAMES.get(move.promotion) if move.promotion else None
    return ForcingMoveEvidence(
        uci=move.uci(),
        san=board.san(move),
        is_check=board.gives_check(move),
        is_capture=board.is_capture(move),
        captured_piece=_piece_label(captured),
        promotion=promotion,
    )


def build_position_fingerprint(board: chess.Board) -> PositionFingerprint:
    canonical_fen = board.fen()
    piece_map: dict[str, dict[str, list[str]]] = {
        "white": {name: [] for name in PIECE_NAMES.values()},
        "black": {name: [] for name in PIECE_NAMES.values()},
    }
    material = {"white": 0, "black": 0}
    for square, piece in board.piece_map().items():
        color = _color_name(piece.color)
        name = PIECE_NAMES[piece.piece_type]
        piece_map[color][name].append(chess.square_name(square))
        material[color] += PIECE_VALUES[piece.piece_type]
    for side in piece_map.values():
        for squares in side.values():
            squares.sort()
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else None
    return PositionFingerprint(
        canonical_fen=canonical_fen,
        side_to_move=_color_name(board.turn),
        piece_map=piece_map,
        material=material,
        castling_rights=board.castling_xfen() or "-",
        en_passant=ep,
        in_check=board.is_check(),
        legal_move_count=board.legal_moves.count(),
        position_hash=hashlib.sha256(canonical_fen.encode("utf-8")).hexdigest()[:16],
    )


def _piece_evidence(board: chess.Board, square: chess.Square, piece: chess.Piece) -> PieceEvidence:
    attackers = len(board.attackers(not piece.color, square))
    defenders = len(board.attackers(piece.color, square))
    return PieceEvidence(
        color=_color_name(piece.color),
        piece=PIECE_NAMES[piece.piece_type],
        square=chess.square_name(square),
        attackers=attackers,
        defenders=defenders,
    )


def build_tactical_snapshot(board: chess.Board) -> TacticalSnapshot:
    checks: list[ForcingMoveEvidence] = []
    captures: list[ForcingMoveEvidence] = []
    en_prise_squares: set[chess.Square] = set()
    for move in board.legal_moves:
        if board.gives_check(move):
            checks.append(_move_evidence(board, move))
        if board.is_capture(move):
            captures.append(_move_evidence(board, move))
            if not board.is_en_passant(move) and board.piece_at(move.to_square) is not None:
                en_prise_squares.add(move.to_square)

    loose: list[PieceEvidence] = []
    en_prise: list[PieceEvidence] = []
    pinned: list[PieceEvidence] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        evidence = _piece_evidence(board, square, piece)
        if evidence.defenders == 0:
            loose.append(evidence)
        if square in en_prise_squares:
            en_prise.append(evidence)
        if board.is_pinned(piece.color, square):
            pinned.append(evidence)

    key = lambda item: (item.color, item.square, item.piece)
    return TacticalSnapshot(
        side_to_move=_color_name(board.turn),
        checks=sorted(checks, key=lambda item: item.san),
        captures=sorted(captures, key=lambda item: item.san),
        loose_pieces=sorted(loose, key=key),
        en_prise_pieces=sorted(en_prise, key=key),
        pinned_pieces=sorted(pinned, key=key),
    )


def parse_candidate_move(board: chess.Board, text: str) -> chess.Move:
    raw = text.strip()
    try:
        move = chess.Move.from_uci(raw.lower())
        if move in board.legal_moves:
            return move
    except (ValueError, chess.InvalidMoveError):
        pass
    try:
        return board.parse_san(raw)
    except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError) as exc:
        raise ValueError(f"INVALID_COMPARE_MOVE: {text}") from exc


def build_position_delta(
    before: chess.Board,
    after: chess.Board,
    *,
    before_snapshot: TacticalSnapshot | None = None,
    after_snapshot: TacticalSnapshot | None = None,
) -> PositionDelta:
    before_fp = build_position_fingerprint(before)
    after_fp = build_position_fingerprint(after)
    before_snapshot = before_snapshot or build_tactical_snapshot(before)
    after_snapshot = after_snapshot or build_tactical_snapshot(after)

    before_pieces = {
        f"{_color_name(piece.color)}_{PIECE_NAMES[piece.piece_type]}@{chess.square_name(square)}"
        for square, piece in before.piece_map().items()
    }
    after_pieces = {
        f"{_color_name(piece.color)}_{PIECE_NAMES[piece.piece_type]}@{chess.square_name(square)}"
        for square, piece in after.piece_map().items()
    }
    before_loose = {f"{p.color}_{p.piece}@{p.square}" for p in before_snapshot.loose_pieces}
    after_loose = {f"{p.color}_{p.piece}@{p.square}" for p in after_snapshot.loose_pieces}
    before_pinned = {f"{p.color}_{p.piece}@{p.square}" for p in before_snapshot.pinned_pieces}
    after_pinned = {f"{p.color}_{p.piece}@{p.square}" for p in after_snapshot.pinned_pieces}

    return PositionDelta(
        material_delta_white=after_fp.material["white"] - before_fp.material["white"],
        material_delta_black=after_fp.material["black"] - before_fp.material["black"],
        removed_pieces=sorted(before_pieces - after_pieces),
        added_pieces=sorted(after_pieces - before_pieces),
        newly_loose_pieces=sorted(after_loose - before_loose),
        newly_pinned_pieces=sorted(after_pinned - before_pinned),
        check_state_changed=before.is_check() != after.is_check(),
    )


def _reply_from_eval(
    board: chess.Board,
    eval_obj: Any,
    *,
    eval_after_reply: Any | None = None,
) -> StrongestReplyEvidence | None:
    best = getattr(eval_obj, "best_move", None)
    if not best or board.is_game_over(claim_draw=False):
        return None
    try:
        move = chess.Move.from_uci(str(best).lower())
    except (ValueError, chess.InvalidMoveError):
        return None
    if move not in board.legal_moves:
        return None
    captured = _captured_piece(board, move)
    san = board.san(move)
    is_check = board.gives_check(move)
    is_capture = board.is_capture(move)
    reply_board = board.copy(stack=True)
    reply_board.push(move)
    return StrongestReplyEvidence(
        uci=move.uci(),
        san=san,
        is_check=is_check,
        is_capture=is_capture,
        is_forcing=is_check or is_capture,
        captured_piece=_piece_label(captured),
        resulting_fen=reply_board.fen(),
        eval_after_reply_cp=getattr(eval_after_reply, "cp", None),
        eval_after_reply_mate=getattr(eval_after_reply, "mate", None),
        searched_depth=getattr(eval_after_reply, "depth", None),
    )


async def _strongest_reply(
    board: chess.Board,
    eval_obj: Any,
    *,
    pool: Any,
    depth: int,
    deep: bool,
) -> StrongestReplyEvidence | None:
    base = _reply_from_eval(board, eval_obj)
    if base is None or not deep:
        return base
    try:
        move = chess.Move.from_uci(base.uci)
        reply_board = board.copy(stack=True)
        reply_board.push(move)
        if reply_board.is_game_over(claim_draw=False):
            return base
        post_eval = await pool.evaluate(reply_board, depth=min(max(depth, 1) + 2, 24))
    except Exception:
        return base
    return _reply_from_eval(board, eval_obj, eval_after_reply=post_eval)


def _principal_line(board: chess.Board, pv: list[str] | None) -> ForcedLineEvidence:
    if not pv:
        return ForcedLineEvidence()
    b = board.copy(stack=True)
    uci: list[str] = []
    san: list[str] = []
    for raw in list(pv)[:12]:
        try:
            move = chess.Move.from_uci(str(raw).lower())
        except (ValueError, chess.InvalidMoveError):
            return ForcedLineEvidence(
                uci=uci,
                san=san,
                termination_reason="invalid_pv_move",
                tactical_sequence_resolved=False,
            )
        if move not in b.legal_moves:
            return ForcedLineEvidence(
                uci=uci,
                san=san,
                termination_reason="invalid_pv_move",
                tactical_sequence_resolved=False,
            )
        uci.append(move.uci())
        san.append(b.san(move))
        b.push(move)
        if b.is_game_over(claim_draw=False):
            return ForcedLineEvidence(
                uci=uci,
                san=san,
                termination_reason="terminal_position",
                tactical_sequence_resolved=True,
            )
    final_snapshot = build_tactical_snapshot(b)
    resolved = not final_snapshot.checks and not final_snapshot.captures
    return ForcedLineEvidence(
        uci=uci,
        san=san,
        termination_reason="pv_exhausted",
        tactical_sequence_resolved=resolved,
    )


async def _candidate_evidence(
    board: chess.Board,
    requested: str,
    *,
    pool: Any,
    depth: int,
) -> CandidateEvidence:
    move = parse_candidate_move(board, requested)
    san = board.san(move)
    post = board.copy(stack=True)
    post.push(move)
    snapshot = build_tactical_snapshot(post)
    ev = None
    if not post.is_game_over(claim_draw=False):
        ev = await pool.evaluate(post, depth=depth)
    reply = _reply_from_eval(post, ev) if ev is not None else None
    return CandidateEvidence(
        requested=requested,
        uci=move.uci(),
        san=san,
        resulting_fen=post.fen(),
        eval_cp=getattr(ev, "cp", None),
        eval_mate=getattr(ev, "mate", None),
        searched_depth=getattr(ev, "depth", None),
        opponent_best_reply=reply,
        tactical_snapshot_after=snapshot,
    )


def _mechanism_evidence(
    result: MCPMoveAnalysis,
    reply: StrongestReplyEvidence | None,
    tactical_after: TacticalSnapshot,
) -> tuple[list[dict[str, Any]], list[str]]:
    evidence: list[dict[str, Any]] = []
    signatures: list[str] = []
    move_class = result.move_class.value if hasattr(result.move_class, "value") else str(result.move_class)

    if reply is not None and reply.is_forcing:
        evidence.append(
            {
                "mechanism": "forcing_reply",
                "reply": reply.san,
                "is_check": reply.is_check,
                "is_capture": reply.is_capture,
                "captured_piece": reply.captured_piece,
            }
        )
        if move_class in {"mistake", "blunder"}:
            signatures.append("MISSED_FORCING_REPLY_CANDIDATE")
        if reply.is_check:
            signatures.append("FORCING_CHECK_REPLY")
        if reply.is_capture:
            signatures.append("FORCING_CAPTURE_REPLY")

    if reply is not None and reply.is_capture and reply.captured_piece:
        victim_square = None
        try:
            victim_square = chess.Move.from_uci(reply.uci).to_square
        except (ValueError, chess.InvalidMoveError):
            pass
        if victim_square is not None:
            sq_name = chess.square_name(victim_square)
            matching = [p for p in tactical_after.loose_pieces if p.square == sq_name]
            if matching:
                piece = matching[0]
                evidence.append(
                    {
                        "mechanism": "loose_piece_capture",
                        "target": f"{piece.color}_{piece.piece}@{piece.square}",
                        "defenders": piece.defenders,
                        "reply": reply.san,
                    }
                )
                signatures.append("LOOSE_PIECE_PUNISHED")

    if tactical_after.pinned_pieces:
        evidence.append(
            {
                "mechanism": "pin_present_after_played_move",
                "pieces": [f"{p.color}_{p.piece}@{p.square}" for p in tactical_after.pinned_pieces],
            }
        )

    return evidence, sorted(set(signatures))


async def enrich_move_analysis(
    result: MCPMoveAnalysis,
    *,
    board_before: chess.Board,
    played_move: chess.Move | None,
    pool: Any,
    depth: int,
    detail: Literal["coach", "forensic"],
    compare_moves: list[str] | None = None,
) -> ForensicMoveAnalysis:
    board_after = board_before.copy(stack=True)
    if played_move is not None:
        board_after.push(played_move)

    fp_before = build_position_fingerprint(board_before)
    fp_after = build_position_fingerprint(board_after)
    tactical_before = build_tactical_snapshot(board_before)
    tactical_after = build_tactical_snapshot(board_after)
    delta = build_position_delta(
        board_before,
        board_after,
        before_snapshot=tactical_before,
        after_snapshot=tactical_after,
    )
    reply = await _strongest_reply(
        board_after,
        result.eval_after,
        pool=pool,
        depth=depth,
        deep=detail == "forensic",
    )
    mechanisms, signatures = _mechanism_evidence(result, reply, tactical_after)
    forced_line = _principal_line(board_after, result.eval_after.pv)

    requested_candidates: list[str] = []
    if detail == "forensic":
        if result.played_san:
            requested_candidates.append(result.played_san)
        if result.best_move_san and result.best_move_san != result.played_san:
            requested_candidates.append(result.best_move_san)
    for requested in compare_moves or []:
        if requested not in requested_candidates:
            requested_candidates.append(requested)
    if len(requested_candidates) > 8:
        requested_candidates = requested_candidates[:8]

    comparisons: list[CandidateEvidence] = []
    for requested in requested_candidates:
        comparisons.append(
            await _candidate_evidence(board_before, requested, pool=pool, depth=depth)
        )

    evidence = ForensicEvidence(
        detail=detail,
        position_before=fp_before,
        position_after_played=fp_after,
        tactical_before=tactical_before,
        tactical_after_played=tactical_after,
        strongest_reply=reply,
        position_delta=delta,
        mechanism_evidence=mechanisms,
        evidence_signatures=signatures,
        forced_line=forced_line,
        candidate_comparisons=comparisons,
        stability={
            "classification_verified": result.classification_verified,
            "action_equivalent": result.action_equivalent,
            "is_engine_best": result.is_engine_best,
            "requested_depth": depth,
            "searched_depth_before": result.eval_before.searched_depth or result.eval_before.depth,
            "searched_depth_after": result.eval_after.searched_depth or result.eval_after.depth,
        },
    )
    payload = result.model_dump(
        exclude={"same_action_type", "same_outcome", "within_cp_threshold"}
    )
    return ForensicMoveAnalysis(**payload, forensics=evidence)
