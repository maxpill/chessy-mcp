"""Deterministic position-integrity and defender-geometry evidence.

This module builds on the existing forensic primitives without adding engine
searches. Its purpose is to make screenshot/FEN verification and common
"but that piece was defended" questions auditable from board facts alone.
"""

from __future__ import annotations

from typing import Literal

import chess

from mcp_server.analysis.forensics import (
    PIECE_NAMES,
    build_position_delta,
    build_position_fingerprint,
    build_tactical_snapshot,
)
from mcp_server.models.forensics import (
    DefenderLoadEvidence,
    ForensicEval,
    ForcingMoveEvidence,
    PieceEvidence,
    PieceSafetyDelta,
    PositionDelta,
    PositionForensicEvidence,
    TacticalHangingEvidence,
    TacticalSnapshot,
)
from mcp_server.models.mcpeval import MCPEval


def _color_name(color: chess.Color) -> Literal["white", "black"]:
    return "white" if color == chess.WHITE else "black"


def _piece_label(piece: chess.Piece, square: chess.Square) -> str:
    return f"{_color_name(piece.color)}_{PIECE_NAMES[piece.piece_type]}@{chess.square_name(square)}"


def _piece_evidence(board: chess.Board, square: chess.Square, piece: chess.Piece) -> PieceEvidence:
    return PieceEvidence(
        color=_color_name(piece.color),
        piece=PIECE_NAMES[piece.piece_type],
        square=chess.square_name(square),
        attackers=len(board.attackers(not piece.color, square)),
        defenders=len(board.attackers(piece.color, square)),
    )


def _captured_piece(board: chess.Board, move: chess.Move) -> chess.Piece | None:
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        return board.piece_at(move.to_square + offset)
    return board.piece_at(move.to_square)


def _capture_evidence(board: chess.Board, move: chess.Move) -> ForcingMoveEvidence:
    captured = _captured_piece(board, move)
    return ForcingMoveEvidence(
        uci=move.uci(),
        san=board.san(move),
        is_check=board.gives_check(move),
        is_capture=True,
        captured_piece=(
            f"{_color_name(captured.color)}_{PIECE_NAMES[captured.piece_type]}"
            if captured is not None
            else None
        ),
        promotion=PIECE_NAMES.get(move.promotion) if move.promotion else None,
    )


def _defender_load(board: chess.Board, square: chess.Square, piece: chess.Piece) -> DefenderLoadEvidence:
    defended_targets: list[str] = []
    attacked_targets: list[str] = []
    sole_defense_targets: list[str] = []
    for target_square in board.attacks(square):
        target = board.piece_at(target_square)
        if target is None or target.color != piece.color or target.piece_type == chess.KING:
            continue
        label = _piece_label(target, target_square)
        defended_targets.append(label)
        if board.attackers(not piece.color, target_square):
            attacked_targets.append(label)
            defenders = board.attackers(piece.color, target_square)
            if len(defenders) == 1 and square in defenders:
                sole_defense_targets.append(label)
    return DefenderLoadEvidence(
        color=_color_name(piece.color),
        piece=PIECE_NAMES[piece.piece_type],
        square=chess.square_name(square),
        attacked_by=len(board.attackers(not piece.color, square)),
        defended_targets=sorted(defended_targets),
        attacked_targets=sorted(attacked_targets),
        sole_defense_targets=sorted(sole_defense_targets),
    )


def _tactical_hanging_candidates(board: chess.Board) -> list[TacticalHangingEvidence]:
    out: list[TacticalHangingEvidence] = []
    seen: set[tuple[chess.Square, str]] = set()
    for capture in list(board.legal_moves):
        if not board.is_capture(capture) or board.is_en_passant(capture):
            continue
        target = board.piece_at(capture.to_square)
        if target is None or target.piece_type == chess.KING:
            continue
        nominal_defenders = len(board.attackers(target.color, capture.to_square))
        if nominal_defenders <= 0:
            continue

        post = board.copy(stack=True)
        post.push(capture)
        legal_recaptures: list[str] = []
        for reply in post.legal_moves:
            if post.is_capture(reply) and reply.to_square == capture.to_square:
                legal_recaptures.append(post.san(reply))
        if legal_recaptures:
            continue

        key = (capture.to_square, capture.uci())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            TacticalHangingEvidence(
                target=_piece_evidence(board, capture.to_square, target),
                capture=_capture_evidence(board, capture),
                nominal_defenders=nominal_defenders,
                legal_immediate_recaptures=[],
            )
        )
    return sorted(out, key=lambda item: (item.target.square, item.capture.san))


def _defender_sort_key(item: DefenderLoadEvidence) -> tuple[str, str, str]:
    return item.color, item.square, item.piece


def build_rich_tactical_snapshot(board: chess.Board) -> TacticalSnapshot:
    base = build_tactical_snapshot(board)
    attacked_defenders: list[DefenderLoadEvidence] = []
    overloaded: list[DefenderLoadEvidence] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        load = _defender_load(board, square, piece)
        if load.attacked_by > 0 and load.attacked_targets:
            attacked_defenders.append(load)
        if len(load.attacked_targets) >= 2:
            overloaded.append(load)

    return base.model_copy(
        update={
            "tactically_hanging_candidates": _tactical_hanging_candidates(board),
            "attacked_defenders": sorted(attacked_defenders, key=_defender_sort_key),
            "overloaded_defender_candidates": sorted(overloaded, key=_defender_sort_key),
        }
    )


def _labels(items: list[PieceEvidence]) -> set[str]:
    return {f"{item.color}_{item.piece}@{item.square}" for item in items}


def _open_files(board: chess.Board) -> set[str]:
    files: set[str] = set()
    for file_index in range(8):
        has_pawn = any(
            board.piece_at(chess.square(file_index, rank)) is not None
            and board.piece_at(chess.square(file_index, rank)).piece_type == chess.PAWN  # type: ignore[union-attr]
            for rank in range(8)
        )
        if not has_pawn:
            files.add(chess.FILE_NAMES[file_index])
    return files


def _pawn_squares(board: chess.Board, color: chess.Color) -> set[str]:
    return {chess.square_name(square) for square in board.pieces(chess.PAWN, color)}


def _king_ring_attacks(board: chess.Board, color: chess.Color) -> int:
    king = board.king(color)
    if king is None:
        return 0
    ring = set(board.attacks(king)) | {king}
    return sum(1 for square in ring if board.attackers(not color, square))


def _piece_safety_changes(before: chess.Board, after: chess.Board) -> list[PieceSafetyDelta]:
    changes: list[PieceSafetyDelta] = []
    for square, before_piece in before.piece_map().items():
        after_piece = after.piece_at(square)
        if after_piece != before_piece or before_piece.piece_type == chess.KING:
            continue
        attackers_before = len(before.attackers(not before_piece.color, square))
        attackers_after = len(after.attackers(not before_piece.color, square))
        defenders_before = len(before.attackers(before_piece.color, square))
        defenders_after = len(after.attackers(before_piece.color, square))
        if (attackers_before, defenders_before) == (attackers_after, defenders_after):
            continue
        changes.append(
            PieceSafetyDelta(
                target=_piece_label(before_piece, square),
                attackers_before=attackers_before,
                attackers_after=attackers_after,
                defenders_before=defenders_before,
                defenders_after=defenders_after,
            )
        )
    return sorted(changes, key=lambda item: item.target)


def build_rich_position_delta(
    before: chess.Board,
    after: chess.Board,
    *,
    before_snapshot: TacticalSnapshot | None = None,
    after_snapshot: TacticalSnapshot | None = None,
) -> PositionDelta:
    before_snapshot = before_snapshot or build_rich_tactical_snapshot(before)
    after_snapshot = after_snapshot or build_rich_tactical_snapshot(after)
    base = build_position_delta(
        before,
        after,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
    )

    before_en_prise = _labels(before_snapshot.en_prise_pieces)
    after_en_prise = _labels(after_snapshot.en_prise_pieces)
    before_pins = _labels(before_snapshot.pinned_pieces)
    after_pins = _labels(after_snapshot.pinned_pieces)
    before_open = _open_files(before)
    after_open = _open_files(after)

    pawn_changes: list[str] = []
    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        before_pawns = _pawn_squares(before, color)
        after_pawns = _pawn_squares(after, color)
        pawn_changes.extend(f"{name}_pawn_removed@{sq}" for sq in sorted(before_pawns - after_pawns))
        pawn_changes.extend(f"{name}_pawn_added@{sq}" for sq in sorted(after_pawns - before_pawns))

    return base.model_copy(
        update={
            "newly_en_prise_pieces": sorted(after_en_prise - before_en_prise),
            "resolved_en_prise_pieces": sorted(before_en_prise - after_en_prise),
            "removed_pins": sorted(before_pins - after_pins),
            "piece_safety_changes": _piece_safety_changes(before, after),
            "opened_files": sorted(after_open - before_open),
            "closed_files": sorted(before_open - after_open),
            "pawn_structure_changes": pawn_changes,
            "king_ring_attack_delta_white": _king_ring_attacks(after, chess.WHITE)
            - _king_ring_attacks(before, chess.WHITE),
            "king_ring_attack_delta_black": _king_ring_attacks(after, chess.BLACK)
            - _king_ring_attacks(before, chess.BLACK),
        }
    )


def enrich_position_eval(
    result: MCPEval,
    board: chess.Board,
    *,
    detail: Literal["coach", "forensic"],
) -> ForensicEval:
    snapshot = build_rich_tactical_snapshot(board)
    forensic = PositionForensicEvidence(
        detail=detail,
        position=build_position_fingerprint(board),
        tactical_snapshot=snapshot,
    )

    if detail == "forensic" and result.best_move and not board.is_game_over(claim_draw=False):
        try:
            move = chess.Move.from_uci(result.best_move.lower())
        except (ValueError, chess.InvalidMoveError):
            move = None
        if move is not None and move in board.legal_moves:
            post = board.copy(stack=True)
            san = board.san(move)
            post.push(move)
            post_snapshot = build_rich_tactical_snapshot(post)
            forensic = forensic.model_copy(
                update={
                    "best_move_uci": move.uci(),
                    "best_move_san": san,
                    "position_after_best": build_position_fingerprint(post),
                    "tactical_after_best": post_snapshot,
                    "best_move_delta": build_rich_position_delta(
                        board,
                        post,
                        before_snapshot=snapshot,
                        after_snapshot=post_snapshot,
                    ),
                }
            )

    return ForensicEval(**result.model_dump(), forensics=forensic)
