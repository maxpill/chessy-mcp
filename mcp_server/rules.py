"""Central rules and action evaluation engine for Chess MCP.

Provides unified rule checks, FIDE draw claims, dead position detection,
action outcomes, candidate move ranking, and metadata validation across all tools.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import chess
import chess.pgn

_STATUS_FLAG_NAMES = [
    (getattr(chess, "STATUS_NO_WHITE_KING", 1 << 0), "NO_WHITE_KING"),
    (getattr(chess, "STATUS_NO_BLACK_KING", 1 << 1), "NO_BLACK_KING"),
    (getattr(chess, "STATUS_TOO_MANY_KINGS", 1 << 2), "TOO_MANY_KINGS"),
    (getattr(chess, "STATUS_TOO_MANY_WHITE_PAWNS", 1 << 3), "TOO_MANY_WHITE_PAWNS"),
    (getattr(chess, "STATUS_TOO_MANY_BLACK_PAWNS", 1 << 4), "TOO_MANY_BLACK_PAWNS"),
    (getattr(chess, "STATUS_PAWNS_ON_BACKRANK", 1 << 5), "PAWNS_ON_BACKRANK"),
    (getattr(chess, "STATUS_TOO_MANY_WHITE_PIECES", 1 << 6), "TOO_MANY_WHITE_PIECES"),
    (getattr(chess, "STATUS_TOO_MANY_BLACK_PIECES", 1 << 7), "TOO_MANY_BLACK_PIECES"),
    (getattr(chess, "STATUS_BAD_CASTLING_RIGHTS", 1 << 8), "INVALID_CASTLING_RIGHTS"),
    (getattr(chess, "STATUS_INVALID_EP_SQUARE", 1 << 9), "INVALID_EP_SQUARE"),
    (getattr(chess, "STATUS_OPPOSITE_CHECK", 1 << 10), "OPPOSITE_CHECK"),
    (getattr(chess, "STATUS_EMPTY", 1 << 11), "EMPTY_BOARD"),
    (getattr(chess, "STATUS_RACE_CHECK", 1 << 12), "RACE_CHECK"),
    (getattr(chess, "STATUS_RACE_OVER", 1 << 13), "RACE_OVER"),
    (getattr(chess, "STATUS_TOO_MANY_CHECKERS", 1 << 14), "TOO_MANY_CHECKERS"),
]


def format_fen_status_errors(status_mask: int) -> str:
    """Format numeric python-chess status bitmask into human-readable error reasons."""
    reasons = [name for flag, name in _STATUS_FLAG_NAMES if flag and (status_mask & flag)]
    if not reasons:
        return f"INVALID_POSITION_STATUS_{status_mask}"
    return ", ".join(reasons)


class ChessActionType(StrEnum):
    PLAY_MOVE = "play_move"
    CLAIM_DRAW_NOW = "claim_draw"
    CLAIM_DRAW_WITH_INTENDED_MOVE = "claim_draw_with_intended_move"
    GAME_OVER = "game_over"


@dataclass
class RuleStatus:
    terminal: str | None = None
    winner: str | None = None
    can_claim_now: bool = False
    claim_reasons_now: list[str] = field(default_factory=list)
    can_claim_with_intended_move: bool = False
    intended_claim_moves: list[chess.Move] = field(default_factory=list)
    intended_claim_sans: list[str] = field(default_factory=list)
    intended_claim_ucis: list[str] = field(default_factory=list)
    intended_claim_reasons_by_uci: dict[str, list[str]] = field(default_factory=dict)
    claim_reasons: list[str] = field(default_factory=list)
    can_claim_draw: bool = False
    claim_moves: list[str] = field(default_factory=list)
    claim_move: str | None = None
    claim_move_uci: str | None = None
    claim_move_san: str | None = None
    recommended_action: str = "play_move"
    history_dependent_status: bool = False
    requires_move_stack: bool = False
    fen_sufficient_for_status: bool = True
    history_completeness: str = "incomplete"
    repetition_status: str = "unknown"

def is_locked_dead_position(board: chess.Board) -> bool:
    """Detect dead positions caused by completely locked pawn structures
    where NEITHER player can checkmate by any series of legal moves (FIDE 5.2.2).

    A position is dead iff BOTH colors:
    - have no Q, R, or N on the board;
    - have no pawn that can move (from either color's perspective);
    - have no bishop that can move (from either color's perspective);
    - cannot reach an enemy pawn or cross to the opposite side via king maneuvers.
    """
    # If standard insufficient material, it's already dead
    if board.is_insufficient_material():
        return True

    # If any heavy pieces (Q, R) or knights are present on EITHER side, the position
    # is not locked — a knight can jump or a piece can sacrifice to mate.
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.ROOK, chess.KNIGHT):
            if board.pieces(pt, color):
                return False

    # Inspect pawn/bishop mobility for BOTH sides (turn-independent).
    # board.legal_moves only enumerates the side-to-move, so we must flip turn.
    saved_turn = board.turn
    try:
        for color in (chess.WHITE, chess.BLACK):
            board.turn = color
            for m in board.legal_moves:
                if board.piece_type_at(m.from_square) == chess.PAWN:
                    return False
        any_bishop = bool(board.pieces(chess.BISHOP, chess.WHITE) or board.pieces(chess.BISHOP, chess.BLACK))
        if any_bishop:
            for color in (chess.WHITE, chess.BLACK):
                board.turn = color
                for m in board.legal_moves:
                    if board.piece_type_at(m.from_square) == chess.BISHOP:
                        return False
    finally:
        board.turn = saved_turn

    # In King + Pawn positions with no pawn moves:
    # Check if any king can reach an enemy pawn or infiltrate to the other side.
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    if white_king_sq is None or black_king_sq is None:
        return False

    # Pawn attack squares
    white_pawn_attacks = chess.SquareSet()
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        white_pawn_attacks |= board.attacks_mask(sq)

    black_pawn_attacks = chess.SquareSet()
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        black_pawn_attacks |= board.attacks_mask(sq)

    # Flood fill reachable squares for White King:
    # White King can only step on empty squares that are not attacked by Black pawns/Black King
    occupied_or_attacked_by_black = set(chess.SquareSet(board.occupied)) | set(
        chess.SquareSet(black_pawn_attacks)
    )
    white_reachable: set[int] = {white_king_sq}
    queue: deque[int] = deque([white_king_sq])
    while queue:
        curr = queue.popleft()
        for neighbor in chess.SquareSet(chess.BB_KING_ATTACKS[curr]):
            if neighbor not in white_reachable and neighbor not in occupied_or_attacked_by_black:
                white_reachable.add(neighbor)
                queue.append(neighbor)

    # Flood fill reachable squares for Black King:
    occupied_or_attacked_by_white = set(chess.SquareSet(board.occupied)) | set(
        chess.SquareSet(white_pawn_attacks)
    )
    black_reachable: set[int] = {black_king_sq}
    queue = deque([black_king_sq])
    while queue:
        curr = queue.popleft()
        for neighbor in chess.SquareSet(chess.BB_KING_ATTACKS[curr]):
            if neighbor not in black_reachable and neighbor not in occupied_or_attacked_by_white:
                black_reachable.add(neighbor)
                queue.append(neighbor)

    # Can White King attack any Black pawn?
    black_pawns = set(board.pieces(chess.PAWN, chess.BLACK))
    white_can_attack_black_pawn = any(
        bool(chess.SquareSet(chess.BB_KING_ATTACKS[sq]) & white_reachable) for sq in black_pawns
    )
    if white_can_attack_black_pawn:
        return False

    # Can Black King attack any White pawn?
    white_pawns = set(board.pieces(chess.PAWN, chess.WHITE))
    black_can_attack_white_pawn = any(
        bool(chess.SquareSet(chess.BB_KING_ATTACKS[sq]) & black_reachable) for sq in white_pawns
    )
    if black_can_attack_white_pawn:
        return False

    # Can either king cross to the opposite rank area?
    # If White King cannot reach rank 7/8 and Black King cannot reach rank 1/2
    white_max_rank = max((chess.square_rank(sq) for sq in white_reachable), default=0)
    black_min_rank = min((chess.square_rank(sq) for sq in black_reachable), default=7)
    if white_max_rank < black_min_rank:
        return True

    return False


def _can_side_force_checkmate(board: chess.Board, color: chess.Color) -> bool:
    """Conservative FIDE mating-possibility predicate.

    False is returned only when checkmate is impossible by every legal
    continuation. This matters for Laws 5.1.2, 6.9 and 7.5.5 because a false
    negative converts a win on time, resignation or rules infraction into a draw.
    """
    pawns = len(board.pieces(chess.PAWN, color))
    rooks = len(board.pieces(chess.ROOK, color))
    queens = len(board.pieces(chess.QUEEN, color))
    bishops = list(board.pieces(chess.BISHOP, color))
    knights = len(board.pieces(chess.KNIGHT, color))

    if not (pawns or rooks or queens or bishops or knights):
        return False
    if pawns or rooks or queens:
        return True

    opponent_nonking = sum(
        len(board.pieces(pt, not color))
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )

    # Opponent material can legally block escape squares. Therefore K+N or
    # K+B is not generally impossible when the opponent is not a bare king.
    if opponent_nonking:
        return True

    # From here the opponent is a bare king.
    if knights >= 2:
        return True
    if knights == 1 and not bishops:
        return False

    if bishops:
        if len(bishops) == 1 and knights == 0:
            return False
        if knights:
            return True
        complexes = {
            (chess.square_rank(sq) + chess.square_file(sq)) & 1
            for sq in bishops
        }
        return len(complexes) >= 2

    return False


def can_checkmate(board: chess.Board, color: chess.Color) -> bool:
    """Return whether `color` can mate by some legal continuation."""
    return _can_side_force_checkmate(board, color)

def is_terminal_position(board: chess.Board) -> bool:
    """Single source of truth for "the position is game over".

    Combines python-chess's terminal checks with FIDE 5.2.2 dead-position
    detection. Every MCP tool that short-circuits on game-over MUST use this
    helper so a position's terminality cannot disagree between
    evaluate_position, top_moves, and classify_move.
    """
    if board.is_checkmate():
        return True
    if board.is_stalemate():
        return True
    if board.is_insufficient_material():
        return True
    if board.is_seventyfive_moves():
        return True
    if board.is_fivefold_repetition():
        return True
    if is_locked_dead_position(board):
        return True
    return False


def choose_recommended_action(
    board: chess.Board,
    *,
    can_claim_now: bool,
    can_claim_with_intended_move: bool,
    mover_score: int | None = None,
    mate_for_mover: int | None = None,
) -> str:
    """Choose one canonical legal root action for every MCP endpoint."""
    if not (can_claim_now or can_claim_with_intended_move):
        return "play_move"
    if mate_for_mover is not None and mate_for_mover > 0:
        return "play_move"

    piece_vals = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
    }
    mover_mat = sum(
        len(board.pieces(pt, board.turn)) * value
        for pt, value in piece_vals.items()
    )
    opp_mat = sum(
        len(board.pieces(pt, not board.turn)) * value
        for pt, value in piece_vals.items()
    )
    is_down_material = opp_mat - mover_mat >= 200

    if mover_score is None:
        claim_preferred = is_down_material
    else:
        claim_preferred = not (mover_score > 50 and not is_down_material)

    if not claim_preferred:
        return "play_move"
    if can_claim_now:
        return "claim_draw"
    return "claim_draw_with_intended_move"


def choose_recommended_action(
    board: chess.Board,
    *,
    can_claim_now: bool,
    can_claim_with_intended_move: bool,
    mover_score: int | None = None,
    mate_for_mover: int | None = None,
) -> str:
    """Choose one canonical legal root action for every MCP endpoint."""
    if not (can_claim_now or can_claim_with_intended_move):
        return "play_move"
    if mate_for_mover is not None and mate_for_mover > 0:
        return "play_move"

    piece_vals = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
    }
    mover_mat = sum(
        len(board.pieces(pt, board.turn)) * value
        for pt, value in piece_vals.items()
    )
    opp_mat = sum(
        len(board.pieces(pt, not board.turn)) * value
        for pt, value in piece_vals.items()
    )
    is_down_material = opp_mat - mover_mat >= 200

    if mover_score is None:
        claim_preferred = is_down_material
    else:
        claim_preferred = not (mover_score > 50 and not is_down_material)

    if not claim_preferred:
        return "play_move"
    if can_claim_now:
        return "claim_draw"
    return "claim_draw_with_intended_move"


def evaluate_rule_status(
    board: chess.Board,
    mover_score: int | None = None,
    mate_for_mover: int | None = None,
    history_complete: str | bool = "incomplete",
) -> RuleStatus:
    """Evaluate terminal rules and optional draw claims with explicit history provenance."""
    if isinstance(history_complete, bool):
        history_state = "complete" if history_complete else "incomplete"
    else:
        history_state = history_complete
    if history_state not in {"complete", "partial", "incomplete", "not_required"}:
        raise ValueError(f"INVALID_HISTORY_PROVENANCE: {history_state}")

    has_history = history_state in {"complete", "partial"}
    full_history = history_state == "complete"

    def _make(
        terminal: str | None,
        winner: str | None = None,
        recommended_action: str = "game_over",
        history_dep: bool = False,
        repetition: str = "none",
    ) -> RuleStatus:
        return RuleStatus(
            terminal=terminal,
            winner=winner,
            recommended_action=recommended_action,
            history_dependent_status=history_dep,
            requires_move_stack=history_dep,
            fen_sufficient_for_status=not history_dep,
            history_completeness=(
                "not_required"
                if terminal in {
                    "checkmate",
                    "stalemate",
                    "insufficient_material",
                    "seventyfive_moves",
                    "dead_position",
                }
                else history_state
            ),
            repetition_status=repetition,
        )

    if board.is_checkmate():
        winner = "black" if board.turn == chess.WHITE else "white"
        return _make("checkmate", winner=winner)
    if board.is_stalemate():
        return _make("stalemate")
    if board.is_insufficient_material():
        return _make("insufficient_material")
    if board.is_seventyfive_moves():
        return _make("seventyfive_moves")
    if has_history and board.is_fivefold_repetition():
        return _make(
            "fivefold_repetition",
            history_dep=True,
            repetition="fivefold",
        )
    if is_locked_dead_position(board):
        return _make("dead_position")
    if board.is_game_over(claim_draw=False):
        return _make("game_over")

    can_claim_now = False
    claim_reasons_now: list[str] = []
    if board.is_fifty_moves():
        can_claim_now = True
        claim_reasons_now.append("fifty_moves")
    if has_history and board.is_repetition(3):
        can_claim_now = True
        claim_reasons_now.append("threefold_repetition")

    intended_claim_moves: list[chess.Move] = []
    intended_claim_sans: list[str] = []
    intended_claim_ucis: list[str] = []
    intended_claim_reasons: list[str] = []
    intended_claim_reasons_by_uci: dict[str, list[str]] = {}

    for cand in board.legal_moves:
        cand_uci = cand.uci()
        intended_50 = False
        intended_3 = False

        if "fifty_moves" not in claim_reasons_now:
            is_pawn = board.piece_type_at(cand.from_square) == chess.PAWN
            is_capture = board.is_capture(cand)
            if not is_pawn and not is_capture and board.halfmove_clock + 1 >= 100:
                intended_50 = True

        if "threefold_repetition" not in claim_reasons_now and has_history:
            child = board.copy(stack=True)
            child.push(cand)
            if child.is_repetition(3):
                intended_3 = True

        if not (intended_50 or intended_3):
            continue

        try:
            cand_san = board.san(cand)
        except Exception:
            cand_san = cand_uci

        reasons: list[str] = []
        if intended_50:
            reasons.append("fifty_moves")
            if "fifty_moves" not in intended_claim_reasons:
                intended_claim_reasons.append("fifty_moves")
        if intended_3:
            reasons.append("threefold_repetition")
            if "threefold_repetition" not in intended_claim_reasons:
                intended_claim_reasons.append("threefold_repetition")

        intended_claim_moves.append(cand)
        intended_claim_sans.append(cand_san)
        intended_claim_ucis.append(cand_uci)
        intended_claim_reasons_by_uci[cand_uci] = reasons

    can_claim_with_intended_move = bool(intended_claim_moves)
    all_claim_reasons = list(
        dict.fromkeys(claim_reasons_now + intended_claim_reasons)
    )
    can_claim_draw = can_claim_now or can_claim_with_intended_move
    claim_move_san = intended_claim_sans[0] if intended_claim_sans else None
    claim_move_uci = intended_claim_ucis[0] if intended_claim_ucis else None

    recommended_action = choose_recommended_action(
        board,
        can_claim_now=can_claim_now,
        can_claim_with_intended_move=can_claim_with_intended_move,
        mover_score=mover_score,
        mate_for_mover=mate_for_mover,
    )

    repetition_proven = bool(
        "threefold_repetition" in claim_reasons_now
        or "threefold_repetition" in intended_claim_reasons
    )
    if repetition_proven:
        repetition_status = "threefold_claimable"
    elif full_history:
        repetition_status = "none"
    else:
        repetition_status = "unknown"

    requires_stack = repetition_proven
    return RuleStatus(
        terminal=None,
        winner=None,
        can_claim_now=can_claim_now,
        claim_reasons_now=claim_reasons_now,
        can_claim_with_intended_move=can_claim_with_intended_move,
        intended_claim_moves=intended_claim_moves,
        intended_claim_sans=intended_claim_sans,
        intended_claim_ucis=intended_claim_ucis,
        intended_claim_reasons_by_uci=intended_claim_reasons_by_uci,
        claim_reasons=all_claim_reasons,
        can_claim_draw=can_claim_draw,
        claim_moves=intended_claim_sans,
        claim_move=claim_move_san,
        claim_move_uci=claim_move_uci,
        claim_move_san=claim_move_san,
        recommended_action=recommended_action,
        history_dependent_status=requires_stack,
        requires_move_stack=requires_stack,
        fen_sufficient_for_status=not requires_stack,
        history_completeness=history_state,
        repetition_status=repetition_status,
    )

def truncate_pv_at_terminal(board: chess.Board, pv_uci: list[str]) -> list[str]:
    """Ensure a principal variation (PV) does not continue past an automatic terminal state."""
    b = board.copy(stack=True)
    truncated: list[str] = []
    for uci in pv_uci:
        try:
            m = chess.Move.from_uci(uci.lower())
            if m not in b.legal_moves:
                break
            truncated.append(m.uci())
            b.push(m)
            # Check if this move reached an automatic terminal
            if (
                b.is_checkmate()
                or b.is_stalemate()
                or b.is_insufficient_material()
                or b.is_seventyfive_moves()
                or b.is_fivefold_repetition()
                or is_locked_dead_position(b)
            ):
                break
        except Exception:
            break
    return truncated


def validate_mating_possibility(
    board: chess.Board,
    result: str | None,
    termination: str | None,
) -> tuple[str | None, list[str]]:
    """Validate resignation, time-forfeit, and rules-infraction (second illegal move) outcomes
    under FIDE Laws 5.1.2, 6.9 & 7.5.5.
    If the declared winning player cannot checkmate by any possible series of legal moves,
    the game is drawn and a metadata warning is generated.
    """
    warnings: list[str] = []
    normalized_result = result

    if not termination or not result:
        return normalized_result, warnings

    term_clean = termination.strip().lower()
    # P1 audit fix: this used to interpret every string containing "time" as a
    # time forfeit, including innocuous phrases like "Normal time control".
    # Use the SAME strict regex as `normalize_termination()`: only an explicit
    # forfeit marker (forfeit / expired / out of / flag fell / clock flagged)
    # qualifies. Plain "time control" or "increment" alone does NOT.
    is_resignation = "resign" in term_clean
    is_time_forfeit = bool(
        re.search(
            r"\btime\s*(?:forfeit|expired|exhausted|loss)\b"
            r"|\bout\s+of\s+time\b"
            r"|\bflag\s*(?:fell|fall|dropped)\b"
            r"|\blost\s+on\s+time\b"
            r"|\bclock\s+(?:flagged|expired)\b",
            term_clean,
        )
    )
    is_rules_infraction = bool(
        re.search(
            r"\brules?\s+infraction\b|\b(?:second\s+)?illegal\s+move\b|\binfraction\b|\billegal\b",
            term_clean,
        )
    )

    if is_resignation or is_time_forfeit or is_rules_infraction:
        if is_resignation:
            term_name = "Resignation"
            article = "5.1.2"
        elif is_time_forfeit:
            term_name = "Time forfeit"
            article = "6.9"
        else:
            term_name = "Rules infraction / second illegal move"
            article = "7.5.5"

        # P1 audit fix: FIDE 5.1.2 / 6.9 / 7.5.5 require evaluating whether the
        # declared winning player can EVER deliver checkmate by ANY series of
        # legal moves — not just whether they currently have sufficient
        # material by python-chess's narrow definition. `has_insufficient_material`
        # only looks at the static material count; a king + 2 knights vs lone
        # king is `has_insufficient_material=False` but a 1-0 result is still
        # legal because checkmate is achievable. Conversely, K+B vs K with the
        # bishops on the same color as the king is "insufficient" by FIDE but
        # NOT by python-chess (it doesn't distinguish bishop color vs promotion
        # constraints). Use the rules-aware `can_checkmate(board, color)` helper
        # below for a sound check.
        if result == "1-0":
            # White is declared winner. Check if White can mate.
            if not can_checkmate(board, chess.WHITE):
                warnings.append(
                    f"{term_name} by Black declared 1-0, but White has insufficient material to deliver checkmate; normalized to draw (1/2-1/2) under FIDE Article {article}."
                )
                normalized_result = "1/2-1/2"
        elif result == "0-1":
            # Black is declared winner. Check if Black can mate.
            if not can_checkmate(board, chess.BLACK):
                warnings.append(
                    f"{term_name} by White declared 0-1, but Black has insufficient material to deliver checkmate; normalized to draw (1/2-1/2) under FIDE Article {article}."
                )
                normalized_result = "1/2-1/2"

    return normalized_result, warnings
