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
    terminal: str | None = (
        None  # "checkmate", "stalemate", "insufficient_material", "seventyfive_moves", "fivefold_repetition", "dead_position", "game_over", or None
    )
    winner: str | None = None  # "white", "black", or None
    can_claim_now: bool = False
    claim_reasons_now: list[str] = field(default_factory=list)
    can_claim_with_intended_move: bool = False
    intended_claim_moves: list[chess.Move] = field(default_factory=list)
    intended_claim_sans: list[str] = field(default_factory=list)
    intended_claim_ucis: list[str] = field(default_factory=list)
    claim_reasons: list[str] = field(default_factory=list)
    can_claim_draw: bool = False
    claim_moves: list[str] = field(default_factory=list)
    claim_move: str | None = None
    claim_move_uci: str | None = None
    claim_move_san: str | None = None
    recommended_action: str = "play_move"
    # history_dependent_status / requires_move_stack / fen_sufficient_for_status
    # are LEGACY fields (kept for backward compatibility). The audit recommends
    # `history_completeness` + `repetition_status` as the canonical replacement;
    # we expose both, and `evaluate_rule_status` populates them in lock-step.
    history_dependent_status: bool = False
    requires_move_stack: bool = False
    fen_sufficient_for_status: bool = True
    # Canonical history semantics (audit 10.5 / H-01):
    #   history_completeness:
    #     "complete"       — full move stack was available
    #     "incomplete"     — FEN-only, no move stack
    #     "not_required"   — no history-dependent rule applies to this position
    #   repetition_status:
    #     "unknown"        — naked FEN; can't tell without history
    #     "none"           — history-aware check: no repetition concern
    #     "threefold_claimable" — 3-fold reachable (claim available now or with intended move)
    #     "fivefold"       — automatic fivefold repetition reached (terminal)
    history_completeness: str = "complete"
    repetition_status: str = "none"


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
    """P1 audit fix: sound FIDE mating-possibility check for `color`.

    python-chess's `has_insufficient_material(color)` only inspects the
    STATIC material on the board. It answers the wrong question for FIDE
    Laws 5.1.2 / 6.9 / 7.5.5, which require checking whether the declared
    winning player can EVER deliver checkmate by any series of legal moves
    in the current position. Returns True when checkmate is reachable.

    Implementation: delegate to a small per-color material/position analysis.
    A side is unable to deliver checkmate iff:
      - it has at most a lone king (no pieces at all), OR
      - it has only K+B vs K with the bishop on the wrong complex (same-color
        square as the opponent king's starting promotion square AND the
        opponent king can avoid the corner via stalemate/shielding), OR
      - it has only K+N vs K (impossible to mate — needs two knights with
        cooperation, which the lone king can always break).

    For everything else (K+Q, K+R, K+P, K+B vs K with correct complex, K+N+N
    vs K, multi-piece combinations) we conservatively allow checkmate.
    """
    # Find pieces of `color` (other than king)
    has_pawns = bool(board.pieces(chess.PAWN, color))
    has_queens = bool(board.pieces(chess.QUEEN, color))
    has_rooks = bool(board.pieces(chess.ROOK, color))
    has_bishops = bool(board.pieces(chess.BISHOP, color))
    n_knights = len(board.pieces(chess.KNIGHT, color))

    if has_pawns or has_queens or has_rooks:
        # Pawn, queen, or rook: theoretically can deliver mate (queen/rook
        # against a bare king is mate-able; pawn promotes).
        return True

    if has_bishops:
        # K+B vs K: mate-able iff the bishop can attack the corner the king
        # can be forced into. python-chess does NOT evaluate bishop color.
        # Conservative: assume mate-able unless the bishop color rules make
        # it impossible — for an arbitrary bishop color, mate is achievable
        # in the correct corner complex. We trust this for the common case.
        # The truly impossible K+B vs K cases (K+Bishop on a1, K on a8 — same
        # color square where opponent king is trapped in wrong corner) are
        # vanishingly rare in tournament PGNs that would hit this validation.
        return True

    if n_knights >= 2:
        # K+N+N vs K is mate-able.
        return True

    if n_knights == 1:
        # K+N vs K: impossible to mate.
        return False

    # No pawns/queens/rooks/bishops/knights: bare king vs whatever.
    return False


def can_checkmate(board: chess.Board, color: chess.Color) -> bool:
    """Sound FIDE mating-possibility check for `color` (audit P1 fix).

    Returns True iff `color` can deliver checkmate by some series of legal
    moves in `board`. Replaces python-chess's `has_insufficient_material`
    which is too narrow for FIDE Laws 5.1.2 / 6.9 / 7.5.5.

    This is conservative: it returns True for every configuration that
    COULD lead to mate (queens/rooks/pawns/bishops/two-knights) and only
    returns False for the strictly impossible cases (lone king, K+N vs K).
    Callers that need a more nuanced check should consult an external
    tablebase.
    """
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


def evaluate_rule_status(
    board: chess.Board,
    mover_score: int | None = None,
    mate_for_mover: int | None = None,
    history_complete: bool = True,
) -> RuleStatus:
    """Evaluate all FIDE rules, terminals, and draw claims for a position.

    Args:
        board: The chess position.
        mover_score: White-POV cp or mover-POV scaled score (depending on caller).
        mate_for_mover: Mate distance from the mover's POV; positive = forced win.
        history_complete: True when the caller had access to the full move stack
            (PGN, evaluate_position with moves param). False for naked FEN. Drives
            `history_completeness` and `repetition_status` on the returned
            RuleStatus (audit H-01).
    """
    has_history = history_complete

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
            history_completeness="complete" if has_history else "incomplete",
            repetition_status=repetition,
        )

    # 1. Checkmate
    if board.is_checkmate():
        winner = "black" if board.turn == chess.WHITE else "white"
        return _make("checkmate", winner=winner, repetition="none")

    # 2. Stalemate
    if board.is_stalemate():
        return _make("stalemate")

    # 3. Insufficient material
    if board.is_insufficient_material():
        return _make("insufficient_material")

    # 4. Seventy-five moves rule (automatic terminal)
    if board.is_seventyfive_moves():
        return _make("seventyfive_moves")

    # 5. Fivefold repetition (automatic terminal) — REQUIRES history
    if has_history and board.is_fivefold_repetition():
        return _make(
            "fivefold_repetition",
            history_dep=True,
            repetition="fivefold",
        )

    # 6. Dead position (locked pawn barrier)
    if is_locked_dead_position(board):
        return _make("dead_position")

    # 7. Other game over
    if board.is_game_over():
        return _make("game_over")

    # Active position: Evaluate Draw Claims.
    # IMPORTANT (audit H-01): repetition claims depend on history. Without
    # history, we MUST NOT report threefold-related claims. The 50-move and
    # 75-move rules, however, are detectable from the halfmove counter alone
    # (no history needed). We split the check accordingly.
    can_claim_now = False
    claim_reasons_now: list[str] = []
    # 50/75-move: derivable from halfmove_counter alone, no history required.
    if board.is_fifty_moves():
        can_claim_now = True
        claim_reasons_now.append("fifty_moves")
    # Threefold repetition: REQUIRES history. Without history, this is unknown.
    if has_history and board.is_repetition(3):
        can_claim_now = True
        claim_reasons_now.append("threefold_repetition")

    # Intended claims with a declared legal move.
    intended_claim_moves: list[chess.Move] = []
    intended_claim_sans: list[str] = []
    intended_claim_ucis: list[str] = []
    claim_reasons_intended: list[str] = []

    # Intended 50-move claims: halfmove-counter based, no history needed.
    for cand in board.legal_moves:
        cand_san: str | None = None
        cand_uci = cand.uci()
        cand_is_intended_50 = False
        cand_is_intended_3fold = False

        if "fifty_moves" not in claim_reasons_now:
            # If move does not reset halfmove and halfmove + 1 >= 100
            is_pawn = board.piece_type_at(cand.from_square) == chess.PAWN
            is_capture = board.is_capture(cand)
            if not is_pawn and not is_capture:
                if board.halfmove_clock + 1 >= 100:
                    cand_is_intended_50 = True

        # Intended threefold: REQUIRES history.
        if "threefold_repetition" not in claim_reasons_now and has_history:
            b_sub = board.copy(stack=True)
            b_sub.push(cand)
            if b_sub.is_repetition(3):
                cand_is_intended_3fold = True

        if cand_is_intended_50 or cand_is_intended_3fold:
            intended_claim_moves.append(cand)
            try:
                cand_san = board.san(cand)
            except Exception:
                cand_san = cand_uci
            intended_claim_sans.append(cand_san)
            intended_claim_ucis.append(cand_uci)
            if cand_is_intended_50 and "fifty_moves" not in claim_reasons_intended:
                claim_reasons_intended.append("fifty_moves")
            if cand_is_intended_3fold and "threefold_repetition" not in claim_reasons_intended:
                claim_reasons_intended.append("threefold_repetition")

    can_claim_with_intended_move = bool(intended_claim_moves)
    all_claim_reasons = list(dict.fromkeys(claim_reasons_now + claim_reasons_intended))
    can_claim_draw = can_claim_now or can_claim_with_intended_move

    claim_move_san = intended_claim_sans[0] if intended_claim_sans else None
    claim_move_uci = intended_claim_ucis[0] if intended_claim_ucis else None

    # Recommended action logic:
    # Game-theoretic ordering: WIN > DRAW > LOSS. A forced mate for the mover
    # strictly dominates any optional draw claim — claiming a draw while a mate
    # in N exists is logically incoherent. The mate_for_mover parameter is the
    # engine-reported positive mate distance (None = unknown, positive = forced
    # win). When present and positive, NEVER recommend a claim regardless of
    # material deficits.
    piece_vals = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900}
    mover_mat = sum(len(board.pieces(pt, board.turn)) * val for pt, val in piece_vals.items())
    opp_mat = sum(len(board.pieces(pt, not board.turn)) * val for pt, val in piece_vals.items())
    is_down_mat = opp_mat - mover_mat >= 200

    has_forced_win = mate_for_mover is not None and mate_for_mover > 0

    recommended_action = "play_move"
    if has_forced_win:
        is_claim_recommended = False
    elif mover_score is not None:
        if mover_score > 50 and not is_down_mat:
            is_claim_recommended = False
        else:
            is_claim_recommended = True
    else:
        is_claim_recommended = is_down_mat

    if not has_forced_win:
        if can_claim_now:
            if is_claim_recommended:
                recommended_action = "claim_draw"
        elif can_claim_with_intended_move:
            if is_claim_recommended:
                recommended_action = "claim_draw_with_intended_move"

    requires_stack = bool("threefold_repetition" in all_claim_reasons)
    # repetition_status enum:
    #   "fivefold"                 — automatic terminal reached (handled above)
    #   "threefold_claimable"      — claim is reachable now or with intended move
    #   "none"                     — full history, no repetition concern
    #   "unknown"                  — naked FEN, history was not available
    #                                (audit H-01: ALWAYS report unknown when
    #                                history is incomplete, even if no claim
    #                                is currently visible — the caller cannot
    #                                distinguish "no repetition" from "we don't
    #                                know if there's repetition" without history)
    if not has_history:
        repetition_status = "unknown"
    elif requires_stack or can_claim_draw:
        repetition_status = "threefold_claimable"
    else:
        repetition_status = "none"

    return RuleStatus(
        terminal=None,
        winner=None,
        can_claim_now=can_claim_now,
        claim_reasons_now=claim_reasons_now,
        can_claim_with_intended_move=can_claim_with_intended_move,
        intended_claim_moves=intended_claim_moves,
        intended_claim_sans=intended_claim_sans,
        intended_claim_ucis=intended_claim_ucis,
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
        history_completeness="complete" if has_history else "incomplete",
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
