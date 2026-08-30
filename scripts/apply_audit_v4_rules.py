from __future__ import annotations

from pathlib import Path


PATH = Path("mcp_server/rules.py")


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    i = text.find(start)
    if i < 0:
        raise RuntimeError(f"start marker not found: {start!r}")
    j = text.find(end, i)
    if j < 0:
        raise RuntimeError(f"end marker not found: {end!r}")
    return text[:i] + replacement.rstrip() + "\n\n" + text[j:]


text = PATH.read_text(encoding="utf-8")

rule_status = '''@dataclass
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
'''
text = replace_block(text, "@dataclass\nclass RuleStatus:", "def is_locked_dead_position", rule_status)

mating = '''def _can_side_force_checkmate(board: chess.Board, color: chess.Color) -> bool:
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
'''
text = replace_block(text, "def _can_side_force_checkmate", "def is_terminal_position", mating)

rule_eval = '''def choose_recommended_action(
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
'''
text = replace_block(text, "def evaluate_rule_status", "def truncate_pv_at_terminal", rule_eval)

PATH.write_text(text, encoding="utf-8")
print("audit v4 rules migration applied")
