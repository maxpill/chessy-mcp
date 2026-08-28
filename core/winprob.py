from __future__ import annotations

import math

from core.engines.types import Eval, white_cp

_K = 0.00368208  # Lichess win-probability logistic constant
_CP_CLAMP = 2000  # beyond ±20 pawns the logistic is saturated; clamp so mate scores don't overflow exp


def win_prob(cp: float) -> float:
    """Win probability (0-100) from a player-POV centipawn eval — the standard Lichess logistic."""
    c = max(-_CP_CLAMP, min(_CP_CLAMP, cp))
    return 50 + 50 * (2 / (1 + math.exp(-_K * c)) - 1)


def win_prob_eval(ev: Eval, *, mover_is_white: bool = True) -> float:
    """Win probability (0-100) for the given side, from an Eval (mate encoded as a large cp)."""
    cp = white_cp(ev) * (1 if mover_is_white else -1)
    return win_prob(cp)


def move_loss_cp(before_cp: int, after_cp: int) -> float:
    """Win% a move gave up, from per-move player-POV centipawn evals (0 if the move did not lose ground)."""
    return max(0.0, win_prob(before_cp) - win_prob(after_cp))


def move_loss(before: Eval, after: Eval, *, mover_is_white: bool) -> float:
    """Win% the mover gave up across their move, from White-POV evals before and after it."""
    wp_before = win_prob_eval(before, mover_is_white=mover_is_white)
    wp_after = win_prob_eval(after, mover_is_white=mover_is_white)
    return max(0.0, wp_before - wp_after)
