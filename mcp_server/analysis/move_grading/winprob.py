"""Win-probability helper used by move-grading loss calculation.

The dispatcher fetches win probability via the injected function so tests can
swap in a fake. Originally called directly via ``core.winprob.win_prob``;
preserved here for back-compat.
"""

from __future__ import annotations

from core.winprob import win_prob

__all__ = ["win_prob_fn"]


def win_prob_fn(cp: float) -> float:
    """White-POV win-probability from centipawn advantage. 0..100.

    Delegates to :func:`core.winprob.win_prob`; lifted into the
    move-grading package so the strategies don't need to know about
    ``core`` directly.
    """
    return win_prob(cp)
