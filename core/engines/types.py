from enum import StrEnum

from pydantic import BaseModel


class MoveClass(StrEnum):
    BEST = "best"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"


class Arrow(BaseModel):
    """A coaching arrow to draw on the board (squares as 'e2'/'e4'; kind drives colour client-side)."""

    from_sq: str
    to_sq: str
    kind: str


class Eval(BaseModel):
    """Engine evaluation from White's point of view.

    wdl is a 3-tuple of (wins, draws, losses) per-mille (i.e. 1000 = 100%)
    when Stockfish UCI_ShowWDL is enabled. Only populated when the engine
    surfaces WDL in its info lines; otherwise None.
    """

    cp: int | None = None
    mate: int | None = None
    best_move: str | None = None
    pv: list[str] = []
    depth: int = 0
    wdl: tuple[int, int, int] | None = None


MATE_SCORE = 100_000


def white_cp(ev: Eval) -> int:
    """Collapse an Eval to a single White-POV centipawn number (mate encoded as a large value)."""
    if ev.mate is not None:
        if ev.mate == 0:
            return ev.cp if ev.cp is not None else MATE_SCORE
        return (MATE_SCORE - abs(ev.mate)) * (1 if ev.mate > 0 else -1)
    return ev.cp if ev.cp is not None else 0


class MoveAnalysis(BaseModel):
    played: str
    move_class: MoveClass
    centipawn_loss: int
    eval_before: Eval
    eval_after: Eval
    best_move_san: str | None = None
    best_line_san: str | None = None
    played_line_san: str | None = None
