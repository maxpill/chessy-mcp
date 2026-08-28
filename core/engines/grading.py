from .types import MoveClass


def classify_centipawn_loss(loss: int) -> MoveClass:
    """Grade a move by how much eval it gave up vs. the engine's best (mover's POV, centipawns)."""
    if loss < 20:
        return MoveClass.BEST
    if loss < 50:
        return MoveClass.GOOD
    if loss < 100:
        return MoveClass.INACCURACY
    if loss < 300:
        return MoveClass.MISTAKE
    return MoveClass.BLUNDER
