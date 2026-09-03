"""Typed evaluation value object.

The codebase's `core.engines.types.Eval` is a stateful dataclass internal to the
engine pool — it's not a clean domain type to expose beyond that boundary.
`Evaluation` here is the immutable value object all higher-level layers
(position evaluator, top-moves, analyze-game, MCPEval builder) work with.

It's intentionally narrow: only the post-state fields a single Position produces.
Aggregate multi-position analyses belong at the service layer (GameAnalyzer),
not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import chess


@dataclass(frozen=True)
class ScoreLine:
    """A single engine score report for a position.

    Score semantics (White-POV, matching Stockfish's UCI convention):

    - `cp`: integer centipawns (positive = White advantage). None if mate reported.
    - `mate`: integer mate distance (positive = White mates). None if cp reported.
    - `depth`: searched depth at the time of the score.
    - `wdl`: Win/Draw/Loss per-mille, White-POV, when the engine surfaced it.
    """

    cp: int | None
    mate: int | None
    depth: int = 0
    wdl: tuple[int, int, int] | None = None

    @property
    def is_mate(self) -> bool:
        return self.mate is not None

    @property
    def is_centipawn(self) -> bool:
        return self.cp is not None and self.mate is None

    def mover_pov_score(self, turn: chess.Color) -> int | None:
        """Score reported from the mover's perspective instead of White's."""
        sign = 1 if turn == chess.WHITE else -1
        if self.cp is not None:
            return sign * self.cp
        if self.mate is not None:
            return sign * self.mate * 1000
        return None


@dataclass(frozen=True)
class Evaluation:
    """Complete evaluation result for one position.

    Carries both the static score AND the candidate moves considered (PV[0]).
    All fields are optional because callers may construct partially-populated
    eval objects (e.g. terminal positions with no candidate moves).
    """

    fen: str
    score: ScoreLine
    best_move: chess.Move | None = None
    pv: tuple[chess.Move, ...] = field(default_factory=tuple)
    pv_uci: tuple[str, ...] = field(default_factory=tuple)
    searched_depth: int = 0
    requested_depth: int | None = None
    terminal: str | None = None  # one of TERMINAL_STATUSES; None if non-terminal
    winner: str | None = None  # "white" | "black" | None for terminal positions

    def with_terminal(self, terminal: str, winner: str | None = None) -> Evaluation:
        return replace(self, terminal=terminal, winner=winner)

    def with_pv_extension(self, extra_uci: tuple[str, ...]) -> Evaluation:
        return replace(self, pv_uci=self.pv_uci + extra_uci)
