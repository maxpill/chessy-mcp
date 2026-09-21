"""Beam-search rerank over per-cell OCR candidates.

The sidecar fires multiple M3 OCR passes on the same image; for every
half-move (``ply``, ``side``) each pass proposes at most one SAN. We
aggregate those proposals into a list of
``CellCandidates = [(ply, side, [(san, ocr_score), ...])]`` and rerank
them through this module.

Scoring per candidate SAN:

    score(sequence, ply) =
        Σ ocr_score(i)   for i ∈ [0..ply]
      + downstream_bonus    (small positive signal when the next ply's
                              top candidate remains legal)
      - engine_plausibility_delta   (only when ``engine_plausibility``
                              is "tiebreak_only" and the top-2 candidates
                              are within 0.15 score of each other)

Stockfish is used as a **weak tiebreaker**, not a correctness gate: a
``-8.4`` cp loss is a legitimate human blunder and must not invalidate
the OCR.

The rerank supports three resolve modes:

    - ``strict``       — any per-cell ambiguity with delta < 0.2 raises
                         :class:`NeedsReviewError`.
    - ``auto``         — default. Resolves silently if exactly one legal
                         full sequence survives at top score; otherwise
                         returns ``needs_review`` with a populated
                         :attr:`BeamResult.uncertainties`.
    - ``best_effort``  — always returns the top-scoring legal sequence;
                         never raises; low-confidence plies are surfaced
                         with confidence < 0.5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal
from collections.abc import Callable, Sequence

import chess


log = logging.getLogger("chessy_mcp.ocr.beam_search")


ResolveMode = Literal["strict", "auto", "best_effort"]
EnginePlausibility = Literal["off", "tiebreak_only"]

DOWNSTREAM_BONUS: float = 0.3
DOWNSTREAM_LOOKAHEAD: int = 1
TIEBREAK_DELTA: float = 0.15
AMBIGUITY_DELTA: float = 0.2
LOW_CONFIDENCE_THRESHOLD: float = 0.5
MAX_ENGINE_EVALS: int = 5


@dataclass(frozen=True)
class CellCandidate:
    """One candidate SAN at one half-move position."""

    ply: int
    side: Literal["white", "black"]
    san: str
    score: float


@dataclass(frozen=True)
class Uncertainty:
    """One ambiguous half-move flagged by the legal-sequence beam reranker."""

    ply: int
    side: Literal["white", "black"]
    selected: str
    selected_confidence: float
    sequence_confidence: float | None = None
    alternatives: tuple[tuple[str, float], ...] = ()
    reason: str = "handwriting_ambiguity"
    crop_base64: str | None = None


@dataclass(frozen=True)
class BeamResult:
    """Final output of the beam reranker."""

    selected_path: tuple[str, ...]
    uncertainties: tuple[Uncertainty, ...]
    confidence: float
    status: Literal["ok", "needs_review"]
    unique_legal_path: bool = True
    all_moves_legal: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)


class NeedsReviewError(Exception):
    """Raised when ``resolve_ambiguities="strict"`` and any ply is ambiguous."""

    def __init__(self, uncertainties: Sequence[Uncertainty]) -> None:
        super().__init__("Strict mode: one or more plies could not be resolved unambiguously")
        self.uncertainties = tuple(uncertainties)


class BeamRerankError(Exception):
    """Raised when the beam empties — every continuation became illegal."""


type EngineEvalFn = Callable[[chess.Board, chess.Board, str], float]
"""Optional Stockfish callback. Receives (before, after, last_san) and
returns centipawn score from the side-to-move's perspective."""

type BeamEntry = tuple[chess.Board, tuple[str, ...], float, chess.Board | None, str | None]
"""One beam-search entry — the board after the last move, the SAN path
through the prefix, its accumulated score, the parent board, and the SAN
that was just played (carried so engine tiebreaks don't need to replay
the prefix)."""


def _side_at(ply: int) -> Literal["white", "black"]:
    return "white" if ply % 2 == 1 else "black"


def _parse_with_disambiguation(board: chess.Board, san: str) -> chess.Move:
    """Parse SAN, falling back to a piece-type scan when disambiguation is missing."""
    try:
        return board.parse_san(san)
    except chess.AmbiguousMoveError:
        pass
    except (chess.InvalidMoveError, ValueError):
        raise

    if not san or san[0] not in "KQRBN":
        raise chess.AmbiguousMoveError(san, list(board.legal_moves))
    piece_type_map = {
        "K": chess.KING,
        "Q": chess.QUEEN,
        "R": chess.ROOK,
        "B": chess.BISHOP,
        "N": chess.KNIGHT,
    }
    piece_type = piece_type_map[san[0]]
    tail = san[-2:]
    if tail not in chess.SQUARE_NAMES:
        raise chess.AmbiguousMoveError(san, list(board.legal_moves))
    target_square = chess.SQUARE_NAMES.index(tail)
    stripped_target = san.lstrip("KQRBN").replace("x", "")

    for move in board.legal_moves:
        if move.to_square != target_square:
            continue
        piece = board.piece_at(move.from_square)
        if piece is None or piece.piece_type != piece_type:
            continue
        candidate_san = board.san(move)
        if candidate_san.lstrip("KQRBN").replace("x", "") == stripped_target:
            return move
    raise chess.AmbiguousMoveError(san, list(board.legal_moves))


def _try_legal(board: chess.Board, san: str) -> chess.Move | None:
    try:
        return _parse_with_disambiguation(board, san)
    except (chess.AmbiguousMoveError, chess.InvalidMoveError, ValueError, IndexError):
        return None


def _find_elimination_reason(
    ply_idx: int,
    alt_san: str,
    best_path: tuple[str, ...],
) -> tuple[str, float]:
    """Determine at which downstream ply an alternative candidate becomes illegal.

    Returns (reason_str, sequence_confidence).
    """
    board = chess.Board()
    for i in range(ply_idx):
        mv = _try_legal(board, best_path[i])
        if mv is None:
            return "alternative path illegal prefix", 0.99
        board.push(mv)

    alt_mv = _try_legal(board, alt_san)
    if alt_mv is None:
        return "immediately illegal at this ply", 1.0
    board.push(alt_mv)

    for j in range(ply_idx + 1, len(best_path)):
        downstream_san = best_path[j]
        downstream_mv = _try_legal(board, downstream_san)
        if downstream_mv is None:
            move_num = (j // 2) + 1
            dot = "." if (j % 2 == 0) else "..."
            return f"alternative makes later {move_num}{dot}{downstream_san} impossible", 0.999
        board.push(downstream_mv)

    return "alternative has lower overall score", 0.85


def _aggregate_plies(
    cell_candidates: Sequence[CellCandidate],
    *,
    expand_visual: bool = False,
) -> dict[int, list[CellCandidate]]:
    """Group candidates by ply; normalize SAN and optionally expand confusions."""
    from mcp_server.parsers.san_normalize import expand_visual_hypotheses, normalize_token

    grouped: dict[int, list[CellCandidate]] = {}
    for cand in cell_candidates:
        existing = grouped.setdefault(cand.ply, [])
        norm_san, _ = normalize_token(cand.san, language="pl")
        clean_cand = CellCandidate(
            ply=cand.ply,
            side=cand.side,
            san=norm_san,
            score=cand.score,
        )
        if not any(e.san == clean_cand.san for e in existing):
            existing.append(clean_cand)

    if expand_visual:
        for cands in list(grouped.values()):
            if len(cands) == 1:
                cand = cands[0]
                for alt_san, _ in expand_visual_hypotheses(cand.san, cand.score):
                    if not any(e.san == alt_san for e in cands):
                        cands.append(
                            CellCandidate(
                                ply=cand.ply,
                                side=cand.side,
                                san=alt_san,
                                score=round(cand.score * 0.25, 3),
                            )
                        )
    return grouped


def _max_engine_evals(num_cells: int) -> int:
    return min(MAX_ENGINE_EVALS, max(0, num_cells))


def beam_rescore(
    cell_candidates: Sequence[CellCandidate],
    *,
    beam_width: int = 15,
    resolve: ResolveMode = "auto",
    engine_plausibility: EnginePlausibility = "off",
    engine_eval: EngineEvalFn | None = None,
    expand_visual: bool = False,
) -> BeamResult:
    """Rerank the per-cell candidates and return a single best legal path."""
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")
    if not cell_candidates:
        return BeamResult(
            selected_path=(),
            uncertainties=(),
            confidence=0.0,
            status="ok",
            unique_legal_path=True,
            all_moves_legal=True,
            notes=("empty_candidates",),
        )

    grouped = _aggregate_plies(cell_candidates, expand_visual=expand_visual)
    plies = sorted(grouped.keys())
    if not plies:
        return BeamResult(
            selected_path=(),
            uncertainties=(),
            confidence=0.0,
            status="ok",
            unique_legal_path=True,
            all_moves_legal=True,
            notes=("empty_grouped",),
        )

    initial_board = chess.Board()
    beam: list[BeamEntry] = [(initial_board.copy(stack=False), (), 0.0, None, None)]
    used_engine_evals = 0
    notes: list[str] = []
    engine_budget = _max_engine_evals(len(plies)) if engine_plausibility == "tiebreak_only" else 0

    for idx, ply in enumerate(plies):
        next_beam: list[BeamEntry] = []
        cell_options = grouped[ply]
        for board, path, score, _parent, _last_san in beam:
            for cand in cell_options:
                move = _try_legal(board, cand.san)
                if move is None:
                    continue
                new_board = board.copy(stack=False)
                new_board.push(move)
                bonus = 0.0
                if idx + DOWNSTREAM_LOOKAHEAD < len(plies):
                    next_ply_options = grouped[plies[idx + DOWNSTREAM_LOOKAHEAD]]
                    if next_ply_options:
                        top_next = next_ply_options[0]
                        if _try_legal(new_board, top_next.san) is not None:
                            bonus = DOWNSTREAM_BONUS
                new_score = score + cand.score + bonus
                next_beam.append((new_board, (*path, cand.san), new_score, board, cand.san))

        if not next_beam:
            raise BeamRerankError(
                f"Beam emptied at ply {ply}: no candidate SAN was legal from any prefix."
            )

        if (
            engine_plausibility == "tiebreak_only"
            and engine_eval is not None
            and used_engine_evals < engine_budget
        ):
            scored = sorted(next_beam, key=lambda x: x[2], reverse=True)
            if len(scored) >= 2 and (scored[0][2] - scored[1][2]) < TIEBREAK_DELTA:
                top_after, _, _, top_before, top_last_san = scored[0]
                if top_before is not None and top_last_san is not None:
                    cp_before = engine_eval(top_before, top_before, top_last_san)
                    cp_after = engine_eval(top_before, top_after, top_last_san)
                    if cp_after < cp_before - 3.0:
                        next_beam = [
                            (
                                (b, p, s - 1.0, parent, last)
                                if b is top_after
                                else (b, p, s, parent, last)
                            )
                            for (b, p, s, parent, last) in next_beam
                        ]
                        notes.append(f"engine_tiebreak_demoted@ply{ply}")
                    used_engine_evals += 1

        next_beam.sort(key=lambda x: x[2], reverse=True)
        beam = next_beam[:beam_width]

    best_path = beam[0][1]
    best_score = beam[0][2]
    second_score = beam[1][2] if len(beam) > 1 else best_score - 1.0

    uncertainties: list[Uncertainty] = []
    has_ambiguous_surviving_path = False

    for idx, ply in enumerate(plies):
        side = _side_at(ply)
        selected_san = best_path[idx] if idx < len(best_path) else ""
        if not selected_san:
            continue

        per_san: dict[str, list[float]] = {}
        for _board, path, score, _parent, _last_san in beam:
            if idx >= len(path):
                continue
            san = path[idx]
            per_san.setdefault(san, []).append(score)

        top_score = max(per_san.get(selected_san, [best_score - 1.0]))
        competitive_sans = {
            san for san, scores in per_san.items() if max(scores) >= top_score - AMBIGUITY_DELTA
        }

        # Case 1: Multiple candidates survived competitively in the beam
        if len(competitive_sans) > 1:
            runner_up = max(
                (s for san, scores in per_san.items() if san != selected_san for s in scores),
                default=top_score - 1.0,
            )
            delta = top_score - runner_up
            has_ambiguous_surviving_path = True
            alternatives = sorted(
                ((san, max(scores)) for san, scores in per_san.items() if san != selected_san),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            selected_conf = max(0.0, min(1.0, 1.0 - delta))
            reason = (
                "low_confidence"
                if selected_conf < LOW_CONFIDENCE_THRESHOLD
                else "handwriting_ambiguity"
            )
            uncertainties.append(
                Uncertainty(
                    ply=ply,
                    side=side,
                    selected=selected_san,
                    selected_confidence=round(selected_conf, 3),
                    sequence_confidence=0.5,
                    alternatives=tuple((san, round(score, 3)) for san, score in alternatives),
                    reason=reason,
                )
            )
        # Case 2: Only 1 SAN survived competitively; report eliminated alternatives
        else:
            initial_cands = grouped[ply]
            for alt in initial_cands:
                if alt.san != selected_san and alt.score >= 0.2:
                    reason, seq_conf = _find_elimination_reason(idx, alt.san, best_path)
                    uncertainties.append(
                        Uncertainty(
                            ply=ply,
                            side=side,
                            selected=selected_san,
                            selected_confidence=round(max(0.5, 1.0 - alt.score), 3),
                            sequence_confidence=round(seq_conf, 3),
                            alternatives=((alt.san, round(alt.score, 3)),),
                            reason=reason,
                        )
                    )

    confidence = max(0.0, min(1.0, (best_score - second_score) / max(1.0, abs(best_score))))
    confidence = round(confidence, 3)

    unique_legal_path = not has_ambiguous_surviving_path

    status: Literal["ok", "needs_review"]
    if resolve == "strict" and has_ambiguous_surviving_path:
        ambig_uncertainties = [
            u for u in uncertainties if u.sequence_confidence is None or u.sequence_confidence < 0.9
        ]
        raise NeedsReviewError(ambig_uncertainties or uncertainties)

    if resolve == "best_effort":
        status = "ok"
    else:
        status = "needs_review" if has_ambiguous_surviving_path else "ok"

    return BeamResult(
        selected_path=best_path,
        uncertainties=tuple(uncertainties),
        confidence=confidence,
        status=status,
        unique_legal_path=unique_legal_path,
        all_moves_legal=True,
        notes=tuple(notes),
    )


def candidates_from_pairs(
    pairs: Sequence[tuple[int, str, float]],
) -> list[CellCandidate]:
    """Build ``CellCandidate`` list from ``(ply, san, score)`` triples."""
    out: list[CellCandidate] = []
    for ply, san, score in pairs:
        out.append(
            CellCandidate(
                ply=ply,
                side=_side_at(ply),
                san=san,
                score=float(score),
            )
        )
    return out


__all__ = [
    "AMBIGUITY_DELTA",
    "DOWNSTREAM_BONUS",
    "LOW_CONFIDENCE_THRESHOLD",
    "MAX_ENGINE_EVALS",
    "TIEBREAK_DELTA",
    "BeamRerankError",
    "BeamResult",
    "CellCandidate",
    "EngineEvalFn",
    "EnginePlausibility",
    "NeedsReviewError",
    "ResolveMode",
    "Uncertainty",
    "beam_rescore",
    "candidates_from_pairs",
]
