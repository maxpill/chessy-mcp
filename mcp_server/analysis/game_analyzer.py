"""``GameAnalyzer`` service class: orchestrates full-game PGN analysis.

The analyzer uses constructor-injected dependencies so the MCP entry point can
stay thin and tests can supply engine/cache stubs without booting Stockfish.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import chess
import chess.pgn

from core.engines.openings import lookup_opening

if TYPE_CHECKING:
    from mcp.server.mcpserver import Context

from mcp_server.analysis.game_coaching import build_game_coaching_evidence
from mcp_server.analysis.game_critical_forensics import enrich_game_critical_forensics
from mcp_server.analysis.game_termination import build_game_termination_assessment
from mcp_server.analysis.game_validation import GameMetadata, extract_game_metadata
from mcp_server.analysis.mainline_parser import parse_mainline
from mcp_server.analysis.result_reconciliation import reconcile_result
from mcp_server.analysis.trailing_ply_reconciliation import reconcile_trailing_plies
from mcp_server.actions import build_best_action, build_legal_actions
from mcp_server.rules import evaluate_rule_status
from mcp_server.urls import lichess_urls
from mcp_server.engine import (
    _build_identity,
    _gather_evaluate_positions_bounded,
    _get_analyzer_pool,
)
from mcp_server.models import MCPEval
from mcp_server.models.game_coaching import (
    FinalPositionAssessment,
    ForensicGameAnalysisResult,
    GameCoachingEvidence,
)
from mcp_server.parsers import (
    _check_multiple_games,
    _extract_canonical_pgn_text,
    _extract_game_inner,
    _find_movetext_result,
    _sanitize_malformed_pgn_header_lines,
    _validate_strict_header_syntax,
    _validate_strict_mainline_surface,
)

# The service deliberately accepts local/TCP pools plus test doubles. Their
# runtime contract is structural at this injection boundary.
type EnginePool = Any
GameDetail = Literal["standard", "coach", "forensic"]
GamePerspective = Literal["white", "black"]


@dataclass
class GameMetrics:
    white_accuracy: float | None
    black_accuracy: float | None
    white_acpl: float | None
    black_acpl: float | None
    white_raw_acpl: float | None
    black_raw_acpl: float | None
    white_effective_acpl: float | None
    black_effective_acpl: float | None
    white_blunders: int
    white_mistakes: int
    white_inaccuracies: int
    black_blunders: int
    black_mistakes: int
    black_inaccuracies: int
    turning_points: list[Any]


def _finalize_coaching_evidence(
    pgn: str,
    coaching: GameCoachingEvidence,
) -> GameCoachingEvidence:
    """Attach post-mortem metadata at the analyzer boundary.

    Keeping this here makes ``GameAnalyzer`` and the MCP tool return the same
    rich evidence. The public tool remains a transport/error-translation layer
    instead of acquiring a second, subtly different coaching pipeline.
    """
    termination = build_game_termination_assessment(
        pgn,
        final_position=coaching.final_position,
    )
    signature_counts = Counter(
        signature
        for moment in coaching.critical_moments
        for signature in moment.evidence_signatures
    )
    reason_counts = Counter(
        reason for moment in coaching.critical_moments for reason in moment.reasons
    )
    self_reported = sorted(
        moment.ply for moment in coaching.critical_moments if moment.user_comment_raw
    )
    return coaching.model_copy(
        update={
            "termination": termination,
            "critical_evidence_signature_counts": dict(sorted(signature_counts.items())),
            "critical_reason_counts": dict(sorted(reason_counts.items())),
            "self_reported_critical_plies": self_reported,
        }
    )


def _compact_coaching_evidence(
    coaching: GameCoachingEvidence,
    *,
    is_minimal: bool,
) -> GameCoachingEvidence:
    compact_moments = []
    for m in coaching.critical_moments:
        updates: dict[str, Any] = {
            "opponent_forcing_moves_after_played": [],
            "newly_enabled_opponent_forcing_moves_after_played": [],
            "resolved_opponent_forcing_threat_candidates": [],
            "strengthened_opponent_forcing_moves": [],
            "weakened_opponent_forcing_moves": [],
            "forcing_move_semantic_transitions": [],
            "mate_in_one_moves_before": [],
            "opponent_mate_in_one_threats_if_pass_before": [],
            "opponent_mate_in_one_moves_after_played": [],
        }
        if is_minimal:
            updates["causal_trace"] = None
        compact_moments.append(m.model_copy(update=updates))

    coaching_updates: dict[str, Any] = {
        "critical_moments": compact_moments,
    }
    if is_minimal:
        coaching_updates["advantage_events"] = []
        coaching_updates["root_cause_links"] = []
        coaching_updates["failure_corpus"] = None
        coaching_updates["critical_evidence_signature_counts"] = {}
        coaching_updates["critical_reason_counts"] = {}
    return coaching.model_copy(update=coaching_updates)


def _build_fifty_move_short_circuit(
    *,
    game_result: str,
    requested_depth: int,
    positions: list[chess.Board],
) -> Callable[[chess.Board, int], MCPEval | None]:
    """Build a ``short_circuit`` callback for the FINAL 50-move-rule position.

    2026-09-08 audit Bug 8: when a PGN ends with a 50-move-rule draw,
    the final position has ``board.is_fifty_moves()`` True. Calling the
    engine on that position is wasted work — the player will claim a
    draw rather than play on. This callback synthesizes a terminal
    MCPEval (``status="fifty_moves"``, ``searched_depth=0``) for that
    specific position so the gather path skips the engine call.

    Only the LAST position in ``positions`` is short-circuited. An
    intermediate position with ``is_fifty_moves()`` True (e.g. halfmove
    149 with a winning continuation that walks into 75-move-draw on the
    next move) still needs a real engine eval so blunder detection can
    compare eval_before vs eval_after.

    Non-50-move positions and non-last positions return ``None``.
    """

    final_fen = positions[-1].fen() if positions else None

    def _cb(board: chess.Board, index: int) -> MCPEval | None:
        if game_result != "1/2-1/2":
            return None
        if index != len(positions) - 1:
            return None
        if not board.is_fifty_moves():
            return None
        if final_fen is not None and board.fen() != final_fen:
            return None
        rule_status = evaluate_rule_status(board, history_complete="complete")
        canonical_fen = board.fen()
        url, img = lichess_urls(canonical_fen)
        best_action = build_best_action(
            recommended_action="claim_draw",
            rule_status=rule_status,
            engine_eval=None,
            board=board,
            sign=1 if board.turn == chess.WHITE else -1,
        )
        legal_actions = build_legal_actions(
            rule_status=rule_status,
            engine_eval=None,
            board=board,
            legal_engine_moves=None,
        )
        rule_actions = [
            a
            for a in legal_actions
            if a.get("type") in ("claim_draw", "claim_draw_with_intended_move")
        ]
        return MCPEval(
            status="fifty_moves",
            cp=0,
            mate=None,
            best_move=None,
            pv=[],
            depth=0,
            requested_depth=requested_depth,
            searched_depth=0,
            can_claim_draw=True,
            claim_reasons=["fifty_moves"],
            can_claim_now=True,
            claim_reasons_now=["fifty_moves"],
            can_claim_with_intended_move=False,
            claim_moves=[],
            recommended_action="claim_draw",
            best_action="claim_draw",
            best_action_type="claim_draw",
            best_action_obj=best_action,
            legal_actions=legal_actions,
            legal_rule_actions=rule_actions,
            canonical_fen=canonical_fen,
            fen_was_canonicalized=False,
            decision_value={
                "outcome": "draw",
                "cp_equivalent": 0,
                "best_action": "claim_draw",
                "perspective": "white",
            },
            engine_eval={
                "cp": 0,
                "mate": None,
                "best_move": None,
                "pv": [],
                "depth": 0,
            },
            history_dependent_status=False,
            lichess_url_reproduces_history=True,
            requires_move_stack=False,
            fen_sufficient_for_status=True,
            history_completeness="complete",
            repetition_status="none",
            lichess_url=url,
            lichess_image=img,
        )

    return _cb


def _build_zero_ply_coaching(
    board: chess.Board,
    *,
    detail: Literal["coach", "forensic"],
    perspective: GamePerspective,
    scan_depth: int,
    pgn: str,
) -> GameCoachingEvidence:
    """Construct a minimal but real coaching block for a zero-ply PGN.

    The position is the initial/final board (the same since no moves
    were played). No engine call is made — ``searched_depth=0`` is set on
    the outer response. The coaching block carries a ``final_position``
    derived from the board's terminal status and a synthesized empty
    record set so the rich-mode response schema is satisfied.
    """
    legal_count = board.legal_moves.count()
    rule_status = evaluate_rule_status(board, history_complete="complete")
    is_terminal = board.is_game_over(claim_draw=False)
    continued_play = (not is_terminal) and legal_count > 0
    final_position = FinalPositionAssessment(
        perspective=perspective,
        position_terminal_by_rules=is_terminal,
        checkmate=board.is_checkmate(),
        stalemate=board.is_stalemate(),
        forced_mate=board.is_checkmate(),
        mate_distance=None,
        effective_cp=0,
        wdl=None,
        side_to_move="white" if board.turn == chess.WHITE else "black",
        legal_move_count=legal_count,
        board_legal_move_count=legal_count,
        continued_play_legal_under_rules=continued_play,
        can_claim_draw=rule_status.can_claim_draw,
        can_claim_now=rule_status.can_claim_now,
        claim_reasons_now=rule_status.claim_reasons_now,
        can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
        claim_moves=rule_status.claim_moves,
        recommended_action=rule_status.recommended_action,
        best_move_uci=None,
        best_move_san=None,
        defensive_resources_exist=continued_play,
        reasonable_resource_count=None,
        verification_depth=scan_depth if detail == "forensic" else None,
    )
    coaching = GameCoachingEvidence(
        detail=detail,
        perspective=perspective,
        critical_moments=[],
        game_segments=[],
        advantage_events=[],
        positive_moments=[],
        root_cause_links=[],
        final_position=final_position,
        scan_depth=scan_depth,
        verification_depth=scan_depth if detail == "forensic" else None,
        adaptive_escalation_depth=None,
    )
    return _finalize_coaching_evidence(pgn, coaching)


class GameAnalyzer:
    """End-to-end ``analyze_game`` orchestrator."""

    def __init__(
        self,
        get_pool: Callable[[Context | None], Awaitable[EnginePool]],
        evaluate_positions: Callable[..., Awaitable[list[tuple[MCPEval, bool]]]],
        compute_metrics: Callable[
            [list[chess.Board], list[chess.Move], list[MCPEval]], GameMetrics
        ],
        identity: Callable[[EnginePool], dict[str, Any]],
        engine_version: Callable[[EnginePool], str],
    ) -> None:
        self._get_pool = get_pool
        self._evaluate_positions = evaluate_positions
        self._compute_metrics = compute_metrics
        self._identity = identity
        self._engine_version = engine_version

    @classmethod
    def with_defaults(cls) -> GameAnalyzer:
        return cls(
            get_pool=_get_analyzer_pool,
            evaluate_positions=_gather_evaluate_positions_bounded,
            compute_metrics=_wrap_compute_game_metrics,
            identity=_build_identity,
            engine_version=_engine_version_str,
        )

    async def analyze(
        self,
        pgn: str,
        depth: int,
        *,
        strict: bool,
        ctx: Context | None,
        detail: GameDetail = "standard",
        perspective: GamePerspective = "white",
        max_critical_moments: int = 6,
        raw_requested_max_critical_moments: int | None = None,
        verbosity_mode: str = "full",
        metrics: Any | None = None,
    ) -> ForensicGameAnalysisResult:
        t0 = time.time()
        raw_requested_depth = depth
        depth = max(1, min(depth, 30))
        max_critical_moments = max(1, min(max_critical_moments, 7))

        sanitized_pgn, lexical_header_warnings = _sanitize_malformed_pgn_header_lines(
            pgn, strict=strict
        )
        _check_multiple_games(sanitized_pgn)
        if strict:
            _validate_strict_header_syntax(sanitized_pgn)
        canonical_pgn = _extract_canonical_pgn_text(sanitized_pgn)
        game = _extract_game_inner(canonical_pgn, strict=strict)
        if strict:
            _validate_strict_mainline_surface(canonical_pgn, game)

        (
            positions,
            moves,
            syntax_warnings,
            _ignored_from_parse,
            cleaned_movetext,
        ) = parse_mainline(canonical_pgn, game, strict=strict)

        result_movetext = _find_movetext_result(canonical_pgn)
        is_comment_only_input = bool(getattr(game, "comment_only_input", False))

        ignored_trailing_plies = reconcile_trailing_plies(
            canonical_pgn=canonical_pgn,
            cleaned_movetext=cleaned_movetext,
            moves=moves,
            game=game,
        )

        metadata = extract_game_metadata(
            canonical_pgn,
            game,
            strict=strict,
            lexical_warnings=lexical_header_warnings,
            syntax_warnings=syntax_warnings,
            is_comment_only_input=is_comment_only_input,
            result_movetext=result_movetext,
        )

        if ignored_trailing_plies > 0:
            ply_word = "ply" if ignored_trailing_plies == 1 else "plies"
            metadata.metadata_warnings.append(
                f"Movetext contained moves after game termination; "
                f"ignored {ignored_trailing_plies} trailing {ply_word}."
            )

        final_board = positions[-1]
        reconciled = reconcile_result(
            final_board,
            metadata,
            result_movetext=result_movetext,
            moves_count=len(moves),
            strict=strict,
        )
        metadata.metadata_warnings = reconciled.warnings

        if strict and not moves:
            if metadata.syntax_warnings:
                raise ValueError(
                    "STRICT_PGN_ERROR: PGN contains syntax normalization or move number mismatch: "
                    f"{metadata.syntax_warnings[0]}"
                )
            if metadata.metadata_warnings:
                raise ValueError(
                    "STRICT_PGN_ERROR: PGN contains metadata inconsistency: "
                    f"{metadata.metadata_warnings[0]}"
                )

        is_standard_start = game.board().fen() == chess.STARTING_FEN
        pool = await self._get_pool(ctx)
        engine_name_str = self._engine_version(pool)
        identity = self._identity(pool)

        if not moves:
            detected_opening, detected_eco = (
                lookup_opening([])[:2] if is_standard_start else (None, None)
            )
            # 2026-09-08 audit Bug 5: rich zero-ply PGNs must return a real
            # coaching block, not ``coaching=None``. Standard detail keeps
            # the cheap shape; coach/forensic attach a minimal coaching
            # payload built from the initial board state with no engine
            # call (searched_depth stays 0).
            zero_ply_coaching = None
            if detail != "standard":
                zero_ply_coaching = _build_zero_ply_coaching(
                    positions[-1],
                    detail=detail,
                    perspective=perspective,
                    scan_depth=raw_requested_depth,
                    pgn=pgn,
                )
            if zero_ply_coaching is not None and verbosity_mode in ("compact", "minimal"):
                zero_ply_coaching = _compact_coaching_evidence(
                    zero_ply_coaching, is_minimal=(verbosity_mode == "minimal")
                )
            if metrics is not None:
                await metrics.record("analyze_game", (time.time() - t0) * 1000, cache_hit=True)
            return ForensicGameAnalysisResult(
                total_plies=0,
                white_accuracy=None,
                black_accuracy=None,
                white_acpl=None,
                black_acpl=None,
                white_raw_acpl=None,
                black_raw_acpl=None,
                white_effective_acpl=None,
                black_effective_acpl=None,
                white_average_effective_loss=None,
                black_average_effective_loss=None,
                white_blunders=0,
                white_mistakes=0,
                white_inaccuracies=0,
                black_blunders=0,
                black_mistakes=0,
                black_inaccuracies=0,
                turning_points=[],
                white=metadata.white,
                black=metadata.black,
                event=metadata.event,
                site=metadata.site,
                date=metadata.date,
                round=metadata.round,
                result=reconciled.result or metadata.result_header or "*",
                result_header=metadata.result_header,
                result_header_raw=metadata.result_header_raw,
                result_movetext=result_movetext,
                result_inferred=reconciled.result_inferred,
                white_elo=metadata.white_elo,
                black_elo=metadata.black_elo,
                time_control=metadata.time_control,
                variant=metadata.variant,
                eco=detected_eco or metadata.eco_header,
                opening=detected_opening or metadata.opening_header,
                opening_header=metadata.opening_header,
                eco_header=metadata.eco_header,
                metadata_warnings=metadata.metadata_warnings,
                syntax_warnings=metadata.syntax_warnings,
                termination=reconciled.termination,
                termination_header=metadata.termination_header,
                requested_depth=raw_requested_depth,
                searched_depth=0,
                engine="Stockfish",
                engine_version=engine_name_str,
                **identity,
                accuracy_method="win_probability_logistic",
                mate_penalty_policy="1000_cp_mate_transition",
                coaching=zero_ply_coaching,
                requested_max_critical_moments=raw_requested_max_critical_moments or max_critical_moments,
                clamped_max_critical_moments=max_critical_moments,
                returned_critical_moments=len(zero_ply_coaching.critical_moments) if zero_ply_coaching else 0,
                empty_game_reason=getattr(metadata, "empty_game_reason", None),
                is_compact=(verbosity_mode in ("compact", "minimal")),
                is_minimal=(verbosity_mode == "minimal"),
            )

        batch_depth = depth
        if len(positions) > 70 and batch_depth > 12:
            batch_depth = 11 if len(positions) > 100 else 12

        eval_pairs = await self._evaluate_positions(
            positions,
            batch_depth,
            pool,
            requested_depth=raw_requested_depth,
            history_complete="complete",
            short_circuit=_build_fifty_move_short_circuit(
                game_result=reconciled.result,
                requested_depth=raw_requested_depth,
                positions=positions,
            ),
        )
        evals: list[MCPEval] = [ep[0] for ep in eval_pairs]
        all_cached = all(ep[1] for ep in eval_pairs)

        game_metrics = self._compute_metrics(positions, moves, evals)
        final_opening, final_eco, opening_disagreement, eco_disagreement = _detect_opening(
            moves, is_standard_start, metadata
        )
        if opening_disagreement is not None:
            metadata.metadata_warnings.append(opening_disagreement)
        if eco_disagreement is not None:
            metadata.metadata_warnings.append(eco_disagreement)

        if strict:
            if metadata.syntax_warnings:
                raise ValueError(
                    "STRICT_PGN_ERROR: PGN contains syntax normalization or move number mismatch: "
                    f"{metadata.syntax_warnings[0]}"
                )
            if metadata.metadata_warnings:
                raise ValueError(
                    "STRICT_PGN_ERROR: PGN contains metadata inconsistency: "
                    f"{metadata.metadata_warnings[0]}"
                )

        coaching = None
        if detail != "standard":
            coaching = await build_game_coaching_evidence(
                positions=positions,
                moves=moves,
                evals=evals,
                game=game,
                perspective=perspective,
                detail=detail,
                max_critical_moments=max_critical_moments,
                scan_depth=depth,
                pool=pool,
                evaluate_positions=self._evaluate_positions,
            )
            coaching = enrich_game_critical_forensics(
                coaching,
                positions=positions,
                evals=evals,
            )
            coaching = _finalize_coaching_evidence(pgn, coaching)

        if metrics is not None:
            await metrics.record(
                "analyze_game",
                (time.time() - t0) * 1000,
                cache_hit=all_cached,
            )

        if coaching is not None and verbosity_mode in ("compact", "minimal"):
            coaching = _compact_coaching_evidence(
                coaching, is_minimal=(verbosity_mode == "minimal")
            )
        turning_pts = [] if verbosity_mode == "minimal" else game_metrics.turning_points

        return ForensicGameAnalysisResult(
            total_plies=len(moves),
            white_accuracy=game_metrics.white_accuracy,
            black_accuracy=game_metrics.black_accuracy,
            white_acpl=game_metrics.white_acpl,
            black_acpl=game_metrics.black_acpl,
            white_raw_acpl=game_metrics.white_raw_acpl,
            black_raw_acpl=game_metrics.black_raw_acpl,
            white_effective_acpl=game_metrics.white_effective_acpl,
            black_effective_acpl=game_metrics.black_effective_acpl,
            white_average_effective_loss=game_metrics.white_effective_acpl,
            black_average_effective_loss=game_metrics.black_effective_acpl,
            white_blunders=game_metrics.white_blunders,
            white_mistakes=game_metrics.white_mistakes,
            white_inaccuracies=game_metrics.white_inaccuracies,
            black_blunders=game_metrics.black_blunders,
            black_mistakes=game_metrics.black_mistakes,
            black_inaccuracies=game_metrics.black_inaccuracies,
            turning_points=turning_pts,
            white=metadata.white,
            black=metadata.black,
            event=metadata.event,
            site=metadata.site,
            date=metadata.date,
            round=metadata.round,
            result=reconciled.result,
            result_header=metadata.result_header,
            result_header_raw=metadata.result_header_raw,
            result_movetext=result_movetext,
            result_inferred=reconciled.result_inferred,
            white_elo=metadata.white_elo,
            black_elo=metadata.black_elo,
            time_control=metadata.time_control,
            variant=metadata.variant,
            eco=final_eco,
            opening=final_opening,
            opening_header=metadata.opening_header,
            eco_header=metadata.eco_header,
            metadata_warnings=metadata.metadata_warnings,
            syntax_warnings=metadata.syntax_warnings,
            termination=reconciled.termination,
            termination_header=metadata.termination_header,
            requested_depth=raw_requested_depth,
            searched_depth=depth,
            engine="Stockfish",
            engine_version=engine_name_str,
            **identity,
            accuracy_method="win_probability_logistic",
            mate_penalty_policy="1000_cp_mate_transition",
            coaching=coaching,
            requested_max_critical_moments=raw_requested_max_critical_moments or max_critical_moments,
            clamped_max_critical_moments=max_critical_moments,
            returned_critical_moments=len(coaching.critical_moments) if coaching else 0,
            empty_game_reason=getattr(metadata, "empty_game_reason", None),
            is_compact=(verbosity_mode in ("compact", "minimal")),
            is_minimal=(verbosity_mode == "minimal"),
        )


def _engine_version_str(pool: Any) -> str:
    return getattr(pool, "engine_version", getattr(pool, "name", "Stockfish"))


def _wrap_compute_game_metrics(
    positions: list[chess.Board],
    moves: list[chess.Move],
    evals: list[MCPEval],
) -> GameMetrics:
    from mcp_server.tools.game_metrics import _compute_game_metrics as _impl

    (
        white_acc,
        black_acc,
        white_acpl,
        black_acpl,
        white_raw_acpl,
        black_raw_acpl,
        white_avg_eff,
        black_avg_eff,
        (white_blunders, white_mistakes, white_inaccuracies),
        (black_blunders, black_mistakes, black_inaccuracies),
        turning_points,
    ) = _impl(positions, moves, evals)
    return GameMetrics(
        white_accuracy=white_acc,
        black_accuracy=black_acc,
        white_acpl=white_acpl,
        black_acpl=black_acpl,
        white_raw_acpl=white_raw_acpl,
        black_raw_acpl=black_raw_acpl,
        white_effective_acpl=white_avg_eff,
        black_effective_acpl=black_avg_eff,
        white_blunders=white_blunders,
        white_mistakes=white_mistakes,
        white_inaccuracies=white_inaccuracies,
        black_blunders=black_blunders,
        black_mistakes=black_mistakes,
        black_inaccuracies=black_inaccuracies,
        turning_points=turning_points,
    )


def _detect_opening(
    moves: list[chess.Move],
    is_standard_start: bool,
    metadata: GameMetadata,
) -> tuple[str | None, str | None, str | None, str | None]:
    uci_moves = [m.uci() for m in moves]
    if is_standard_start:
        detected_opening, detected_eco, _ = lookup_opening(uci_moves)
    else:
        detected_opening, detected_eco = None, None
    final_opening = detected_opening or metadata.opening_header
    final_eco = detected_eco or metadata.eco_header

    opening_disagreement: str | None = None
    if detected_opening and metadata.opening_header:
        det_clean = detected_opening.strip().lower()
        hdr_clean = metadata.opening_header.strip().lower()
        det_base = det_clean.split(":")[0].strip()
        hdr_base = hdr_clean.split(":")[0].strip()
        is_parent_child = (
            det_clean.startswith(hdr_clean)
            or hdr_clean.startswith(det_clean)
            or det_base == hdr_base
        )
        if not is_parent_child:
            opening_disagreement = (
                f"Opening header '{metadata.opening_header}' disagrees with "
                f"detected opening '{detected_opening}'"
            )

    eco_disagreement: str | None = None
    if (
        detected_eco
        and metadata.eco_header
        and detected_eco.strip().upper() != metadata.eco_header.strip().upper()
    ):
        eco_disagreement = (
            f"ECO header '{metadata.eco_header}' disagrees with detected ECO '{detected_eco}'"
        )

    return final_opening, final_eco, opening_disagreement, eco_disagreement
