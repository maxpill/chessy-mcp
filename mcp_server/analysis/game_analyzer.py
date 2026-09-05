"""``GameAnalyzer`` service class: orchestrates full-game PGN analysis.

The analyzer uses constructor-injected dependencies so the MCP entry point can
stay thin and tests can supply engine/cache stubs without booting Stockfish.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import chess
import chess.pgn

from core.engines.openings import lookup_opening

if TYPE_CHECKING:
    from mcp.server.mcpserver import Context

from mcp_server.analysis.game_coaching import build_game_coaching_evidence
from mcp_server.analysis.game_validation import GameMetadata, extract_game_metadata
from mcp_server.analysis.mainline_parser import parse_mainline
from mcp_server.analysis.result_reconciliation import reconcile_result
from mcp_server.analysis.trailing_ply_reconciliation import reconcile_trailing_plies
from mcp_server.engine import (
    _build_identity,
    _gather_evaluate_positions_bounded,
    _get_analyzer_pool,
)
from mcp_server.models import MCPEval
from mcp_server.models.game_coaching import ForensicGameAnalysisResult
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
                result_inferred=None,
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
                coaching=None,
            )

        eval_pairs = await self._evaluate_positions(
            positions,
            depth,
            pool,
            requested_depth=raw_requested_depth,
            history_complete="complete",
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

        if metrics is not None:
            await metrics.record(
                "analyze_game",
                (time.time() - t0) * 1000,
                cache_hit=all_cached,
            )

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
            turning_points=game_metrics.turning_points,
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
