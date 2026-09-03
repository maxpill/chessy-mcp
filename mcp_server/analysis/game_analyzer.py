"""``GameAnalyzer`` service class — orchestrates full-game PGN analysis.

Extracted from ``mcp_server.tools.analyze_game``. Encapsulates the
end-to-end ``analyze_game`` flow as a class with constructor-injected
dependencies (engine pool callable, cache, grader) so the tool entry
point in ``mcp_server.tools.analyze_game`` can shrink to a thin
dispatcher.

Public surface:

  * :class:`GameAnalyzer` — the service. ``analyze(pgn, depth, strict)``
    returns a populated :class:`GameAnalysisResult`.

Dependencies (injected via the constructor):

  * ``get_pool``: async callable ``(ctx: Context | None) -> AnalyzerPool | TCPAnalyzerPool``
  * ``evaluate_positions``: async callable
    ``(positions, depth, pool, requested_depth, history_complete) -> list[tuple[MCPEval, bool]]``
  * ``compute_metrics``: pure function
    ``(positions, moves, evals) -> GameMetrics``
  * ``identity``: pure function ``(pool) -> dict``
  * ``engine_version``: pure function ``(pool) -> str``

The default values wire the service to the live tool entry points via
:meth:`GameAnalyzer.with_defaults`, while tests can construct a
``GameAnalyzer`` with stub callables to exercise the orchestration logic
without booting an engine.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TYPE_CHECKING

import chess
import chess.pgn

from core.engines.openings import lookup_opening

if TYPE_CHECKING:
    from mcp.server.mcpserver import Context

    from mcp_server.models import GameAnalysisResult, MCPEval
    from mcp_server.tcp_analyzer import TCPAnalyzerPool
    from core.engines.pool import AnalyzerPool

from mcp_server.analysis.game_validation import GameMetadata, extract_game_metadata
from mcp_server.analysis.result_reconciliation import reconcile_result
from mcp_server.engine import (
    _build_identity,
    _gather_evaluate_positions_bounded,
    _get_analyzer_pool,
)
from mcp_server.models import GameAnalysisResult, MCPEval
from mcp_server.parsers import (
    TAG_PAIR_REGEX,
    _check_multiple_games,
    _extract_canonical_pgn_text,
    _extract_game_inner,
    _find_movetext_result,
    _normalize_movetext_figurines,
    _sanitize_malformed_pgn_header_lines,
    _strip_pgn_escape_lines,
    _strip_promotion_eq,
    _truncate_movetext_at_result,
    _validate_strict_header_syntax,
    _validate_strict_mainline_surface,
)
from mcp_server.rules import (
    evaluate_rule_status,
    format_fen_status_errors,
)
from mcp_server.tools.game_metrics import _compute_game_metrics

EnginePool = "AnalyzerPool | TCPAnalyzerPool"


@dataclass
class GameMetrics:
    """Typed return shape for the game-metrics pure function.

    Mirrors the existing ``_compute_game_metrics`` tuple but with named
    attributes so the analyzer can construct the response without
    positional unpacking.
    """

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
    """End-to-end ``analyze_game`` orchestrator.

    Constructor-injected dependencies make the class testable without
    booting an engine; ``with_defaults`` is the production factory.
    """

    def __init__(
        self,
        get_pool: Callable[["Context | None"], Awaitable["EnginePool"]],
        evaluate_positions: Callable[..., Awaitable[list[tuple["MCPEval", bool]]]],
        compute_metrics: Callable[
            [list[chess.Board], list[chess.Move], list["MCPEval"]], GameMetrics
        ],
        identity: Callable[["EnginePool"], dict[str, Any]],
        engine_version: Callable[["EnginePool"], str],
    ) -> None:
        self._get_pool = get_pool
        self._evaluate_positions = evaluate_positions
        self._compute_metrics = compute_metrics
        self._identity = identity
        self._engine_version = engine_version

    @classmethod
    def with_defaults(cls) -> GameAnalyzer:
        """Production wiring — plugs the analyzer into the live engine layer."""
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
        ctx: "Context | None",
        metrics: Any | None = None,
    ) -> GameAnalysisResult:
        """Run a full PGN game analysis. Returns a populated
        :class:`GameAnalysisResult` on success; raises ``ValueError`` for
        recoverable strict-validation errors and ``Exception`` for
        engine-level failures.
        """
        t0 = time.time()
        raw_requested_depth = depth
        depth = max(1, min(depth, 30))

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
        ) = _parse_mainline(canonical_pgn, game, strict=strict)

        result_movetext = _find_movetext_result(canonical_pgn)
        is_comment_only_input = bool(getattr(game, "comment_only_input", False))

        # Reconcile trailing-ply count BEFORE header validation warnings
        # so the "N plies ignored" line lands in metadata_warnings once.
        ignored_trailing_plies = _reconcile_trailing_plies(
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
                    f"STRICT_PGN_ERROR: PGN contains syntax normalization or move number mismatch: {metadata.syntax_warnings[0]}"
                )
            if metadata.metadata_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains metadata inconsistency: {metadata.metadata_warnings[0]}"
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
            return GameAnalysisResult(
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

        # Compute metrics, then enrich with opening detection + strict pass.
        # The pure ``compute_metrics`` function returns a typed structure;
        # ``_wrap_compute_game_metrics`` translates the legacy tuple shape
        # for backwards compatibility with the existing tests.
        game_metrics = _wrap_compute_game_metrics(positions, moves, evals)
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
                    f"STRICT_PGN_ERROR: PGN contains syntax normalization or move number mismatch: {metadata.syntax_warnings[0]}"
                )
            if metadata.metadata_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains metadata inconsistency: {metadata.metadata_warnings[0]}"
                )

        if metrics is not None:
            await metrics.record(
                "analyze_game",
                (time.time() - t0) * 1000,
                cache_hit=all_cached,
            )

        return GameAnalysisResult(
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
        )


def _engine_version_str(pool: Any) -> str:
    return getattr(pool, "engine_version", getattr(pool, "name", "Stockfish"))


def _wrap_compute_game_metrics(
    positions: list[chess.Board],
    moves: list[chess.Move],
    evals: list[MCPEval],
) -> GameMetrics:
    """Translate the legacy tuple-return shape of
    :func:`mcp_server.tools.game_metrics._compute_game_metrics` into a
    typed :class:`GameMetrics` for the analyzer's response builder.
    """
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
    ) = _compute_game_metrics(positions, moves, evals)
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
    """Return ``(final_opening, final_eco, opening_disagreement, eco_disagreement)``."""
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


def _parse_mainline(
    canonical_pgn: str,
    game: chess.pgn.Game,
    *,
    strict: bool,
) -> tuple[list[chess.Board], list[chess.Move], list[str], int, str]:
    """Walk the game mainline, validating tokens and surfacing strict/non-
    strict normalization warnings. Returns ``(positions, moves, syntax_warnings,
    ignored_trailing_plies, cleaned_movetext)``.

    The implementation preserves every audit-invariant (U-15, R4-§C, B-01,
    P3) — the only changes vs. the inline version are extraction into a
    function and routing warnings through the return tuple instead of
    enclosing-scope mutation.
    """
    positions: list[chess.Board] = []
    moves: list[chess.Move] = []
    syntax_warnings: list[str] = []
    ignored_trailing_plies = 0

    curr_board = game.board()
    if not curr_board.is_valid() or curr_board.status() != chess.STATUS_VALID:
        raise ValueError(
            f"INVALID_FEN: Initial position '{curr_board.fen()}' in PGN is not a valid chess position ({format_fen_status_errors(curr_board.status())})."
        )

    positions.append(curr_board.copy(stack=True))
    auto_termination: str | None = None
    reached_terminal = False

    initial_rule = evaluate_rule_status(curr_board, history_complete="complete")
    if initial_rule.terminal is not None:
        auto_termination = initial_rule.terminal
        reached_terminal = True
        if strict:
            raise ValueError(
                f"STRICT_PGN_ERROR: Initial FEN '{curr_board.fen()}' is already "
                f"terminal ({initial_rule.terminal}); cannot execute movetext."
            )
        syntax_warnings.append(
            f"Initial FEN is terminal ({initial_rule.terminal}); "
            f"all movetext moves will be ignored."
        )

    for nag_match in re.finditer(r"\$([0-9]+)", canonical_pgn):
        nag_val = int(nag_match.group(1))
        if nag_val > 255:
            syntax_warnings.append(f"NAG value ${nag_val} outside the PGN-supported range 0..255.")

    header_end = 0
    first_header = TAG_PAIR_REGEX.search(canonical_pgn)
    first_mv = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", canonical_pgn)
    if first_header and (not first_mv or first_header.start() < first_mv.start()):
        header_end = first_header.end()
        for m in TAG_PAIR_REGEX.finditer(canonical_pgn):
            if m.start() < header_end:
                continue
            if canonical_pgn[header_end : m.start()].strip() == "":
                header_end = m.end()
            else:
                break

    movetext_section = canonical_pgn[header_end:]

    movetext_section = re.sub(
        r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)(\$\d+)",
        r"\1 \2",
        movetext_section,
    )
    movetext_section = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)(\$\d+)", r"\1\2 \3", movetext_section)
    cleaned_movetext = _normalize_movetext_figurines(movetext_section)
    if re.search(
        r"(?:^|\s)\(?e\.?p\.?\)?(?=\s|$)",
        cleaned_movetext,
        flags=re.IGNORECASE,
    ):
        syntax_warnings.append("En-passant marker 'e.p.' normalized to canonical SAN.")
    while "{" in cleaned_movetext and "}" in cleaned_movetext:
        prev = cleaned_movetext
        cleaned_movetext = re.sub(r"\{[^{}]*\}", " ", cleaned_movetext, flags=re.DOTALL)
        if cleaned_movetext == prev:
            break
    cleaned_movetext = re.sub(r";[^\r\n]*", " ", cleaned_movetext)
    while "(" in cleaned_movetext and ")" in cleaned_movetext:
        prev = cleaned_movetext
        cleaned_movetext = re.sub(r"\([^()]*\)", " ", cleaned_movetext, flags=re.DOTALL)
        if cleaned_movetext == prev:
            break
    cleaned_movetext = re.sub(
        r"(?<!\S)(\d+)\s+(\.+)(?=\s|$)",
        lambda m: f" {m.group(1)}{m.group(2)} ",
        cleaned_movetext,
    )

    movetext_tokens = cleaned_movetext.split()
    tok_idx = 0
    expected_fullmove = curr_board.fullmove_number

    for i, tok in enumerate(movetext_tokens):
        clean_tok = tok.strip(".,;:!?")
        num_m = re.match(r"^(\d+)[\.\:]*$", clean_tok)
        if num_m:
            tok_idx = i
            break
        try:
            curr_board.parse_san(clean_tok)
            tok_idx = i
            break
        except Exception:
            continue

    for node in game.mainline():
        if reached_terminal:
            ignored_trailing_plies += 1
            continue

        move = node.move
        if move not in curr_board.legal_moves:
            ignored_trailing_plies += 1
            reached_terminal = True
            continue

        canonical_san = curr_board.san(move)

        while tok_idx < len(movetext_tokens):
            raw_tok = movetext_tokens[tok_idx]
            num_m = re.match(r"^(\d+)(\.+)$", raw_tok)
            if num_m:
                move_num = int(num_m.group(1))
                if move_num != expected_fullmove:
                    syntax_warnings.append(
                        f"Move number mismatch: found '{movetext_tokens[tok_idx]}' but expected move {expected_fullmove}."
                    )
                expected_dots = "..." if curr_board.turn == chess.BLACK else "."
                actual_dots = num_m.group(2) or ""
                if actual_dots != expected_dots:
                    syntax_warnings.append(
                        f"Wrong side marker: found '{movetext_tokens[tok_idx]}' "
                        f"but expected '{expected_dots}' for the side to move."
                    )
                tok_idx += 1
                continue
            if raw_tok in ("1-0", "0-1", "1/2-1/2", "*") or re.match(r"^\$[0-9]+$", raw_tok):
                tok_idx += 1
                continue
            break

        if tok_idx < len(movetext_tokens):
            raw_tok = movetext_tokens[tok_idx].strip(".,;:!?")
            raw_tok_san = raw_tok.rstrip("!?")
            raw_tok_promotionless = _strip_promotion_eq(raw_tok_san)
            canonical_promotionless = _strip_promotion_eq(canonical_san)
            if raw_tok_promotionless != canonical_promotionless and not re.fullmatch(
                r"[a-h][1-8][a-h][1-8][qrbn]?", raw_tok_san.lower()
            ):
                syntax_warnings.append(
                    f"Input SAN '{movetext_tokens[tok_idx]}' normalized to '{canonical_san}'"
                )
            tok_idx += 1

        moves.append(move)
        curr_board.push(move)
        positions.append(curr_board.copy(stack=True))
        if curr_board.turn == chess.WHITE:
            expected_fullmove += 1

        if curr_board.is_repetition(5):
            reached_terminal = True
            auto_termination = "fivefold_repetition"
        else:
            rule_after = evaluate_rule_status(curr_board, history_complete="complete")
            if rule_after.terminal is not None:
                reached_terminal = True
                auto_termination = rule_after.terminal

    return positions, moves, syntax_warnings, ignored_trailing_plies, cleaned_movetext


def _reconcile_trailing_plies(
    *,
    canonical_pgn: str,
    cleaned_movetext: str,
    moves: list[chess.Move],
    game: chess.pgn.Game,
) -> int:
    """Reconcile ignored trailing plies (audit P2): count tokens after the
    board-trace ends so callers see the actual number of unused ply tokens
    rather than ``len(game.errors)`` (which underreports — usually 1 even
    when several trailing tokens were ignored)."""
    ignored_trailing = 0

    # Branch 1: python-chess flagged parse errors — count tokens after the
    # last successfully executed ply.
    if game.errors:
        consumed_plies = len(moves)
        san_pattern = (
            r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*"
            r"|O-O-O[\+#\?!]*|O-O[\+#\?!]*)\b"
        )
        total_ply_tokens = len(re.findall(san_pattern, cleaned_movetext))
        trailing_from_errors = max(0, total_ply_tokens - consumed_plies)
        if trailing_from_errors > 0:
            ignored_trailing += trailing_from_errors
        else:
            # Fallback when no recoverable tokens exist after the error —
            # surface at least ``len(game.errors)`` so the count is never
            # below the parser's own signal.
            ignored_trailing = max(ignored_trailing, len(game.errors))

    # Branch 2: tokens that survive AFTER the result token in the raw PGN
    # (e.g. trailing junk after ``1-0`` / ``*`` that python-chess silently
    # accepted). Audit U-15 ensures side markers are caught upstream; this
    # branch catches ply tokens themselves.
    raw_pgn_clean = _strip_pgn_escape_lines(canonical_pgn)
    raw_truncated = _truncate_movetext_at_result(raw_pgn_clean)
    if len(raw_truncated) < len(raw_pgn_clean):
        after_part = raw_pgn_clean[len(raw_truncated) :]
        after_clean = re.sub(r"\{[^{}]*\}", " ", after_part)
        after_clean = re.sub(r";[^\r\n]*", " ", after_clean)
        tokens_after = re.findall(
            r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*"
            r"|O-O-O[\+#\?!]*|O-O[\+#\?!]*)\b",
            after_clean,
        )
        if tokens_after:
            ignored_trailing += len(tokens_after)

    return ignored_trailing
