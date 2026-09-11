"""Semantic oracles and invariant validators for Chess MCP output."""

from __future__ import annotations

import io
from typing import Any

import chess
import chess.pgn

from scripts.audit.case_spec import CaseSpec
from scripts.audit.response_normalization import NormalizedResponse

ERROR_CODE_ALIASES: dict[str, set[str]] = {
    "invalid_fen": {"invalid_fen", "invalid_position"},
    "invalid_position": {"invalid_fen", "invalid_position"},
    "invalid_move": {"invalid_move", "illegal_move"},
    "illegal_move": {"invalid_move", "illegal_move"},
    "strict_san_error": {"strict_san_error", "strict_validation_error"},
    "strict_validation_error": {"strict_san_error", "strict_validation_error"},
}


def validate_expected_error(spec: CaseSpec, response: NormalizedResponse) -> list[str]:
    """Strict structured error matching (R2-005).

    Matches canonical error codes without loose substring fallbacks into prose.
    """
    errors: list[str] = []
    if not response.is_error:
        return [f"expected error '{spec.expected_error_code}', but call succeeded"]

    expected_code = (spec.expected_error_code or "").strip().lower()
    actual_code = (response.structured_error_code or "").strip().lower()

    if not actual_code:
        return [
            f"expected structured error code '{expected_code}', but no structured code found in response: {response.text_content[:150]}"
        ]

    allowed = ERROR_CODE_ALIASES.get(expected_code, {expected_code})
    if actual_code not in allowed:
        return [
            f"wrong error code: expected '{expected_code}' (allowed: {sorted(allowed)}), but got '{actual_code}' (text: {response.text_content[:150]})"
        ]

    return errors


def check_forcing_evidence(data: Any, board: chess.Board | None = None) -> list[str]:
    """Verify check/mate invariants and board ground truth if board context is available."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return errors

    san = data.get("san")
    is_mate = data.get("is_mate")
    is_check = data.get("is_check")
    uci = data.get("uci")

    if isinstance(san, str) and "is_mate" in data:
        if san.endswith("#") and is_mate is not True:
            errors.append(f"san='{san}' ends with '#' but is_mate is {is_mate}")
        if not san.endswith("#") and is_mate is True:
            errors.append(f"san='{san}' does not end with '#' but is_mate is True")
        if is_mate is True and is_check is not True:
            errors.append(f"is_mate is True but is_check is {is_check} for san='{san}'")

    if board is not None and isinstance(uci, str):
        try:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                child = board.copy(stack=False)
                child.push(m)
                if is_mate is not None and child.is_checkmate() != is_mate:
                    errors.append(
                        f"uci='{uci}' child.is_checkmate()={child.is_checkmate()} != is_mate={is_mate}"
                    )
                if is_check is not None and child.is_check() != is_check:
                    errors.append(
                        f"uci='{uci}' child.is_check()={child.is_check()} != is_check={is_check}"
                    )
        except Exception as exc:
            errors.append(f"invalid UCI move '{uci}' on board: {exc}")

    return errors


def validate_evaluate_position(spec: CaseSpec, data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    fen = spec.arguments["fen"]
    try:
        b = chess.Board(fen)
        status = data.get("status")
        winner = data.get("winner")
        is_over = (
            status
            in (
                "checkmate",
                "stalemate",
                "insufficient_material",
                "seventyfive_moves",
                "fivefold_repetition",
            )
            or data.get("game_over") is True
        )
        if b.is_checkmate():
            if status != "checkmate" and not is_over:
                errors.append(f"position is checkmate but status is {status!r}")
            expected_winner = "white" if b.turn == chess.BLACK else "black"
            if winner != expected_winner:
                errors.append(f"checkmate winner expected {expected_winner}, got {winner}")
        elif b.is_stalemate():
            if status != "stalemate" and not is_over:
                errors.append(f"position is stalemate but status is {status!r}")
            if winner is not None:
                errors.append(f"stalemate winner expected None, got {winner}")
        elif b.is_seventyfive_moves():
            if status != "seventyfive_moves" and not is_over:
                errors.append(f"position is 75-moves rule but status is {status!r}")
        elif not is_over:
            best_move = data.get("best_move")
            if best_move:
                try:
                    m = chess.Move.from_uci(best_move)
                    if m not in b.legal_moves:
                        errors.append(f"best_move '{best_move}' is not legal")
                except ValueError:
                    errors.append(f"best_move '{best_move}' is not valid UCI")
            pv = data.get("pv")
            if pv and len(pv) > 0 and best_move and pv[0] != best_move:
                errors.append(f"pv[0]='{pv[0]}' != best_move='{best_move}'")
            score = data.get("score") or {}
            cp_val = data.get("cp") if "cp" in data else score.get("cp")
            mate_val = data.get("mate") if "mate" in data else score.get("mate")
            if cp_val is not None and mate_val is not None:
                errors.append(f"contradictory score: both cp={cp_val} and mate={mate_val}")

        # Check tactical snapshot if present
        forensics = data.get("forensics") or {}
        tactical = forensics.get("tactical_snapshot") or data.get("tactical_snapshot")
        if isinstance(tactical, dict):
            errors.extend(check_forcing_evidence(tactical, b))
    except Exception as exc:
        errors.append(f"oracle evaluate_position error: {exc}")

    return errors


def validate_top_moves(spec: CaseSpec, data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    fen = spec.arguments["fen"]
    try:
        b = chess.Board(fen)
        moves = data.get("result") or data.get("moves") or []
        returned_n = data.get("returned_n")
        if returned_n is not None and returned_n != len(moves):
            errors.append(f"returned_n={returned_n} != len(moves)={len(moves)}")

        ucis = [
            m.get("best_move")
            or m.get("executable_move")
            or m.get("uci")
            or ((m.get("pv") or [None])[0] if isinstance(m.get("pv"), list) else None)
            for m in moves
            if isinstance(m, dict)
        ]
        ucis = [u for u in ucis if isinstance(u, str)]
        if len(ucis) != len(set(ucis)):
            errors.append(f"candidate root UCIs not unique: {ucis}")

        for m in moves:
            if not isinstance(m, dict):
                continue
            uci = (
                m.get("best_move")
                or m.get("executable_move")
                or m.get("uci")
                or ((m.get("pv") or [None])[0] if isinstance(m.get("pv"), list) else None)
            )
            if uci and isinstance(uci, str):
                try:
                    m_obj = chess.Move.from_uci(uci)
                    if m_obj not in b.legal_moves:
                        errors.append(f"candidate move '{uci}' not legal")

                    # R2-013: Candidate model exposes 'candidate_san' (with 'san' fallback)
                    cand_san = m.get("candidate_san") or m.get("san")
                    if cand_san and b.san(m_obj) != cand_san:
                        errors.append(f"candidate SAN '{cand_san}' != board.san '{b.san(m_obj)}'")

                    post_fen = m.get("post_fen")
                    if post_fen:
                        child = b.copy(stack=False)
                        child.push(m_obj)
                        if post_fen != child.fen():
                            errors.append(f"post_fen '{post_fen}' != child.fen() '{child.fen()}'")
                except Exception as exc:
                    errors.append(f"candidate move error '{uci}': {exc}")
    except Exception as exc:
        errors.append(f"oracle top_moves error: {exc}")

    return errors


def validate_classify_move(spec: CaseSpec, data: dict[str, Any]) -> list[str]:
    """R2-012: Validate actual MCPMoveAnalysis fields 'played' and 'played_san'."""
    errors: list[str] = []
    fen = spec.arguments["fen"]
    move_str = spec.arguments.get("move")
    action_type = spec.arguments.get("action_type", "play_move")

    try:
        b = chess.Board(fen)
        if action_type == "play_move" and move_str:
            played_uci = data.get("played")
            played_san = data.get("played_san")

            if not played_uci:
                errors.append(f"Missing required 'played' UCI in classify_move response for move '{move_str}'")
            else:
                try:
                    req_move = b.parse_san(move_str) if move_str not in [m.uci() for m in b.legal_moves] else chess.Move.from_uci(move_str)
                    if played_uci != req_move.uci():
                        errors.append(f"played UCI '{played_uci}' != requested move UCI '{req_move.uci()}'")
                    if played_san and played_san != b.san(req_move):
                        errors.append(f"played SAN '{played_san}' != expected board SAN '{b.san(req_move)}'")
                except Exception as exc:
                    errors.append(f"Requested move '{move_str}' could not be resolved on root board: {exc}")

            if data.get("is_engine_best"):
                best_move = data.get("best_move")
                best_uci = best_move.get("uci") if isinstance(best_move, dict) else (best_move if isinstance(best_move, str) else None)
                if best_uci and played_uci and played_uci != best_uci:
                    errors.append(f"is_engine_best is True but played '{played_uci}' != best '{best_uci}'")

        # Tactical evidence verification
        forensics = data.get("forensics") or {}
        tactical = forensics.get("tactical_snapshot")
        if isinstance(tactical, dict):
            errors.extend(check_forcing_evidence(tactical, b))
    except Exception as exc:
        errors.append(f"oracle classify_move error: {exc}")

    return errors


def validate_analyze_game(spec: CaseSpec, data: dict[str, Any]) -> list[str]:
    """R2-015: Step-by-step game replay and path-aware critical moment board validation."""
    errors: list[str] = []
    pgn = spec.arguments["pgn"]

    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
        if game:
            mainline = list(game.mainline_moves())
            total_plies = data.get("total_plies")
            if total_plies is not None and total_plies != len(mainline):
                errors.append(f"total_plies={total_plies} != mainline length {len(mainline)}")

            b = game.board()
            board_at_ply: dict[int, chess.Board] = {0: b.copy(stack=False)}
            for ply_idx, m in enumerate(mainline, 1):
                b.push(m)
                board_at_ply[ply_idx] = b.copy(stack=False)

            final_fen = data.get("final_fen")
            if final_fen and final_fen != b.fen():
                errors.append(f"final_fen mismatch: got '{final_fen}', expected '{b.fen()}'")

            # Validate critical moments with their exact board state at each ply
            critical_moments = data.get("critical_moments") or []
            for cm in critical_moments:
                if not isinstance(cm, dict):
                    continue
                ply = cm.get("ply")
                if isinstance(ply, int) and ply in board_at_ply:
                    # Board before player move
                    pre_board = board_at_ply[ply - 1] if ply > 0 else board_at_ply[0]
                    # Board after player move
                    post_board = board_at_ply.get(ply)

                    # Validate strongest reply on post_board
                    strongest_reply = cm.get("strongest_reply") or {}
                    if isinstance(strongest_reply, dict) and post_board is not None:
                        errors.extend(check_forcing_evidence(strongest_reply, post_board))

                    # Validate tactical snapshot on pre_board
                    snapshot = cm.get("tactical_snapshot")
                    if isinstance(snapshot, dict):
                        errors.extend(check_forcing_evidence(snapshot, pre_board))
    except Exception as exc:
        errors.append(f"oracle analyze_game error: {exc}")

    return errors


def run_semantic_oracle(spec: CaseSpec, data: dict[str, Any] | None) -> list[str]:
    """Route to tool-specific semantic oracle."""
    if not data:
        return ["no parsed JSON payload in tool response"]

    if spec.tool == "evaluate_position":
        return validate_evaluate_position(spec, data)
    elif spec.tool == "top_moves":
        return validate_top_moves(spec, data)
    elif spec.tool == "classify_move":
        return validate_classify_move(spec, data)
    elif spec.tool == "analyze_game":
        return validate_analyze_game(spec, data)

    return []
