"""Case generation for the ultra-hard 600-call Chess MCP stress audit."""

from __future__ import annotations

import io
from typing import Any

import chess
import chess.pgn

from scripts.audit.case_spec import CaseSpec, DepthTier, ExpectedKind


def _gen_long_game_pgn(plies: int, headers: dict[str, str] | None = None) -> str:
    """Generate a legal ongoing chess game PGN with exact move count without early draw."""
    import random

    rng = random.Random(plies)
    b = chess.Board()
    game = chess.pgn.Game()
    if headers:
        for k, v in headers.items():
            game.headers[k] = v

    node = game
    for _ in range(plies):
        legals = list(b.legal_moves)
        valid: list[chess.Move] = []
        for m in legals:
            b.push(m)
            if not b.is_game_over(claim_draw=False) and not b.is_fivefold_repetition():
                valid.append(m)
            b.pop()
        chosen = rng.choice(valid if valid else legals)
        b.push(chosen)
        node = node.add_variation(chosen)

    out = io.StringIO()
    exporter = chess.pgn.FileExporter(out)
    game.accept(exporter)
    return out.getvalue().strip()


# Standard positions
pos_start = chess.STARTING_FEN
pos_open = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5"
pos_closed = "r1b1kb1r/pp1n1ppp/2p1pn2/q2p4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w KQkq - 2 7"
pos_iqp = "r1bq1rk1/pp3ppp/2n1pn2/3p4/2PP4/2NB1N2/PP3PPP/R1BQ1RK1 w - - 0 9"
pos_hanging = "r2q1rk1/pb1nbppp/1p2p3/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 10"
pos_opp_castle = "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/2KR1B1R b - - 2 9"
pos_exposed_k = "r1b1k2r/pppp1ppp/8/4P3/1b1q1P2/2NB4/PPP3PP/R2QK2R b KQkq - 1 11"
pos_endgame_r = "4r3/5pk1/4p1p1/7p/7P/8/5PK1/4R3 w - - 0 45"
pos_endgame_p = "8/5k2/4p1p1/5p1p/5P1P/6K1/8/8 w - - 0 50"
pos_mate_in_1 = "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4"
pos_mate_in_2 = "r5rk/5p1p/5R2/4Q3/8/8/PPP3PP/7K w - - 0 1"
pos_stalemate = "k7/8/1Q6/8/8/8/8/7K b - - 0 1"
pos_checkmate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
pos_insufficient = "8/8/5k2/8/8/8/3K1B2/8 w - - 0 1"
pos_75_moves = "4k2r/8/8/8/8/8/8/4K2R w Kk - 150 76"
pos_50_moves = "4k2r/8/8/8/8/8/8/4K2R w Kk - 100 51"
pos_50_claimable = "4k2r/8/8/8/8/8/8/4K2R w Kk - 99 50"
pos_promotion = "8/4P3/8/8/8/5k2/8/4K3 w - - 0 1"
pos_underpromo = "8/5P2/8/8/8/5k2/8/4K3 w - - 0 1"


def build_600_case_specs() -> list[CaseSpec]:
    """Generate exactly 600 schema-valid, distinct CaseSpec objects."""
    cases: list[CaseSpec] = []
    ordinal = 0

    def add_case(
        tool: str,
        case_id: str,
        arguments: dict[str, Any],
        expected_kind: ExpectedKind = "success",
        expected_error_code: str | None = None,
        semantic_profile: str | None = None,
        tags: list[str] | None = None,
        depth_tier: DepthTier = "low",
    ) -> None:
        nonlocal ordinal
        ordinal += 1
        cases.append(
            CaseSpec(
                case_id=case_id,
                tool=tool,
                arguments=arguments,
                expected_kind=expected_kind,
                case_ordinal=ordinal,
                expected_error_code=expected_error_code,
                semantic_profile=semantic_profile,
                tags=tags or [],
                depth_tier=depth_tier,
            )
        )

    # =========================================================================
    # A. EVALUATE_POSITION: 140 cases (10 error, 130 success)
    # =========================================================================
    eval_errors = [
        ("eval_err_001", {"fen": "invalid_fen_string"}, "tool_error", "invalid_fen", "malformed_fen"),
        ("eval_err_002", {"fen": "8/8/8/8/8/8/8/8 w - - 0 1"}, "tool_error", "invalid_fen", "missing_kings"),
        ("eval_err_003", {"fen": "4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1", "moves": ["e2e8"]}, "tool_error", "invalid_move", "illegal_history_move"),
        ("eval_err_004", {"fen": pos_start, "moves": ["e4!?"], "strict": True}, "tool_error", "strict_san_error", "strict_annotation"),
        ("eval_err_005", {"fen": pos_start, "verbosity": "minimal", "detail": "forensic"}, "tool_error", "invalid_argument", "minimal_forensic_reject"),
        ("eval_err_006", {"fen": pos_start, "verbosity": "min", "detail": "coach"}, "tool_error", "invalid_argument", "min_coach_reject"),
        ("eval_err_007", {"fen": pos_start, "moves": ["e5"]}, "tool_error", "illegal_move", "illegal_first_move"),
        ("eval_err_008", {"fen": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "moves": ["e4"]}, "tool_error", "game_already_over", "move_after_terminal"),
        ("eval_err_009", {"fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "moves": ["bad_move"]}, "tool_error", "invalid_move", "unparseable_move"),
        ("eval_err_010", {"fen": pos_start, "verbosity": "minimal", "detail": "forensic", "strict": True}, "tool_error", "invalid_argument", "minimal_forensic_strict"),
    ]
    for cid, args, ekind, ecode, prof in eval_errors:
        add_case("evaluate_position", cid, args, expected_kind=ekind, expected_error_code=ecode, semantic_profile=prof, tags=["error_boundary"])

    # Valid evaluate_position cases (130)
    eval_fens = [
        pos_start, pos_open, pos_closed, pos_iqp, pos_hanging, pos_opp_castle,
        pos_exposed_k, pos_endgame_r, pos_endgame_p, pos_mate_in_1, pos_mate_in_2,
        pos_stalemate, pos_checkmate, pos_insufficient, pos_75_moves, pos_50_moves,
        pos_50_claimable, pos_promotion, pos_underpromo,
    ]
    eval_depths = [1, 4, 8, 12, 14, 16, 18, 20, 22]
    eval_verbs = ["full", "compact", "minimal", "standard", "default"]
    eval_details = ["standard", "coach", "forensic"]

    eval_idx = 10
    for f_i, f in enumerate(eval_fens):
        for d_i, d in enumerate(eval_depths):
            if eval_idx >= 140:
                break
            eval_idx += 1
            # Avoid minimal with coach/forensic; avoid forensic at extreme depth
            det = eval_details[(f_i + d_i) % len(eval_details)]
            if d >= 18 and det == "forensic":
                det = "coach"
            vb = eval_verbs[(f_i * 2 + d_i) % len(eval_verbs)]
            if vb in ("minimal", "min") and det in ("coach", "forensic"):
                vb = "full"
            tier: DepthTier = "hard" if d >= 22 else ("medium" if d >= 14 else "low")
            args: dict[str, Any] = {"fen": f, "depth": d, "detail": det, "verbosity": vb}
            if f_i % 3 == 0:
                args["strict"] = (f_i % 2 == 0)
            add_case("evaluate_position", f"eval_{eval_idx:03d}", args, depth_tier=tier, tags=["matrix", f"depth_{d}"])

    while eval_idx < 140:
        eval_idx += 1
        d = 10 + (eval_idx % 15)
        tier = "hard" if d >= 22 else "medium"
        add_case("evaluate_position", f"eval_{eval_idx:03d}", {"fen": pos_open, "depth": d, "detail": "standard", "verbosity": "full"}, depth_tier=tier)

    # =========================================================================
    # B. TOP_MOVES: 150 cases (10 error, 140 success)
    # =========================================================================
    top_errors = [
        ("top_err_001", {"fen": "invalid_fen_string", "n": 3}, "tool_error", "invalid_fen", "malformed_fen"),
        ("top_err_002", {"fen": pos_start, "include_moves": ["e5"]}, "tool_error", "invalid_move", "opponent_include_move"),
        ("top_err_003", {"fen": pos_start, "include_moves": ["e4!?"], "strict": True}, "tool_error", "strict_san_error", "strict_include_move"),
        ("top_err_004", {"fen": pos_start, "proof_mode": "invalid_proof_mode"}, "schema_error", "schema_validation_error", "bad_proof_mode"),
        ("top_err_005", {"fen": pos_start, "include_moves": ["e4", "d4", "c4", "Nf3", "Nc3", "g3", "b3", "f4", "a3"]}, "tool_error", "invalid_argument", "too_many_include_moves"),
        ("top_err_006", {"fen": pos_start, "moves": ["e2e5"]}, "tool_error", "illegal_move", "illegal_history_move"),
        ("top_err_007", {"fen": pos_start, "moves": ["1. e4"], "strict": True}, "tool_error", "strict_san_error", "strict_san_syntax"),
        ("top_err_008", {"fen": pos_stalemate, "n": 3}, "success", None, "stalemate_top_moves"),
        ("top_err_009", {"fen": pos_checkmate, "n": 3}, "success", None, "checkmate_top_moves"),
        ("top_err_010", {"fen": pos_start, "include_moves": ["e2e4", "e4", "invalid_move"]}, "tool_error", "invalid_move", "bad_include_move"),
    ]
    for cid, args, ekind, ecode, prof in top_errors:
        add_case("top_moves", cid, args, expected_kind=ekind, expected_error_code=ecode, semantic_profile=prof, tags=["error_boundary"])

    top_n_values = [1, 2, 3, 5, 8, 10, 15, 20]
    top_depths = [2, 6, 10, 14, 16, 18, 20]

    top_idx = 10
    for f_i, f in enumerate(eval_fens):
        for n_val in top_n_values:
            for d in top_depths:
                if top_idx >= 150:
                    break
                top_idx += 1
                pm = "tactical" if (n_val <= 3 and (f_i + n_val) % 2 == 1) else "none"
                det = "forensic" if (pm == "none" and n_val <= 5) else "standard"
                tier: DepthTier = "hard" if d >= 18 else ("medium" if d >= 14 else "low")
                args = {"fen": f, "n": n_val, "depth": d, "proof_mode": pm, "detail": det}
                if f == pos_start and n_val <= 3:
                    args["include_moves"] = ["e4", "d4"]
                elif f == pos_open and n_val <= 2:
                    args["include_moves"] = ["O-O", "d3"]
                add_case("top_moves", f"top_{top_idx:03d}", args, depth_tier=tier, tags=["matrix", f"n_{n_val}"])
            if top_idx >= 150:
                break
        if top_idx >= 150:
            break

    while top_idx < 150:
        top_idx += 1
        d = 10 + (top_idx % 10)
        tier = "hard" if d >= 18 else "medium"
        add_case("top_moves", f"top_{top_idx:03d}", {"fen": pos_open, "n": 3, "depth": d}, depth_tier=tier)

    # =========================================================================
    # C. CLASSIFY_MOVE: 170 cases (15 error, 155 success)
    # =========================================================================
    classify_errors = [
        ("cls_err_001", {"fen": "invalid_fen", "move": "e4"}, "tool_error", "invalid_fen", "bad_fen"),
        ("cls_err_002", {"fen": pos_start, "move": "e5"}, "tool_error", "invalid_move", "opponent_move"),
        ("cls_err_003", {"fen": pos_start, "move": "e9"}, "tool_error", "invalid_move", "unparseable_square"),
        ("cls_err_004", {"fen": pos_start, "action_type": "invalid_action_type"}, "schema_error", "schema_validation_error", "bad_action_type"),
        ("cls_err_005", {"fen": pos_start, "action_type": "claim_draw"}, "tool_error", "illegal_action", "unclaimable_draw"),
        ("cls_err_006", {"fen": pos_start, "action_type": "claim_draw_with_intended_move", "move": "e4"}, "tool_error", "illegal_action", "unclaimable_intended"),
        ("cls_err_007", {"fen": pos_start, "move": "e4!?", "strict": True}, "tool_error", "strict_san_error", "strict_annotation"),
        ("cls_err_008", {"fen": pos_start, "move": "e4", "compare_moves": ["e5"]}, "tool_error", "invalid_move", "opponent_compare_move"),
        ("cls_err_009", {"fen": pos_start, "move": "e4", "compare_moves": ["e4!?"], "strict": True}, "tool_error", "strict_san_error", "strict_compare_move"),
        ("cls_err_010", {"fen": pos_start, "move": "e4", "action_type": "invalid_action"}, "schema_error", "schema_validation_error", "bad_action"),
        ("cls_err_011", {"fen": pos_start, "move": "e4", "detail": "invalid_detail"}, "schema_error", "schema_validation_error", "bad_detail"),
        ("cls_err_012", {"fen": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "move": "e2e3"}, "tool_error", "game_already_over", "terminal_classify"),
        ("cls_err_013", {"fen": pos_open, "move": "d4", "moves": ["e4", "e5", "invalid_move"]}, "tool_error", "invalid_move", "bad_history"),
        ("cls_err_014", {"fen": pos_start, "move": "e4", "compare_moves": ["d4", "d4", "invalid_move"]}, "tool_error", "invalid_move", "bad_compare"),
        ("cls_err_015", {"fen": pos_start, "move": "e4", "action_type": "play_move", "moves": ["d4"], "strict": True}, "tool_error", "invalid_move", "mismatch_history"),
    ]
    for cid, args, ekind, ecode, prof in classify_errors:
        add_case("classify_move", cid, args, expected_kind=ekind, expected_error_code=ecode, semantic_profile=prof, tags=["error_boundary"])

    # Tactical & move-quality spectrum fixtures
    tactical_cases = [
        (pos_start, "e4", "standard", 12, "play_move", "best_e4"),
        (pos_start, "d4", "standard", 12, "play_move", "best_d4"),
        (pos_start, "Nf3", "standard", 12, "play_move", "best_nf3"),
        (pos_start, "c4", "coach", 12, "play_move", "good_c4"),
        (pos_start, "a3", "coach", 12, "play_move", "inaccuracy_a3"),
        (pos_start, "h4", "coach", 12, "play_move", "mistake_h4"),
        (pos_start, "g4", "forensic", 14, "play_move", "blunder_g4"),
        (pos_mate_in_1, "Qxf7#", "forensic", 10, "play_move", "delivered_mate"),
        (pos_mate_in_2, "Rxf7+", "forensic", 14, "play_move", "mate_in_2_discovered_check"),
        (pos_open, "O-O", "coach", 16, "play_move", "castle_kingside"),
        (pos_open, "d3", "coach", 16, "play_move", "quiet_d3"),
        (pos_exposed_k, "Qxf4", "forensic", 18, "play_move", "queen_capture"),
        (pos_endgame_r, "Re3", "standard", 20, "play_move", "rook_endgame_move"),
        (pos_endgame_p, "Kf3", "coach", 20, "play_move", "pawn_endgame_king"),
        (pos_promotion, "e8=Q", "forensic", 14, "play_move", "queen_promotion"),
        (pos_underpromo, "f8=N", "forensic", 14, "play_move", "knight_underpromotion"),
        (pos_50_claimable, "Ke2", "standard", 10, "claim_draw_with_intended_move", "claim_draw_intended"),
    ]

    cls_idx = 15
    for fen, mv, det, d, act, prof in tactical_cases:
        cls_idx += 1
        tier: DepthTier = "hard" if d >= 18 else ("medium" if d >= 14 else "low")
        args = {"fen": fen, "move": mv, "detail": det, "depth": d, "action_type": act}
        add_case("classify_move", f"cls_{cls_idx:03d}", args, semantic_profile=prof, depth_tier=tier, tags=["tactical", act])

    # Fill up to 170 with varied depths, compare_moves, histories
    cls_depths = [6, 8, 10, 12, 14, 16]
    for fen in [pos_start, pos_open, pos_closed, pos_iqp, pos_hanging, pos_opp_castle, pos_exposed_k]:
        b = chess.Board(fen)
        legal_moves = [b.san(m) for m in list(b.legal_moves)[:4]]
        for mv in legal_moves:
            for d in cls_depths:
                if cls_idx >= 170:
                    break
                cls_idx += 1
                det = "forensic" if (d <= 12 and (cls_idx % 2 == 0)) else ("coach" if d >= 14 else "standard")
                tier = "hard" if d >= 16 else ("medium" if d >= 10 else "low")
                compare = [m for m in legal_moves if m != mv][:2]
                args = {"fen": fen, "move": mv, "depth": d, "detail": det}
                if compare:
                    args["compare_moves"] = compare
                add_case("classify_move", f"cls_{cls_idx:03d}", args, depth_tier=tier, tags=["spectrum", f"depth_{d}"])
            if cls_idx >= 170:
                break
        if cls_idx >= 170:
            break

    # =========================================================================
    # D. ANALYZE_GAME: 140 cases (10 error, 130 success)
    # =========================================================================
    analyze_errors = [
        ("ana_err_001", {"pgn": "invalid_pgn_text"}, "tool_error", "invalid_pgn", "unparseable_pgn"),
        ("ana_err_002", {"pgn": "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. d4 exd4 5. e8=Q *"}, "tool_error", "invalid_pgn", "illegal_mainline_move"),
        ("ana_err_003", {"pgn": "1. e4 2. e5 *", "strict": True}, "tool_error", "strict_san_error", "strict_move_number_mismatch"),
        ("ana_err_004", {"pgn": "1. e4 e5 1-0 2. d4", "strict": True}, "tool_error", "strict_san_error", "strict_trailing_tokens"),
        ("ana_err_005", {"pgn": "1. e4 e5 2. Nf3 d6 3. d4 exd4 4. c3 dxc3 5. e.p. *", "strict": True}, "tool_error", "strict_san_error", "strict_ep_normalization"),
        ("ana_err_006", {"pgn": "1. e4 e5 *", "perspective": "invalid_perspective"}, "schema_error", "schema_validation_error", "schema_perspective"),
        ("ana_err_007", {"pgn": "1. e4 e5 *", "detail": "invalid_detail"}, "schema_error", "schema_validation_error", "schema_detail"),
        ("ana_err_008", {"pgn": "1. e4 e5 2. e2e4 *", "strict": True}, "tool_error", "strict_san_error", "strict_uci_in_pgn"),
        ("ana_err_009", {"pgn": '[SetUp "1"]\n[FEN "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"]\n\n1. e4 *', "strict": True}, "tool_error", "invalid_pgn", "terminal_initial_fen"),
        ("ana_err_010", {"pgn": "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 5. 0-0 *", "strict": True}, "tool_error", "strict_san_error", "zeros_castling"),
    ]
    for cid, args, ekind, ecode, prof in analyze_errors:
        add_case("analyze_game", cid, args, expected_kind=ekind, expected_error_code=ecode, semantic_profile=prof, tags=["error_boundary"])

    # Base PGN corpuses: short, medium, long, annotated, SetUp
    pgn_fools = "1. f3 e5 2. g4 Qh4# 0-1"
    pgn_scholars = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
    pgn_resignation = '[Termination "White resigns"]\n[Result "0-1"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. d3 Nf6 0-1'
    pgn_annotated = '1. e4 {Best by test} 1... e5 2. Nf3 $1 2... Nc6 3. Bc4 Bc5 4. d3 d6 5. O-O Nf6 *'
    pgn_setup_fen = '[SetUp "1"]\n[FEN "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4"]\n\n4. Qxf7# 1-0'

    pgn_medium_italian = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d3 d6 6. O-O O-O 7. Nbd2 a6 8. Bb3 Ba7 9. Re1 Re8 10. Nf1 Be6 *"
    pgn_medium_sicilian = "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 *"
    pgn_medium_qgd = "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. cxd5 exd5 5. Bg5 Be7 6. e3 O-O 7. Bd3 Nbd7 8. Nge2 Re8 9. O-O Nf8 10. Qc2 c6 *"

    # True long games (80, 120, 160, 200 plies)
    pgn_long_80 = _gen_long_game_pgn(80, {"Event": "Ultra Hard 80 Plies"})
    pgn_long_120 = _gen_long_game_pgn(120, {"Event": "Ultra Hard 120 Plies"})
    pgn_long_160 = _gen_long_game_pgn(160, {"Event": "Ultra Hard 160 Plies"})
    pgn_long_200 = _gen_long_game_pgn(200, {"Event": "Ultra Hard 200 Plies"})

    all_pgns = [
        (pgn_fools, "fools_mate", [4, 6, 8, 10, 14, 18, 20]),
        (pgn_scholars, "scholars_mate", [4, 6, 8, 10, 14, 18, 20]),
        (pgn_resignation, "clean_resignation", [4, 6, 8, 10, 14, 18, 20]),
        (pgn_annotated, "annotated_comments_nags", [4, 6, 8, 10, 14, 16]),
        (pgn_setup_fen, "setup_fen_game", [4, 6, 8, 10, 14, 18, 20]),
        (pgn_medium_italian, "medium_italian", [4, 6, 8, 10, 12, 14]),
        (pgn_medium_sicilian, "medium_sicilian", [4, 6, 8, 10, 12, 14]),
        (pgn_medium_qgd, "medium_qgd", [4, 6, 8, 10, 12, 14]),
        (pgn_long_80, "long_80_plies", [4, 6, 8]),
        (pgn_long_120, "long_120_plies", [4, 6, 8]),
        (pgn_long_160, "ultralong_160_plies", [4, 6, 8]),
        (pgn_long_200, "ultralong_200_plies", [4, 6, 8]),
    ]

    ana_details = ["standard", "coach", "forensic"]
    ana_perspectives = ["white", "black"]
    ana_max_cm = [1, 2, 4, 6, 7]

    pgn_candidates: dict[str, list[tuple[dict[str, Any], str, DepthTier]]] = {prof: [] for _, prof, _ in all_pgns}
    for p_idx, (pgn_str, prof, depths) in enumerate(all_pgns):
        for d_i, d in enumerate(depths):
            for det_i, det in enumerate(ana_details):
                if det == "forensic" and d > 10:
                    continue
                persp = ana_perspectives[(p_idx + d_i) % 2]
                mcm = ana_max_cm[(p_idx + det_i) % len(ana_max_cm)]
                strict = (p_idx % 2 == 0 and d_i % 2 == 0)
                args = {
                    "pgn": pgn_str,
                    "depth": d,
                    "detail": det,
                    "perspective": persp,
                    "max_critical_moments": mcm,
                }
                if strict:
                    args["strict"] = True
                tier: DepthTier = "hard" if d >= 18 else ("medium" if d >= 10 else "low")
                pgn_candidates[prof].append((args, prof, tier))

    selected_ana: list[tuple[dict[str, Any], str, DepthTier]] = []
    max_per_pgn = max(len(c) for c in pgn_candidates.values())
    for round_i in range(max_per_pgn):
        for _, prof, _ in all_pgns:
            if len(selected_ana) >= 130:
                break
            cand_list = pgn_candidates[prof]
            if round_i < len(cand_list):
                selected_ana.append(cand_list[round_i])

    for i, (args, prof, tier) in enumerate(selected_ana, start=11):
        add_case("analyze_game", f"ana_{i:03d}", args, semantic_profile=prof, depth_tier=tier, tags=["pgn", prof])

    assert len(cases) == 600, f"Expected exactly 600 cases, got {len(cases)}"
    return cases


def build_adversarial_1000_case_specs() -> list[CaseSpec]:
    """Generate 1000 schema-valid, distinct CaseSpec objects with deep adversarial cases."""
    cases = list(build_600_case_specs())
    ordinal = len(cases)

    def add_adv(
        tool: str,
        case_id: str,
        arguments: dict[str, Any],
        expected_kind: ExpectedKind = "success",
        expected_error_code: str | None = None,
        semantic_profile: str | None = None,
        tags: list[str] | None = None,
        depth_tier: DepthTier = "low",
    ) -> None:
        nonlocal ordinal
        ordinal += 1
        cases.append(
            CaseSpec(
                case_id=case_id,
                tool=tool,
                arguments=arguments,
                expected_kind=expected_kind,
                case_ordinal=ordinal,
                expected_error_code=expected_error_code,
                semantic_profile=semantic_profile,
                tags=tags or [],
                depth_tier=depth_tier,
            )
        )

    # 1. Legal's Mate Invariant A & B: 100 cases
    fen_legals = "rn1qkbnr/ppp2B1p/3p2p1/4N3/4P3/2N5/PPPP1PPP/R1BbK2R b KQkq - 0 6"
    legals_depths = [1, 2, 4, 6]
    legals_details = ["standard", "coach", "forensic"]
    legals_stricts = [False, True]
    legals_moves = ["Ke7", "e8e7"]
    legals_idx = 1
    for d in legals_depths:
        for det in legals_details:
            for st in legals_stricts:
                for mv in legals_moves:
                    if legals_idx > 48:
                        break
                    tier: DepthTier = "medium" if d >= 6 else "low"
                    add_adv(
                        "classify_move",
                        f"adv_legals_{legals_idx:03d}",
                        {"fen": fen_legals, "move": mv, "depth": d, "detail": det, "strict": st},
                        semantic_profile="legals_mate_invariant_a",
                        tags=["adversarial", "invariant_a", "forced_move"],
                        depth_tier=tier,
                    )
                    legals_idx += 1

    # 2 cases with compare_moves to reach 50 classify_move cases
    add_adv(
        "classify_move",
        f"adv_legals_{legals_idx:03d}",
        {"fen": fen_legals, "move": "Ke7", "depth": 2, "detail": "standard", "compare_moves": ["e8e7"]},
        semantic_profile="legals_mate_invariant_a",
        tags=["adversarial", "invariant_a", "forced_move"],
        depth_tier="low",
    )
    legals_idx += 1
    add_adv(
        "classify_move",
        f"adv_legals_{legals_idx:03d}",
        {"fen": fen_legals, "move": "Ke7", "depth": 4, "detail": "coach", "compare_moves": ["e8e7"]},
        semantic_profile="legals_mate_invariant_a",
        tags=["adversarial", "invariant_a", "forced_move"],
        depth_tier="low",
    )
    legals_idx += 1

    # 50 cases on analyze_game (51 to 100)
    while legals_idx <= 100:
        d = (legals_idx % 4) + 1
        persp = "black" if legals_idx % 2 == 0 else "white"
        det = "coach" if legals_idx % 3 == 0 else "standard"
        pgn_legals_unique = (
            f'[Event "Legal\'s Mate Check"]\n'
            f'[Round "{legals_idx}"]\n\n'
            f'1. e4 e5 2. Nf3 d6 3. Bc4 Bg4 4. Nc3 g6 5. Nxe5 Bxd1 6. Bxf7+ Ke7 7. Nd5# 1-0'
        )
        add_adv(
            "analyze_game",
            f"adv_legals_{legals_idx:03d}",
            {"pgn": pgn_legals_unique, "depth": d, "detail": det, "perspective": persp},
            semantic_profile="legals_mate_game_invariant_a",
            tags=["adversarial", "invariant_a", "legals_game"],
            depth_tier="low",
        )
        legals_idx += 1

    # 2. Contradictory PGN Invariant D: 100 cases
    contra_idx = 1
    contra_templates = [
        '[Result "1-0"]\n[Round "{i}"]\n\n1. f3 e5 2. g4 Qh4# 0-1',
        '[Result "0-1"]\n[Round "{i}"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0',
        '[Result "1/2-1/2"]\n[Round "{i}"]\n\n1. f3 e5 2. g4 Qh4# 0-1',
        '[Result "1/2-1/2"]\n[Round "{i}"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0',
        '[Result "1-0"]\n[Termination "Time forfeit"]\n[Round "{i}"]\n\n1. f3 e5 2. g4 Qh4# 0-1',
    ]
    contra_profiles = [
        "fools_mate_inverted_result",
        "scholars_mate_inverted_result",
        "fools_mate_draw_header",
        "scholars_mate_draw_header",
        "fools_mate_forfeit_conflict",
    ]
    while contra_idx <= 100:
        tmpl = contra_templates[contra_idx % len(contra_templates)]
        prof = contra_profiles[contra_idx % len(contra_profiles)]
        pgn_contra = tmpl.format(i=contra_idx)
        d = (contra_idx % 3) + 1
        persp = "white" if (contra_idx // len(contra_templates)) % 2 == 0 else "black"
        det = "coach" if contra_idx % 2 == 0 else "standard"
        add_adv(
            "analyze_game",
            f"adv_contra_{contra_idx:03d}",
            {"pgn": pgn_contra, "depth": d, "detail": det, "perspective": persp},
            semantic_profile=prof,
            tags=["adversarial", "invariant_d", "contradictory_pgn"],
            depth_tier="low",
        )
        contra_idx += 1

    # 3. Machine Comments Invariant E: 80 cases
    machine_idx = 1
    machine_templates = [
        '[Round "{i}"]\n\n1. e4 {{[%clk 0:05:00] [%eval +0.20]}} e5 {{[%clk 0:04:55]}} 2. Nf3 Nc6 1/2-1/2',
        '[Round "{i}"]\n\n1. e4 {{[%emt 0:00:03] [%csl Gf3,Re4]}} e5 {{[%cal Gf3f7]}} 2. Nf3 Nc6 1/2-1/2',
        '[Round "{i}"]\n\n1. e4 {{[%clk 0:05:00] Played e4 for central control}} e5 2. Nf3 1/2-1/2',
        '[Round "{i}"]\n\n1. e4 {{%clk 0:05:00 %eval +0.20}} e5 {{%clk 0:04:50}} 2. Nf3 Nc6 1/2-1/2',
    ]
    machine_profiles = [
        "pure_clock_eval",
        "emt_csl_cal",
        "mixed_clk_human",
        "unbracketed_clk",
    ]
    while machine_idx <= 80:
        tmpl = machine_templates[machine_idx % len(machine_templates)]
        prof = machine_profiles[machine_idx % len(machine_profiles)]
        pgn_mach = tmpl.format(i=machine_idx)
        d = (machine_idx % 3) + 1
        persp = "white" if machine_idx % 2 == 0 else "black"
        det = "coach" if machine_idx % 3 == 0 else "standard"
        add_adv(
            "analyze_game",
            f"adv_mach_{machine_idx:03d}",
            {"pgn": pgn_mach, "depth": d, "detail": det, "perspective": persp},
            semantic_profile=prof,
            tags=["adversarial", "invariant_e", "machine_annotations"],
            depth_tier="low",
        )
        machine_idx += 1

    # 4. Repetition & Invariant F: 40 cases
    # 20 threefold (2 cycles) and 20 fivefold (4 cycles) on evaluate_position with unique move stacks
    base_pairs = [
        ([], ["g1f3", "g8f6", "f3g1", "f6g8"]),
        ([], ["b1c3", "b8c6", "c3b1", "c6b8"]),
        ([], ["g1h3", "g8h6", "h3g1", "h6g8"]),
        ([], ["b1a3", "b8a6", "a3b1", "a6b8"]),
        ([], ["g1f3", "b8c6", "f3g1", "c6b8"]),
        ([], ["g1f3", "g8h6", "f3g1", "h6g8"]),
        ([], ["g1f3", "b8a6", "f3g1", "a6b8"]),
        ([], ["b1c3", "g8f6", "c3b1", "f6g8"]),
        ([], ["b1c3", "g8h6", "c3b1", "h6g8"]),
        ([], ["b1c3", "b8a6", "c3b1", "a6b8"]),
        ([], ["g1h3", "g8f6", "h3g1", "f6g8"]),
        ([], ["g1h3", "b8c6", "h3g1", "c6b8"]),
        ([], ["g1h3", "b8a6", "h3g1", "a6b8"]),
        ([], ["b1a3", "g8f6", "a3b1", "f6g8"]),
        ([], ["b1a3", "b8c6", "a3b1", "c6b8"]),
        ([], ["b1a3", "g8h6", "a3b1", "h6g8"]),
        (["e2e4", "e7e5"], ["g1f3", "g8f6", "f3g1", "f6g8"]),
        (["d2d4", "d7d5"], ["g1f3", "g8f6", "f3g1", "f6g8"]),
        (["e2e4", "e7e5"], ["b1c3", "b8c6", "c3b1", "c6b8"]),
        (["d2d4", "d7d5"], ["b1c3", "b8c6", "c3b1", "c6b8"]),
    ]
    rep_idx = 1
    # 20 threefold cases (startpos + 2 cycles = 3 occurrences)
    for prefix, rep_cycle in base_pairs:
        moves = prefix + (rep_cycle * 2)
        add_adv(
            "evaluate_position",
            f"adv_rep_{rep_idx:03d}",
            {"fen": "startpos", "moves": moves, "depth": 1},
            semantic_profile="threefold_repetition_stack",
            tags=["adversarial", "invariant_f", "threefold"],
            depth_tier="low",
        )
        rep_idx += 1

    # 20 fivefold cases (startpos + 4 cycles = 5 occurrences)
    for prefix, rep_cycle in base_pairs:
        moves = prefix + (rep_cycle * 4)
        add_adv(
            "evaluate_position",
            f"adv_rep_{rep_idx:03d}",
            {"fen": "startpos", "moves": moves, "depth": 1},
            semantic_profile="fivefold_repetition_stack",
            tags=["adversarial", "invariant_f", "fivefold"],
            depth_tier="low",
        )
        rep_idx += 1

    # 5. Parameter Boundary Fuzzing & Clamping: 80 cases
    bound_idx = 1
    # 4 tool_error boundary cases on top_moves proof_defenses
    for p_def in [0, -1, -5, -10]:
        add_adv(
            "top_moves",
            f"adv_bound_{bound_idx:03d}",
            {"fen": "startpos", "proof_mode": "tactical", "proof_defenses": p_def},
            expected_kind="tool_error",
            expected_error_code="invalid_argument",
            semantic_profile="proof_defenses_negative_error",
            tags=["adversarial", "boundary", "proof_defenses"],
            depth_tier="low",
        )
        bound_idx += 1

    # 30 top_moves n-clamping cases across distinct FENs
    n_fens = [pos_start, pos_open, pos_closed, pos_iqp, pos_hanging, pos_endgame_r]
    for fen in n_fens:
        for raw_n in [-1, 0, 20, 21, 100]:
            add_adv(
                "top_moves",
                f"adv_bound_{bound_idx:03d}",
                {"fen": fen, "n": raw_n, "depth": 1},
                semantic_profile="top_moves_n_clamping",
                tags=["adversarial", "boundary", "n_clamping"],
                depth_tier="low",
            )
            bound_idx += 1

    # 46 evaluate_position depth-clamping & verbosity cases across distinct generated FENs
    # (Generating distinct positions guarantees zero collision with any base cases in eval_fens)
    clamp_fens: list[str] = []
    base_b = chess.Board()
    for m in list(base_b.legal_moves):
        b_child = base_b.copy(stack=False)
        b_child.push(m)
        clamp_fens.append(b_child.fen())
    b_e4 = chess.Board()
    b_e4.push_san("e4")
    for m in list(b_e4.legal_moves):
        b_child = b_e4.copy(stack=False)
        b_child.push(m)
        clamp_fens.append(b_child.fen())
    b_d4 = chess.Board()
    b_d4.push_san("d4")
    for m in list(b_d4.legal_moves):
        b_child = b_d4.copy(stack=False)
        b_child.push(m)
        clamp_fens.append(b_child.fen())

    depth_samples = [-5, 0, 1, 30, 31]
    verb_samples = ["min", "compact", "full"]
    for i in range(46):
        fen = clamp_fens[i]
        d = depth_samples[i % len(depth_samples)]
        v = verb_samples[i % len(verb_samples)]
        add_adv(
            "evaluate_position",
            f"adv_bound_{bound_idx:03d}",
            {"fen": fen, "depth": d, "verbosity": v, "strict": True},
            semantic_profile="depth_and_verbosity_boundary",
            tags=["adversarial", "boundary", "clamping"],
            depth_tier="low",
        )
        bound_idx += 1

    assert len(cases) == 1000, f"Expected exactly 1000 cases, got {len(cases)}"
    return cases

