"""Property-based and chess-rule invariant suite for forcing move evidence.

Verifies that ``build_forcing_move_evidence(board, move).is_mate == child.is_checkmate()``
is universally true across legal moves, randomly played game trajectories, and
diverse tactical edge cases (en passant mates, promotion mates, castling mates,
discovered mates, double checks).
"""

from __future__ import annotations

import random
import chess
from mcp_server.analysis.forensics import build_forcing_move_evidence


def test_property_forcing_evidence_random_walks() -> None:
    """Across 50 random game walks and sampled legal moves, is_mate matches python-chess."""
    rng = random.Random(42)
    sample_count = 0

    for _ in range(50):
        board = chess.Board()
        # Play up to 40 plies
        for _ in range(40):
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # Test a random sample of legal moves from this position
            sampled_moves = rng.sample(legal_moves, min(len(legal_moves), 4))
            for move in sampled_moves:
                sample_count += 1
                child = board.copy(stack=False)
                child.push(move)

                evidence = build_forcing_move_evidence(board, move)

                # Fundamental P0 invariants
                assert evidence.is_mate == child.is_checkmate(), (
                    f"Invariant violated at FEN={board.fen()} move={move.uci()}: "
                    f"evidence.is_mate={evidence.is_mate} != child.is_checkmate()={child.is_checkmate()}"
                )
                assert evidence.is_check == child.is_check(), (
                    f"Invariant violated at FEN={board.fen()} move={move.uci()}: "
                    f"evidence.is_check={evidence.is_check} != child.is_check()={child.is_check()}"
                )
                if evidence.is_mate:
                    assert evidence.is_check, "Checkmate must imply check"
                    assert evidence.san.endswith("#"), f"Checkmate SAN must end in #: {evidence.san}"

            # Step forward with one random move
            board.push(rng.choice(legal_moves))

    assert sample_count > 200, f"Expected >200 sampled moves, got {sample_count}"


def test_edge_case_en_passant_mate() -> None:
    """Discovered en passant capture that results in checkmate."""
    # FEN: White pawn on e5, Black pawn on d5, White bishop on b1, White bishop on f6.
    # Discovered check via e5xd6 e.p.#
    board = chess.Board("8/8/5B2/3pP3/8/8/8/kBR1K3 w - d6 0 1")
    assert board.is_valid()
    ep_move = chess.Move.from_uci("e5d6")
    assert ep_move in board.legal_moves
    assert board.is_en_passant(ep_move)

    child = board.copy(stack=False)
    child.push(ep_move)
    assert child.is_checkmate()

    evidence = build_forcing_move_evidence(board, ep_move)
    assert evidence.is_mate is True
    assert evidence.is_check is True
    assert evidence.is_capture is True
    assert evidence.san == "exd6#"


def test_edge_case_promotion_mates() -> None:
    """Pawn promotion to Queen delivering checkmate vs non-mating promotion."""
    board = chess.Board("k7/2R2P2/8/8/8/8/8/K7 w - - 0 1")
    assert board.is_valid()

    move_q = chess.Move.from_uci("f7f8q")
    assert move_q in board.legal_moves
    child_q = board.copy(stack=False)
    child_q.push(move_q)
    assert child_q.is_checkmate()

    evidence_q = build_forcing_move_evidence(board, move_q)
    assert evidence_q.is_mate is True
    assert evidence_q.is_check is True
    assert evidence_q.san == "f8=Q#"

    move_n = chess.Move.from_uci("f7f8n")
    assert move_n in board.legal_moves
    child_n = board.copy(stack=False)
    child_n.push(move_n)
    assert not child_n.is_checkmate()

    evidence_n = build_forcing_move_evidence(board, move_n)
    assert evidence_n.is_mate is False
    assert evidence_n.san == "f8=N"


def test_edge_case_castling_check_and_invariants() -> None:
    """Castling move preserves forcing move invariants."""
    board = chess.Board("rnbqk2r/pppp1ppp/5n2/4p3/1bB1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    castle_move = chess.Move.from_uci("e1g1")
    assert castle_move in board.legal_moves
    child = board.copy(stack=False)
    child.push(castle_move)
    evidence = build_forcing_move_evidence(board, castle_move)
    assert evidence.is_mate == child.is_checkmate()
    assert evidence.is_check == child.is_check()
    assert evidence.san == "O-O"


def test_edge_case_discovered_and_double_check() -> None:
    """Discovered check and double check moves."""
    # Discovered check:
    # White: Kh1, Re1, Nd5. Black: Ke8.
    # Nd5 moves to f6: double check! (Knight gives check from f6, Rook from e1).
    board = chess.Board("4k3/8/8/3N4/8/8/8/4R2K w - - 0 1")
    double_check_move = chess.Move.from_uci("d5f6")
    assert double_check_move in board.legal_moves
    child = board.copy(stack=False)
    child.push(double_check_move)
    assert child.is_check()

    evidence = build_forcing_move_evidence(board, double_check_move)
    assert evidence.is_check is True
    assert evidence.is_mate == child.is_checkmate()
    assert evidence.san.endswith("+") or evidence.san.endswith("#")
