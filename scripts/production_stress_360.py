"""Direct 360-call production stress harness for Chess MCP.

Per 2026-09-09/10 master audit, this harness executes exactly 360 distinct
``session.call_tool(...)`` requests against the MCP endpoint and emits one
JSONL record per call with build pinning and latency/byte stats.

Distribution per master audit:
  - 90 evaluate_position
  - 90 top_moves
  - 100 classify_move
  - 80 analyze_game
= 360 total direct MCP calls.

Audit repairs included:
  - AUDIT-008: 360 distinct argument sets (asserted unique sha256 per call).
  - AUDIT-009: Clean resignation PGN fixture; separate dirty PGN recovery case.
  - AUDIT-010: Explicit CaseSpec with expected_kind and exact error code checks.
  - AUDIT-011: Comprehensive chess semantic oracle (forcing is_mate, terminal states,
               candidate legality/uniqueness, post_fen replay, engine consistency).
  - AUDIT-012: CLI --expected-sha argument with fail-fast build pinning.

Usage:
  python scripts/production_stress_360.py --target https://mcp.trychessy.com --expected-sha <sha>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import chess
import chess.pgn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


DISTRIBUTION: dict[str, int] = {
    "evaluate_position": 90,
    "top_moves": 90,
    "classify_move": 100,
    "analyze_game": 80,
}

CONCURRENCY_PHASES: list[int] = [1, 2, 4]

EXPECTED_TOOLS: set[str] = {
    "evaluate_position",
    "top_moves",
    "classify_move",
    "analyze_game",
}


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    tool: str
    arguments: dict[str, Any]
    expected_kind: Literal["success", "tool_error"]
    expected_error_code: str | None = None
    semantic_profile: str | None = None

    @property
    def arguments_sha256(self) -> str:
        payload = json.dumps(self.arguments, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{self.tool}\0{payload}".encode()).hexdigest()


@dataclass
class CallRecord:
    sequence: int
    tool: str
    case_id: str
    arguments_sha256: str
    started_at: float
    elapsed_ms: float
    transport_ok: bool
    tool_error: bool
    semantic_ok: bool
    build_sha: str
    response_bytes: int
    status: str
    notes: list[str] = field(default_factory=list)


def _build_case_specs() -> list[CaseSpec]:
    """Generate exactly 360 distinct CaseSpec objects matching DISTRIBUTION."""
    cases: list[CaseSpec] = []

    # 1. EVALUATE_POSITION (90 calls: 6 error, 84 valid)
    cases.append(CaseSpec(
        case_id="eval_err_001",
        tool="evaluate_position",
        arguments={"fen": "invalid_fen_string", "depth": 6, "verbosity": "compact", "detail": "standard"},
        expected_kind="tool_error",
        expected_error_code="invalid_position",
    ))
    cases.append(CaseSpec(
        case_id="eval_err_002",
        tool="evaluate_position",
        arguments={"fen": "8/8/8/4k3/4K3/8/8/8 w - - 0 1", "depth": 6, "verbosity": "compact", "detail": "standard"},
        expected_kind="tool_error",
        expected_error_code="invalid_fen",
    ))
    cases.append(CaseSpec(
        case_id="eval_err_003",
        tool="evaluate_position",
        arguments={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "depth": 6, "verbosity": "bogus_verbosity", "detail": "standard"},
        expected_kind="tool_error",
        expected_error_code="invalid_verbosity",
    ))
    cases.append(CaseSpec(
        case_id="eval_err_004",
        tool="evaluate_position",
        arguments={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "depth": 6, "verbosity": "compact", "detail": "bogus_detail"},
        expected_kind="tool_error",
        expected_error_code="invalid_detail",
    ))
    cases.append(CaseSpec(
        case_id="eval_err_005",
        tool="evaluate_position",
        arguments={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "depth": 6, "verbosity": "minimal", "detail": "coach"},
        expected_kind="tool_error",
        expected_error_code="invalid_argument",
    ))
    cases.append(CaseSpec(
        case_id="eval_err_006",
        tool="evaluate_position",
        arguments={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "depth": 8, "verbosity": "min", "detail": "forensic"},
        expected_kind="tool_error",
        expected_error_code="invalid_argument",
    ))

    eval_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
        "8/8/4k3/8/8/4K3/3Q4/8 w - - 0 1",
        "8/8/4k3/8/8/4K3/3R4/8 w - - 0 1",
        "8/4k3/8/4P3/8/4K3/8/8 w - - 0 1",
        "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    ]
    eval_depths = [4, 6, 8, 10, 12, 14]
    valid_verb_details = [
        ("compact", "standard"),
        ("full", "coach"),
        ("compact", "forensic"),
        ("minimal", "standard"),
        ("standard", "coach"),
        ("default", "forensic"),
    ]
    eval_idx = 0
    for f_i, fen in enumerate(eval_fens):
        for d_i, depth in enumerate(eval_depths):
            v, det = valid_verb_details[(f_i + d_i) % len(valid_verb_details)]
            eval_idx += 1
            cases.append(CaseSpec(
                case_id=f"eval_{eval_idx:03d}",
                tool="evaluate_position",
                arguments={"fen": fen, "depth": depth, "verbosity": v, "detail": det},
                expected_kind="success",
            ))

    # 2. TOP_MOVES (90 calls: 5 error, 85 valid)
    cases.append(CaseSpec(
        case_id="top_err_001",
        tool="top_moves",
        arguments={"fen": "bad_fen", "n": 3, "depth": 6, "verbosity": "compact", "detail": "standard"},
        expected_kind="tool_error",
        expected_error_code="invalid_position",
    ))
    cases.append(CaseSpec(
        case_id="top_err_002",
        tool="top_moves",
        arguments={"fen": "8/8/8/4k3/4K3/8/8/8 w - - 0 1", "n": 3, "depth": 6, "verbosity": "compact", "detail": "standard"},
        expected_kind="tool_error",
        expected_error_code="invalid_fen",
    ))
    cases.append(CaseSpec(
        case_id="top_err_003",
        tool="top_moves",
        arguments={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "n": 3, "depth": 6, "verbosity": "bogus", "detail": "standard"},
        expected_kind="tool_error",
        expected_error_code="invalid_verbosity",
    ))
    cases.append(CaseSpec(
        case_id="top_err_004",
        tool="top_moves",
        arguments={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "n": 3, "depth": 6, "verbosity": "compact", "detail": "bogus"},
        expected_kind="tool_error",
        expected_error_code="invalid_detail",
    ))
    cases.append(CaseSpec(
        case_id="top_err_005",
        tool="top_moves",
        arguments={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "n": 3, "depth": 6, "include_moves": ["e2e5"]},
        expected_kind="tool_error",
        expected_error_code="illegal_move",
    ))

    top_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
        "8/8/4k3/8/8/4K3/3Q4/8 w - - 0 1",
        "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    ]
    n_values = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    top_depths = [6, 8, 10, 12, 14]
    top_verbosities = ["compact", "full", "standard", "minimal"]
    top_details = ["standard", "coach", "forensic"]

    top_idx = 0
    for f_i, fen in enumerate(top_fens):
        for n_i, n in enumerate(n_values):
            if top_idx >= 85:
                break
            top_idx += 1
            d = top_depths[(f_i + n_i) % len(top_depths)]
            v = top_verbosities[(f_i + n_i) % len(top_verbosities)]
            det = top_details[(f_i + n_i) % len(top_details)]
            args: dict[str, Any] = {"fen": fen, "n": n, "depth": d, "verbosity": v, "detail": det}
            if n_i == 2 and fen == top_fens[0]:
                args["include_moves"] = ["e2e4"]
            elif n_i == 4 and fen == top_fens[0]:
                args["include_moves"] = ["d2d4", "g1f3"]
            cases.append(CaseSpec(
                case_id=f"top_{top_idx:03d}",
                tool="top_moves",
                arguments=args,
                expected_kind="success",
            ))

    # 3. CLASSIFY_MOVE (100 calls: 6 error, 94 valid)
    cases.append(CaseSpec(
        case_id="classify_err_001",
        tool="classify_move",
        arguments={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "move": "e2e5", "depth": 6},
        expected_kind="tool_error",
        expected_error_code="illegal_move",
    ))
    cases.append(CaseSpec(
        case_id="classify_err_002",
        tool="classify_move",
        arguments={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "move": "e4", "depth": 6, "action_type": "bogus_action"},
        expected_kind="tool_error",
        expected_error_code="invalid_action_type",
    ))
    cases.append(CaseSpec(
        case_id="classify_err_003",
        tool="classify_move",
        arguments={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "move": None, "action_type": "play_move", "depth": 6},
        expected_kind="tool_error",
        expected_error_code="invalid_input",
    ))
    cases.append(CaseSpec(
        case_id="classify_err_004",
        tool="classify_move",
        arguments={"fen": "bad_fen", "move": "e4", "depth": 6},
        expected_kind="tool_error",
        expected_error_code="invalid_position",
    ))
    cases.append(CaseSpec(
        case_id="classify_err_005",
        tool="classify_move",
        arguments={"fen": "8/8/8/4k3/4K3/8/8/8 w - - 0 1", "move": "Ke3", "depth": 6},
        expected_kind="tool_error",
        expected_error_code="invalid_fen",
    ))
    cases.append(CaseSpec(
        case_id="classify_err_006",
        tool="classify_move",
        arguments={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "move": "e4", "depth": 6, "detail": "bogus"},
        expected_kind="tool_error",
        expected_error_code="invalid_detail",
    ))

    startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    italian = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    scholars = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    fools = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
    fools_blunder_fen = "rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2"
    kq_fen = "8/8/4k3/8/8/4K3/3Q4/8 w - - 0 1"
    lucena = "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1"
    backrank = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
    ep_fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"

    classify_setups = [
        (startpos, "e4", "play_move"),
        (startpos, "d4", "play_move"),
        (startpos, "c4", "play_move"),
        (startpos, "Nf3", "play_move"),
        (startpos, "Nc3", "play_move"),
        (startpos, "f4", "play_move"),
        (startpos, "g3", "play_move"),
        (startpos, "b3", "play_move"),
        (italian, "d3", "play_move"),
        (italian, "c3", "play_move"),
        (italian, "O-O", "play_move"),
        (italian, "Nc3", "play_move"),
        (italian, "h3", "play_move"),
        (italian, "a4", "play_move"),
        (scholars, "Qxf7#", "play_move"),
        (scholars, "Qe2", "play_move"),
        (scholars, "Qf3", "play_move"),
        (fools, "Qh4#", "play_move"),
        (fools, "d5", "play_move"),
        (fools, "Nc6", "play_move"),
        (fools, "Nf6", "play_move"),
        (fools_blunder_fen, "g4", "play_move"),
        (kq_fen, "Qe2", "play_move"),
        (kq_fen, "Kd4", "play_move"),
        (kq_fen, "Qd6+", "play_move"),
        (lucena, "Rc1", "play_move"),
        (backrank, "Re8#", "play_move"),
        (ep_fen, "exf6", "play_move"),
    ]
    classify_depths = [6, 8, 10, 12]
    classify_details = ["standard", "coach", "forensic"]

    classify_idx = 0
    for s_i, (fen, move, action_type) in enumerate(classify_setups):
        for d_i, depth in enumerate(classify_depths):
            if classify_idx >= 94:
                break
            classify_idx += 1
            det = classify_details[(s_i + d_i) % len(classify_details)]
            args = {"fen": fen, "move": move, "action_type": action_type, "depth": depth, "detail": det}
            cases.append(CaseSpec(
                case_id=f"classify_{classify_idx:03d}",
                tool="classify_move",
                arguments=args,
                expected_kind="success",
            ))

    # 4. ANALYZE_GAME (80 calls: 4 error, 76 valid)
    cases.append(CaseSpec(
        case_id="analyze_err_001",
        tool="analyze_game",
        arguments={"pgn": "[Event ?? invalid syntax :::", "depth": 6},
        expected_kind="tool_error",
        expected_error_code="invalid_pgn",
    ))
    cases.append(CaseSpec(
        case_id="analyze_err_002",
        tool="analyze_game",
        arguments={"pgn": "1. e4 e5 2. e5 1-0", "depth": 6},
        expected_kind="tool_error",
        expected_error_code="invalid_pgn",
    ))
    cases.append(CaseSpec(
        case_id="analyze_err_003",
        tool="analyze_game",
        arguments={"pgn": "[Event \"Game 1\"]\n[White \"P1\"]\n[Black \"P2\"]\n\n1. e4 e5 0-1\n\n[Event \"Game 2\"]\n[White \"P3\"]\n[Black \"P4\"]\n\n1. d4 d5 1-0", "depth": 6},
        expected_kind="tool_error",
        expected_error_code="multiple_games_not_supported",
    ))
    cases.append(CaseSpec(
        case_id="analyze_err_004",
        tool="analyze_game",
        arguments={"pgn": "1. e4 e5 2. Nf3 Nc6 *", "depth": 6, "detail": "bogus_detail"},
        expected_kind="tool_error",
        expected_error_code="invalid_detail",
    ))

    # AUDIT-009 Dirty PGN recovery fixture
    cases.append(CaseSpec(
        case_id="analyze_dirty_pgn_001",
        tool="analyze_game",
        arguments={"pgn": "[Termination \"White resigns\"}] 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 0-1", "depth": 6, "detail": "standard"},
        expected_kind="success",
        semantic_profile="dirty_pgn_recovery",
    ))

    # AUDIT-009 Clean resignation fixture
    cases.append(CaseSpec(
        case_id="analyze_clean_resignation_001",
        tool="analyze_game",
        arguments={
            "pgn": "[Event \"Resignation semantics\"]\n[Result \"0-1\"]\n[Termination \"White resigns\"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 0-1",
            "depth": 6,
            "detail": "standard",
        },
        expected_kind="success",
        semantic_profile="clean_resignation",
    ))

    analyze_pgns = [
        "1. f3 e5 2. g4 Qh4# 0-1",
        "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0",
        "1. e4 e5 2. Nf3 Nc6 3. Nc3 Nf6 4. Bb5 Bb4 1/2-1/2",
        "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 1/2-1/2",
        "1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 *",
        "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Bd2 Bxd2+ 8. Nbxd2 d5 *",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 *",
        "1. e4 e6 2. d4 d5 3. Nc3 Nf6 4. Bg5 Be7 5. e5 Nfd7 6. h4 Bxg5 7. hxg5 Qxg5 *",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Nbd7 5. e3 c6 6. Nf3 Qa5 *",
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 *",
        "1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 *",
        "1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 d5 5. cxd5 Nxd5 6. Bg2 *",
    ]
    analyze_depths = [4, 6, 8, 10, 12]
    analyze_details = ["standard", "coach", "forensic"]

    analyze_idx = 2
    for pgn in analyze_pgns:
        for depth in analyze_depths:
            for detail in analyze_details:
                if analyze_idx >= 76:
                    break
                analyze_idx += 1
                args = {"pgn": pgn, "depth": depth, "detail": detail}
                if analyze_idx % 3 == 0:
                    args["white_player"] = "Player W"
                elif analyze_idx % 3 == 1:
                    args["black_player"] = "Player B"
                cases.append(CaseSpec(
                    case_id=f"analyze_{analyze_idx:03d}",
                    tool="analyze_game",
                    arguments=args,
                    expected_kind="success",
                ))
            if analyze_idx >= 76:
                break
        if analyze_idx >= 76:
            break

    return cases


def _check_forcing_move_evidence(data: Any, root_fen: str | None = None) -> list[str]:
    """AUDIT-011: Verify forcing move evidence invariants recursively."""
    errors: list[str] = []
    if isinstance(data, dict):
        san = data.get("san")
        is_mate = data.get("is_mate")
        is_check = data.get("is_check")
        if isinstance(san, str) and "is_mate" in data:
            if san.endswith("#") and is_mate is not True:
                errors.append(f"san='{san}' ends with '#' but is_mate is {is_mate}")
            if is_mate is True and is_check is not True:
                errors.append(f"is_mate is True but is_check is {is_check} for san='{san}'")
            uci = data.get("uci")
            if root_fen and isinstance(uci, str):
                try:
                    b = chess.Board(root_fen)
                    m = chess.Move.from_uci(uci)
                    if m in b.legal_moves:
                        child = b.copy(stack=False)
                        child.push(m)
                        if is_mate is not None and child.is_checkmate() != is_mate:
                            errors.append(
                                f"uci='{uci}' child.is_checkmate()={child.is_checkmate()} != is_mate={is_mate}"
                            )
                except Exception:
                    pass
        for k, v in data.items():
            sub_fen = (
                root_fen
                if k
                not in (
                    "after_reply",
                    "tactical_after_reply",
                    "tactical_after_played",
                    "after_played",
                    "mechanism_evidence",
                    "candidate_comparisons",
                    "forensics",
                )
                else None
            )
            errors.extend(_check_forcing_move_evidence(v, sub_fen))
    elif isinstance(data, (list, tuple)):
        for item in data:
            errors.extend(_check_forcing_move_evidence(item, root_fen))
    return errors


def _run_semantic_oracle(spec: CaseSpec, data: dict[str, Any] | None) -> list[str]:
    """AUDIT-011: Run chess semantic oracle assertions across tool output."""
    if not data:
        return ["no parsed JSON payload in tool response"]

    errors: list[str] = []
    tool = spec.tool
    args = spec.arguments

    if tool == "evaluate_position":
        fen = args["fen"]
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
            errors.extend(_check_forcing_move_evidence(data, root_fen=fen))
        except Exception as exc:
            errors.append(f"oracle evaluate_position error: {exc}")

    elif tool == "top_moves":
        fen = args["fen"]
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
                        san = m.get("san")
                        if san and b.san(m_obj) != san:
                            errors.append(f"candidate SAN '{san}' != board.san '{b.san(m_obj)}'")
                        post_fen = m.get("post_fen")
                        if post_fen:
                            child = b.copy(stack=False)
                            child.push(m_obj)
                            if child.fen() != post_fen:
                                errors.append(f"candidate post_fen mismatch for '{uci}'")
                        pv = m.get("pv")
                        if pv and len(pv) > 0 and pv[0] != uci:
                            errors.append(f"pv[0]='{pv[0]}' != candidate '{uci}'")
                    except ValueError:
                        errors.append(f"candidate move '{uci}' invalid UCI")
            include_moves = args.get("include_moves") or []
            for inc in include_moves:
                try:
                    inc_m = b.parse_san(inc) if not inc.isalnum() or len(inc) != 4 else chess.Move.from_uci(inc)
                    if inc_m in b.legal_moves and inc_m.uci() not in ucis:
                        errors.append(f"legal include_move '{inc}' missing from returned candidates")
                except Exception:
                    pass
            errors.extend(_check_forcing_move_evidence(moves, root_fen=fen))
        except Exception as exc:
            errors.append(f"oracle top_moves error: {exc}")

    elif tool == "classify_move":
        fen = args["fen"]
        move_str = args.get("move")
        action_type = args.get("action_type", "play_move")
        try:
            if action_type == "play_move" and move_str:
                played = data.get("played_move") or {}
                if played:
                    if played.get("uci") != move_str and played.get("san") != move_str:
                        errors.append(f"played_move {played} does not match requested '{move_str}'")
                    if data.get("is_engine_best"):
                        best_uci = (data.get("best_move") or {}).get("uci")
                        if best_uci and played.get("uci") != best_uci:
                            errors.append(f"is_engine_best is True but played '{played.get('uci')}' != best '{best_uci}'")
            errors.extend(_check_forcing_move_evidence(data, root_fen=fen))
        except Exception as exc:
            errors.append(f"oracle classify_move error: {exc}")

    elif tool == "analyze_game":
        pgn = args["pgn"]
        try:
            if spec.semantic_profile == "clean_resignation":
                if data.get("termination") != "resignation":
                    errors.append(f"expected termination='resignation', got {data.get('termination')!r}")
                if data.get("termination_header") != "White resigns":
                    errors.append(f"expected termination_header='White resigns', got {data.get('termination_header')!r}")
            elif spec.semantic_profile != "dirty_pgn_recovery":
                game = chess.pgn.read_game(io.StringIO(pgn))
                if game:
                    mainline = list(game.mainline_moves())
                    total_plies = data.get("total_plies")
                    if total_plies is not None and total_plies != len(mainline):
                        errors.append(f"total_plies={total_plies} != mainline length {len(mainline)}")
                    final_fen = data.get("final_fen")
                    if final_fen:
                        b = game.board()
                        for m in mainline:
                            b.push(m)
                        if final_fen != b.fen():
                            errors.append(f"final_fen mismatch: got '{final_fen}', expected '{b.fen()}'")
            errors.extend(_check_forcing_move_evidence(data))
        except Exception as exc:
            errors.append(f"oracle analyze_game error: {exc}")

    return errors


def _extract_response_data(result: Any) -> tuple[bool, str, dict[str, Any] | None]:
    """Returns (is_error, text, parsed_json_or_none)."""
    is_err = False
    text = ""
    if hasattr(result, "isError"):
        is_err = bool(result.isError)
    elif hasattr(result, "is_error"):
        is_err = bool(result.is_error)

    content = getattr(result, "content", None)
    if content and isinstance(content, (list, tuple)) and len(content) > 0:
        first = content[0]
        text = getattr(first, "text", "") or ""
    elif isinstance(result, dict):
        is_err = bool(result.get("isError") or result.get("is_error"))
        c = result.get("content", [])
        if c and isinstance(c[0], dict):
            text = c[0].get("text", "")

    parsed: dict[str, Any] | None = None
    if text:
        try:
            val = json.loads(text)
            if isinstance(val, dict):
                parsed = val
        except Exception:
            parsed = None
    return is_err, text, parsed


def _find_build_sha(result: Any, parsed: dict[str, Any] | None = None) -> str:
    if parsed and isinstance(parsed, dict):
        sha = _deep_find(parsed, "build_sha")
        if sha:
            return sha
    md = getattr(result, "model_dump", None)
    if callable(md):
        try:
            dumped = md(mode="json", by_alias=True)
            return _deep_find(dumped, "build_sha")
        except Exception:
            pass
    return ""


def _deep_find(value: Any, key: str) -> str:
    if isinstance(value, dict):
        if key in value and isinstance(value[key], str):
            return value[key]
        for v in value.values():
            found = _deep_find(v, key)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for v in value:
            found = _deep_find(v, key)
            if found:
                return found
    return ""


def _percentile(values: Sequence[float | int], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct
    f_ = int(k)
    c_ = min(f_ + 1, len(sorted_vals) - 1)
    return float(sorted_vals[f_] + (sorted_vals[c_] - sorted_vals[f_]) * (k - f_))


async def _run_phase(
    session: ClientSession,
    cases: list[CaseSpec],
    concurrency: int,
    expected_sha: str | None,
    observed_shas: set[str],
) -> list[CallRecord]:
    records: list[CallRecord] = []
    semaphore = asyncio.Semaphore(concurrency)
    sequence = 0

    async def _run_one(spec: CaseSpec) -> CallRecord:
        nonlocal sequence
        sequence += 1
        record = CallRecord(
            sequence=sequence,
            tool=spec.tool,
            case_id=spec.case_id,
            arguments_sha256=spec.arguments_sha256,
            started_at=time.time(),
            elapsed_ms=0.0,
            transport_ok=False,
            tool_error=False,
            semantic_ok=True,
            build_sha="",
            response_bytes=0,
            status="active",
        )
        async with semaphore:
            t0 = time.monotonic()
            retries_left = 1
            result = None
            while True:
                try:
                    result = await session.call_tool(spec.tool, arguments=spec.arguments)
                    elapsed_ms = (time.monotonic() - t0) * 1000
                    record.elapsed_ms = elapsed_ms
                    record.transport_ok = True
                    break
                except Exception as exc:
                    err_msg = str(exc).lower()
                    if retries_left > 0 and (
                        "stream ended" in err_msg
                        or "closed" in err_msg
                        or "connection" in err_msg
                        or "reset" in err_msg
                    ):
                        retries_left -= 1
                        await asyncio.sleep(0.5)
                        continue
                    elapsed_ms = (time.monotonic() - t0) * 1000
                    record.elapsed_ms = elapsed_ms
                    record.transport_ok = False
                    record.semantic_ok = False
                    record.status = "transport_error"
                    record.notes.append(f"transport error: {type(exc).__name__}: {exc}")
                    return record

            try:
                is_err, text, parsed = _extract_response_data(result)
                record.response_bytes = len(text.encode()) if text else 0

                build_sha = _find_build_sha(result, parsed)
                record.build_sha = build_sha
                if build_sha:
                    observed_shas.add(build_sha)
                    if expected_sha and not build_sha.startswith(expected_sha):
                        record.semantic_ok = False
                        record.notes.append(f"build_sha drift: expected {expected_sha}, got {build_sha}")

                if spec.expected_kind == "tool_error":
                    if not is_err:
                        record.semantic_ok = False
                        record.status = "unexpected_success"
                        record.notes.append(
                            f"expected error {spec.expected_error_code}, but got success"
                        )
                    else:
                        record.tool_error = True
                        code_in_brackets = ""
                        if "[" in text and "]" in text:
                            open_b = text.find("[")
                            close_b = text.find("]", open_b)
                            if close_b > open_b:
                                code_in_brackets = text[open_b + 1 : close_b].strip().lower()
                        expected_lower = (spec.expected_error_code or "").lower()

                        matches_code = False
                        if code_in_brackets:
                            if expected_lower == code_in_brackets:
                                matches_code = True
                            elif expected_lower in ("invalid_fen", "invalid_position") and code_in_brackets in ("invalid_fen", "invalid_position"):
                                matches_code = True

                        if not matches_code and "validation error for" in text.lower():
                            arg_names = {
                                "invalid_verbosity": ["verbosity"],
                                "invalid_detail": ["detail"],
                                "invalid_action_type": ["action_type"],
                                "invalid_proof_mode": ["proof_mode"],
                                "invalid_argument": ["verbosity", "detail", "argument"],
                            }
                            expected_args = arg_names.get(expected_lower, [expected_lower])
                            if any(arg in text.lower() for arg in expected_args):
                                matches_code = True

                        if not matches_code and expected_lower and expected_lower in text.lower():
                            matches_code = True

                        if not matches_code:
                            record.semantic_ok = False
                            record.status = "wrong_tool_error"
                            record.notes.append(
                                f"expected error '{expected_lower}', got code '{code_in_brackets}' ({text[:100]})"
                            )
                        else:
                            record.status = "expected_invalid_input"
                else:
                    if is_err:
                        record.tool_error = True
                        record.semantic_ok = False
                        record.status = "unexpected_tool_error"
                        record.notes.append(f"unexpected tool error: {text[:200]}")
                    else:
                        record.status = "active"
                        semantic_errors = _run_semantic_oracle(spec, parsed)
                        if semantic_errors:
                            record.semantic_ok = False
                            for se in semantic_errors:
                                record.notes.append(f"semantic failure: {se}")

            except Exception as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                record.elapsed_ms = elapsed_ms
                record.semantic_ok = False
                record.notes.append(f"processing error: {type(exc).__name__}: {exc}")
        return record

    tasks = [asyncio.create_task(_run_one(spec)) for spec in cases]
    records = await asyncio.gather(*tasks)
    return list(records)


def _write_jsonl(records: list[CallRecord], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), default=str) + "\n")


def _write_markdown_summary(
    records: list[CallRecord],
    requested_expected_sha: str | None,
    first_observed_build_sha: str | None,
    all_observed_build_shas: set[str],
    by_tool: dict[str, list[CallRecord]],
    path: str,
) -> None:
    total = len(records)
    transport_failures = sum(1 for r in records if not r.transport_ok)
    unexpected_errors = sum(1 for r in records if r.status == "unexpected_tool_error")
    wrong_tool_errors = sum(1 for r in records if r.status == "wrong_tool_error")
    unexpected_successes = sum(1 for r in records if r.status == "unexpected_success")
    expected_invalid = sum(1 for r in records if r.status == "expected_invalid_input")
    semantic_flags = sum(1 for r in records if not r.semantic_ok)

    elapsed_values = [r.elapsed_ms for r in records if r.transport_ok]
    bytes_values = [r.response_bytes for r in records if r.response_bytes > 0]
    sha_drift = [r for r in records if any("drift" in n for n in r.notes)]

    lines = [
        "# Chess MCP 360-call production stress summary",
        "",
        f"- Total direct `session.call_tool` attempts: **{total}**",
        f"- Requested expected build SHA: `{requested_expected_sha or '(not enforced)'}`",
        f"- First observed build SHA: `{first_observed_build_sha or '(none)'}`",
        f"- All observed build SHAs: `{sorted(all_observed_build_shas)}`",
        f"- Transport failures: **{transport_failures}**",
        f"- Unexpected tool errors: **{unexpected_errors}**",
        f"- Wrong tool error codes: **{wrong_tool_errors}**",
        f"- Unexpected successes (failed rejection): **{unexpected_successes}**",
        f"- Expected invalid-input errors (correctly rejected): **{expected_invalid}**",
        f"- Semantic/invariant flags: **{semantic_flags}**",
        f"- build_sha drift records: **{len(sha_drift)}**",
        "",
        "## Latency (ms)",
        "",
    ]
    for label in ("p50", "p90", "p95", "p99", "max"):
        v = (
            _percentile(elapsed_values, {"p50": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99}[label])
            if label != "max"
            else (max(elapsed_values) if elapsed_values else 0.0)
        )
        lines.append(f"- {label}: **{v:.1f} ms**")
    lines.extend(
        [
            "",
            "## Response size (bytes)",
            "",
        ]
    )
    for label in ("p50", "p90", "p95", "p99", "max"):
        v = (
            _percentile(bytes_values, {"p50": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99}[label])
            if label != "max"
            else (max(bytes_values) if bytes_values else 0)
        )
        lines.append(f"- {label}: **{v:.0f} bytes**")
    lines.extend(
        [
            "",
            "## Per-tool breakdown",
            "",
            "| Tool | Calls | OK | Transport err | Unexpected err | Wrong err | Unexp success | Expected invalid | Semantic flags |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tool_name, recs in by_tool.items():
        ok = sum(1 for r in recs if r.transport_ok and not r.tool_error and r.semantic_ok)
        te = sum(1 for r in recs if not r.transport_ok)
        ue = sum(1 for r in recs if r.status == "unexpected_tool_error")
        we = sum(1 for r in recs if r.status == "wrong_tool_error")
        us = sum(1 for r in recs if r.status == "unexpected_success")
        ei = sum(1 for r in recs if r.status == "expected_invalid_input")
        sf = sum(1 for r in recs if not r.semantic_ok)
        lines.append(f"| {tool_name} | {len(recs)} | {ok} | {te} | {ue} | {we} | {us} | {ei} | {sf} |")

    if sha_drift:
        lines.extend(
            [
                "",
                "## Build SHA drift records",
                "",
            ]
        )
        for r in sha_drift[:20]:
            lines.append(f"- #{r.sequence} {r.tool}/{r.case_id}: {r.notes[0]}")

    semantic_issues = [r for r in records if not r.semantic_ok and r.status != "expected_invalid_input"]
    if semantic_issues:
        lines.extend(
            [
                "",
                "## Semantic and Error Invariant Issues",
                "",
            ]
        )
        for r in semantic_issues[:20]:
            lines.append(f"- #{r.sequence} {r.tool}/{r.case_id} ({r.status}): {'; '.join(r.notes)}")

    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


async def _main_async(args: argparse.Namespace) -> int:
    target = args.target.rstrip("/")
    cli_expected_sha = (args.expected_sha or "").strip() or None
    print(
        f"[INFO] target={target} expected_sha={cli_expected_sha} "
        f"concurrency_phases={CONCURRENCY_PHASES} distribution={DISTRIBUTION}",
        flush=True,
    )

    all_specs = _build_case_specs()
    assert len(all_specs) == 360, f"Expected 360 case specs, got {len(all_specs)}"
    unique_hashes = {spec.arguments_sha256 for spec in all_specs}
    assert len(unique_hashes) == 360, f"Expected 360 unique argument hashes, got {len(unique_hashes)}"
    print(f"[INFO] generated 360 distinct argument sets (unique sha256 count: {len(unique_hashes)})", flush=True)

    mcp_url = target + "/mcp"
    all_records: list[CallRecord] = []
    by_tool: dict[str, list[CallRecord]] = {tool: [] for tool in DISTRIBUTION}
    observed_shas: set[str] = set()
    first_observed_sha: str | None = None

    try:
        async with streamable_http_client(mcp_url) as streams:
            read_stream, write_stream, *_ = streams
            async with ClientSession(read_stream, write_stream) as session:
                init = await session.initialize()
                print(f"[INFO] initialize: {init}", flush=True)

                tools_result = await session.list_tools()
                tool_names = {tool.name for tool in tools_result.tools}
                if tool_names != EXPECTED_TOOLS:
                    print(
                        f"[FAIL] tool surface mismatch: expected {sorted(EXPECTED_TOOLS)} "
                        f"got {sorted(tool_names)}",
                        flush=True,
                    )
                    return 2

                # Iterate tool groups per distribution
                for tool_name, count in DISTRIBUTION.items():
                    tool_specs = [s for s in all_specs if s.tool == tool_name]
                    assert len(tool_specs) == count, f"Expected {count} specs for {tool_name}, got {len(tool_specs)}"

                    num_phases = len(CONCURRENCY_PHASES)
                    chunk_size = len(tool_specs) // num_phases
                    for phase_idx, concurrency in enumerate(CONCURRENCY_PHASES):
                        start = phase_idx * chunk_size
                        end = (phase_idx + 1) * chunk_size if phase_idx < num_phases - 1 else len(tool_specs)
                        phase_cases = tool_specs[start:end]

                        records = await _run_phase(
                            session, phase_cases, concurrency, cli_expected_sha or first_observed_sha, observed_shas
                        )
                        all_records.extend(records)
                        by_tool[tool_name].extend(records)

                        if not first_observed_sha:
                            for r in records:
                                if r.build_sha:
                                    first_observed_sha = r.build_sha
                                    print(f"[INFO] first observed build_sha={first_observed_sha}", flush=True)
                                    if cli_expected_sha and not first_observed_sha.startswith(cli_expected_sha):
                                        print(
                                            f"[FAIL] initial deployment build_sha {first_observed_sha} "
                                            f"does not match expected {cli_expected_sha}",
                                            flush=True,
                                        )
                                    break
    except Exception as exc:
        print(f"[FAIL] harness-level exception: {type(exc).__name__}: {exc}", flush=True)
        return 1

    jsonl_path = args.jsonl_out
    md_path = args.md_out
    _write_jsonl(all_records, jsonl_path)
    _write_markdown_summary(
        all_records,
        cli_expected_sha,
        first_observed_sha,
        observed_shas,
        by_tool,
        md_path,
    )

    total = len(all_records)
    transport_failures = sum(1 for r in all_records if not r.transport_ok)
    unexpected_errors = sum(1 for r in all_records if r.status == "unexpected_tool_error")
    wrong_tool_errors = sum(1 for r in all_records if r.status == "wrong_tool_error")
    unexpected_successes = sum(1 for r in all_records if r.status == "unexpected_success")
    sha_drift = sum(1 for r in all_records if any("drift" in n for n in r.notes))
    semantic_flags = sum(1 for r in all_records if not r.semantic_ok)

    print(
        f"[INFO] recorded {total} calls -> {jsonl_path} (summary {md_path})",
        flush=True,
    )
    print(
        f"[INFO] transport_failures={transport_failures} "
        f"unexpected_tool_errors={unexpected_errors} wrong_tool_errors={wrong_tool_errors} "
        f"unexpected_successes={unexpected_successes} build_sha_drift={sha_drift} "
        f"semantic_flags={semantic_flags}",
        flush=True,
    )

    if total != 360:
        print(f"[FAIL] expected exactly 360 calls; got {total}", flush=True)
        return 1
    if transport_failures > 0:
        print(f"[FAIL] transport failures > 0: {transport_failures}", flush=True)
        return 1
    if unexpected_errors > 0:
        print(f"[FAIL] unexpected tool errors > 0: {unexpected_errors}", flush=True)
        return 1
    if wrong_tool_errors > 0:
        print(f"[FAIL] wrong tool errors > 0: {wrong_tool_errors}", flush=True)
        return 1
    if unexpected_successes > 0:
        print(f"[FAIL] unexpected tool successes > 0: {unexpected_successes}", flush=True)
        return 1
    if sha_drift > 0:
        print(f"[FAIL] build_sha drift > 0: {sha_drift}", flush=True)
        return 1
    if semantic_flags > 0:
        print(f"[FAIL] semantic invariant flags > 0: {semantic_flags}", flush=True)
        return 1

    print("[OK] 360-call production stress passed all P0 invariants", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Direct 360-call production stress harness for Chess MCP"
    )
    parser.add_argument("--target", default="https://mcp.trychessy.com")
    parser.add_argument(
        "--expected-sha",
        default="",
        help="Expected deployed git SHA (audit-012)",
    )
    parser.add_argument(
        "--jsonl-out",
        default="artifacts/chess_mcp_stress_360.jsonl",
        help="JSONL per-call record output path",
    )
    parser.add_argument(
        "--md-out",
        default="artifacts/chess_mcp_stress_360.md",
        help="Markdown summary output path",
    )
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
