from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import chess
import chess.pgn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


@dataclass(frozen=True)
class Case:
    seq: int
    tool: str
    case_id: str
    args: dict[str, Any]
    expect_error: bool = False
    check: str | None = None


def dump_model(value: Any) -> Any:
    fn = getattr(value, "model_dump", None)
    if callable(fn):
        try:
            return fn(mode="json", by_alias=True)
        except TypeError:
            return fn()
    return value


def recursive_values(value: Any, key: str) -> list[Any]:
    value = dump_model(value)
    out: list[Any] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k == key:
                out.append(v)
            out.extend(recursive_values(v, key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.extend(recursive_values(item, key))
    elif isinstance(value, str):
        s = value.strip()
        if s.startswith(("{", "[")):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                pass
            else:
                out.extend(recursive_values(parsed, key))
    return out


def result_is_error(value: Any) -> bool:
    d = dump_model(value)
    return isinstance(d, dict) and bool(d.get("isError") or d.get("is_error"))


def json_bytes(value: Any) -> bytes:
    return json.dumps(dump_model(value), ensure_ascii=False, separators=(",", ":"), default=str).encode()


def first(value: Any, key: str, default: Any = None) -> Any:
    vals = recursive_values(value, key)
    return vals[0] if vals else default


def generate_long_pgn(target_plies: int, seed: int) -> str:
    # Deterministic search for a random legal game that survives to target_plies.
    for attempt in range(500):
        rng = random.Random(seed + attempt * 100003)
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = f"Synthetic stress {target_plies}"
        game.headers["Result"] = "*"
        node = game
        for _ in range(target_plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            rng.shuffle(moves)
            chosen = None
            for move in moves:
                board.push(move)
                terminal = board.is_game_over(claim_draw=False)
                board.pop()
                if not terminal:
                    chosen = move
                    break
            if chosen is None:
                chosen = moves[0]
            board.push(chosen)
            node = node.add_variation(chosen)
            if board.is_game_over(claim_draw=False):
                break
        if board.ply() >= target_plies:
            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
            return game.accept(exporter)
    raise RuntimeError(f"could not generate {target_plies}-ply game")


def fixtures() -> dict[str, str]:
    opera = """[Event "Opera Game"]
[White "Paul Morphy"]
[Black "Duke Karl / Count Isouard"]
[Result "1-0"]

1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6
7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7
12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6 15. Bxd7+ Nxd7
16. Qb8+ Nxb8 17. Rd8# 1-0"""
    fools = """[Event "Fools Mate"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1"""
    explicit_resign = """[Event "Resign"]
[Result "0-1"]
[Termination "White resigned"]

1. e4 e5 2. Nf3 Nc6 0-1"""
    candidate_resign = """[Event "Candidate"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 0-1"""
    repetition = """[Result "*"]

1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 *"""
    return {
        "opera": opera,
        "fools": fools,
        "explicit_resign": explicit_resign,
        "candidate_resign": candidate_resign,
        "repetition": repetition,
        "long80": generate_long_pgn(80, 1001),
        "long120": generate_long_pgn(120, 2002),
        "long160": generate_long_pgn(160, 3003),
    }


def build_cases() -> list[Case]:
    fx = fixtures()
    cases: list[Case] = []
    seq = 0

    def add(tool: str, cid: str, args: dict[str, Any], *, expect_error: bool = False, check: str | None = None) -> None:
        nonlocal seq
        seq += 1
        cases.append(Case(seq, tool, cid, args, expect_error, check))

    # 90 evaluate_position calls.
    eval_positions = [
        ("start", "startpos"),
        ("after_e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        ("mate1", "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"),
        ("checkmated", "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"),
        ("stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
        ("kings", "8/8/8/8/8/8/6k1/4K3 w - - 0 1"),
        ("fifty", "7k/8/6K1/8/8/8/8/Q7 b - - 100 51"),
        ("seventyfive", "7k/8/6K1/8/8/8/8/Q7 b - - 150 76"),
        ("ep", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
        ("promotion", "7k/P7/6K1/8/8/8/8/8 w - - 0 1"),
        ("castle", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        ("queen_end", "8/8/8/3k4/8/4K3/3Q4/8 w - - 0 1"),
    ]
    eval_depths = [1, 2, 4, 6, 10]
    for i in range(60):
        name, fen = eval_positions[i % len(eval_positions)]
        detail = ("standard", "coach", "forensic")[i % 3]
        add("evaluate_position", f"eval_{i:03d}_{name}_{detail}", {
            "fen": fen, "depth": eval_depths[i % len(eval_depths)],
            "verbosity": "compact" if i % 2 == 0 else "full", "detail": detail,
        })
    replay_sets = [
        ["e4"], ["d4"], ["Nf3"], ["c4"],
        ["e4", "e5", "Nf3", "Nc6"],
        ["d4", "d5", "c4", "e6"],
        ["e4", "c6", "d4", "d5"],
        ["Nf3", "Nf6", "Ng1", "Ng8"],
    ]
    for i in range(20):
        add("evaluate_position", f"eval_replay_{i:02d}", {
            "fen": "startpos", "moves": replay_sets[i % len(replay_sets)],
            "depth": 2 + (i % 3), "verbosity": "compact",
            "strict": bool(i % 2), "detail": "standard",
        })
    invalid_eval = [
        "",
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "8/8/8/8/8/8/7k/K6k w - - 0 1",
        "8/8/8/8/8/8/8/K6x w - - 0 1",
        "8/8/8/8/8/8/8/K6k x - - 0 1",
        "8/8/8/8/8/8/8/K6k w - - -1 1",
        "8/8/8/8/8/8/8/K6k w - - 0 0",
        "not a chess position",
        "8/8/8/8/8/8/8/K6k w - -",
        "8/8/8/8/8/8/8/K6k w - - 0",
    ]
    for i, fen in enumerate(invalid_eval):
        add("evaluate_position", f"eval_invalid_{i:02d}", {
            "fen": fen, "depth": 1, "strict": True, "verbosity": "compact",
        }, expect_error=True)

    # 90 top_moves calls.
    top_fens = [
        ("start", "startpos"),
        ("mate1", "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"),
        ("stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
        ("fifty", "7k/8/6K1/8/8/8/8/Q7 b - - 100 51"),
        ("ep", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
        ("promotion", "7k/P7/6K1/8/8/8/8/8 w - - 0 1"),
        ("castle", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ]
    nvals = [-10, 0, 1, 2, 3, 5, 10, 11, 20, 999]
    for i in range(50):
        name, fen = top_fens[i % len(top_fens)]
        add("top_moves", f"top_{i:03d}_{name}_n{nvals[i % len(nvals)]}", {
            "fen": fen, "n": nvals[i % len(nvals)], "depth": 1 + (i % 4),
            "verbosity": "compact", "detail": "standard",
        }, check="n_contract" if nvals[i % len(nvals)] > 10 else None)
    for i in range(20):
        if i % 2 == 0:
            args = {
                "fen": "startpos", "n": 2, "depth": 2, "verbosity": "compact",
                "proof_mode": "tactical", "proof_defenses": [0, 1, 3, 8, 9][i % 5],
                "detail": "forensic",
            }
        else:
            args = {
                "fen": "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
                "n": 2, "depth": 2, "verbosity": "compact",
                "proof_mode": "tactical", "proof_defenses": [1, 3, 8][i % 3],
                "detail": "forensic",
            }
        add("top_moves", f"top_proof_{i:02d}", args, check="proof_label")
    include_sets = [
        ["e4"], ["d4"], ["e4", "d4"], ["Nf3", "c4"],
        ["e4", "d4", "Nf3", "c4"], ["a3", "h3"], ["Na3", "Nh3"],
    ]
    for i in range(10):
        add("top_moves", f"top_include_{i:02d}", {
            "fen": "startpos", "n": 1 + i % 3, "depth": 2,
            "include_moves": include_sets[i % len(include_sets)],
            "verbosity": "compact", "detail": "forensic",
        })
    invalid_top = [
        {"fen": "startpos", "include_moves": ["e5"], "depth": 1},
        {"fen": "startpos", "include_moves": ["e4"] * 9, "depth": 1},
        {"fen": "startpos", "include_moves": ["1. e4"], "strict": True, "depth": 1},
        {"fen": "not fen", "depth": 1},
        {"fen": "startpos", "moves": ["e4", "e4"], "depth": 1},
        {"fen": "", "depth": 1},
        {"fen": "8/8/8/8/8/8/8/8 w - - 0 1", "depth": 1},
        {"fen": "startpos", "moves": ["Nf3+"], "strict": True, "depth": 1},
        {"fen": "startpos", "moves": ["E2E4"], "strict": True, "depth": 1},
        {"fen": "startpos", "verbosity": "banana", "depth": 1},
    ]
    for i, args in enumerate(invalid_top):
        add("top_moves", f"top_invalid_{i:02d}", args, expect_error=True)

    # 100 classify_move calls.
    classify_base = [
        ("start_e4", "startpos", "e4"),
        ("start_d4", "startpos", "d4"),
        ("start_a3", "startpos", "a3"),
        ("mate_qg7", "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", "Qg7#"),
        ("mate_qxf7", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "Qxf7#"),
        ("ep", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "exd6"),
        ("promote", "7k/P7/6K1/8/8/8/8/8 w - - 0 1", "a8=Q"),
        ("castle_k", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "O-O"),
        ("castle_q", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "O-O-O"),
        ("fifty_play", "7k/8/6K1/8/8/8/8/Q7 b - - 100 51", "Kg8"),
    ]
    for i in range(60):
        name, fen, move = classify_base[i % len(classify_base)]
        detail = ("standard", "coach", "forensic")[i % 3]
        check = "mate_equivalence" if name.startswith("mate_") else None
        add("classify_move", f"classify_{i:03d}_{name}_{detail}", {
            "fen": fen, "move": move, "depth": 1 + (i % 5), "detail": detail,
        }, check=check)
    for i in range(10):
        add("classify_move", f"classify_claim_now_{i:02d}", {
            "fen": "7k/8/6K1/8/8/8/8/Q7 b - - 100 51",
            "action_type": "claim_draw", "depth": 2, "detail": "standard",
        }, check="claim_draw")
    for i in range(10):
        add("classify_move", f"classify_claim_intended_{i:02d}", {
            "fen": "7k/8/6K1/8/8/8/8/Q7 b - - 99 50",
            "move": "Kg8", "action_type": "claim_draw_with_intended_move",
            "depth": 2, "detail": "coach",
        }, check="claim_draw")
    compare_sets = [
        ["e4", "d4"], ["e4", "Nf3"], ["d4", "c4"],
        ["a3", "h3"], ["Na3", "Nh3"], ["e4", "d4", "Nf3", "c4"],
    ]
    for i in range(10):
        add("classify_move", f"classify_compare_{i:02d}", {
            "fen": "startpos", "move": "e4", "depth": 2,
            "detail": "forensic", "compare_moves": compare_sets[i % len(compare_sets)],
        })
    invalid_classify = [
        {"fen": "startpos", "move": "e5", "depth": 1},
        {"fen": "startpos", "move": "Nf3??", "strict": True, "depth": 1},
        {"fen": "startpos", "move": "0-0", "strict": True, "depth": 1},
        {"fen": "startpos", "move": "E2E4", "strict": True, "depth": 1},
        {"fen": "startpos", "move": None, "action_type": "play_move", "depth": 1},
        {"fen": "startpos", "action_type": "claim_draw", "depth": 1},
        {"fen": "not fen", "move": "e4", "depth": 1},
        {"fen": "startpos", "move": "e4", "compare_moves": ["d4"] * 9, "depth": 1},
        {"fen": "startpos", "move": "e4", "compare_moves": ["e5"], "depth": 1},
        {"fen": "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "move": "Kh7", "depth": 1},
    ]
    for i, args in enumerate(invalid_classify):
        add("classify_move", f"classify_invalid_{i:02d}", args, expect_error=True)

    # 80 analyze_game calls.
    short_pgns = [
        ("fools", fx["fools"]),
        ("opera", fx["opera"]),
        ("explicit_resign", fx["explicit_resign"]),
        ("candidate_resign", fx["candidate_resign"]),
        ("repetition", fx["repetition"]),
        ("quiet", "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *"),
        ("comment", "1. e4 {central space} e5 2. Nf3 Nc6 *"),
        ("castle", "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 *"),
    ]
    for i in range(40):
        name, pgn = short_pgns[i % len(short_pgns)]
        detail = ("standard", "coach", "forensic")[i % 3]
        add("analyze_game", f"analyze_{i:03d}_{name}_{detail}", {
            "pgn": pgn, "depth": 1 + (i % 3), "detail": detail,
            "perspective": "white" if i % 2 == 0 else "black",
            "max_critical_moments": [0, 1, 3, 6, 7, 99][i % 6],
        }, check="critical_cap")
    for i in range(20):
        name = ("long80", "long120", "long160")[i % 3]
        add("analyze_game", f"analyze_long_{i:02d}_{name}", {
            "pgn": fx[name], "depth": 1, "detail": "standard",
            "perspective": "white", "max_critical_moments": 6,
        })
    strict_pgns = [
        ("strict_ok", "1. e4 e5 2. Nf3 Nc6 *", False),
        ("strict_nag_glued", "1. e4?? e5 *", False),
        ("strict_false_check", "1. e4+ e5 *", True),
        ("strict_uci", "1. e2e4 e7e5 *", True),
        ("strict_zero_castle", "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. 0-0 *", True),
        ("strict_malformed_header", "[Event \"x\"\n\n1. e4 e5 *", True),
        ("empty", "", True),
        ("garbage", "foo bar baz", True),
        ("illegal", "1. e4 e4 *", True),
        ("two_games", "1. e4 e5 1-0\n\n1. d4 d5 1/2-1/2", True),
    ]
    for i, (name, pgn, expected_error) in enumerate(strict_pgns):
        add("analyze_game", f"analyze_strict_{i:02d}_{name}", {
            "pgn": pgn, "depth": 1, "strict": True, "detail": "standard",
        }, expect_error=expected_error, check="strict_nag" if name == "strict_nag_glued" else None)
    lenient_pgns = [
        "[Event \"x\"\n\n1. e4 e5 *",
        "1. e4! e5 2. Nf3?? Nc6 *",
        "\ufeff[Event \"BOM\"]\n[Result \"*\"]\n\n1. d4 d5 *",
        "[Event \"A\"]\n[Event \"B\"]\n[Result \"*\"]\n\n1. e4 e5 *",
        "[SetUp \"1\"]\n[FEN \"7k/5Q2/6K1/8/8/8/8/8 w - - 0 1\"]\n[Result \"1-0\"]\n\n1. Qg7# 1-0",
        "[Termination \"White resigned\"]\n[Result \"0-1\"]\n\n1. e4 e5 0-1",
        "[Result \"*\"]\n\n1. c4 e5 2. Nc3 Nf6 *",
        "1. Nf3 d5 2. g3 c5 3. Bg2 Nc6 *",
        "1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 *",
        "1. e4 c6 2. d4 d5 3. exd5 cxd5 *",
    ]
    for i, pgn in enumerate(lenient_pgns):
        add("analyze_game", f"analyze_lenient_{i:02d}", {
            "pgn": pgn, "depth": 1, "strict": False, "detail": "coach",
            "max_critical_moments": 2,
        }, check="termination_resources" if i in (5, 6) else None)

    assert seq == 360, seq
    return cases


def semantic_check(case: Case, result: Any) -> list[str]:
    issues: list[str] = []
    if case.check == "mate_equivalence":
        is_best = first(result, "is_engine_best")
        if is_best is None:
            is_best = first(result, "is_best_engine_move")
        eq = first(result, "action_equivalent")
        best_action = first(result, "is_best_action")
        if is_best is True and best_action is True and eq is not True:
            issues.append("F001 exact engine-best mating action reports action_equivalent != true")
    elif case.check == "n_contract":
        requested = case.args.get("n")
        clamped = first(result, "clamped_n")
        if isinstance(requested, int) and requested > 10 and isinstance(clamped, int) and clamped > 10:
            issues.append(f"F002 public schema says n<=10 but runtime clamped_n={clamped}")
    elif case.check == "proof_label":
        statuses = recursive_values(result, "proof_status")
        if not statuses:
            status = first(result, "status")
            if status == "active":
                issues.append("active tactical proof omitted proof_status")
    elif case.check == "claim_draw":
        best_action = first(result, "best_action")
        if case.args["action_type"].startswith("claim_draw") and best_action not in (
            "claim_draw", "claim_draw_with_intended_move", "play_move"
        ):
            issues.append(f"unexpected best_action for claim case: {best_action!r}")
    elif case.check == "critical_cap":
        vals = recursive_values(result, "critical_moments")
        for v in vals:
            if isinstance(v, list) and len(v) > 7:
                issues.append(f"critical_moments returned {len(v)} > 7")
                break
    elif case.check == "termination_resources":
        fps = recursive_values(result, "final_position")
        terms = recursive_values(result, "termination")
        if fps and terms and isinstance(fps[0], dict) and isinstance(terms[0], dict):
            fp = fps[0]
            term = terms[0]
            if fp.get("defensive_resources_exist") is True and term.get("defensive_resources_exist") is False:
                if term.get("status") == "ongoing_or_unknown":
                    issues.append("F005 ongoing termination block reports no defensive resources while final_position says resources exist")
    return issues


async def execute_case(session: ClientSession, case: Case, sem: asyncio.Semaphore, expected_sha: str | None) -> tuple[dict[str, Any], str | None]:
    async with sem:
        started = time.perf_counter()
        args_json = json.dumps(case.args, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        rec: dict[str, Any] = {
            "sequence": case.seq,
            "tool": case.tool,
            "case_id": case.case_id,
            "arguments_sha256": hashlib.sha256(args_json.encode()).hexdigest(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "expected_error": case.expect_error,
        }
        try:
            result = await session.call_tool(case.tool, arguments=case.args)
            elapsed = (time.perf_counter() - started) * 1000
            dumped = dump_model(result)
            is_err = result_is_error(result)
            b = json_bytes(result)
            rec.update({
                "elapsed_ms": round(elapsed, 2),
                "transport_ok": True,
                "tool_error": is_err,
                "response_bytes": len(b),
            })
            build_values = [str(x) for x in recursive_values(result, "build_sha") if x not in (None, "", "unknown")]
            build_sha = build_values[0] if build_values else None
            rec["build_sha"] = build_sha
            notes: list[str] = []
            if case.expect_error:
                if not is_err:
                    notes.append("expected structured tool error but call succeeded")
            else:
                if is_err:
                    notes.append("unexpected structured tool error")
            if not is_err:
                notes.extend(semantic_check(case, result))
            rec["notes"] = notes
            rec["semantic_ok"] = not notes
            if is_err:
                rec["error_preview"] = json.dumps(dumped, ensure_ascii=False, default=str)[:1000]
            if expected_sha and build_sha and not (build_sha.startswith(expected_sha) or expected_sha.startswith(build_sha)):
                rec["semantic_ok"] = False
                rec["notes"].append(f"build SHA changed: expected {expected_sha}, got {build_sha}")
            return rec, build_sha
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            rec.update({
                "elapsed_ms": round(elapsed, 2),
                "transport_ok": False,
                "tool_error": False,
                "semantic_ok": False,
                "response_bytes": 0,
                "build_sha": None,
                "notes": [f"transport/protocol exception: {type(exc).__name__}: {exc}"],
            })
            return rec, None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def write_summary(records: list[dict[str, Any]], out_md: Path, build_sha: str | None) -> None:
    times = [float(r["elapsed_ms"]) for r in records]
    sizes = [int(r["response_bytes"]) for r in records]
    failures = [r for r in records if not r.get("semantic_ok", False)]
    transport_fail = [r for r in records if not r.get("transport_ok", False)]
    tool_err = [r for r in records if r.get("tool_error")]
    unexpected_tool_err = [r for r in records if r.get("tool_error") and not r.get("expected_error")]
    expected_err_missed = [r for r in records if r.get("expected_error") and not r.get("tool_error") and r.get("transport_ok")]
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_tool.setdefault(r["tool"], []).append(r)
    issue_lines: list[str] = []
    for r in records:
        for note in r.get("notes", []):
            if note:
                issue_lines.append(f"- #{r['sequence']} `{r['tool']}` `{r['case_id']}`: {note}")
    slowest = sorted(records, key=lambda r: r["elapsed_ms"], reverse=True)[:20]
    largest = sorted(records, key=lambda r: r["response_bytes"], reverse=True)[:20]

    lines = [
        "# Chess MCP production ultra stress 360",
        "",
        f"- UTC generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Exact direct `session.call_tool` attempts: **{len(records)}**",
        f"- Build SHA pinned from responses: `{build_sha or 'not observed'}`",
        f"- Transport failures: **{len(transport_fail)}**",
        f"- Structured tool errors: **{len(tool_err)}** (includes expected invalid-input tests)",
        f"- Unexpected structured tool errors: **{len(unexpected_tool_err)}**",
        f"- Expected errors that unexpectedly succeeded: **{len(expected_err_missed)}**",
        f"- Records with semantic/invariant notes: **{len(failures)}**",
        "",
        "## Calls by tool",
        "",
    ]
    for tool in sorted(by_tool):
        group = by_tool[tool]
        bad = sum(1 for r in group if not r.get("semantic_ok", False))
        lines.append(f"- `{tool}`: {len(group)} calls, {bad} flagged records")
    lines += [
        "",
        "## Latency",
        "",
        f"- p50: {percentile(times, .50):.1f} ms",
        f"- p90: {percentile(times, .90):.1f} ms",
        f"- p95: {percentile(times, .95):.1f} ms",
        f"- p99: {percentile(times, .99):.1f} ms",
        f"- max: {max(times) if times else 0:.1f} ms",
        "",
        "## Response size",
        "",
        f"- p50: {percentile([float(x) for x in sizes], .50):.0f} bytes",
        f"- p90: {percentile([float(x) for x in sizes], .90):.0f} bytes",
        f"- p95: {percentile([float(x) for x in sizes], .95):.0f} bytes",
        f"- p99: {percentile([float(x) for x in sizes], .99):.0f} bytes",
        f"- max: {max(sizes) if sizes else 0} bytes",
        "",
        "## Semantic and contract findings",
        "",
    ]
    lines += issue_lines or ["No semantic notes were generated by the harness."]
    lines += ["", "## 20 slowest calls", ""]
    for r in slowest:
        lines.append(f"- #{r['sequence']} `{r['tool']}` `{r['case_id']}`: {r['elapsed_ms']} ms")
    lines += ["", "## 20 largest responses", ""]
    for r in largest:
        lines.append(f"- #{r['sequence']} `{r['tool']}` `{r['case_id']}`: {r['response_bytes']} bytes")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    cases = build_cases()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_jsonl = out_dir / f"chess_mcp_stress_360_{stamp}.jsonl"
    out_md = out_dir / f"chess_mcp_stress_360_{stamp}.md"

    sem = asyncio.Semaphore(args.concurrency)
    mcp_url = args.target.rstrip("/") + "/mcp"
    records: list[dict[str, Any]] = []
    expected_sha: str | None = None

    async with streamable_http_client(mcp_url) as streams:
        read_stream, write_stream, *_ = streams
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            expected = {"evaluate_position", "top_moves", "classify_move", "analyze_game"}
            if names != expected:
                raise RuntimeError(f"tool surface mismatch: {sorted(names)}")

            first_case = cases[0]
            first_rec, first_sha = await execute_case(session, first_case, sem, None)
            records.append(first_rec)
            expected_sha = first_sha
            remaining = cases[1:]

            for offset in range(0, len(remaining), args.batch_size):
                batch = remaining[offset : offset + args.batch_size]
                results = await asyncio.gather(
                    *(execute_case(session, case, sem, expected_sha) for case in batch)
                )
                for rec, sha in results:
                    records.append(rec)
                    if expected_sha is None and sha:
                        expected_sha = sha
                if args.pause_ms:
                    await asyncio.sleep(args.pause_ms / 1000)

    records.sort(key=lambda r: r["sequence"])
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    write_summary(records, out_md, expected_sha)

    print(f"JSONL={out_jsonl}")
    print(f"MARKDOWN={out_md}")
    print(f"CALLS={len(records)}")
    print(f"BUILD_SHA={expected_sha}")
    flagged = sum(1 for r in records if not r.get("semantic_ok", False))
    transport = sum(1 for r in records if not r.get("transport_ok", False))
    unexpected_errors = sum(1 for r in records if r.get("tool_error") and not r.get("expected_error"))
    print(f"FLAGGED={flagged}")
    print(f"TRANSPORT_FAILURES={transport}")
    print(f"UNEXPECTED_TOOL_ERRORS={unexpected_errors}")

    if len(records) != 360:
        return 3
    if transport or unexpected_errors:
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="https://mcp.trychessy.com")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pause-ms", type=int, default=100)
    parser.add_argument("--output-dir", default="artifacts")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
