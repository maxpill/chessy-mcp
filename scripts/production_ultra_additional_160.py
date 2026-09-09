from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from production_stress_360 import (
    Case,
    dump_model,
    execute_case,
    generate_long_pgn,
    percentile,
    recursive_values,
)


def fen_after(moves: list[str]) -> str:
    board = chess.Board()
    for san in moves:
        board.push_san(san)
    return board.fen()


def legal_sans(fen: str, limit: int = 8) -> list[str]:
    board = chess.Board() if fen == "startpos" else chess.Board(fen)
    moves = sorted(board.legal_moves, key=lambda m: m.uci())
    return [board.san(m) for m in moves[:limit]]


def build_cases() -> list[Case]:
    cases: list[Case] = []
    seq = 0

    def add(tool: str, cid: str, args: dict[str, Any], *, expect_error: bool = False, check: str | None = None) -> None:
        nonlocal seq
        seq += 1
        cases.append(Case(seq, tool, cid, args, expect_error, check))

    positions = {
        "start": "startpos",
        "italian": fen_after(["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4", "cxd4"]),
        "ruy": fen_after(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3", "d6", "c3", "O-O"]),
        "sicilian": fen_after(["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6", "Be3", "e6", "f3", "b5"]),
        "french": fen_after(["e4", "e6", "d4", "d5", "Nc3", "Nf6", "e5", "Nfd7", "f4", "c5", "Nf3", "Nc6"]),
        "qgd": fen_after(["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O", "Nf3", "Nbd7", "Rc1", "c6"]),
        "kid": fen_after(["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3", "O-O", "Be2", "e5", "O-O", "Nc6"]),
        "mate1": "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
        "mate2": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "fifty": "7k/8/6K1/8/8/8/8/Q7 b - - 100 51",
        "fifty99": "7k/8/6K1/8/8/8/8/Q7 b - - 99 50",
        "seventyfive": "7k/8/6K1/8/8/8/8/Q7 b - - 150 76",
        "ep": "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "promo": "7k/P7/6K1/8/8/8/8/8 w - - 0 1",
        "castle": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "rookend": "8/5pk1/6p1/3R4/8/5PK1/8/7r w - - 0 1",
        "pawnrace": "8/2p5/3k4/8/4K3/5P2/8/8 w - - 0 1",
    }

    hard = ["italian", "ruy", "sicilian", "french", "qgd", "kid", "rookend", "pawnrace"]

    # 40 evaluate_position calls. High-depth and edge semantics.
    for i in range(24):
        name = hard[i % len(hard)]
        add(
            "evaluate_position",
            f"extra_eval_hard_{i:02d}_{name}",
            {
                "fen": positions[name],
                "depth": [12, 14, 16, 18, 20, 22][i % 6],
                "verbosity": "compact",
                "detail": ["standard", "coach", "forensic"][i % 3],
            },
        )
    for i, (name, depth) in enumerate([
        ("mate1", 24), ("mate2", 24), ("fifty", 8), ("seventyfive", 8),
        ("ep", 18), ("promo", 18), ("castle", 18), ("rookend", 24),
    ]):
        add("evaluate_position", f"extra_eval_special_{i:02d}_{name}", {
            "fen": positions[name], "depth": depth, "verbosity": "compact", "detail": "forensic"
        })
    for i, depth in enumerate([-100, -1, 0, 31, 99, 999]):
        add("evaluate_position", f"extra_eval_depth_{i:02d}_{depth}", {
            "fen": positions["italian"], "depth": depth, "verbosity": "compact", "detail": "standard"
        })
    for i, bad in enumerate([
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "8/8/8/8/8/8/7k/K6k w - - 0 1",
    ]):
        add("evaluate_position", f"extra_eval_invalid_{i}", {"fen": bad, "depth": 2, "strict": True}, expect_error=True)

    assert sum(c.tool == "evaluate_position" for c in cases) == 40

    # 40 top_moves calls. Heavy MultiPV, tactical proof and explicit candidates.
    for i in range(16):
        name = hard[i % len(hard)]
        add("top_moves", f"extra_top_multipv_{i:02d}_{name}", {
            "fen": positions[name],
            "n": [3, 5, 10, 11, 20, 999][i % 6],
            "depth": [10, 12, 14, 16, 18][i % 5],
            "verbosity": "compact",
            "detail": "standard",
        }, check="n_contract" if [3, 5, 10, 11, 20, 999][i % 6] > 10 else None)
    for i in range(12):
        name = ["mate1", "promo", "ep", "castle", "rookend", "sicilian"][i % 6]
        add("top_moves", f"extra_top_proof_{i:02d}_{name}", {
            "fen": positions[name], "n": 3, "depth": [8, 12, 16, 18][i % 4],
            "verbosity": "compact", "detail": "forensic",
            "proof_mode": "tactical", "proof_defenses": [1, 3, 8][i % 3],
        })
    for i in range(8):
        name = hard[i % len(hard)]
        choices = legal_sans(positions[name], 6)
        add("top_moves", f"extra_top_include_{i:02d}_{name}", {
            "fen": positions[name], "n": 2, "depth": 12, "verbosity": "compact",
            "detail": "forensic", "include_moves": choices[: min(4, len(choices))],
        })
    add("top_moves", "extra_top_invalid_9include", {
        "fen": "startpos", "include_moves": ["e4"] * 9, "depth": 2
    }, expect_error=True)
    add("top_moves", "extra_top_invalid_move", {
        "fen": "startpos", "include_moves": ["e5"], "depth": 2
    }, expect_error=True)
    add("top_moves", "extra_top_invalid_strict", {
        "fen": "startpos", "moves": ["Nf3??"], "strict": True, "depth": 2
    }, expect_error=True)
    add("top_moves", "extra_top_terminal_75", {
        "fen": positions["seventyfive"], "n": 20, "depth": 2, "verbosity": "compact"
    })

    assert sum(c.tool == "top_moves" for c in cases) == 40

    # 40 classify_move calls. Mate invariants, rule actions and candidate comparisons.
    for i in range(12):
        fen, move = (positions["mate1"], "Qg7#") if i % 2 == 0 else (positions["mate2"], "Qxf7#")
        add("classify_move", f"extra_class_exact_mate_{i:02d}", {
            "fen": fen, "move": move, "depth": [4, 8, 12, 16, 20, 24][i % 6],
            "detail": ["standard", "coach", "forensic"][i % 3],
        }, check="mate_equivalence")
    for i in range(8):
        if i % 2 == 0:
            args = {"fen": positions["fifty"], "action_type": "claim_draw", "depth": 8, "detail": "forensic"}
        else:
            args = {"fen": positions["fifty"], "move": "Kg8", "action_type": "play_move", "depth": 8, "detail": "forensic"}
        add("classify_move", f"extra_class_rule_{i:02d}", args)
    for i in range(12):
        name = hard[i % len(hard)]
        choices = legal_sans(positions[name], 7)
        add("classify_move", f"extra_class_compare_{i:02d}_{name}", {
            "fen": positions[name], "move": choices[0], "depth": [8, 12, 16][i % 3],
            "detail": "forensic", "compare_moves": choices[1:5],
        })
    invalid_class = [
        {"fen": "startpos", "move": "e5", "depth": 2},
        {"fen": "startpos", "move": "Nf3??", "strict": True, "depth": 2},
        {"fen": "startpos", "move": "Nf3+", "strict": True, "depth": 2},
        {"fen": "startpos", "action_type": "claim_draw", "depth": 2},
        {"fen": positions["seventyfive"], "move": "Kg8", "depth": 2},
        {"fen": "startpos", "move": "e4", "compare_moves": ["d4"] * 9, "depth": 2},
        {"fen": "not fen", "move": "e4", "depth": 2},
        {"fen": "", "move": "e4", "depth": 2},
    ]
    for i, args in enumerate(invalid_class):
        add("classify_move", f"extra_class_invalid_{i:02d}", args, expect_error=True)

    assert sum(c.tool == "classify_move" for c in cases) == 40

    # 40 analyze_game calls. Rich classic games plus long 200-240 ply inputs.
    opera = """[Event "Opera Game"]
[White "Paul Morphy"]
[Black "Duke Karl / Count Isouard"]
[Result "1-0"]

1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6
7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7
12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6 15. Bxd7+ Nxd7
16. Qb8+ Nxb8 17. Rd8# 1-0"""
    legal_mate = """[Event "Legal Mate"]
[Result "1-0"]

1. e4 e5 2. Nf3 d6 3. Bc4 Bg4 4. Nc3 g6 5. Nxe5 Bxd1 6. Bxf7+ Ke7 7. Nd5# 1-0"""
    annotated = """[Event "Annotated"]
[Result "*"]

1. e4 {central space} e5 2. Nf3?! Nc6 3. Bb5 a6 *"""
    candidate = """[Event "Candidate"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 0-1"""

    for i in range(16):
        pgn = [opera, legal_mate, annotated, candidate][i % 4]
        add("analyze_game", f"extra_analyze_rich_{i:02d}", {
            "pgn": pgn, "depth": [2, 3, 6, 10, 14, 18, 20, 22][i % 8],
            "strict": False, "detail": "forensic",
            "perspective": "white" if i % 2 == 0 else "black",
            "max_critical_moments": [1, 3, 6, 7][i % 4],
        })

    long200 = generate_long_pgn(200, 9200)
    long240 = generate_long_pgn(240, 9240)
    for i in range(16):
        pgn = long200 if i % 2 == 0 else long240
        add("analyze_game", f"extra_analyze_long_{i:02d}_{200 if i % 2 == 0 else 240}", {
            "pgn": pgn, "depth": [1, 2, 3, 4][i % 4], "strict": False,
            "detail": "standard" if i % 4 else "coach",
            "perspective": "white" if i % 2 == 0 else "black",
            "max_critical_moments": [1, 3, 6][i % 3],
        })

    invalid_pgns = [
        "garbage",
        "1. e4 e5 2. BADMOVE",
        "[Event \"broken\"\n1. e4 e5",
        "[Result \"*\"]\n1. e2e4 e5 *",
        "[Result \"*\"]\n1. 0-0 *",
        "[Result \"*\"]\n1. e4 e5 2. Nf3+ *",
        "[Result \"*\"]\n1. e4 e5\n\n[Result \"*\"]\n1. d4 d5 *",
        "12345",
    ]
    for i, pgn in enumerate(invalid_pgns):
        add("analyze_game", f"extra_analyze_invalid_{i:02d}", {
            "pgn": pgn, "depth": 1, "strict": True, "detail": "standard",
            "perspective": "white", "max_critical_moments": 3,
        }, expect_error=True)

    assert sum(c.tool == "analyze_game" for c in cases) == 40
    assert len(cases) == 160
    return cases


def schema_hash(tools: Any) -> str:
    rows = []
    for tool in tools.tools:
        rows.append({"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema})
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_summary(records: list[dict[str, Any]], schema: dict[str, Any], path: Path, build_sha: str | None) -> None:
    times = [float(r["elapsed_ms"]) for r in records]
    sizes = [int(r["response_bytes"]) for r in records]
    flagged = [r for r in records if not r.get("semantic_ok", False)]
    transport = [r for r in records if not r.get("transport_ok", False)]
    unexpected = [r for r in records if r.get("tool_error") and not r.get("expected_error")]
    by_tool: dict[str, int] = {}
    for r in records:
        by_tool[r["tool"]] = by_tool.get(r["tool"], 0) + 1

    lines = [
        "# Chess MCP additional ultra-hard stress 160",
        "",
        f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Exact direct session.call_tool attempts: **{len(records)}**",
        f"- Build SHA: `{build_sha}`",
        f"- Live tools/list schema SHA-256: `{schema['fingerprint']}`",
        f"- Calls by tool: `{by_tool}`",
        f"- Transport failures: **{len(transport)}**",
        f"- Unexpected structured tool errors: **{len(unexpected)}**",
        f"- Flagged records: **{len(flagged)}**",
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
        f"- p95: {percentile([float(x) for x in sizes], .95):.0f} bytes",
        f"- p99: {percentile([float(x) for x in sizes], .99):.0f} bytes",
        f"- max: {max(sizes) if sizes else 0} bytes",
        "",
        "## Flagged records",
        "",
    ]
    for r in flagged:
        lines.append(f"- #{r['sequence']} `{r['tool']}` `{r['case_id']}`: {'; '.join(r.get('notes', []))}")
    lines += ["", "## 20 slowest", ""]
    for r in sorted(records, key=lambda x: x["elapsed_ms"], reverse=True)[:20]:
        lines.append(f"- #{r['sequence']} `{r['tool']}` `{r['case_id']}`: {r['elapsed_ms']} ms")
    lines += ["", "## 20 largest", ""]
    for r in sorted(records, key=lambda x: x["response_bytes"], reverse=True)[:20]:
        lines.append(f"- #{r['sequence']} `{r['tool']}` `{r['case_id']}`: {r['response_bytes']} bytes")
    lines += ["", "## Live schema", "", "```json", json.dumps(schema["tools"], ensure_ascii=False, indent=2), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    cases = build_cases()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl = out_dir / f"chess_mcp_additional_160_{stamp}.jsonl"
    md = out_dir / f"chess_mcp_additional_160_{stamp}.md"
    schema_path = out_dir / f"chess_mcp_live_schema_{stamp}.json"

    sem = asyncio.Semaphore(1)
    records: list[dict[str, Any]] = []
    build_sha: str | None = None
    mcp_url = args.target.rstrip("/") + "/mcp"

    async with streamable_http_client(mcp_url) as streams:
        read_stream, write_stream, *_ = streams
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            rows = [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in tools.tools]
            schema = {"fingerprint": schema_hash(tools), "tools": rows}
            schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

            for case in cases:
                rec, sha = await execute_case(session, case, sem, build_sha)
                records.append(rec)
                if build_sha is None and sha:
                    build_sha = sha
                print(f"{case.seq:03d}/160 {case.tool} {case.case_id} {rec['elapsed_ms']}ms bytes={rec['response_bytes']} notes={rec.get('notes')}", flush=True)

    with jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    write_summary(records, schema, md, build_sha)

    print(f"CALLS={len(records)}")
    print(f"BUILD_SHA={build_sha}")
    print(f"JSONL={jsonl}")
    print(f"MARKDOWN={md}")
    print(f"SCHEMA={schema_path}")
    print(f"TRANSPORT_FAILURES={sum(not r.get('transport_ok', False) for r in records)}")
    print(f"UNEXPECTED_TOOL_ERRORS={sum(r.get('tool_error') and not r.get('expected_error') for r in records)}")
    return 0 if len(records) == 160 else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="https://mcp.trychessy.com")
    parser.add_argument("--output-dir", default="artifacts")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
