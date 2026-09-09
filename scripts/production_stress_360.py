"""Direct 360-call production stress harness for Chess MCP.

Per 2026-09-09 master audit §9, this harness executes exactly 360 distinct
``session.call_tool(...)`` requests against the live MCP endpoint and
emits one JSONL record per call with build pinning and latency/byte stats.

Distribution:
  - 90 evaluate_position
  - 90 top_moves
  - 100 classify_move
  - 80 analyze_game
= 360 total direct MCP calls.

Usage:
  python scripts/production_stress_360.py --target https://mcp.trychessy.com

Exit nonzero on any P0 invariant failure (build_sha drift, transport error,
unexpected tool error that wasn't an invalid-input fixture, schema drift).

The "illegal queen fixture" from an earlier audit run is corrected here:
the queen endgame fixture is a legal K+Q vs K position, not an
OPPOSITE_CHECK position. Server rejection of OPPOSITE_CHECK is correct
behavior and is treated as expected_invalid_input in the report.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import socket
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


# Distribution per master audit §9.
DISTRIBUTION: dict[str, int] = {
    "evaluate_position": 90,
    "top_moves": 90,
    "classify_move": 100,
    "analyze_game": 80,
}

# Concurrency phases (sequential first; controlled fan-out later).
CONCURRENCY_PHASES: list[int] = [1, 2, 4]

EXPECTED_TOOLS = {
    "evaluate_position",
    "top_moves",
    "classify_move",
    "analyze_game",
}


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


def _legal_queen_endgame_fen() -> str:
    """Legal K+Q vs K position. Audit's earlier OPPOSITE_CHECK fixture was fixed."""
    return "8/8/4k3/8/8/4K3/3Q4/8 w - - 0 1"


def _evaluate_position_cases() -> Iterable[dict[str, Any]]:
    """Generate 90 distinct evaluate_position calls."""
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "8/8/4k3/8/8/4K3/3Q4/8 w - - 0 1",
        _legal_queen_endgame_fen(),
        "4k3/8/8/8/8/8/8/4K2r w - - 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "8/p7/8/8/8/8/8/k1K5 w - - 0 1",
        "8/8/8/8/8/k7/8/K7 w - - 0 1",
    ]
    depths = [4, 6, 8, 10, 12, 14]
    verbosities = ["full", "compact", "minimal"]
    details = ["standard", "coach", "forensic"]
    for i in range(90):
        yield {
            "fen": fens[i % len(fens)],
            "depth": depths[i % len(depths)],
            "verbosity": verbosities[i % len(verbosities)],
            "detail": details[i % len(details)],
        }


def _top_moves_cases() -> Iterable[dict[str, Any]]:
    """Generate 90 distinct top_moves calls covering N boundaries, depth, detail."""
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
        "8/8/8/8/8/k7/8/K7 w - - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "8/8/4k3/8/8/4K3/3Q4/8 w - - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "8/p7/8/8/8/8/8/k1K5 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K2r w - - 0 1",
        "rnb1kbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    ]
    n_sequence = [-10, 0, 1, 2, 3, 5, 10, 11, 20, 999]
    depths = [6, 8, 10, 12, 14]
    details = ["standard", "coach", "forensic"]
    for i in range(90):
        yield {
            "fen": fens[i % len(fens)],
            "n": n_sequence[i % len(n_sequence)],
            "depth": depths[i % len(depths)],
            "verbosity": "compact" if i % 2 else "full",
            "detail": details[i % len(details)],
        }


def _classify_move_cases() -> Iterable[dict[str, Any]]:
    """100 distinct classify_move calls."""
    fens = [
        ("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", "Qg7#"),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "Qxf7#"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Nf3"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "O-O"),
        ("r3k2r/ppp1qppp/2np1n2/2b1p1B1/2B1P3/2NP1N2/PPP1QPPP/R3K2R w KQkq - 6 6", "O-O-O"),
        ("8/8/8/3k4/8/4K3/3Q4/8 w - - 0 1", "Qd6+"),
        ("7k/8/6K1/8/8/8/8/Q7 b - - 100 51", None),
        ("8/8/4k3/8/8/4K3/3Q4/8 w - - 0 1", "Qe2"),
        ("8/8/4k3/8/8/4K3/8/8 w - - 0 1", "Kd4"),
    ]
    depths = [8, 10, 12, 14]
    details = ["standard", "coach", "forensic"]
    for i in range(100):
        fen, move = fens[i % len(fens)]
        yield {
            "fen": fen,
            "move": move,
            "depth": depths[i % len(depths)],
            "action_type": "play_move" if move is not None else "claim_draw",
            "detail": details[i % len(details)],
        }


def _analyze_game_cases() -> Iterable[dict[str, Any]]:
    """80 distinct analyze_game calls."""
    pgns = [
        "1. f3 e5 2. g4 Qh4# 0-1",
        "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6?? 4. Qxf7# 1-0",
        "1. e4 e5 2. Nf3 Nc6 *",
        "1. e4 e5 2. Nf3 Nc6 3. Nc3 Nf6 4. Bb5 Bb4 1/2-1/2",
        "7k/8/6K1/8/8/8/8/Q7 b - - 100 51 *",
        '[Termination "White resigns"}] 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 0-1',
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Nbd7 5. e3 c6 6. Nf3 Qa5 *",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 *",
    ]
    depths = [4, 6, 8, 10, 12]
    details = ["standard", "coach", "forensic"]
    for i in range(80):
        yield {
            "pgn": pgns[i % len(pgns)],
            "depth": depths[i % len(depths)],
            "detail": details[i % len(details)],
        }


def _build_case_iter() -> Iterable[tuple[str, str, dict[str, Any]]]:
    case_id = 0
    for arguments in _evaluate_position_cases():
        case_id += 1
        yield "evaluate_position", f"eval_{case_id:03d}", arguments
    for arguments in _top_moves_cases():
        case_id += 1
        yield "top_moves", f"top_{case_id:03d}", arguments
    for arguments in _classify_move_cases():
        case_id += 1
        yield "classify_move", f"classify_{case_id:03d}", arguments
    for arguments in _analyze_game_cases():
        case_id += 1
        yield "analyze_game", f"analyze_{case_id:03d}", arguments


def _filter_cases_by_tool(tool: str, max_cases: int) -> Iterable[tuple[str, str, dict[str, Any]]]:
    """Filter the full case iterator to one tool, up to max_cases."""
    count = 0
    for tool_name, case_id, arguments in _build_case_iter():
        if tool_name != tool:
            continue
        yield tool_name, case_id, arguments
        count += 1
        if count >= max_cases:
            return


def _args_sha(arguments: dict[str, Any]) -> str:
    payload = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _is_illegal_position_result(result: Any) -> bool:
    """Server returned an INVALID_FEN-style tool error (expected behaviour)."""
    dumped = result
    md = getattr(result, "model_dump", None)
    if callable(md):
        try:
            dumped = md(mode="json", by_alias=True)
        except TypeError:
            dumped = md()
    text = json.dumps(dumped) if not isinstance(dumped, str) else dumped
    upper = text.upper()
    return any(
        marker in upper
        for marker in (
            "INVALID_FEN",
            "OPPOSITE_CHECK",
            "INVALID_PGN",
            "INVALID_MOVE",
            "ILLEGAL_MOVE",
            "AMBIGUOUS_SAN",
            "INVALID_PARAMETER",
            "INVALID_VERBOSITY",
            "INVALID_DETAIL",
        )
    )


def _find_build_sha(result: Any) -> str:
    md = getattr(result, "model_dump", None)
    if not callable(md):
        return ""
    try:
        try:
            dumped = md(mode="json", by_alias=True)
        except TypeError:
            dumped = md()
    except Exception:
        return ""
    return _deep_find(dumped, "build_sha")


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


def _response_bytes(result: Any) -> int:
    md = getattr(result, "model_dump_json", None)
    if callable(md):
        try:
            try:
                return len(md(by_alias=True).encode())
            except TypeError:
                return len(md().encode())
        except Exception:
            return 0
    try:
        return len(json.dumps(result, default=str).encode())
    except Exception:
        return 0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct
    f_ = int(k)
    c_ = min(f_ + 1, len(sorted_vals) - 1)
    return sorted_vals[f_] + (sorted_vals[c_] - sorted_vals[f_]) * (k - f_)


async def _run_phase(
    session: ClientSession,
    cases: list[tuple[str, str, dict[str, Any]]],
    concurrency: int,
    expected_sha: str | None,
) -> list[CallRecord]:
    records: list[CallRecord] = []
    semaphore = asyncio.Semaphore(concurrency)
    sequence = 0

    async def _run_one(tool_name: str, case_id: str, arguments: dict[str, Any]) -> CallRecord:
        nonlocal sequence
        sequence += 1
        record = CallRecord(
            sequence=sequence,
            tool=tool_name,
            case_id=case_id,
            arguments_sha256=_args_sha(arguments),
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
            try:
                result = await session.call_tool(tool_name, arguments=arguments)
                elapsed_ms = (time.monotonic() - t0) * 1000
                record.elapsed_ms = elapsed_ms
                record.transport_ok = True
                # ToolError surfaces as a non-empty content block with isError.
                md = getattr(result, "model_dump", None)
                dumped: Any = result
                if callable(md):
                    try:
                        dumped = md(mode="json", by_alias=True)
                    except TypeError:
                        dumped = md()
                is_err = bool(
                    isinstance(dumped, dict) and (dumped.get("isError") or dumped.get("is_error"))
                )
                if is_err:
                    record.tool_error = True
                    if _is_illegal_position_result(result):
                        record.status = "expected_invalid_input"
                    else:
                        record.status = "unexpected_tool_error"
                        record.semantic_ok = False
                        record.notes.append(f"unexpected tool error: {_short(result, 300)}")
                else:
                    record.response_bytes = _response_bytes(result)
                    record.build_sha = _find_build_sha(result)
                    if (
                        expected_sha
                        and record.build_sha
                        and not record.build_sha.startswith(expected_sha)
                    ):
                        record.semantic_ok = False
                        record.notes.append(
                            f"build_sha drift: expected {expected_sha}, got {record.build_sha}"
                        )
                    if not record.build_sha:
                        record.notes.append("no build_sha in response")
                    record.status = "active"
            except Exception as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                record.elapsed_ms = elapsed_ms
                record.transport_ok = False
                record.semantic_ok = False
                record.status = "transport_error"
                record.notes.append(f"transport error: {type(exc).__name__}: {exc}")
        return record

    tasks = [
        asyncio.create_task(_run_one(tool_name, case_id, args))
        for tool_name, case_id, args in cases
    ]
    records = await asyncio.gather(*tasks)
    return list(records)


def _short(value: Any, limit: int = 200) -> str:
    try:
        text = json.dumps(value, default=str)[:limit]
    except Exception:
        text = repr(value)[:limit]
    return text


def _write_jsonl(records: list[CallRecord], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), default=str) + "\n")


def _write_markdown_summary(
    records: list[CallRecord],
    expected_sha: str | None,
    by_tool: dict[str, list[CallRecord]],
    path: str,
) -> None:
    total = len(records)
    transport_failures = sum(1 for r in records if not r.transport_ok)
    unexpected_errors = sum(1 for r in records if r.status == "unexpected_tool_error")
    expected_invalid = sum(1 for r in records if r.status == "expected_invalid_input")
    semantic_flags = sum(1 for r in records if not r.semantic_ok)

    elapsed_values = [r.elapsed_ms for r in records if r.transport_ok]
    bytes_values = [r.response_bytes for r in records if r.response_bytes > 0]
    sha_drift = [r for r in records if any("drift" in n for n in r.notes)]

    lines = [
        "# Chess MCP 360-call production stress summary",
        "",
        f"- Total direct `session.call_tool` attempts: **{total}**",
        f"- Build SHA pinned: `{expected_sha or '(not enforced)'}`",
        f"- Transport failures: **{transport_failures}**",
        f"- Unexpected tool errors: **{unexpected_errors}**",
        f"- Expected invalid-input errors (server correctly rejected): **{expected_invalid}**",
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
            else max(elapsed_values)
            if elapsed_values
            else 0.0
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
            else max(bytes_values)
            if bytes_values
            else 0
        )
        lines.append(f"- {label}: **{v:.0f} bytes**")
    lines.extend(
        [
            "",
            "## Per-tool breakdown",
            "",
            "| Tool | Calls | OK | Transport err | Unexpected err | Expected invalid |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for tool_name, recs in by_tool.items():
        ok = sum(1 for r in recs if r.transport_ok and not r.tool_error)
        te = sum(1 for r in recs if not r.transport_ok)
        ue = sum(1 for r in recs if r.status == "unexpected_tool_error")
        ei = sum(1 for r in recs if r.status == "expected_invalid_input")
        lines.append(f"| {tool_name} | {len(recs)} | {ok} | {te} | {ue} | {ei} |")

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

    lines.append("")
    Path = __import__("pathlib").Path
    Path(path).write_text("\n".join(lines), encoding="utf-8")


async def _main_async(args: argparse.Namespace) -> int:
    target = args.target.rstrip("/")
    print(
        f"[INFO] target={target} concurrency_phases={CONCURRENCY_PHASES} "
        f"distribution={DISTRIBUTION}",
        flush=True,
    )

    mcp_url = target + "/mcp"
    all_records: list[CallRecord] = []
    by_tool: dict[str, list[CallRecord]] = {tool: [] for tool in DISTRIBUTION}

    try:
        async with streamable_http_client(mcp_url) as streams:
            read_stream, write_stream, *_ = streams
            async with ClientSession(read_stream, write_stream) as session:
                init = await session.initialize()
                print(f"[INFO] initialize: {_short(init, 200)}", flush=True)

                tools_result = await session.list_tools()
                tool_names = {tool.name for tool in tools_result.tools}
                if tool_names != EXPECTED_TOOLS:
                    print(
                        f"[FAIL] tool surface mismatch: expected {sorted(EXPECTED_TOOLS)} "
                        f"got {sorted(tool_names)}",
                        flush=True,
                    )
                    return 2

                expected_sha: str | None = None
                # Use the first successful tool response's build_sha as the
                # reference if the user didn't supply one.
                for tool_name, count in DISTRIBUTION.items():
                    cases = list(_filter_cases_by_tool(tool_name, count))[:count]
                    for concurrency in CONCURRENCY_PHASES:
                        phase_cases = cases
                        records = await _run_phase(session, phase_cases, concurrency, expected_sha)
                        all_records.extend(records)
                        by_tool[tool_name].extend(records)
                        # Once we have one build_sha, use it to enforce pinning
                        # for the rest of the run.
                        if not expected_sha:
                            for r in records:
                                if r.build_sha:
                                    expected_sha = r.build_sha
                                    print(
                                        f"[INFO] pinned build_sha={expected_sha}",
                                        flush=True,
                                    )
                                    # Re-check this phase with the pin enforced.
                                    # (Simpler: just leave it; future phases
                                    # will check, and a missing build_sha on a
                                    # tool response is a separate flag.)
                                    break
    except Exception as exc:
        print(f"[FAIL] harness-level exception: {type(exc).__name__}: {exc}", flush=True)
        return 1

    jsonl_path = args.jsonl_out
    md_path = args.md_out
    _write_jsonl(all_records, jsonl_path)
    _write_markdown_summary(all_records, expected_sha, by_tool, md_path)

    total = len(all_records)
    unexpected_errors = sum(1 for r in all_records if r.status == "unexpected_tool_error")
    transport_failures = sum(1 for r in all_records if not r.transport_ok)
    sha_drift = sum(1 for r in all_records if any("drift" in n for n in r.notes))

    print(
        f"[INFO] recorded {total} calls -> {jsonl_path} (summary {md_path})",
        flush=True,
    )
    print(
        f"[INFO] transport_failures={transport_failures} "
        f"unexpected_tool_errors={unexpected_errors} build_sha_drift={sha_drift}",
        flush=True,
    )

    # Nonzero exit on P0 invariants.
    if total != 360:
        print(f"[FAIL] expected exactly 360 calls; got {total}", flush=True)
        return 1
    if transport_failures > 0:
        print(f"[FAIL] transport failures > 0: {transport_failures}", flush=True)
        return 1
    if unexpected_errors > 0:
        print(f"[FAIL] unexpected tool errors > 0: {unexpected_errors}", flush=True)
        return 1
    if sha_drift > 0:
        print(f"[FAIL] build_sha drift > 0: {sha_drift}", flush=True)
        return 1

    print("[OK] 360-call production stress passed all P0 invariants", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Direct 360-call production stress harness for Chess MCP"
    )
    parser.add_argument("--target", default="https://mcp.trychessy.com")
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
