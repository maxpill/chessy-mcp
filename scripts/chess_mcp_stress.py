"""Unified Chess MCP Stress & Certification Harness.

Executes direct MCP tool calls across evaluate_position, top_moves, classify_move,
and analyze_game, validating schemas, semantic invariants, build identity, and
transport reliability.

Usage:
  # Validate case definitions and exit:
  python scripts/chess_mcp_stress.py --validate-cases-only

  # Run local in-process stress:
  python scripts/chess_mcp_stress.py --transport in-process --calls 600

  # Run remote production stress:
  python scripts/chess_mcp_stress.py --target https://mcp.trychessy.com --expected-sha <sha> --calls 600
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit.build_identity import validate_call_build_identity, validate_observed_build_shas  # noqa: E402
from scripts.audit.case_generation import build_600_case_specs  # noqa: E402
from scripts.audit.case_spec import CaseSpec  # noqa: E402
from scripts.audit.reporting import CallRecord, generate_markdown_summary, write_jsonl  # noqa: E402
from scripts.audit.response_normalization import normalize_call_tool_result  # noqa: E402
from scripts.audit.schema_validation import preflight_validate_cases  # noqa: E402
from scripts.audit.semantic_oracles import run_semantic_oracle, validate_expected_error  # noqa: E402

CONCURRENCY_PHASES = [1, 2, 4]


async def _execute_single_call(
    spec: CaseSpec,
    sequence: int,
    call_func: Any,
    concurrency_phase: int,
    effective_hash: str,
    pinned_sha: str,
    strict_certification: bool,
    observed_shas: set[str],
) -> CallRecord:
    call_id = f"call_{uuid4().hex[:12]}"
    t_start = time.time()
    t0 = time.monotonic()

    record = CallRecord(
        call_id=call_id,
        case_ordinal=spec.case_ordinal,
        sequence=sequence,
        tool=spec.tool,
        case_id=spec.case_id,
        arguments=spec.arguments,
        raw_argument_hash=spec.raw_argument_hash,
        effective_argument_hash=effective_hash,
        expected_kind=spec.expected_kind,
        expected_error_code=spec.expected_error_code,
        tags=spec.tags,
        depth_tier=spec.depth_tier,
        semantic_profile=spec.semantic_profile,
        concurrency_phase=concurrency_phase,
        started_at=t_start,
        elapsed_ms=0.0,
        transport_ok=False,
        tool_error=False,
        semantic_ok=True,
        build_sha="",
        wire_bytes=0,
        text_bytes=0,
        structured_bytes=0,
        status="active",
    )

    retries_left = 1
    raw_result = None
    attempt_durations: list[float] = []

    while True:
        att_start = time.monotonic()
        try:
            raw_result = await call_func(spec.tool, spec.arguments)
            elapsed = (time.monotonic() - att_start) * 1000
            attempt_durations.append(elapsed)
            record.transport_ok = True
            break
        except Exception as exc:
            elapsed = (time.monotonic() - att_start) * 1000
            attempt_durations.append(elapsed)
            err_msg = str(exc).lower()

            is_retryable = any(m in err_msg for m in ("stream ended", "closed", "connection", "reset", "timeout"))
            if retries_left > 0 and is_retryable:
                retries_left -= 1
                record.attempt_count += 1
                record.retry_reasons.append(f"{type(exc).__name__}: {exc}")
                await asyncio.sleep(0.5)
                continue

            record.transport_ok = False
            record.semantic_ok = False
            record.status = "transport_error"
            record.notes.append(f"transport error: {type(exc).__name__}: {exc}")
            record.elapsed_ms = (time.monotonic() - t0) * 1000
            record.per_attempt_elapsed_ms = attempt_durations
            return record

    record.elapsed_ms = (time.monotonic() - t0) * 1000
    record.per_attempt_elapsed_ms = attempt_durations

    if strict_certification and record.attempt_count > 1:
        record.semantic_ok = False
        record.notes.append(f"strict certification: transport retried {record.attempt_count - 1} times")

    # Normalize response
    norm = normalize_call_tool_result(raw_result)
    record.tool_error = norm.is_error
    record.wire_bytes = norm.wire_bytes
    record.text_bytes = norm.text_bytes
    record.structured_bytes = norm.structured_payload_bytes
    record.tool_error_code = norm.structured_error_code
    record.build_sha = norm.build_sha

    if norm.build_sha:
        observed_shas.add(norm.build_sha)

    # Validate build identity
    sha_errs = validate_call_build_identity(norm.build_sha, pinned_sha, is_success=not norm.is_error)
    if sha_errs:
        record.semantic_ok = False
        record.notes.extend(sha_errs)

    # Validate error or success expectations
    if spec.expected_kind in ("tool_error", "schema_error"):
        err_match = validate_expected_error(spec, norm)
        if err_match:
            record.semantic_ok = False
            record.status = "wrong_tool_error" if norm.is_error else "unexpected_success"
            record.notes.extend(err_match)
        else:
            record.status = "expected_invalid_input"
    else:
        if norm.is_error:
            record.semantic_ok = False
            record.status = "unexpected_tool_error"
            record.notes.append(f"unexpected tool error: {norm.text_content[:200]}")
        else:
            record.status = "active"
            sem_errs = run_semantic_oracle(spec, norm.parsed_payload)
            if sem_errs:
                record.semantic_ok = False
                record.semantic_failures.extend(sem_errs)
                record.notes.extend([f"semantic failure: {se}" for se in sem_errs])

    return record


async def run_stress_suite(
    cases: list[CaseSpec],
    effective_hashes: dict[str, str],
    call_func: Any,
    pinned_sha: str,
    strict_certification: bool,
    observed_shas: set[str],
) -> list[CallRecord]:
    all_records: list[CallRecord] = []
    global_seq = 0

    tools = ["evaluate_position", "top_moves", "classify_move", "analyze_game"]
    for tool_name in tools:
        tool_specs = [c for c in cases if c.tool == tool_name]
        if not tool_specs:
            continue

        num_phases = len(CONCURRENCY_PHASES)
        chunk_size = max(1, len(tool_specs) // num_phases)

        for phase_idx, concurrency in enumerate(CONCURRENCY_PHASES):
            start = phase_idx * chunk_size
            end = (phase_idx + 1) * chunk_size if phase_idx < num_phases - 1 else len(tool_specs)
            phase_specs = tool_specs[start:end]
            if not phase_specs:
                continue

            sem = asyncio.Semaphore(concurrency)

            async def _worker(
                spec: CaseSpec,
                _sem: asyncio.Semaphore = sem,
                _concurrency: int = concurrency,
            ) -> CallRecord:
                nonlocal global_seq
                global_seq += 1
                seq = global_seq
                eff_hash = effective_hashes.get(spec.case_id, spec.raw_argument_hash)
                async with _sem:
                    return await _execute_single_call(
                        spec=spec,
                        sequence=seq,
                        call_func=call_func,
                        concurrency_phase=_concurrency,
                        effective_hash=eff_hash,
                        pinned_sha=pinned_sha,
                        strict_certification=strict_certification,
                        observed_shas=observed_shas,
                    )

            tasks = [asyncio.create_task(_worker(s)) for s in phase_specs]
            records = await asyncio.gather(*tasks)
            all_records.extend(records)

    return all_records


async def main_async(args: argparse.Namespace) -> int:
    cases = build_600_case_specs()
    print(f"[INFO] Built {len(cases)} case specifications.")

    # Preflight case validation (R2-003, R2-004, R2-017)
    errors, eff_hashes = preflight_validate_cases(cases)
    if errors:
        print(f"[FAIL] Preflight case validation failed with {len(errors)} errors:")
        for err in errors[:15]:
            print(f"  - {err}")
        return 1

    print(f"[OK] Preflight validation passed: {len(eff_hashes)} unique effective argument hashes.")

    if args.validate_cases_only:
        print("[SUCCESS] Cases validated successfully. Exiting per --validate-cases-only.")
        return 0

    cli_expected_sha = (args.expected_sha or "").strip()
    pinned_sha = cli_expected_sha
    observed_shas: set[str] = set()

    # Determine transport
    if args.transport == "in-process":
        from mcp_server import server as server_module

        pool = await server_module._get_analyzer_pool()
        print(f"[INFO] Initialized in-process analyzer pool: {pool.name}")

        async def in_process_caller(tool: str, arguments: dict[str, Any]) -> Any:
            fn = getattr(server_module, tool)
            return await fn(**arguments)

        # Preflight pinning call
        preflight_res = await in_process_caller("evaluate_position", {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "depth": 1})
        norm_pre = normalize_call_tool_result(preflight_res)
        if norm_pre.build_sha:
            pinned_sha = norm_pre.build_sha
            observed_shas.add(pinned_sha)
            print(f"[INFO] Pinned in-process build SHA: {pinned_sha}")

        try:
            records = await run_stress_suite(
                cases=cases,
                effective_hashes=eff_hashes,
                call_func=in_process_caller,
                pinned_sha=pinned_sha,
                strict_certification=args.strict_certification,
                observed_shas=observed_shas,
            )
        finally:
            await server_module.close_analyzer_pool()
    else:
        # Remote HTTP transport
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        mcp_url = args.target.rstrip("/") + "/mcp"
        print(f"[INFO] Connecting to remote MCP: {mcp_url}")

        async with streamable_http_client(mcp_url) as streams:
            read_stream, write_stream, *_ = streams
            async with ClientSession(read_stream, write_stream) as session:
                init_res = await session.initialize()
                print(f"[INFO] Remote session initialized: {init_res}")

                # Preflight call to pin SHA before concurrency
                pre_res = await session.call_tool("evaluate_position", arguments={"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "depth": 1})
                norm_pre = normalize_call_tool_result(pre_res)
                if not norm_pre.build_sha:
                    print("[FAIL] Preflight call failed to return build_sha! Cannot certify.")
                    return 1

                pinned_sha = norm_pre.build_sha
                observed_shas.add(pinned_sha)
                print(f"[INFO] Pinned remote build SHA: {pinned_sha}")
                if cli_expected_sha and not pinned_sha.startswith(cli_expected_sha):
                    print(f"[FAIL] Pinned SHA '{pinned_sha}' does not match expected '{cli_expected_sha}'!")
                    return 1

                async def http_caller(tool: str, arguments: dict[str, Any]) -> Any:
                    return await session.call_tool(tool, arguments=arguments)

                records = await run_stress_suite(
                    cases=cases,
                    effective_hashes=eff_hashes,
                    call_func=http_caller,
                    pinned_sha=pinned_sha,
                    strict_certification=args.strict_certification,
                    observed_shas=observed_shas,
                )

    # Serialize artifacts
    write_jsonl(records, args.jsonl_out)
    generate_markdown_summary(records, pinned_sha, observed_shas, args.md_out)
    print(f"[INFO] Artifacts written to {args.jsonl_out} and {args.md_out}")

    # Validate final build SHAs
    final_sha_errs = validate_observed_build_shas(observed_shas, pinned_sha)
    if final_sha_errs:
        for fse in final_sha_errs:
            print(f"[FAIL] {fse}")
        return 1

    # Check for failures
    failures = [r for r in records if not r.semantic_ok or not r.transport_ok]
    if failures:
        print(f"[FAIL] Stress run finished with {len(failures)} failed calls out of {len(records)}.")
        return 1

    print(f"[SUCCESS] All {len(records)} stress calls passed successfully with 0 failures!")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Chess MCP Ultra Stress & Certification Harness")
    parser.add_argument("--target", default="https://mcp.trychessy.com", help="Target MCP server URL")
    parser.add_argument("--expected-sha", default="", help="Expected deployed git SHA")
    parser.add_argument("--calls", type=int, default=600, help="Total calls to execute")
    parser.add_argument("--profile", default="ultra", help="Stress profile")
    parser.add_argument("--transport", choices=["http", "in-process"], default="http", help="Transport mode")
    parser.add_argument("--validate-cases-only", action="store_true", help="Validate case definitions and exit")
    parser.add_argument("--strict-certification", action="store_true", default=True, help="Enforce strict certification")
    parser.add_argument("--jsonl-out", default="artifacts/chess_mcp_ultra_600.jsonl", help="Output path for JSONL records")
    parser.add_argument("--md-out", default="artifacts/chess_mcp_ultra_600.md", help="Output path for Markdown summary")

    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
