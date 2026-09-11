"""Local in-process stress runner for Chess MCP.

Executes case specifications directly against server_module in-process,
validating responses against the exact expected error codes and chess semantic oracles.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp_server import server as server_module  # noqa: E402
from scripts.audit.case_generation import build_600_case_specs  # noqa: E402
from scripts.audit.case_spec import CaseSpec  # noqa: E402
from scripts.audit.response_normalization import normalize_call_tool_result  # noqa: E402
from scripts.audit.semantic_oracles import run_semantic_oracle, validate_expected_error  # noqa: E402


async def run_local_stress(limit: int | None = None) -> int:
    print("[INFO] Initializing in-process analyzer pool...")
    pool = await server_module._get_analyzer_pool()
    print(f"[INFO] Pool ready: {pool.name}")

    tool_dispatch = {
        "evaluate_position": server_module.evaluate_position,
        "top_moves": server_module.top_moves,
        "classify_move": server_module.classify_move,
        "analyze_game": server_module.analyze_game,
    }

    all_specs = build_600_case_specs()
    cases: list[CaseSpec] = all_specs[:limit] if limit else all_specs
    print(f"[INFO] Running {len(cases)} cases locally...")

    passed = 0
    failed = 0
    errors: list[tuple[str, str, dict[str, Any], str]] = []

    t_start = time.monotonic()

    try:
        for idx, spec in enumerate(cases, 1):
            func = tool_dispatch[spec.tool]
            try:
                raw_res = await func(**spec.arguments)
                norm = normalize_call_tool_result(raw_res)

                if spec.expected_kind in ("tool_error", "schema_error"):
                    # Success when error expected
                    failed += 1
                    msg = f"expected error '{spec.expected_error_code}', but call succeeded"
                    errors.append((spec.tool, spec.case_id, spec.arguments, msg))
                else:
                    sem_errs = run_semantic_oracle(spec, norm.parsed_payload)
                    if sem_errs:
                        failed += 1
                        msg = f"semantic failures: {'; '.join(sem_errs)}"
                        errors.append((spec.tool, spec.case_id, spec.arguments, msg))
                    else:
                        passed += 1

            except Exception as exc:
                norm = normalize_call_tool_result(exc)
                if spec.expected_kind in ("tool_error", "schema_error"):
                    err_match = validate_expected_error(spec, norm)
                    if err_match:
                        failed += 1
                        msg = f"wrong error: {'; '.join(err_match)}"
                        errors.append((spec.tool, spec.case_id, spec.arguments, msg))
                    else:
                        passed += 1
                else:
                    failed += 1
                    msg = f"unexpected exception: {type(exc).__name__}: {exc}"
                    errors.append((spec.tool, spec.case_id, spec.arguments, msg))

            if idx % 5 == 0 or idx == len(cases) or spec.expected_kind != "success":
                print(
                    f"[{idx:03d}/{len(cases)}] {spec.tool:<18} {spec.case_id} OK/FAIL: {passed}/{failed} ({time.monotonic() - t_start:.1f}s)",
                    flush=True,
                )

    finally:
        await server_module.close_analyzer_pool()

    total_time = time.monotonic() - t_start
    print("\n" + "=" * 60, flush=True)
    print(f"LOCAL STRESS COMPLETED in {total_time:.2f}s", flush=True)
    print(f"Total: {len(cases)}, Passed: {passed}, Failed: {failed}", flush=True)
    print("=" * 60, flush=True)

    if failed > 0:
        print("\nFailures:", flush=True)
        for tool, cid, _args, err in errors[:20]:
            print(f"- {tool} {cid}: {err}", flush=True)
        return 1

    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Local in-process stress runner")
    parser.add_argument("--limit", type=int, default=30, help="Number of cases to run (default 30)")
    args = parser.parse_args()
    return asyncio.run(run_local_stress(limit=args.limit))


if __name__ == "__main__":
    sys.exit(main())
