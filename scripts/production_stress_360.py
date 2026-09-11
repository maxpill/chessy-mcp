"""Direct production stress harness for Chess MCP (backward-compatibility wrapper).

Delegates to scripts.audit and scripts.chess_mcp_stress.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit.case_generation import build_600_case_specs  # noqa: E402
from scripts.audit.case_spec import CaseSpec  # noqa: E402
from scripts.chess_mcp_stress import main_async  # noqa: E402


def _build_case_specs() -> list[CaseSpec]:
    """Return 360 case specs matching the historical distribution."""
    all_specs = build_600_case_specs()
    ev = [s for s in all_specs if s.tool == "evaluate_position"][:90]
    top = [s for s in all_specs if s.tool == "top_moves"][:90]
    cls = [s for s in all_specs if s.tool == "classify_move"][:100]
    ana = [s for s in all_specs if s.tool == "analyze_game"][:80]
    return ev + top + cls + ana


def main() -> int:
    parser = argparse.ArgumentParser(description="Chess MCP 360-call production stress wrapper")
    parser.add_argument("--target", default="https://mcp.trychessy.com")
    parser.add_argument("--expected-sha", default="", help="Expected deployed git SHA")
    parser.add_argument("--jsonl-out", default="artifacts/chess_mcp_stress_360.jsonl")
    parser.add_argument("--md-out", default="artifacts/chess_mcp_stress_360.md")
    parser.add_argument("--validate-cases-only", action="store_true")
    args = parser.parse_args()

    # Pass through to unified runner
    runner_args = argparse.Namespace(
        target=args.target,
        expected_sha=args.expected_sha,
        calls=360,
        profile="production_360",
        transport="http",
        validate_cases_only=args.validate_cases_only,
        strict_certification=True,
        jsonl_out=args.jsonl_out,
        md_out=args.md_out,
    )
    return asyncio.run(main_async(runner_args))


if __name__ == "__main__":
    sys.exit(main())
