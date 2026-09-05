"""MCP request cost estimator (audit P1 admission control).

Extracted from :mod:`mcp_server.middleware.request_logger`. Owns
:func:`estimate_mcp_request_cost`, a per-tool approximation of the CPU
admission cost given the request body. Used by
:class:`ASGIRequestLoggerMiddleware` to charge the rate limiter fairly per tool.
"""

from __future__ import annotations

import json
from typing import Any, cast


def estimate_mcp_request_cost(body: bytes) -> float:
    """Approximate CPU admission cost from tool, depth, MultiPV and PGN size."""
    try:
        payload_any: Any = json.loads(body.decode("utf-8"))
        payload = cast(dict[str, Any], payload_any) if isinstance(payload_any, dict) else {}
        params_any: Any = payload.get("params")
        params = cast(dict[str, Any], params_any) if isinstance(params_any, dict) else {}
        tool_name = str(params.get("name") or params.get("tool") or "")
        args_any: Any = params.get("arguments")
        args = cast(dict[str, Any], args_any) if isinstance(args_any, dict) else {}
        depth = max(1, min(int(args.get("depth", 18)), 30))
        if tool_name == "evaluate_position":
            return 1.0 + depth / 14.0
        if tool_name == "top_moves":
            n = max(1, min(int(args.get("n", 3)), 20))
            base = 1.0 + (depth * n) / 14.0
            detail = str(args.get("detail", "standard"))
            include_any = args.get("include_moves")
            include_moves = include_any if isinstance(include_any, list) else []
            proof_mode = str(args.get("proof_mode", "none"))
            if detail == "coach" and not include_moves and proof_mode == "none":
                return base + 0.5
            extra_searches = min(len(include_moves), 8)
            if detail == "forensic":
                # Forensic top-moves evaluates each returned root candidate's
                # resulting position in addition to explicit alternatives.
                extra_searches += min(n, 8)
            if proof_mode == "tactical":
                proof_defenses = max(1, min(int(args.get("proof_defenses", 3)), 8))
                # One opponent MultiPV search in sampled mode plus one post-state
                # evaluation per sampled defense. Exhaustive mode is bounded at
                # eight replies, so this charge is deliberately conservative.
                extra_searches += proof_defenses + 1
            return base + (depth * extra_searches) / 14.0
        if tool_name == "classify_move":
            base = 2.0 + depth / 10.0
            detail = str(args.get("detail", "standard"))
            compare_any = args.get("compare_moves")
            compare_moves = compare_any if isinstance(compare_any, list) else []
            if detail == "coach" and not compare_moves:
                return base + 0.5
            if detail == "forensic" or compare_moves:
                # Forensic mode may perform one verification search after the
                # strongest reply plus one post-position search for the played
                # move, engine-best move and each explicit comparison.
                explicit = min(len(compare_moves), 8)
                candidate_searches = min(2 + explicit, 8)
                return base + (depth * (1 + candidate_searches)) / 14.0
            return base
        if tool_name == "analyze_game":
            pgn = str(args.get("pgn", ""))
            estimated_plies = max(1.0, min(200.0, len(pgn) / 24.0))
            return 5.0 + (depth * estimated_plies) / 28.0
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    return 1.0


# Back-compat shim.
_estimate_mcp_request_cost = estimate_mcp_request_cost
