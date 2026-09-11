"""Audit reporting, JSONL serialization, and coverage distribution diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CallRecord:
    call_id: str
    case_ordinal: int
    sequence: int
    tool: str
    case_id: str
    arguments: dict[str, Any]
    raw_argument_hash: str
    effective_argument_hash: str
    expected_kind: str
    expected_error_code: str | None
    started_at: float
    elapsed_ms: float
    transport_ok: bool
    tool_error: bool
    semantic_ok: bool
    build_sha: str
    wire_bytes: int
    text_bytes: int
    structured_bytes: int
    status: str
    tags: list[str] = field(default_factory=list)
    depth_tier: str = "low"
    semantic_profile: str | None = None
    concurrency_phase: int = 1
    attempt_count: int = 1
    retry_reasons: list[str] = field(default_factory=list)
    per_attempt_elapsed_ms: list[float] = field(default_factory=list)
    tool_error_code: str | None = None
    semantic_failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def arguments_sha256(self) -> str:
        return self.raw_argument_hash

    @property
    def response_bytes(self) -> int:
        return self.wire_bytes


def write_jsonl(records: list[CallRecord], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), default=str) + "\n")


def percentile(values: Sequence[float | int], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct
    f_ = int(k)
    c_ = min(f_ + 1, len(sorted_vals) - 1)
    return float(sorted_vals[f_] + (sorted_vals[c_] - sorted_vals[f_]) * (k - f_))


def generate_markdown_summary(
    records: list[CallRecord],
    pinned_sha: str,
    observed_shas: set[str],
    path: Path | str,
) -> None:
    total = len(records)
    transport_failures = sum(1 for r in records if not r.transport_ok)
    unexpected_errors = sum(1 for r in records if r.status == "unexpected_tool_error")
    wrong_tool_errors = sum(1 for r in records if r.status == "wrong_tool_error")
    unexpected_successes = sum(1 for r in records if r.status == "unexpected_success")
    expected_invalid = sum(1 for r in records if r.status == "expected_invalid_input")
    semantic_flags = sum(1 for r in records if not r.semantic_ok and r.status != "expected_invalid_input")
    retried_calls = sum(1 for r in records if r.attempt_count > 1)
    sha_drift = sum(1 for r in records if any("drift" in n for n in r.notes))

    unique_raw = len({r.raw_argument_hash for r in records})
    unique_eff = len({r.effective_argument_hash for r in records})

    elapsed_values = [r.elapsed_ms for r in records if r.transport_ok]
    bytes_values = [r.wire_bytes for r in records if r.wire_bytes > 0]

    lines = [
        "# Chess MCP Ultra Audit Summary",
        "",
        f"- Total calls: **{total}**",
        f"- Unique raw argument hashes: **{unique_raw}**",
        f"- Unique effective argument hashes: **{unique_eff}**",
        f"- Pinned build SHA: `{pinned_sha or '(none)'}`",
        f"- All observed build SHAs: `{sorted(observed_shas)}`",
        f"- Transport failures: **{transport_failures}**",
        f"- Retried calls: **{retried_calls}**",
        f"- Unexpected tool errors: **{unexpected_errors}**",
        f"- Wrong tool error codes: **{wrong_tool_errors}**",
        f"- Unexpected successes: **{unexpected_successes}**",
        f"- Expected invalid inputs correctly rejected: **{expected_invalid}**",
        f"- Semantic invariant failures: **{semantic_flags}**",
        f"- Build SHA drift records: **{sha_drift}**",
        "",
        "## Latency (ms)",
        "",
    ]

    for label, pct in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
        v = percentile(elapsed_values, pct)
        lines.append(f"- {label}: **{v:.1f} ms**")
    lines.append(f"- max: **{(max(elapsed_values) if elapsed_values else 0.0):.1f} ms**")

    lines.extend([
        "",
        "## Response Size (Wire Bytes)",
        "",
    ])
    for label, pct in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
        v = percentile(bytes_values, pct)
        lines.append(f"- {label}: **{v:.0f} bytes**")
    lines.append(f"- max: **{(max(bytes_values) if bytes_values else 0):.0f} bytes**")

    # Coverage distribution tables (R2-025)
    lines.extend([
        "",
        "## Tool Distribution",
        "",
        "| Tool | Calls | Success OK | Expected Invalid | Unexpected Err | Wrong Err | Unexp Success | Semantic Fail | Retries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    tools = sorted({r.tool for r in records})
    for t in tools:
        recs = [r for r in records if r.tool == t]
        ok = sum(1 for r in recs if r.transport_ok and not r.tool_error and r.semantic_ok)
        ei = sum(1 for r in recs if r.status == "expected_invalid_input")
        ue = sum(1 for r in recs if r.status == "unexpected_tool_error")
        we = sum(1 for r in recs if r.status == "wrong_tool_error")
        us = sum(1 for r in recs if r.status == "unexpected_success")
        sf = sum(1 for r in recs if not r.semantic_ok and r.status != "expected_invalid_input")
        ret = sum(1 for r in recs if r.attempt_count > 1)
        lines.append(f"| {t} | {len(recs)} | {ok} | {ei} | {ue} | {we} | {us} | {sf} | {ret} |")

    # Depth bucket distribution
    lines.extend([
        "",
        "## Depth Distribution",
        "",
        "| Depth Bucket | Calls |",
        "|---|---:|",
    ])
    depth_buckets: Counter[str] = Counter()
    for r in records:
        d = r.arguments.get("depth")
        if d is None:
            depth_buckets["default/omitted"] += 1
        elif d <= 8:
            depth_buckets["low (<=8)"] += 1
        elif d <= 16:
            depth_buckets["medium (9-16)"] += 1
        elif d <= 24:
            depth_buckets["high (17-24)"] += 1
        else:
            depth_buckets["hard (25-30)"] += 1
    for bucket, cnt in depth_buckets.most_common():
        lines.append(f"| {bucket} | {cnt} |")

    # Detail distribution
    lines.extend([
        "",
        "## Detail Distribution",
        "",
        "| Detail | Calls |",
        "|---|---:|",
    ])
    detail_counts = Counter(str(r.arguments.get("detail", "omitted")) for r in records)
    for det, cnt in detail_counts.most_common():
        lines.append(f"| {det} | {cnt} |")

    # Long game distribution for analyze_game
    analyze_recs = [r for r in records if r.tool == "analyze_game"]
    if analyze_recs:
        lines.extend([
            "",
            "## Analyze Game PGN Length Distribution",
            "",
            "| Length Bucket | Calls |",
            "|---|---:|",
        ])
        lg_buckets: Counter[str] = Counter()
        for r in analyze_recs:
            pgn = str(r.arguments.get("pgn", ""))
            moves = [w for w in pgn.split() if "." in w or w in ("O-O", "e4", "d4", "Nf3")]
            approx_plies = len(moves) * 2
            if approx_plies <= 20:
                lg_buckets["short (<=20 plies)"] += 1
            elif approx_plies <= 79:
                lg_buckets["medium (21-79 plies)"] += 1
            elif approx_plies <= 159:
                lg_buckets["long (80-159 plies)"] += 1
            else:
                lg_buckets["ultralong (>=160 plies)"] += 1
        for b, cnt in lg_buckets.most_common():
            lines.append(f"| {b} | {cnt} |")

    # Semantic failure details
    failures = [r for r in records if not r.semantic_ok and r.status != "expected_invalid_input"]
    if failures:
        lines.extend([
            "",
            "## Failed Cases (Top 25)",
            "",
        ])
        for f in failures[:25]:
            reasons = "; ".join(f.semantic_failures or f.notes)
            lines.append(f"- `#{f.sequence}` **{f.tool}** `{f.case_id}`: {reasons} (args: `{json.dumps(f.arguments)}`)")

    lines.append("")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
