# Chess MCP Ultra-Hard Audit Round 2 Remediation Report

Date: 2026-09-11  
Repository: `maxpill/chessy-mcp`  
Branch: `fix/ultra-hard-round2-audit`  
Author: Antigravity / Claude  

---

## Executive Summary

The Round 2 audit examined the integrity of the Chess MCP testing, stress, and certification infrastructure. A high test count is meaningless if the certification harness itself can silently accept wrong error codes, bypass board reconstruction, fail to pin build identities, produce false positives, or execute invalid GitHub workflows.

All 25 findings (`R2-001` through `R2-025`) have been fully remediated and verified through unit tests and local schema validation. The certification harness has been modularized into `scripts/audit/`, and the case matrix expanded from 360 to 600 schema-valid, effectively distinct requests with bounded high-depth tiers (18–30) and legal long games (80–200+ plies).

---

## Remediation Status Matrix (R2-001 – R2-025)

| ID | Severity | Reproduced | Root Cause | Fix | Tests | Local Verified | CI Verified | Production Verified | Host Verified |
|---|---|---|---|---|---|---|---|---|---|
| **R2-001** | P0/P1 | Yes | Static expression `${{ github.sha }}` in `inputs.expected_sha.description` caused immediate 0-job workflow failure. | Replaced with static description; defaulted environment variables for PR triggers. | `test_workflow_syntax.py` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_DISPATCH` | `N/A` |
| **R2-002** | P1 | Yes | `local_stress_360.py` imported removed `_build_case_iter` and used broad string allowlists. | Refactored `local_stress_360.py` to delegate to shared `scripts/audit/` specifications and exact oracles. | `test_22_local_stress_imports_current_case_generator` | `UNIT_VERIFIED` | `CI_VERIFIED` | `N/A` | `N/A` |
| **R2-003** | P1 | Yes | `analyze_game` stress generator sent undeclared `white_player` and `black_player` arguments. | Removed undeclared arguments; tested real arguments (`perspective`, `max_critical_moments`, `strict`). | `test_03_every_success_case_schema_valid`, `test_20` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-004** | P1/P2 | Yes | Argument hashing used raw JSON string, allowing ignored/unknown arguments to game uniqueness. | Implemented `effective_argument_hash` derived from schema-normalized, alias-resolved, default-canonicalized arguments. | `test_02_effective_argument_uniqueness` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-005** | P0/P1 | Yes | Fuzzy error matching used loose substring fallback (`expected_lower in text.lower()`), allowing wrong codes to pass. | Replaced with structured error identity matching and exact bracketed code checking. | `test_05_substring_error_collision_fails` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-006** | P0 | Yes | Missing `build_sha` on success responses was not recorded as certification failure. | Added mandatory `validate_call_build_identity`: missing `build_sha` on success triggers failure. | `test_06_missing_build_sha_fails_certification` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-007** | P0 | Yes | No SHA comparison occurred in phase 1 when CLI expected SHA was omitted, allowing mixed SHAs to pass. | Pin build SHA before concurrency via preflight call; enforce `observed_shas == {pinned_sha}` invariant. | `test_07_two_shas_in_run_fails_certification` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-008** | P2 | Yes | Build mismatch only failed after running expensive concurrent phases. | Preflight pinning fails fast before starting concurrent stress phases. | `test_08_sha_drift_fails_certification` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-009** | P2 | Yes | `CallRecord.sequence` reset to 0 in each concurrency phase. | Monotonic global sequence and unique `call_id` assigned per call across all phases. | `test_18_global_call_ids_unique_across_records` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-010** | P1/P2 | Yes | JSONL lacked arguments, expected kind, error codes, and failure details needed to reproduce failures. | Enriched `CallRecord` with sanitized arguments, hashes, attempts, durations, error codes, and semantic failures. | `test_21_jsonl_output_includes_sanitized_arguments` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-011** | P2 | Yes | Parser inspected only the first text content block, ignoring `structuredContent` and multi-block responses. | Canonical `normalize_call_tool_result` parses structured content, concatenates text blocks, and tracks byte metrics. | `test_16_structured_response_content_parsed`, `test_17` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-012** | P1 | Yes | Classify oracle read dead `played_move` dict instead of `played` (UCI) and `played_san` (SAN). | Classify oracle verifies `played` and `played_san` on root board and validates engine best consistency. | `test_09_classify_oracle_catches_wrong_played_uci_and_san` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-013** | P1 | Yes | Top moves oracle checked `san` instead of canonical `candidate_san`. | Updated top moves oracle to assert `candidate_san` matches `board.san(move)`. | `test_10_top_oracle_catches_wrong_candidate_san` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-014** | P1 | Yes | Forcing move oracle used root FEN for nested replies, causing false positives and negatives. | Path-aware validators carry exact board states for snapshots, candidates, and replies. | `test_11`, `test_12_is_mate_true_on_nonmate_fails` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-015** | P1 | Yes | Analyze game oracle did not reconstruct boards for nested critical moment evidence. | Step-by-step game replay creates exact board states for each critical moment's pre-move and post-move checks. | `scripts/audit/semantic_oracles.py:validate_analyze_game` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-016** | P2 | Yes | Broad `except Exception: pass` swallowed parser and legality errors in stress checks. | Removed broad exception swallowing; all parser failures are explicitly expected or flagged. | `test_14_illegal_candidate_is_caught` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-017** | P1 | Yes | Cases were not validated against canonical tool schemas prior to execution. | Preflight schema validator validates 100% of cases against `tools/list` schema before running. | `test_03`, `test_20` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-018** | P1 | Yes | Stress harness internals lacked dedicated unit tests. | Created `tests/test_production_stress_harness_round2.py` with 22 comprehensive unit tests. | `tests/test_production_stress_harness_round2.py` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-019** | P1 | Yes | Normal CI did not validate workflows or stress case definitions. | Added `test_workflow_syntax.py` and `chess_mcp_stress.py --validate-cases-only` to `.github/workflows/ci.yml`. | `.github/workflows/ci.yml` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-020** | P2 | Yes | Remediation report overstated certification status before production run. | Adopted strict certification taxonomy (`SOURCE_FIXED`, `UNIT_VERIFIED`, `CI_VERIFIED`, `PRODUCTION_STRESS_CERTIFIED`). | This document | `DOCS_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `EXTERNAL_HOST_ISSUE` |
| **R2-021** | P1 | Yes | 360-call matrix was too narrow and lacked high-depth and long-game representation. | Scaled to 600 distinct schema-valid calls (140 eval, 150 top, 170 classify, 140 analyze). | `test_01_exact_case_count`, `test_02` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-022** | P1/P2 | Yes | Stress matrix depths were mostly <= 14, lacking hard-depth verification. | Added bounded high-depth tier (depths 18, 20, 22, 24, 26, 30) for ~60 calls across tools. | `scripts/audit/case_generation.py` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-023** | P1 | Yes | Analyze game corpus lacked true long games (80–200+ plies). | Added programmatic legal long-game generation (80, 120, 160, 200 plies) with endgames and annotations. | `scripts/audit/case_generation.py` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-024** | P2 | Yes | Transport retries were hidden inside success records without attempt telemetry. | Tracked `attempt_count`, `retry_reasons`, and `per_attempt_elapsed_ms` in `CallRecord`. | `test_19_transport_retry_telemetry_visible` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |
| **R2-025** | P2 | Yes | Global markdown summary lacked coverage distribution diagnostics. | Added distribution tables by tool, detail, verbosity, depth bucket, strict, action_type, and PGN length. | `scripts/audit/reporting.py:generate_markdown_summary` | `UNIT_VERIFIED` | `CI_VERIFIED` | `PENDING_STRESS` | `N/A` |

---

## Architectural Improvements

### 1. Modular Architecture (`scripts/audit/`)
The previous monolithic `production_stress_360.py` has been decomposed into testable, single-responsibility modules:
- `case_spec.py`: Data model with raw and effective argument hashing.
- `schema_validation.py`: Strict preflight schema validation disallowing undeclared fields on success cases.
- `response_normalization.py`: Comprehensive CallToolResult normalizer handling text, structuredContent, and error codes.
- `semantic_oracles.py`: Ground-truth chess validation carrying exact board contexts.
- `build_identity.py`: Fail-fast build SHA pinning before concurrent phases.
- `reporting.py`: Sanitized JSONL recording and detailed markdown distribution summaries.
- `case_generation.py`: 600 schema-valid, distinct cases.

### 2. Unified CLI Runner (`scripts/chess_mcp_stress.py`)
Provides a single interface for:
- Preflight validation: `python scripts/chess_mcp_stress.py --validate-cases-only`
- Local in-process testing: `python scripts/chess_mcp_stress.py --transport in-process`
- Remote production certification: `python scripts/chess_mcp_stress.py --target https://mcp.trychessy.com --expected-sha <sha>`

---

## External ChatGPT Host Integration Analysis

During the audit, direct MCP calls from the external ChatGPT host runtime failed after resource discovery:
1. `tools/list` succeeded and discovered all 4 tools.
2. The initial call to `evaluate_position` resulted in the host disabling/dropping the tool surface with an external runtime error.
3. Subsequent direct calls were not routed to Chess MCP.

**Root Cause Analysis**:
- The Chess MCP server protocol response format is valid Streamable HTTP MCP (verified with Python MCP client and curl).
- The issue is isolated to the host-level tool router / gateway in the external client environment.
- Chess MCP will not alter valid chess semantics to accommodate opaque external gateway failures.
