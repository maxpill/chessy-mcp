# Chess MCP Ultra Hard Production Audit Remediation Report
**Audit Date**: 2026-09-10  
**Branch**: `fix/2026-09-10-ultra-audit-remediation`  
**Target Specification**: `/Users/max/Downloads/chess_mcp_ultra_audit_claude_code_2026-09-10.md`  
**Test Suite Status**: 1220 passed, 6 skipped  
**Type Check Status**: Pyright 0 errors, 0 warnings  
**Linter Status**: Ruff clean on modified codebase  

---

## Executive Summary

On 2026-09-10, an ultra-hard audit of the Chess MCP service identified 15 findings spanning chess-rule semantic correctness, MCP FastMCP transport and schema reflection, server runtime diagnostics, forensic accounting, production stress harness coverage, and CI workflow pinning.

All 15 findings have been systematically resolved, verified with comprehensive unit, property-based, transport contract, and integration tests, and certified on the dedicated branch `fix/2026-09-10-ultra-audit-remediation`.

---

## Certification Matrix

| Finding ID | Severity | Category | Status | Primary Artifacts & Verification |
|---|---|---|---|---|
| **AUDIT-001** | P0 | Chess-Rule Correctness | **RESOLVED** | `mcp_server/analysis/forensics.py`, `tests/test_audit_2026_09_10_forcing_mate.py`, `tests/test_property_forcing_evidence.py` |
| **AUDIT-002** | P1 | Contract & Transport | **RESOLVED** | `mcp_server/tools/_common.py`, `mcp_server/tools/evaluate_position.py`, `mcp_server/tools/top_moves.py`, `tests/test_mcp_transport_contract.py` |
| **AUDIT-003** | P1 | Server Diagnostics | **RESOLVED** | `mcp_server/server.py`, `tests/test_mcp_transport_contract.py` |
| **AUDIT-004** | P0/P1 | Protocol & Transport | **RESOLVED** | `tests/test_mcp_transport_contract.py` (in-process client-server transport verification) |
| **AUDIT-005** | P2/P1 | Forensic Accounting | **RESOLVED** | `mcp_server/analysis/top_moves_forensics.py`, `tests/test_public_cross_tool_invariants_2026_09_09.py` |
| **AUDIT-006** | P1 | Contract Discrepancy | **RESOLVED** | Option A implemented: `mcp_server/tools/evaluate_position.py`, `mcp_server/tools/_common.py`, `tests/test_mcp_transport_contract.py` |
| **AUDIT-007** | P1 | Architecture Clarity | **VERIFIED** | Baseline `TOP_MOVES_MAX_N=20` preserved; commit `eb9e6e28` confirmed as intentional canonical spec. |
| **AUDIT-008** | P0 | Stress Harness Integrity | **RESOLVED** | `scripts/production_stress_360.py` (asserted 360/360 unique argument sets) |
| **AUDIT-009** | P2 | Stress Fixture Quality | **RESOLVED** | `scripts/production_stress_360.py` (valid resignation header + separate dirty PGN recovery case) |
| **AUDIT-010** | P0 | Stress Error Rigor | **RESOLVED** | `scripts/production_stress_360.py` (`CaseSpec` with exact declared error code matching) |
| **AUDIT-011** | P1/P0 | Stress Semantic Oracle | **RESOLVED** | `scripts/production_stress_360.py` (multi-tier chess semantic oracle verifying mate, legality, and terminal states) |
| **AUDIT-012** | P0/P1 | CI Workflow Pinning | **RESOLVED** | `scripts/production_stress_360.py`, `.github/workflows/stress-360.yml` (`--expected-sha` enforced) |
| **AUDIT-013** | P3 | Code Quality | **RESOLVED** | `scripts/production_probe.py` (duplicate schema fingerprint removed; contract diagnostics added) |
| **AUDIT-014** | P2 | Schema Snapshot Parity | **RESOLVED** | `tests/fixtures/schema_fingerprint_main.json`, `tests/test_p2_2026_09_09_schema_fingerprint.py` |
| **AUDIT-015** | P2 | Cross-Tool Invariants | **RESOLVED** | `tests/test_public_cross_tool_invariants_2026_09_09.py` (Invariants 3 and 5 implemented and passing) |

---

## Detailed Remediation Findings

### 1. AUDIT-001 (P0): `ForcingMoveEvidence.is_mate` Chess-Rule Oracle Failure
- **Root Cause**: `ForcingMoveEvidence` previously defaulted `is_mate=False` and was constructed via heuristic checks that never executed `child.is_checkmate()`. Consequently, mating moves (such as `Qh4#` in Fool's Mate or `Qxf7#` in Scholar's Mate) produced `san="Qh4#"` with `is_mate=False`.
- **Fix Implemented**:
  1. Implemented centralized `build_forcing_move_evidence(board: chess.Board, move: chess.Move) -> ForcingMoveEvidence` in `mcp_server/analysis/forensics.py`.
  2. The function clones the board (`child = board.copy(stack=False); child.push(move)`), executes `is_mate = child.is_checkmate()`, and derives `is_check = child.is_check()`.
  3. Replaced ad-hoc and incomplete `_capture_evidence` / `_threat_probe` constructors across `mcp_server/analysis/forensics.py`, `mcp_server/analysis/tactical_snapshot_extensions.py`, and `mcp_server/analysis/position_integrity.py`.
- **Verification**:
  - `tests/test_audit_2026_09_10_forcing_mate.py`: 6/6 deterministic tests passing for Fool's mate, Scholar's mate, KQK mate, back-rank rook mate, mating capture, promotion mate, and non-mating checks.
  - `tests/test_property_forcing_evidence.py`: 5/5 property-based tests verifying `evidence.is_mate == child.is_checkmate()` across 50 random game walks (>200 legal moves) and tactical edge cases (discovered en passant checkmate `exd6#`, pawn promotion mate `f8=Q#`, castling, discovered/double checks).

---

### 2. AUDIT-002 & AUDIT-006 (P1): Verbosity Aliases & Option A Precedence
- **Root Cause**: Tool signatures exposed only `Literal["minimal", "compact", "full"]`, causing FastMCP input schemas to reject valid aliases (`"min"`, `"standard"`, `"default"`) during LLM tool invocation. Additionally, a contract contradiction existed between `verbosity="minimal"` and `detail="coach"` / `detail="forensic"`.
- **Fix Implemented**:
  1. Defined `VerbosityInput = Literal["minimal", "compact", "full", "min", "standard", "default"]` and `SUPPORTED_VERBOSITY_INPUTS` in `mcp_server/tools/_common.py`.
  2. Applied `VerbosityInput` to parameters in `mcp_server/tools/evaluate_position.py` and `mcp_server/tools/top_moves.py`.
  3. Implemented Option A precedence in `evaluate_position`: rejecting `verbosity in ("minimal", "min")` when `detail in ("coach", "forensic")` with an immediate `INVALID_ARGUMENT` `ToolError`.
  4. Added `("INVALID_ARGUMENT", "invalid_argument")` to `_ERROR_CODE_PREFIXES` in `mcp_server/tools/_common.py`.
- **Verification**:
  - `tests/test_mcp_transport_contract.py`: tested all 6 verbosity aliases over in-memory JSON-RPC FastMCP transport; verified exact rejection of `minimal + coach` and `min + forensic`.

---

### 3. AUDIT-003 & AUDIT-004 (P1/P0): Server Runtime Diagnostics & In-Process Transport
- **Root Cause**: The `/health` HTTP endpoint only returned `{"status": "ok"}` without exposing deployment metadata (`build_sha`, `package_version`, tool list). Furthermore, no regression test exercised the in-process JSON-RPC transport contract.
- **Fix Implemented**:
  1. Updated `/health` in `mcp_server/server.py` to return `build_sha`, `package_version`, and sorted `tools`.
  2. Created `tests/test_mcp_transport_contract.py` utilizing `create_client_server_memory_streams` to initialize a low-level MCP server and `ClientSession`.
- **Verification**:
  - Verified schema reflection for all 4 public tools (`evaluate_position`, `top_moves`, `classify_move`, `analyze_game`), alias handling, and error response formatting over transport.

---

### 4. AUDIT-005 (P2/P1): `included_move_count` Accounting
- **Root Cause**: In `top_moves`, `included_move_count` was previously set to `len(include_moves)` regardless of how many requested moves were actually included or whether duplicates were requested.
- **Fix Implemented**:
  - In `mcp_server/analysis/top_moves_forensics.py` (`enrich_top_moves_result`), canonicalized requested moves to UCI, deduplicated them, and counted only unique root candidate moves that were actually merged into the returned candidate list.
- **Verification**:
  - `tests/test_public_cross_tool_invariants_2026_09_09.py` verified exact accounting.

---

### 5. AUDIT-007 (P1): Canonical Boundary Clarity (`TOP_MOVES_MAX_N = 20`)
- **Root Cause**: Audit flagged potential reversion to `TOP_MOVES_MAX_N = 10`.
- **Resolution**: Confirmed that commit `eb9e6e286247aa26c609e03d6b1cfb0863613a56` intentionally raised the canonical maximum to 20. Maintained `TOP_MOVES_MAX_N = 20` across documentation, schemas, and validators.

---

### 6. AUDIT-008, 009, 010, 011, 012: Production Stress Harness Overhaul
- **Root Causes**:
  - `production_stress_360.py` used modulo arithmetic that generated only ~30-60 unique argument sets repeated across 360 calls (AUDIT-008).
  - Malformed resignation PGN header `[Termination "White resigns"}]` swallowed game moves in python-chess (AUDIT-009).
  - Generic allowlist allowed unexpected errors to pass as `expected_invalid_input` (AUDIT-010).
  - Lack of a semantic chess oracle allowed invalid chess states to pass silently (AUDIT-011).
  - `.github/workflows/stress-360.yml` resolved deployment SHA but never passed `--expected-sha` to the script (AUDIT-012).
- **Fixes Implemented**:
  1. Rewrote `scripts/production_stress_360.py` with immutable `CaseSpec` objects.
  2. Generated exactly 360 distinct argument sets (90 evaluate_position, 90 top_moves, 100 classify_move, 80 analyze_game) and asserted `len(unique_sha256) == 360`.
  3. Replaced malformed resignation fixture with valid PGN; added a separate explicit dirty PGN recovery fixture.
  4. Enforced strict error code checking against `spec.expected_error_code`; flagged unexpected errors, wrong error codes, or unexpected successes as failures.
  5. Implemented comprehensive `_run_semantic_oracle`:
     - Asserting `evidence.is_mate == child.is_checkmate()`.
     - Validating checkmate/stalemate winner semantics.
     - Validating candidate root UCI uniqueness and legality.
     - Validating candidate `post_fen` against board move push.
     - Validating game analysis plies and final FEN against replayed mainline.
  6. Added CLI argument `--expected-sha` to `scripts/production_stress_360.py` and updated `.github/workflows/stress-360.yml` to pass `--expected-sha "${{ steps.deployment.outputs.expected_sha }}"`.

---

### 7. AUDIT-013 (P3): Duplicate Schema Fingerprint Cleanup
- **Root Cause**: `scripts/production_probe.py` contained redundant schema fingerprint computation before and after tool calls.
- **Fix Implemented**: Removed duplicate schema calculation block in lines 278-299; added contract diagnostics for `top_moves.n` max, `classify_move.depth` default, and `verbosity` enum options.

---

### 8. AUDIT-014 & AUDIT-015 (P2): Schema Parity & Invariants 3 & 5
- **Fixes Implemented**:
  1. Regenerated `tests/fixtures/schema_fingerprint_main.json` with updated `VerbosityInput` schemas (sha256: `70e8e4d8a48e25f8becbe2d2428379ee02a8408189889150e1c4e1ef3a0cdbeb`).
  2. Implemented `test_invariant_3_rule_action_agreement` and `test_invariant_5_history_provenance` in `tests/test_public_cross_tool_invariants_2026_09_09.py`. All 8 cross-tool invariant tests pass.

---

## Verification Evidence

### 1. Pytest Full Suite
```text
1220 passed, 6 skipped, 1913 warnings in 156.35s (0:02:36)
```

### 2. Pyright Type Check
```text
0 errors, 0 warnings, 0 informations
```

### 3. Ruff Linter
```text
All checks passed!
```

### 4. Stress Generator Argument Uniqueness Check
```python
from scripts.production_stress_360 import _build_case_specs, DISTRIBUTION
cases = _build_case_specs()
assert len(cases) == 360
assert len({c.arguments_sha256 for c in cases}) == 360
assert {c.tool: sum(1 for x in cases if x.tool == c.tool) for c in cases} == DISTRIBUTION
# Result: All 360 case specs valid, unique, and match distribution!
```

---

## Next Steps

1. **Code Review**: Conduct review of diff on branch `fix/2026-09-10-ultra-audit-remediation`.
2. **PR Creation**: Create pull request targeting `main`.
3. **CI Run**: Verify GitHub Actions workflow runs `stress-360.yml` with `--expected-sha` pinning against staging/production before final merge.
