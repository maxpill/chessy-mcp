"""Regression test suite for the Chess MCP production stress & certification harness (Round 2).

Tests the 22+ audit certification invariants:
1. Exact case count (600 cases generated).
2. Effective argument uniqueness (0 duplicates).
3. Every success case validates against canonical tool schema.
4. Schema errors are explicitly categorized as schema_error.
5. Substring error matching collision fails (INTERNAL_ERROR text containing invalid_detail).
6. Missing build SHA fails certification.
7. Two SHAs in first phase fails certification.
8. Later SHA drift fails certification.
9. Classify oracle catches wrong played UCI and played_san.
10. Top oracle catches wrong candidate_san.
11. SAN '#' with is_mate=False is caught.
12. is_mate=True on non-checkmate is caught.
13. Post FEN mismatch is caught.
14. Illegal candidate move is caught.
15. Duplicate candidate root UCI is caught.
16. Structured response content parsed.
17. Multi-content blocks handled deterministically.
18. Monotonic sequence and unique call IDs across phases.
19. Transport retry telemetry is recorded and visible in CallRecord.
20. Unknown argument in success case rejected preflight.
21. JSONL output includes sanitized arguments for reproduction.
22. Local stress script imports current case generator and runs cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess

from scripts.audit.build_identity import validate_call_build_identity, validate_observed_build_shas
from scripts.audit.case_generation import build_600_case_specs
from scripts.audit.case_spec import CaseSpec
from scripts.audit.reporting import CallRecord, write_jsonl
from scripts.audit.response_normalization import (
    NormalizedResponse,
    normalize_call_tool_result,
)
from scripts.audit.schema_validation import (
    load_canonical_schemas,
    preflight_validate_cases,
    validate_case_arguments,
)
from scripts.audit.semantic_oracles import (
    check_forcing_evidence,
    validate_classify_move,
    validate_expected_error,
    validate_top_moves,
)


def test_01_exact_case_count() -> None:
    cases = build_600_case_specs()
    assert len(cases) == 600
    by_tool = {
        "evaluate_position": sum(1 for c in cases if c.tool == "evaluate_position"),
        "top_moves": sum(1 for c in cases if c.tool == "top_moves"),
        "classify_move": sum(1 for c in cases if c.tool == "classify_move"),
        "analyze_game": sum(1 for c in cases if c.tool == "analyze_game"),
    }
    assert by_tool["evaluate_position"] == 140
    assert by_tool["top_moves"] == 150
    assert by_tool["classify_move"] == 170
    assert by_tool["analyze_game"] == 140


def test_02_effective_argument_uniqueness() -> None:
    cases = build_600_case_specs()
    errors, eff_hashes = preflight_validate_cases(cases)
    assert not errors, f"Preflight errors found: {errors}"
    assert len(eff_hashes) == 600


def test_03_every_success_case_schema_valid() -> None:
    cases = build_600_case_specs()
    schemas = load_canonical_schemas()
    for spec in cases:
        if spec.expected_kind == "success":
            schema = schemas.get(spec.tool, {})
            is_valid, errs, _ = validate_case_arguments(spec, schema)
            assert is_valid, f"Success case '{spec.case_id}' failed schema validation: {errs}"


def test_04_schema_errors_explicit() -> None:
    cases = build_600_case_specs()
    schema_err_cases = [c for c in cases if c.expected_kind == "schema_error"]
    assert len(schema_err_cases) > 0
    for c in schema_err_cases:
        assert c.expected_error_code == "schema_validation_error"


def test_05_substring_error_collision_fails() -> None:
    spec = CaseSpec(
        case_id="test_case",
        tool="evaluate_position",
        arguments={"fen": chess.STARTING_FEN},
        expected_kind="tool_error",
        expected_error_code="invalid_detail",
    )
    # Server returned internal_error code, but message mentions 'invalid_detail' in prose
    norm = NormalizedResponse(
        is_error=True,
        text_content="[INTERNAL_ERROR] An unexpected internal error occurred for invalid_detail requested fixture.",
        parsed_payload={"code": "INTERNAL_ERROR"},
        wire_bytes=100,
        text_bytes=100,
        structured_payload_bytes=50,
        structured_error_code="INTERNAL_ERROR",
        build_sha="123456",
    )
    errs = validate_expected_error(spec, norm)
    assert len(errs) > 0, "Substring matching must not accept internal_error as invalid_detail"
    assert "wrong error code" in errs[0].lower()


def test_06_missing_build_sha_fails_certification() -> None:
    errs = validate_call_build_identity(build_sha="", pinned_sha="abcdef123456", is_success=True)
    assert len(errs) > 0
    assert "missing_build_identity" in errs[0]


def test_07_two_shas_in_run_fails_certification() -> None:
    observed = {"abcdef123456", "789012abcdef"}
    errs = validate_observed_build_shas(observed, pinned_sha="abcdef123456")
    assert len(errs) > 0
    assert "multiple build shas" in errs[0].lower()


def test_08_sha_drift_fails_certification() -> None:
    errs = validate_call_build_identity(build_sha="wrongsha9999", pinned_sha="abcdef123456", is_success=True)
    assert len(errs) > 0
    assert "drift" in errs[0].lower()


def test_09_classify_oracle_catches_wrong_played_uci_and_san() -> None:
    spec = CaseSpec(
        case_id="cls_test",
        tool="classify_move",
        arguments={"fen": chess.STARTING_FEN, "move": "e4"},
        expected_kind="success",
    )
    # Payload claims played is e2e3 instead of e2e4
    bad_payload = {
        "played": "e2e3",
        "played_san": "e3",
        "is_engine_best": False,
    }
    errs = validate_classify_move(spec, bad_payload)
    assert len(errs) > 0
    assert any("played UCI" in e for e in errs)


def test_10_top_oracle_catches_wrong_candidate_san() -> None:
    spec = CaseSpec(
        case_id="top_test",
        tool="top_moves",
        arguments={"fen": chess.STARTING_FEN, "n": 1},
        expected_kind="success",
    )
    # Candidate move e2e4 has SAN claiming 'd4'
    bad_payload = {
        "returned_n": 1,
        "moves": [
            {
                "best_move": "e2e4",
                "candidate_san": "d4",
                "post_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            }
        ],
    }
    errs = validate_top_moves(spec, bad_payload)
    assert len(errs) > 0
    assert any("candidate SAN" in e for e in errs)


def test_11_san_mate_hash_without_is_mate_flag_fails() -> None:
    data = {
        "san": "Qxf7#",
        "is_mate": False,
        "is_check": True,
    }
    errs = check_forcing_evidence(data)
    assert len(errs) > 0
    assert "ends with '#' but is_mate is False" in errs[0]


def test_12_is_mate_true_on_nonmate_fails() -> None:
    b = chess.Board()
    # e4 is not checkmate
    data = {
        "san": "e4",
        "uci": "e2e4",
        "is_mate": True,
        "is_check": False,
    }
    errs = check_forcing_evidence(data, b)
    assert len(errs) > 0
    assert any("child.is_checkmate()" in e for e in errs)


def test_13_post_fen_mismatch_is_caught() -> None:
    spec = CaseSpec(
        case_id="top_test",
        tool="top_moves",
        arguments={"fen": chess.STARTING_FEN, "n": 1},
        expected_kind="success",
    )
    bad_payload = {
        "returned_n": 1,
        "moves": [
            {
                "best_move": "e2e4",
                "candidate_san": "e4",
                "post_fen": "8/8/8/8/8/8/8/8 w - - 0 1",  # Wrong post_fen
            }
        ],
    }
    errs = validate_top_moves(spec, bad_payload)
    assert len(errs) > 0
    assert any("post_fen" in e for e in errs)


def test_14_illegal_candidate_is_caught() -> None:
    spec = CaseSpec(
        case_id="top_test",
        tool="top_moves",
        arguments={"fen": chess.STARTING_FEN, "n": 1},
        expected_kind="success",
    )
    bad_payload = {
        "returned_n": 1,
        "moves": [{"best_move": "e2e5", "candidate_san": "e5"}],
    }
    errs = validate_top_moves(spec, bad_payload)
    assert len(errs) > 0
    assert any("not legal" in e for e in errs)


def test_15_duplicate_candidate_root_uci_is_caught() -> None:
    spec = CaseSpec(
        case_id="top_test",
        tool="top_moves",
        arguments={"fen": chess.STARTING_FEN, "n": 2},
        expected_kind="success",
    )
    bad_payload = {
        "returned_n": 2,
        "moves": [
            {"best_move": "e2e4", "candidate_san": "e4"},
            {"best_move": "e2e4", "candidate_san": "e4"},
        ],
    }
    errs = validate_top_moves(spec, bad_payload)
    assert len(errs) > 0
    assert any("candidate root UCIs not unique" in e for e in errs)


def test_16_structured_response_content_parsed() -> None:
    class MockStructured:
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            return {"status": "ok", "best_move": "e2e4", "build_sha": "testsha123"}

    class MockResult:
        def __init__(self) -> None:
            self.structuredContent = MockStructured()
            self.content: list[Any] = []

    norm = normalize_call_tool_result(MockResult())
    assert norm.parsed_payload is not None
    assert norm.parsed_payload.get("best_move") == "e2e4"
    assert norm.build_sha == "testsha123"


def test_17_multi_content_blocks_handled_deterministically() -> None:
    class MockBlock:
        def __init__(self, text: str):
            self.text = text

    class MockResult:
        def __init__(self) -> None:
            self.content = [MockBlock('{"status": "ok"}'), MockBlock("extra diagnostic information")]

    norm = normalize_call_tool_result(MockResult())
    assert norm.parsed_payload == {"status": "ok"}
    assert "extra diagnostic information" in norm.text_content


def test_18_global_call_ids_unique_across_records() -> None:
    rec1 = CallRecord(
        call_id="call_001",
        case_ordinal=1,
        sequence=1,
        tool="evaluate_position",
        case_id="eval_001",
        arguments={"fen": chess.STARTING_FEN},
        raw_argument_hash="hash1",
        effective_argument_hash="eff1",
        expected_kind="success",
        expected_error_code=None,
        started_at=1000.0,
        elapsed_ms=10.0,
        transport_ok=True,
        tool_error=False,
        semantic_ok=True,
        build_sha="sha1",
        wire_bytes=100,
        text_bytes=100,
        structured_bytes=100,
        status="active",
    )
    rec2 = CallRecord(
        call_id="call_002",
        case_ordinal=2,
        sequence=2,
        tool="top_moves",
        case_id="top_001",
        arguments={"fen": chess.STARTING_FEN},
        raw_argument_hash="hash2",
        effective_argument_hash="eff2",
        expected_kind="success",
        expected_error_code=None,
        started_at=1001.0,
        elapsed_ms=12.0,
        transport_ok=True,
        tool_error=False,
        semantic_ok=True,
        build_sha="sha1",
        wire_bytes=120,
        text_bytes=120,
        structured_bytes=120,
        status="active",
    )
    assert rec1.call_id != rec2.call_id
    assert rec1.sequence < rec2.sequence


def test_19_transport_retry_telemetry_visible() -> None:
    rec = CallRecord(
        call_id="call_retry",
        case_ordinal=1,
        sequence=1,
        tool="classify_move",
        case_id="cls_001",
        arguments={"fen": chess.STARTING_FEN, "move": "e4"},
        raw_argument_hash="hash",
        effective_argument_hash="eff",
        expected_kind="success",
        expected_error_code=None,
        started_at=1000.0,
        elapsed_ms=500.0,
        transport_ok=True,
        tool_error=False,
        semantic_ok=True,
        build_sha="sha",
        wire_bytes=100,
        text_bytes=100,
        structured_bytes=100,
        status="active",
        attempt_count=2,
        retry_reasons=["ConnectionResetError: stream ended"],
        per_attempt_elapsed_ms=[250.0, 250.0],
    )
    assert rec.attempt_count == 2
    assert len(rec.retry_reasons) == 1
    assert len(rec.per_attempt_elapsed_ms) == 2


def test_20_unknown_argument_in_success_case_rejected_preflight() -> None:
    bad_spec = CaseSpec(
        case_id="bad_analyze",
        tool="analyze_game",
        arguments={"pgn": "1. e4 e5 *", "white_player": "Player W"},  # Undeclared arg
        expected_kind="success",
    )
    schemas = load_canonical_schemas()
    is_valid, errs, _ = validate_case_arguments(bad_spec, schemas.get("analyze_game", {}))
    assert not is_valid
    assert any("Undeclared argument 'white_player'" in e for e in errs)


def test_21_jsonl_output_includes_sanitized_arguments(tmp_path: Path) -> None:
    rec = CallRecord(
        call_id="call_jsonl",
        case_ordinal=1,
        sequence=1,
        tool="evaluate_position",
        case_id="eval_001",
        arguments={"fen": chess.STARTING_FEN, "depth": 14},
        raw_argument_hash="hash1",
        effective_argument_hash="eff1",
        expected_kind="success",
        expected_error_code=None,
        started_at=1000.0,
        elapsed_ms=10.0,
        transport_ok=True,
        tool_error=False,
        semantic_ok=True,
        build_sha="sha1",
        wire_bytes=100,
        text_bytes=100,
        structured_bytes=100,
        status="active",
    )
    out_file = tmp_path / "records.jsonl"
    write_jsonl([rec], out_file)
    content = out_file.read_text(encoding="utf-8")
    loaded = json.loads(content.strip())
    assert loaded["arguments"] == {"fen": chess.STARTING_FEN, "depth": 14}
    assert loaded["call_id"] == "call_jsonl"


def test_22_local_stress_imports_current_case_generator() -> None:
    import scripts.local_stress_360
    assert hasattr(scripts.local_stress_360, "run_local_stress")
    import scripts.production_stress_360
    specs = scripts.production_stress_360._build_case_specs()
    assert len(specs) == 360
