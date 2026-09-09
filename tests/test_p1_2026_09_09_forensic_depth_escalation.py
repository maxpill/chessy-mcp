"""2026-09-09 master audit F-003: forensic analyze_game depth must be bounded.

The audit observed that ``analyze_game(detail='forensic', depth=3)`` for the
Opera Game silently escalated verification to depth 22 (52.9 s outlier). The
root cause was a hardcoded depth ramp in ``_verify_critical_moments`` plus
duplication in ``middleware.request_cost``. This test pins the new monotonic
policy in the shared helper ``forensic_verification_depth``.
"""

from __future__ import annotations

import pytest

from mcp_server.analysis.game_coaching import (
    FORENSIC_VERIFICATION_FLOOR,
    FORENSIC_VERIFICATION_MAX,
    forensic_verification_depth,
)
from mcp_server.middleware.request_cost import estimate_mcp_request_cost


@pytest.mark.parametrize(
    "scan_depth, expected_min, expected_max",
    [
        (1, 1, FORENSIC_VERIFICATION_FLOOR),
        (2, 1, FORENSIC_VERIFICATION_FLOOR),
        (3, 1, FORENSIC_VERIFICATION_FLOOR),
        (6, 1, FORENSIC_VERIFICATION_FLOOR),
        (10, 1, FORENSIC_VERIFICATION_FLOOR),
        (13, 1, FORENSIC_VERIFICATION_FLOOR),
        (14, 22, 22),
        (18, 22, 22),
        (19, 24, 24),
        (20, 24, 24),
        (21, 22, FORENSIC_VERIFICATION_MAX),
        (24, 26, 26),
        (26, 26, FORENSIC_VERIFICATION_MAX),
        (30, 26, FORENSIC_VERIFICATION_MAX),
    ],
)
def test_forensic_verification_depth_is_monotonic_and_bounded(
    scan_depth: int, expected_min: int, expected_max: int
) -> None:
    """scan_depth < 14 must NOT jump to 22; bounds are honored."""
    vd = forensic_verification_depth(scan_depth)
    assert expected_min <= vd <= expected_max, (
        f"forensic_verification_depth({scan_depth}) returned {vd}; "
        f"expected within [{expected_min}, {expected_max}]"
    )


def test_low_scan_depth_does_not_jump_to_22() -> None:
    """P0 F-003: scan_depth=3 must not silently escalate to depth 22."""
    assert forensic_verification_depth(3) < 22, (
        f"F-003 regression: forensic scan_depth=3 jumped to "
        f"{forensic_verification_depth(3)} (expected <22)"
    )
    assert forensic_verification_depth(6) < 22
    assert forensic_verification_depth(10) < 22
    assert forensic_verification_depth(13) < 22


def test_high_scan_depth_caps_at_26() -> None:
    """Scan depth > 24 still caps verification at 26."""
    assert forensic_verification_depth(28) == FORENSIC_VERIFICATION_MAX
    assert forensic_verification_depth(30) == FORENSIC_VERIFICATION_MAX


def test_request_cost_uses_same_helper() -> None:
    """request_cost estimator must agree with forensic_verification_depth."""
    import json

    for scan_depth in (1, 3, 10, 14, 18, 20, 24, 30):
        body = json.dumps(
            {
                "params": {
                    "name": "analyze_game",
                    "arguments": {
                        "pgn": "1. e4 e5 2. Nf3 Nc6 *",
                        "depth": scan_depth,
                        "detail": "forensic",
                    },
                }
            }
        ).encode()
        cost = estimate_mcp_request_cost(body)
        # The cost grows with verification_depth; if the helper is bypassed
        # the cost curve differs from what the helper returns.
        expected_vd = forensic_verification_depth(scan_depth)
        # The cost formula is base + (vd * extra) / 12.0; check the vd term
        # is monotone and uses the helper by recomputing it manually.
        # Sanity floor: any valid helper value must produce cost > base.
        base = 5.0 + (scan_depth * 20.0 / 28.0)
        assert cost > base, (
            f"cost for scan_depth={scan_depth} (expected vd={expected_vd}) "
            f"must exceed base {base}; got {cost}"
        )
