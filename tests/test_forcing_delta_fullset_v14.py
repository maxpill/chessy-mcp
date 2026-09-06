from __future__ import annotations

from mcp_server.analysis.game_threat_forensics import _compare_forcing_lists
from mcp_server.models.forensics import ForcingMoveEvidence


def _fact(index: int) -> ForcingMoveEvidence:
    return ForcingMoveEvidence(
        uci=f"a{index % 8 + 1}a{(index + 1) % 8 + 1}",
        san=f"M{index}",
        is_check=index % 2 == 0,
        is_capture=index % 3 == 0,
    )


def test_full_set_delta_is_not_limited_by_wire_cap() -> None:
    # Use unique synthetic UCI keys because this helper's contract is set
    # comparison, independent of move-generation legality.
    baseline = [
        ForcingMoveEvidence(uci=f"base-{index}", san=f"B{index}")
        for index in range(30)
    ]
    after = [
        ForcingMoveEvidence(uci=f"base-{index}", san=f"B{index}")
        for index in range(5, 30)
    ] + [
        ForcingMoveEvidence(uci=f"new-{index}", san=f"N{index}", is_check=True)
        for index in range(8)
    ]

    newly, resolved, persistent = _compare_forcing_lists(baseline, after)

    assert {item.uci for item in newly} == {f"new-{index}" for index in range(8)}
    assert {item.uci for item in resolved} == {f"base-{index}" for index in range(5)}
    assert {item.uci for item in persistent} == {f"base-{index}" for index in range(5, 30)}
