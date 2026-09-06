from __future__ import annotations

from mcp_server.analysis.game_coaching import _build_segments
from mcp_server.models import MCPEval


def _evals(*cps: int) -> list[MCPEval]:
    return [MCPEval(cp=cp, depth=12, searched_depth=12) for cp in cps]


def test_one_ply_threshold_flicker_does_not_create_fake_game_phase() -> None:
    # Position 0 is the root. The after-ply states are:
    # equal, slight-worse, equal, slight-worse, slight-worse, slight-worse.
    # The isolated ply-2 dip is noise. The later repeated dip is confirmed.
    segments = _build_segments(
        _evals(0, 0, -80, 0, -80, -90, -90),
        perspective="white",
    )

    assert [(item.start_ply, item.end_ply, item.state) for item in segments] == [
        (1, 3, "approximately_equal"),
        (4, 6, "slightly_worse"),
    ]
    assert segments[1].transition_cause_ply == 4
    assert segments[1].transition_confirmed_ply == 5
    assert segments[0].raw_state_change_count == 2
    assert segments[0].stability == "medium"
    assert segments[1].stability == "high"


def test_large_decisive_jump_is_not_delayed_by_persistence_filter() -> None:
    segments = _build_segments(
        _evals(0, 0, -420, -450),
        perspective="white",
    )

    assert [(item.start_ply, item.end_ply, item.state) for item in segments] == [
        (1, 1, "approximately_equal"),
        (2, 3, "decisively_worse"),
    ]
    assert segments[1].transition_cause_ply == 2
    assert segments[1].transition_confirmed_ply == 2
    assert segments[1].eval_peak_effective_cp == -420
    assert segments[1].eval_trough_effective_cp == -450


def test_black_perspective_story_flips_eval_direction_before_segmentation() -> None:
    segments = _build_segments(
        _evals(0, 0, 180, 200),
        perspective="black",
    )

    assert segments[-1].state == "worse"
    assert segments[-1].eval_end_effective_cp == -200
