import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pts.offline import delayed_replay, execute_prefix


SETTINGS = {"execution_horizon": 8,
            "pts_core": {"replan_steps": 8, "rate_grid": 0.25, "latency_steps": 2,
                         "gripper_min_steps": 2, "gripper_max_steps": 9, "gripper_command_threshold": 0.02},
            "offline_actuator_box": {"positive": [0.035] * 3, "negative": [0.035] * 3,
                                     "max_translation_norm": 0.035, "max_rotation_travel": 0.06}}


def trajectory():
    actions = np.zeros((50, 7))
    actions[:, 0] = np.arange(1, 51) * 0.002
    actions[:, 1] = np.sin(np.arange(1, 51) * 0.3) * 0.001
    actions[:, 5] = np.arange(1, 51) * 0.002
    return actions, np.zeros(7)


def test_prefix_retains_full_horizon_and_rounds_replan_threshold_up():
    actions, state = trajectory()
    out, progress, info = execute_prefix(actions, state, "pts_1p5x", SETTINGS)
    np.testing.assert_allclose(progress, [1.5, 3, 4.5, 6, 7.5, 9])
    np.testing.assert_allclose(out[-1, :3], actions[8, :3])
    assert info["consumed_progress"] == 9
    identity, p, _ = execute_prefix(actions, state, "pts_1x", SETTINGS)
    np.testing.assert_allclose(identity, actions[:8], atol=1e-14)


@pytest.mark.parametrize("delays", [(2,), (2, 3, 40)])
def test_cached_replay_switches_at_sum_of_executed_rates(delays):
    actions, state = trajectory()
    replay = delayed_replay(actions, state, SETTINGS, delays=delays, ticks=150)
    assert replay["summary"]["switches"] > 0
    assert replay["summary"]["max_entry_error"] < 1e-12
    assert replay["summary"]["moving_steps_during_inference"] > 0
    if 40 in delays:
        assert replay["summary"]["hold_steps"] > 0


def test_gripper_window_changes_prefix_without_dropping_path():
    actions, state = trajectory()
    actions[:, 6] = 0.08
    out, progress, info = execute_prefix(actions, state, "pts_2x", SETTINGS)
    assert info["rates"] == [2, 1, 1, 1, 1, 1, 1]
    assert progress[-1] == 8 and len(out) == 7
