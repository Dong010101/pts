from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pts.clock import ProgressHistory
from pts.core import ActuatorBox, PTSConfig, PTSEngine, RequestToken
from pts.paths import IncrementPath, AbsolutePosePath


def path(x, gripper=0.0):
    actions = np.zeros((len(x), 7))
    actions[:, 0] = x
    actions[:, 6] = gripper
    return IncrementPath(actions)


def engine(config=None, bounds=100):
    return PTSEngine(config or PTSConfig(replan_steps=2, latency_steps=0),
                     ActuatorBox((bounds,) * 3, (bounds,) * 3), initial_gripper=0)


def install(e, p):
    e.accept(e.begin_request(), p)


def test_fractional_sums_conserve_entire_curved_path():
    rng = np.random.default_rng(7)
    p = IncrementPath(rng.normal(size=(50, 7)))
    points = np.r_[0, np.sort(rng.uniform(0, 50, 100)), 50]
    summed = sum(p.delta(a, b) for a, b in zip(points[:-1], points[1:]))
    np.testing.assert_allclose(summed, p.actions[:, :6].sum(axis=0), atol=1e-12)
    corner = IncrementPath([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1]])
    np.testing.assert_allclose(corner.delta(0.5, 1.5)[:3], [0.5, 0.5, 0])


def test_connected_feasible_rejects_reentry():
    p = path([1, 1, -1, -1, 0])
    box = ActuatorBox((1.25,) * 3, (1.25,) * 3)
    assert abs(p.delta(0, 4)[0]) < 1.25  # endpoint alone would wrongly pass
    assert not box.connected_feasible(p, 0, 4)
    e = engine(PTSConfig(replan_steps=1, rate_cap=4, latency_steps=0), bounds=1.25)
    install(e, p)
    assert e.step().rate == 1.25


def test_checks_fractional_path_knots_between_rate_grid_points():
    p = path([0, 1, -1, 0, 0])
    box = ActuatorBox((0.95, 2, 2), (2, 2, 2))
    assert p.delta(0.9, 1.9)[0] == pytest.approx(0.9)
    assert p.delta(0.9, 2.15)[0] == pytest.approx(0.85)
    assert not box.connected_feasible(p, 0.9, 1.25)  # peak at rate 1.1


def test_signed_bounds_and_invalid_one_x_fail_without_clipping():
    p = path([-1] * 10)
    box = ActuatorBox((3, 3, 3), (1.25, 3, 3))
    assert box.connected_feasible(p, 0, 1.25)
    assert not box.connected_feasible(p, 0, 1.5)
    e = engine(bounds=0.9)
    install(e, p)
    with pytest.raises(ValueError, match="1x action"):
        e.step()
    assert e.position == 0


def test_continuation_counts_consumed_progress_including_variable_rates():
    e = engine()
    install(e, path([0.01] * 50))
    first = e.step()
    token = e.begin_request(first.end - 0.25)  # camera captured before request
    e.step()
    e.config = replace(e.config, rate_cap=1)
    e.step()
    m = e.accept(token, path([0.02] * 50))
    assert m == pytest.approx(0.25 + 1.5 + 1)
    assert e.position == m and e.since_switch == 0
    assert not e.request_due
    e.step()
    e.step()
    assert e.request_due


def test_held_ticks_do_not_advance_continuation():
    e = engine()
    token = e.begin_request()
    for _ in range(10):
        assert e.step().rate == 0
    assert e.accept(token, path([0] * 50)) == 0


def test_single_flight_and_stale_response():
    e = engine()
    token = e.begin_request()
    with pytest.raises(RuntimeError):
        e.begin_request()
    with pytest.raises(ValueError):
        e.accept(RequestToken(token.sequence + 1, 0), path([0] * 50))
    e.accept(token, path([0] * 50))
    with pytest.raises(ValueError):
        e.accept(token, path([0] * 50))


@pytest.mark.parametrize("speed,window_steps", [(0, 2), (0.03, 12), (None, 12)])
def test_gripper_flip_window_minimum_feedback_maximum(speed, window_steps):
    e = engine()
    install(e, path([0] * 50, gripper=1))
    assert e.step().rate == 1.5  # flip is t0, window starts t0+1
    for _ in range(window_steps):
        command = e.step(gripper_speed=speed)
        assert command.rate == 1 and command.gripper_window
    assert e.step(gripper_speed=speed).rate == 1.5


def test_gripper_is_command_at_end_of_consumed_interval():
    p = path([0] * 5, gripper=[0, 1, 2, 3, 4])
    assert p.gripper(1) == 0
    assert p.gripper(1.25) == 1
    assert p.gripper(5) == 4


def test_latency_cap_fractional_tail_and_hold():
    e = engine(PTSConfig(replan_steps=1, rate_cap=3, latency_steps=1))
    install(e, path([0.01] * 6))
    assert e.feasibility_cap == 2.5
    assert [e.step().rate for _ in range(4)] == [2.5, 2.5, 1, 0]
    e2 = engine()
    install(e2, path([0.01] * 5))
    assert [e2.step().rate for _ in range(5)] == [1.5, 1.5, 1.5, 0.5, 0]
    e.set_latency(100)
    assert e.request_due
    install(e, path([0.01] * 6))
    assert e.step().reason == "latency_infeasible"


def test_absolute_orientation_composes_noncommuting_rotations():
    r = Rotation.from_euler("xyz", [[0, 0, 0], [80, 0, 0], [80, 50, 30]], degrees=True)
    p = AbsolutePosePath([[1, 0, 0], [1, 1, 0]], r.as_quat()[1:], [0, 1],
                         anchor_position=[0, 0, 0], anchor_quaternion=r.as_quat()[0])
    current = r[0]
    for a, b in [(0, 0.75), (0.75, 1.25), (1.25, 2)]:
        current = Rotation.from_rotvec(p.delta(a, b)[3:]) * current
    assert (current * r[-1].inv()).magnitude() < 1e-12
    assert p.angular_travel(0, 2) > np.linalg.norm(p.delta(0, 2)[3:])


def test_progress_history_interpolation_holds_and_expiry():
    h = ProgressHistory(maxlen=3)
    for sample in [(1, 0), (2, 1.5), (3, 1.5)]:
        h.append(*sample)
    assert h.at(0.9) == 0
    assert h.at(1.5) == 0.75
    assert h.at(2.5) == 1.5 and h.at(4) == 1.5
    h.append(4, 3)
    with pytest.raises(ValueError, match="predates"):
        h.at(1)
    with pytest.raises(ValueError, match="monotonic"):
        h.append(5, 2)
