"""Offline diagnostics on absolute TCP predictions, never hardware execution.

The prediction horizon remains H=50. A prefix ends at the first command whose
consumed progress reaches s=8 (1.5x can reach 9, not artificially truncate at 8).
"""

from dataclasses import asdict

import numpy as np
from scipy.spatial.transform import Rotation

from .core import ActuatorBox, PTSConfig, PTSEngine
from .paths import AbsolutePosePath


CONDITIONS = ("baseline_10step", "pts_1x", "pts_1p5x", "pts_2x")
RATES = {"baseline_10step": 1.0, "pts_1x": 1.0, "pts_1p5x": 1.5, "pts_2x": 2.0}


def pose_path(actions, state):
    actions, state = np.asarray(actions), np.asarray(state)
    if actions.ndim != 2 or actions.shape[1] != 7 or state.shape != (7,):
        raise ValueError("Expected absolute TCP actions (H,7) and state (7,)")
    return AbsolutePosePath(actions[:, :3], Rotation.from_rotvec(actions[:, 3:6]).as_quat(), actions[:, 6],
                            anchor_position=state[:3], anchor_quaternion=Rotation.from_rotvec(state[3:6]).as_quat())


def make_engine(settings, rate, gripper, *, latency=None):
    core = {**settings["pts_core"], "rate_cap": rate}
    if latency is not None:
        core["latency_steps"] = latency
    return PTSEngine(PTSConfig(**core), ActuatorBox(**settings["offline_actuator_box"]), initial_gripper=gripper)


def execute_prefix(actions, state, condition, settings):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition == "baseline_10step":
        n = int(settings["execution_horizon"])
        return np.asarray(actions[:n]).copy(), np.arange(1, n + 1, dtype=float), {
            "rate_cap": 1.0, "rates": [1.0] * n, "reasons": ["baseline"] * n,
            "gripper_windows": [False] * n, "consumed_progress": float(n),
        }
    path = pose_path(actions, state)
    engine = make_engine(settings, RATES[condition], state[6])
    engine.accept(engine.begin_request(), path)
    position, rotation = np.array(state[:3], dtype=float), Rotation.from_rotvec(state[3:6])
    poses, progress, rates, reasons, windows = [], [], [], [], []
    while engine.since_switch < settings["pts_core"]["replan_steps"] - 1e-9:
        # No finger-speed observation in this fixed-input fixture. The production
        # fallback conservatively keeps the complete window after a command flip.
        command = engine.step(gripper_speed=None)
        if command.rate <= 0:
            raise ValueError(f"Offline prefix cannot advance: {command.reason}")
        position += command.arm[:3]
        rotation = Rotation.from_rotvec(command.arm[3:]) * rotation
        poses.append(np.r_[position, rotation.as_rotvec(), command.gripper])
        progress.append(command.end)
        rates.append(command.rate)
        reasons.append(command.reason)
        windows.append(command.gripper_window)
    return np.asarray(poses), np.asarray(progress), {
        "rate_cap": RATES[condition], "rates": rates, "reasons": reasons,
        "gripper_windows": windows, "consumed_progress": engine.since_switch,
    }


def delayed_replay(actions, state, settings, *, rate=1.5, delays=(2,), ticks=120):
    """Deterministic scheduler test with cached, observation-independent chunks.

    This is NOT a closed-loop robot simulation or a sequential dataset replay.
    Each new request receives the same frozen pose path; only bookkeeping is tested.
    """
    if not delays or any(d < 1 or int(d) != d for d in delays):
        raise ValueError("delays must be positive integer control steps")
    path = pose_path(actions, state)
    engine = make_engine(settings, rate, state[6], latency=delays[0])
    engine.accept(engine.begin_request(), path)  # initial inference excluded, as in paper
    pending, events, request_index = None, [], 0
    delivered, max_error, moving_while_pending = 0, 0.0, 0
    for tick in range(ticks):
        if pending is not None and pending[0] == tick:
            _, token, since, expected_sum = pending
            consumed = engine.accept(token, path)
            error = abs(consumed - expected_sum)
            max_error = max(max_error, error)
            events.append({"event": "switch", "tick": tick, "request_tick": since,
                           "entry_progress": consumed, "summed_rates_since_request": expected_sum,
                           "entry_clamped_to_horizon": engine.position, "error": error})
            pending, delivered = None, delivered + 1
        if engine.request_due:
            assert pending is None
            token = engine.begin_request()
            delay = delays[request_index % len(delays)]
            request_index += 1
            pending = [tick + delay, token, tick, 0.0]
            events.append({"event": "request", "tick": tick, "delay": delay,
                           "total_progress": engine.total_consumed, "since_switch": engine.since_switch})
        command = engine.step(gripper_speed=None)
        if pending is not None:
            pending[3] += command.rate
            moving_while_pending += int(command.rate > 0)
        events.append({"event": "step", "tick": tick, "rate": command.rate, "start": command.start,
                       "end": command.end, "reason": command.reason,
                       "pending": pending is not None, "gripper_window": command.gripper_window})
    return {"events": events, "summary": {
        "ticks": ticks, "switches": delivered, "request_count": request_index,
        "max_entry_error": max_error, "moving_steps_during_inference": moving_while_pending,
        "hold_steps": sum(e["rate"] == 0 for e in events if e["event"] == "step"),
        "one_in_flight": True, "delays": list(delays),
        "scope": "cached chunk scheduler fixture; no causal observations or robot dynamics",
    }}
