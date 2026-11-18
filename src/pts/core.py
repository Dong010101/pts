"""PTS R1--R4, independent of model framework, network and robot drivers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .paths import IncrementPath


@dataclass(frozen=True)
class ActuatorBox:
    """Signed translation bounds in *the path's units*, optional real-robot guards."""

    positive: tuple[float, float, float]
    negative: tuple[float, float, float]
    max_translation_norm: float | None = None
    max_rotation_travel: float | None = None

    def __post_init__(self):
        for values in (self.positive, self.negative):
            if len(values) != 3 or not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError("actuator bounds must contain three finite positive values")
        for value in (self.max_translation_norm, self.max_rotation_travel):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("optional norm bounds must be finite and positive")

    def connected_feasible(self, path: IncrementPath, p: float, rate: float) -> bool:
        # Paper requires ALL rates in [1,r], not just the selected grid endpoints.
        # Translation is piecewise linear and a box/ball is convex, so testing
        # every path knot in this interval is sufficient, including non-grid knots.
        lower = min(1.0, rate)
        for end in path.breakpoints(p + lower, p + rate):
            delta = path.delta(p, end)
            if np.any(delta[:3] > np.asarray(self.positive) + 1e-10):
                return False
            if np.any(delta[:3] < -np.asarray(self.negative) - 1e-10):
                return False
            if self.max_translation_norm is not None and np.linalg.norm(delta[:3]) > self.max_translation_norm + 1e-10:
                return False
        # Total angular travel is a conservative bound even for noncommuting SO(3) motion.
        return self.max_rotation_travel is None or path.angular_travel(p, p + rate) <= self.max_rotation_travel + 1e-10


@dataclass(frozen=True)
class PTSConfig:
    replan_steps: float = 8.0
    rate_cap: float = 1.5
    rate_grid: float = 0.25
    latency_steps: float = 2.0
    gripper_min_steps: int = 2
    gripper_max_steps: int = 12
    gripper_speed_threshold: float = 0.02
    gripper_command_threshold: float = 0.0

    def __post_init__(self):
        for name in ("replan_steps", "rate_cap", "rate_grid"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.rate_cap < 1:
            raise ValueError("PTS supports rates >= 1")
        for name in ("latency_steps", "gripper_speed_threshold", "gripper_command_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.gripper_min_steps <= self.gripper_max_steps:
            raise ValueError("invalid gripper window")


@dataclass(frozen=True)
class RequestToken:
    sequence: int
    consumed_at_observation: float


@dataclass(frozen=True)
class Command:
    arm: np.ndarray
    gripper: float | None
    rate: float
    start: float
    end: float
    reason: str
    gripper_window: bool
    feasibility_cap: float


class PTSEngine:
    """One step per action/control interval, as in Algorithm 1.

    total_consumed is commanded path progress, not elapsed steps or measured
    distance. A hardware adapter must finish each command before another step
    or accepting a result. Request tokens can use timestamped earlier progress.
    """

    def __init__(self, config: PTSConfig, box: ActuatorBox, *, initial_gripper=None):
        self.config, self.box = config, box
        self.path: IncrementPath | None = None
        self.position = 0.0
        self.total_consumed = 0.0
        self.since_switch = 0.0
        self.pending: RequestToken | None = None
        self.tick = 0
        self.sequence = 0
        self.last_gripper = initial_gripper
        self.last_flip: int | None = None
        self.latency_steps = config.latency_steps

    @property
    def feasibility_cap(self) -> float:
        if self.path is None:
            return 0.0
        if self.latency_steps == 0:
            return math.inf
        return max(0.0, (self.path.horizon - self.config.replan_steps) / (2 * self.latency_steps))

    @property
    def request_due(self) -> bool:
        return self.pending is None and (
            self.path is None
            or self.since_switch >= self.config.replan_steps - 1e-9
            or self.position >= self.path.horizon - 1e-9
            or self.feasibility_cap < 1
        )

    def begin_request(self, consumed_at_observation: float | None = None) -> RequestToken:
        if self.pending is not None:
            raise RuntimeError("only one inference request may be in flight")
        progress = self.total_consumed if consumed_at_observation is None else consumed_at_observation
        if not math.isfinite(progress) or not 0 <= progress <= self.total_consumed + 1e-8:
            raise ValueError("observation progress must refer to already executed motion")
        self.sequence += 1
        self.pending = RequestToken(self.sequence, float(progress))
        return self.pending

    def accept(self, token: RequestToken, path: IncrementPath) -> float:
        if token != self.pending:
            raise ValueError("inference response does not match the outstanding request")
        if path.horizon <= self.config.replan_steps:
            raise ValueError("prediction horizon must exceed replanning interval")
        consumed = self.total_consumed - token.consumed_at_observation
        self.path = path
        self.position = min(path.horizon, consumed)
        self.since_switch = 0.0
        self.pending = None
        return consumed

    def set_latency(self, steps: float):
        if not math.isfinite(steps) or steps < 0:
            raise ValueError("latency must be finite and nonnegative")
        self.latency_steps = float(steps)

    def step(self, *, gripper_speed: float | None = None) -> Command:
        if gripper_speed is not None and not math.isfinite(gripper_speed):
            raise ValueError("gripper speed must be finite or absent")
        elapsed = None if self.last_flip is None else self.tick - self.last_flip
        window = elapsed is not None and 1 <= elapsed <= self.config.gripper_max_steps and (
            elapsed <= self.config.gripper_min_steps
            or gripper_speed is None  # missing feedback: keep the full window
            or abs(gripper_speed) > self.config.gripper_speed_threshold
        )
        p = self.position
        cap = self.feasibility_cap
        reason = "rate_cap"
        rate = 0.0
        if self.path is None or p >= self.path.horizon - 1e-9:
            reason = "chunk_exhausted"
        elif cap < 1:
            reason = "latency_infeasible"
        else:
            remaining = self.path.horizon - p
            maximum = min(1.0 if window else self.config.rate_cap, cap, remaining)
            base = min(1.0, remaining)
            if not self.box.connected_feasible(self.path, p, base):
                # Silently clipping would destroy R2's conservation property.
                raise ValueError("even the 1x action exceeds actuator bounds; calibrate or fix the base policy")
            rate = base
            for candidate in np.arange(1.0 + self.config.rate_grid, maximum + 1e-9, self.config.rate_grid):
                if not self.box.connected_feasible(self.path, p, float(candidate)):
                    reason = "actuator"
                    break
                rate = float(candidate)
            if window:
                reason = "gripper"
            elif cap < self.config.rate_cap:
                reason = "latency"
            if remaining < 1:
                reason = "final_fraction"
        arm = np.zeros(6) if rate == 0 else self.path.delta(p, p + rate)
        gripper = self.last_gripper if rate == 0 else self.path.gripper(p + rate)
        if gripper is not None:
            if self.last_gripper is not None and abs(gripper - self.last_gripper) > self.config.gripper_command_threshold:
                self.last_flip = self.tick
            self.last_gripper = gripper
        self.position += rate
        self.total_consumed += rate
        self.since_switch += rate
        self.tick += 1
        return Command(arm, gripper, rate, p, self.position, reason, bool(window), cap)

