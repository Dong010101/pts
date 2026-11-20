"""Real-robot settings. Units here are metres, radians and seconds."""

from dataclasses import dataclass
import math
from pathlib import Path

import yaml

from .core import ActuatorBox, PTSConfig


@dataclass(frozen=True)
class RobotSettings:
    task: str | None = None
    action_hz: float = 15.0
    replan_steps: float = 8.0
    rate_cap: float = 1.0
    rate_grid: float = 0.25
    initial_latency_s: float = 0.15
    latency_window: int = 32
    gripper_min_s: float = 0.1
    gripper_max_s: float = 0.6
    gripper_speed_threshold: float = 0.02
    gripper_width_to_finger_speed: float = 0.5
    gripper_command_threshold: float = 0.02
    max_input_age_s: float = 0.25
    max_observation_age_s: float = 1.0
    inference_timeout_s: float = 5.0
    observation_camera: str = "global"
    max_state_camera_skew_s: float = 0.05
    decoder: str = "zero_noise_one_step"
    actuator: dict | None = None
    notes: str = ""

    def __post_init__(self):
        if self.task not in {None, "flip", "polish"}:
            raise ValueError("task must be flip or polish")
        for name in ("action_hz", "max_input_age_s", "max_observation_age_s", "inference_timeout_s", "max_state_camera_skew_s"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("initial_latency_s", "gripper_min_s", "gripper_max_s", "gripper_width_to_finger_speed"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.latency_window < 1 or self.gripper_min_s > self.gripper_max_s:
            raise ValueError("invalid latency history or gripper window")
        if self.decoder not in {"zero_noise_one_step", "stochastic_10_step"}:
            raise ValueError("unsupported decoder")
        self.core_config()

    def core_config(self):
        return PTSConfig(
            replan_steps=self.replan_steps, rate_cap=self.rate_cap, rate_grid=self.rate_grid,
            latency_steps=math.ceil(self.initial_latency_s * self.action_hz),
            gripper_min_steps=math.ceil(self.gripper_min_s * self.action_hz),
            gripper_max_steps=math.ceil(self.gripper_max_s * self.action_hz),
            gripper_speed_threshold=self.gripper_speed_threshold,
            gripper_command_threshold=self.gripper_command_threshold,
        )

    def box(self, profile):
        velocity_limit = profile.max_vel / self.action_hz
        rotation_limit = profile.max_angular_vel / self.action_hz
        if self.actuator is None:
            if self.rate_cap > 1:
                raise ValueError("rate_cap > 1 needs an actuator box measured on this robot (metres per action step)")
            return ActuatorBox((velocity_limit,) * 3, (velocity_limit,) * 3, velocity_limit, rotation_limit)
        data = dict(self.actuator)
        if data.pop("units", None) != "metres_per_action_step":
            raise ValueError("real-robot actuator bounds must use metres_per_action_step, not LIBERO action units")
        data.pop("source", None)
        positive, negative = tuple(data.pop("positive")), tuple(data.pop("negative"))
        translation_norm = min(float(data.pop("max_translation_norm", velocity_limit)), velocity_limit)
        rotation_travel = min(float(data.pop("max_rotation_travel", rotation_limit)), rotation_limit)
        if data:
            raise ValueError(f"unknown actuator settings: {sorted(data)}")
        return ActuatorBox(positive, negative, translation_norm, rotation_travel)


def load_settings(path: str | Path) -> RobotSettings:
    with open(path) as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("PTS config must be a mapping")
    return RobotSettings(**data)

