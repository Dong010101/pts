"""Synthetic fixtures only: no robot SDK, pretrained model or task-success claim."""

import asyncio
from dataclasses import replace
import time

import numpy as np

from forcebench.runner.execution import MockRobotDriver


class FreshMockRobot(MockRobotDriver):
    def get_state(self):
        return replace(self.state, timestamp=time.monotonic(), wall_timestamp=time.time())

    def command_pose(self, position, quaternion, extras=None):
        super().command_pose(position, quaternion, extras)
        if extras and "gripper" in extras:
            self.state = replace(self.state, extras={**self.state.extras, "gripper_state": extras["gripper"]})


class MockTransport:
    def __init__(self, capabilities=None, *, delay=0.12, horizon=50, decoder="zero_noise_one_step"):
        self.capabilities = capabilities or {}
        self.delay, self.horizon = delay, horizon
        self.server_metadata = {"control_frequency_hz": 15, "pts": {
            "decoder": decoder, "num_steps": 1 if decoder == "zero_noise_one_step" else 10,
            "action_horizon": horizon, "action_dim": 32, "action_space": "absolute_tcp",
            "config": "SYNTHETIC_FIXTURE", "checkpoint": None,
        }}
        self.closed = False
        self.requests = []

    async def connect(self):
        self.closed = False
        return self.capabilities

    async def configure(self, config):
        return config

    async def infer(self, observation):
        self.requests.append(time.monotonic())
        await asyncio.sleep(self.delay)
        state = np.asarray(observation["observation/state"])
        action = np.tile(state, (self.horizon, 1)).astype(np.float64)
        # Gentle curved path, 1 mm per training waypoint; stays inside the mock box.
        p = np.arange(1, self.horizon + 1)
        action[:, 0] += 0.001 * p
        action[:, 1] += 0.002 * np.sin(p / 8)
        action[:, 6] = 0.04
        return {"actions": action}

    async def close(self):
        self.closed = True
