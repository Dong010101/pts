import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest

from forcebench.config import load_run_configuration
from forcebench.runner.execution import ExecutionProfile
from pts.forcebench import PTSRunner, check_server_metadata
from pts.mock import FreshMockRobot, MockTransport
from pts.settings import load_settings, RobotSettings

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT.parent / "ForceBenchWeb/configs/runs/peg_in_hole_v2_200sets_15hz_pi05.yaml"


class Cameras:
    def start(self):
        self.running = True
    def read(self):
        stamp = time.monotonic()
        return {name: {"image": np.zeros((224, 224, 3), np.uint8), "monotonic_timestamp": stamp}
                for name in ("global", "wrist")}
    def stop(self):
        self.running = False


def make_runner(tmp_path, transport_type=MockTransport, delay=0.12):
    resolved = load_run_configuration(RUN)
    robot, cameras = FreshMockRobot(), Cameras()
    transport = transport_type(resolved.policy_protocol_config(), delay=delay)
    settings = load_settings(ROOT / "configs/mock_15hz.yaml")
    runner = PTSRunner(pts_settings=settings, pts_log=tmp_path / "events.jsonl",
                       transport=transport, robot=robot, camera_hub=cameras,
                       profile=resolved.runner_execution_profile(), config=resolved.policy_runtime_config())
    return runner


def test_async_runner_moves_during_inference_and_logs_continuation(tmp_path):
    runner = make_runner(tmp_path)
    result = asyncio.run(runner.run(max_planner_steps=240))
    events = [json.loads(x) for x in runner.pts_log.read_text().splitlines()]
    switches = [e for e in events if e["event"] == "switch"]
    assert len(switches) >= 3
    assert switches[0]["entry_progress"] == 0
    assert all(e["entry_progress"] > 0 for e in switches[1:])
    requests = {e["sequence"]: e for e in events if e["event"] == "request"}
    for switch in switches:
        assert switch["entry_progress"] == pytest.approx(
            switch["total_consumed"] - requests[switch["sequence"]]["observed_progress"])
    assert any(e["event"] == "step" and e["rate"] == 1.5 for e in events)
    assert runner.robot.commands[-1][0][0] > runner.robot.commands[0][0][0]
    assert result.commands_sent > 200 and runner.robot.safely_stopped
    assert runner.transport.closed and not runner.camera_hub.running
    steps = [e["monotonic_s"] for e in events if e["event"] == "step"]
    # Scheduler must not systematically round 15 Hz down to 100/7 Hz.
    assert np.median(np.diff(steps)) < 0.070


def test_bad_response_stops_robot_and_closes_resources(tmp_path):
    class BadTransport(MockTransport):
        async def infer(self, obs):
            raw = await super().infer(obs)
            raw["actions"][0, 0] = np.nan
            return raw
    runner = make_runner(tmp_path, BadTransport)
    with pytest.raises(ValueError):
        asyncio.run(runner.run(max_planner_steps=100))
    assert runner.robot.safely_stopped and runner.transport.closed
    assert not runner.camera_hub.running


def test_cancel_in_flight_request_stops_motion(tmp_path):
    runner = make_runner(tmp_path, delay=3)
    async def scenario():
        task = asyncio.create_task(runner.run())
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert runner.robot.safely_stopped and runner.transport.closed
    events = [json.loads(line)["event"] for line in runner.pts_log.read_text().splitlines()]
    assert "cancelled" in events and "error" not in events


def test_rejects_wrong_decoder_before_start(tmp_path):
    runner = make_runner(tmp_path)
    runner.transport.server_metadata["pts"]["decoder"] = "stochastic_10_step"
    with pytest.raises(ValueError, match="server decoder"):
        asyncio.run(runner.run(max_planner_steps=1))
    assert not runner.robot.started


def test_real_settings_require_measured_units_for_speedup():
    with pytest.raises(ValueError, match="measured"):
        RobotSettings(rate_cap=1.5).box(ExecutionProfile())
    with pytest.raises(ValueError, match="units"):
        RobotSettings(rate_cap=1.5, actuator={"positive": [2, 3, 3], "negative": [2, 3, 2]}).box(ExecutionProfile())


def test_rejects_model_frequency_mismatch():
    metadata = MockTransport().server_metadata
    metadata["control_frequency_hz"] = 20
    with pytest.raises(ValueError, match="control_frequency_hz"):
        check_server_metadata(metadata, RobotSettings(action_hz=15))
