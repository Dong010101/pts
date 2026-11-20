"""Real-robot PTS execution using ForceBench's existing drivers and adapters.

No DTW or fixed-frequency replanning is used. Low-rate PTS decisions are
expanded along the predicted pose path at the existing driver command rate.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, replace
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from forcebench.runner.adapters import ActionAdapter, ObservationAdapter
from forcebench.runner.execution import MotionExecutor, ScheduledPose
from forcebench.runner.lifecycle import (
    StandardRunner, RunnerResult, _HookDispatcher,
    _reject_stale_observation, _validate_policy_waypoints, _validate_scheduled_poses,
)

from .clock import ProgressHistory
from .core import PTSEngine
from .paths import AbsolutePosePath
from .settings import RobotSettings


async def call_io(function, *args):
    """Do blocking SDK calls off the loop, drain them before cleanup on cancellation."""
    if inspect.iscoroutinefunction(function):
        return await function(*args)
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
    return await result if inspect.isawaitable(result) else result


def check_server_metadata(metadata, settings):
    if settings.task is not None and metadata.get("task") != settings.task:
        raise ValueError(f"server task must match {settings.task}")
    pts = metadata.get("pts", {})
    if pts.get("decoder") != settings.decoder:
        raise ValueError(f"server decoder {pts.get('decoder')!r} != required {settings.decoder!r}; use pts.server")
    expected_steps = 1 if settings.decoder == "zero_noise_one_step" else 10
    if pts.get("num_steps") != expected_steps:
        raise ValueError("server integration-step metadata is inconsistent")
    if pts.get("action_space") != "absolute_tcp":
        raise ValueError("ForceBench TCP adapter requires an absolute_tcp server")
    if int(pts.get("action_horizon", 0)) <= settings.replan_steps:
        raise ValueError("model horizon must exceed replanning interval")
    if not math.isclose(float(metadata.get("control_frequency_hz", 0)), settings.action_hz):
        raise ValueError("server training control_frequency_hz must match PTS action_hz")


class PTSRunner(StandardRunner):
    def __init__(self, *, pts_settings: RobotSettings, pts_log: str | Path, **kwargs):
        super().__init__(**kwargs)
        self.settings = pts_settings
        self.pts_log = Path(pts_log)
        if not math.isclose(self.profile.action_waypoint_hz, pts_settings.action_hz):
            raise ValueError("PTS action_hz must match the policy's trained waypoint frequency")
        if self.profile.control_hz < pts_settings.action_hz:
            raise ValueError("driver command rate must be >= the PTS action rate")
        if self.profile.trajectory_time_scale != 1:
            raise ValueError("PTS owns time scaling; set ForceBench trajectory_time_scale to 1")
        if self.profile.action_control_mode != "position":
            raise ValueError("this adapter supports TCP position actions; force/impedance action extras need a separate adapter")
        if not math.isclose(self.profile.gripper_command_threshold, pts_settings.gripper_command_threshold):
            raise ValueError("PTS and driver gripper_command_threshold must match")
        self.box = pts_settings.box(self.profile)

    @classmethod
    def from_standard(cls, runner, *, settings, log):
        names = (
            "transport", "robot", "camera_hub", "profile", "config", "observation_adapter",
            "action_adapter", "recorder_hook", "state_recorder_hook", "policy_action_recorder_hook",
            "lifecycle_hook", "visualizer",
        )
        return cls(pts_settings=settings, pts_log=log, **{name: getattr(runner, name) for name in names})

    async def run(self, *, max_planner_steps=None):
        started_at = time.time()
        settings = self.settings
        camera_started, robot_started = self._camera_prestarted, self._robot_prestarted
        inference = sampler = None
        hooks = _HookDispatcher()
        log = None
        error = None
        planner_steps = commands_sent = 0
        io_lock = asyncio.Lock()
        states = deque(maxlen=max(100, int((self.profile.state_sample_hz or 200) * 10)))
        progress = ProgressHistory()
        latencies = deque(maxlen=settings.latency_window)
        first_state = asyncio.Event()

        async def robot_call(name, *args):
            async with io_lock:
                return await call_io(getattr(self.robot, name), *args)

        class DriverProxy:
            async def command_pose(_self, *args):
                return await robot_call("command_pose", *args)

        def emit(event, **values):
            record = {"event": event, "monotonic_s": time.monotonic(), "wall_s": time.time(), **values}
            # Infinity is useful mathematically but invalid in portable JSON.
            record = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                      for key, value in record.items()}
            log.write(json.dumps(record, allow_nan=False) + "\n")

        def record_control(pose, measured):
            if self.recorder_hook:
                pose = replace(pose, extras={**pose.extras, "_recorded_monotonic_s": time.monotonic(),
                                             "_recorded_wall_s": time.time()})
                hooks.submit(self.recorder_hook, pose, measured)

        try:
            self.pts_log.parent.mkdir(parents=True, exist_ok=True)
            log = self.pts_log.open("w")
            hooks.start()
            capabilities = self._preflight_capabilities
            if capabilities is None:
                capabilities = await self.transport.connect()
                accepted = await self.transport.configure(self.config)
            else:
                accepted = self._preflight_config or self.config
            check_server_metadata(self.transport.server_metadata, settings)
            self.observation_adapter = self.observation_adapter or ObservationAdapter(protocol=capabilities, runner=accepted)
            self.action_adapter = self.action_adapter or ActionAdapter(protocol=capabilities, runner=accepted)
            emit("metadata", settings=asdict(settings), profile=asdict(self.profile),
                 server=self.transport.server_metadata, rotation_adaptation="SO3_relative_continuation")
            if not camera_started:
                camera_started = True
                await call_io(self.camera_hub.start)
            if not robot_started:
                robot_started = True
                await robot_call("configure", self.profile)
                await robot_call("start")
            if self.visualizer:
                self.visualizer.start()

            async def sample_states():
                period = 1 / (self.profile.state_sample_hz or self.profile.control_hz)
                while True:
                    state = await robot_call("get_state")
                    states.append(state)
                    if self.state_recorder_hook:
                        hooks.submit(self.state_recorder_hook, state)
                    first_state.set()
                    await asyncio.sleep(period)

            sampler = asyncio.create_task(sample_states())
            # Propagate startup sampling errors instead of waiting indefinitely.
            first_wait = asyncio.create_task(first_state.wait())
            try:
                done, _ = await asyncio.wait((sampler, first_wait), timeout=settings.inference_timeout_s,
                                            return_when=asyncio.FIRST_COMPLETED)
                if sampler in done:
                    sampler.result()
                if not first_state.is_set():
                    raise TimeoutError("robot did not provide an initial state")
            finally:
                first_wait.cancel()
                await asyncio.gather(first_wait, return_exceptions=True)
            initial = states[-1]
            engine = PTSEngine(settings.core_config(), self.box, initial_gripper=initial.extras.get("gripper_state"))
            progress.append(time.monotonic(), 0)
            anchor_position = initial.position.copy()
            anchor_rotation = Rotation.from_quat(initial.quaternion)
            last_pose = ScheduledPose(anchor_position, initial.quaternion, 0, {})
            executor = MotionExecutor(DriverProxy(), self.profile, recorder_hook=record_control)
            if getattr(self, "trial", None) is not None:
                self.trial.start()

            async def infer_chunk():
                frames = await call_io(self.camera_hub.read)
                camera = frames.get(settings.observation_camera)
                if not isinstance(camera, dict) or "monotonic_timestamp" not in camera:
                    raise ValueError("PTS requires the reference camera's monotonic capture timestamp")
                observed_at = float(camera["monotonic_timestamp"])
                if not math.isfinite(observed_at) or observed_at > time.monotonic() + 0.01:
                    raise ValueError("camera timestamp is not in the runner's monotonic clock domain")
                state = min(states, key=lambda s: abs(s.timestamp - observed_at))
                if abs(state.timestamp - observed_at) > settings.max_state_camera_skew_s:
                    raise ValueError("robot state and reference image are not sufficiently synchronized")
                timestamps = [state.timestamp] + [float(f["monotonic_timestamp"]) for f in frames.values()]
                if any(not math.isfinite(t) or t > time.monotonic() + 0.01 for t in timestamps):
                    raise ValueError("invalid observation timestamp")
                oldest = min(timestamps)
                _reject_stale_observation(oldest, limit_s=settings.max_input_age_s, context="PTS input")
                token = engine.begin_request(progress.at(observed_at))
                # Snapshot preparation and inference are asynchronous. Progress during
                # image preprocessing is counted as well as network/model latency.
                observation = await asyncio.to_thread(self.observation_adapter.adapt, state.as_mapping(), frames)
                request_time = time.monotonic()
                emit("request", sequence=token.sequence, observed_at=observed_at,
                     observed_progress=token.consumed_at_observation)
                raw = await asyncio.wait_for(self.transport.infer(observation), settings.inference_timeout_s)
                waypoints = await asyncio.to_thread(self.action_adapter.adapt, raw)
                if len(waypoints) != int(self.transport.server_metadata["pts"]["action_horizon"]):
                    raise ValueError("returned horizon differs from the model metadata")
                _validate_policy_waypoints(self.profile, state.position, state.quaternion, waypoints)
                if any(set(w.extras) - {"gripper"} for w in waypoints):
                    raise ValueError("PTS TCP adapter does not silently discard non-gripper action extras")
                path = AbsolutePosePath(
                    [w.position for w in waypoints], [w.quaternion for w in waypoints],
                    [w.extras["gripper"] for w in waypoints],
                    anchor_position=state.position, anchor_quaternion=state.quaternion,
                )
                latency_ms = (time.monotonic() - request_time) * 1000
                if self.policy_action_recorder_hook:
                    hooks.submit(self.policy_action_recorder_hook, raw, waypoints,
                                 {"infer_latency_ms": latency_ms, "recorded_monotonic_s": time.monotonic(),
                                  "recorded_wall_s": time.time()})
                if self.visualizer:
                    self._submit_visualization(frames, state, waypoints)
                return token, path, observed_at, oldest, latency_ms

            action_period = 1 / settings.action_hz
            driver_period = 1 / self.profile.control_hz
            segment = None
            next_tick = time.monotonic()
            previous_width = None
            emit("start")
            await self._emit("START", {"started_at": started_at, "executor": "PTS"})
            while max_planner_steps is None or planner_steps < max_planner_steps:
                if sampler.done():
                    sampler.result()
                now = time.monotonic()
                # Finish the entire previous logical control step before a replan
                # can be installed. Inside it, use its *piecewise* pose path, not a
                # straight chord between endpoints that would cut corners.
                if segment is not None:
                    command, path, t0, total0, position0, rotation0 = segment
                    fraction = min(1.0, max(0.0, (now - t0) / action_period))
                    if command.rate > 0:
                        delta = path.delta(command.start, command.start + command.rate * fraction)
                        position = position0 + delta[:3]
                        rotation = Rotation.from_rotvec(delta[3:]) * rotation0
                    else:
                        position, rotation = position0, rotation0
                    extras = {} if command.gripper is None else {"gripper": command.gripper}
                    pose = ScheduledPose(position, rotation.as_quat(), 0, extras)
                    _validate_scheduled_poses(self.profile, last_pose.position, last_pose.quaternion, [pose])
                    await executor.execute_pose(pose)
                    commands_sent += 1
                    last_pose = pose
                    progress.append(time.monotonic(), total0 + command.rate * fraction)
                    if fraction >= 1:
                        emit("segment_complete", total_consumed=total0 + command.rate, rate=command.rate)
                        anchor_position, anchor_rotation = position.copy(), rotation
                        segment = None
                if segment is None:
                    if inference is not None and inference.done():
                        token, path, observed_at, oldest, latency_ms = inference.result()
                        inference = None
                        _reject_stale_observation(oldest, limit_s=settings.max_observation_age_s, context="PTS response")
                        age_s = time.monotonic() - observed_at
                        # Conservative causal max over recent observed delivery ages.
                        # Paper uses fixed d; this is the explicit variable-latency adaptation.
                        latencies.append(math.ceil(age_s * settings.action_hz))
                        engine.set_latency(max(math.ceil(settings.initial_latency_s * settings.action_hz), max(latencies)))
                        consumed = engine.accept(token, path)
                        emit("switch", sequence=token.sequence, entry_progress=consumed,
                             total_consumed=engine.total_consumed, observation_age_s=age_s,
                             infer_latency_ms=latency_ms, latency_steps=engine.latency_steps)
                    if inference is None and engine.request_due:
                        inference = asyncio.create_task(infer_chunk())
                        # Give snapshot capture a chance before scheduling the next step.
                        await asyncio.sleep(0)
                    state = states[-1]
                    width = state.extras.get("gripper_state")
                    speed = None
                    if width is not None:
                        if previous_width is not None and state.timestamp > previous_width[0]:
                            speed = settings.gripper_width_to_finger_speed * (float(width) - previous_width[1]) / (state.timestamp - previous_width[0])
                        previous_width = (state.timestamp, float(width))
                    total0 = engine.total_consumed
                    command = engine.step(gripper_speed=speed)
                    segment = (command, engine.path, time.monotonic(), total0, anchor_position.copy(), anchor_rotation)
                    emit("step", tick=engine.tick - 1, rate=command.rate, start=command.start, end=command.end,
                         reason=command.reason, gripper_window=command.gripper_window,
                         feasibility_cap=command.feasibility_cap, gripper=command.gripper,
                         arm_delta=command.arm.tolist(), has_plan=engine.path is not None)
                    # Match the paper: the gripper command at the END of the
                    # consumed interval is issued with this step's arm command.
                    pose = ScheduledPose(anchor_position, anchor_rotation.as_quat(), 0,
                                         {} if command.gripper is None else {"gripper": command.gripper})
                    await executor.execute_pose(pose)
                    commands_sent += 1
                planner_steps += 1
                # Wake at both motor ticks and logical action boundaries. Otherwise
                # rounding every 1/15 s step to 7 motor ticks silently yields 14.3 Hz.
                now = time.monotonic()
                if now > next_tick + driver_period:
                    emit("driver_overrun", lag_s=now - next_tick)
                    next_tick = now
                if next_tick <= now:
                    next_tick += driver_period
                deadline = min(next_tick, segment[2] + action_period) if segment is not None else next_tick
                await asyncio.sleep(max(0.0, deadline - time.monotonic()))
        except BaseException as exc:
            error = exc
            if log:
                emit("cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                     kind=type(exc).__name__, message=str(exc))
            raise
        finally:
            execution_end = time.monotonic()
            for task in (inference, sampler):
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(t for t in (inference, sampler) if t is not None), return_exceptions=True)
            cleanup_errors = []
            # Stop motion before draining logging or releasing cameras/network.
            for function in ([self.robot.safe_stop] if robot_started else []) + ([self.camera_hub.stop] if camera_started else []):
                try:
                    await call_io(function)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            try:
                await hooks.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                await self.transport.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
            if self.visualizer:
                try:
                    self.visualizer.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if log:
                emit("stop", planner_steps=planner_steps, commands_sent=commands_sent,
                     execution_end_monotonic_s=execution_end,
                     executed_progress=progress.samples[-1][1] if progress.samples else 0,
                     cleanup_errors=[str(e) for e in cleanup_errors])
                log.close()
            self._preflight_capabilities = self._preflight_config = None
            self._camera_prestarted = self._robot_prestarted = False
            if cleanup_errors:
                raise cleanup_errors[0]
        return RunnerResult(planner_steps, commands_sent, started_at, time.time())


async def prepare_pts_trial(managed, session, settings):
    """Use the existing ready/start/stop, homing, camera and HDF5 lifecycle."""
    await managed.prepare(session)
    managed.runner = PTSRunner.from_standard(managed.runner, settings=settings,
                                            log=managed.artifact_dir / "pts_events.jsonl")
    # Fail mismatched decoders before managed.ready() can home the robot.
    await managed.transport.connect()
    check_server_metadata(managed.transport.server_metadata, settings)
    (managed.artifact_dir / "pts_config.json").write_text(json.dumps(asdict(settings), indent=2))
    return managed.runner
