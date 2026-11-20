"""Run one PTS trial with the existing ForceBench hardware/recording lifecycle."""

import argparse
import asyncio
from dataclasses import asdict, replace
import json
import math
import importlib.util
from pathlib import Path
import time

import numpy as np

from .settings import load_settings
from .trial import (Trial, add_trial_arguments, atomic_json, install_interrupt,
                    remove_interrupt, session_name, supervise)

ROOT = Path(__file__).resolve().parents[2]


async def warmup(managed, settings, timeout):
    """Compile with synthetic inputs BEFORE opening robot/cameras; discard output."""
    from forcebench.runner.adapters import ObservationAdapter, ActionAdapter
    from forcebench.runner.execution import RobotState
    from .forcebench import check_server_metadata

    capabilities = await managed.transport.connect()
    accepted = await managed.transport.configure(managed.policy_config)
    check_server_metadata(managed.transport.server_metadata, settings)
    frames = {str(c["name"]): {"image": np.zeros((224, 224, 3), dtype=np.uint8)}
              for c in managed.resolved.layers["platform"]["cameras"]}
    state = RobotState(np.zeros(3), np.array([0, 0, 0, 1]), extras={"gripper_state": 0.0})
    obs = ObservationAdapter(protocol=capabilities, runner=accepted).adapt(state.as_mapping(), frames)
    previous = getattr(managed.transport, "timeout", None)
    if previous is not None:
        managed.transport.timeout = timeout
    start = time.monotonic()
    try:
        raw = await asyncio.wait_for(managed.transport.infer(obs), timeout)
        points = ActionAdapter(protocol=capabilities, runner=accepted).adapt(raw)
        if len(points) != int(managed.transport.server_metadata["pts"]["action_horizon"]):
            raise ValueError("warmup horizon mismatch")
    finally:
        if previous is not None:
            managed.transport.timeout = settings.inference_timeout_s
    return {"duration_s": time.monotonic() - start, "input": "synthetic_black_images_zero_pose",
            "used_for_control": False, "horizon": len(points)}


async def run_trial(args):
    from forcebench.config import load_run_configuration
    from forcebench.orchestrator.runtime import build_standardized_runtime
    from forcebench.runner.transport import OpenPIWebSocketPolicyClient
    from .forcebench import PTSRunner
    from .mock import FreshMockRobot, MockTransport

    mode = "hardware" if args.hardware else "dry_run" if args.dry_run else "mock"
    if mode == "hardware":
        missing = [name for name in ("easyrobot", "flexivrdk", "pyrealsense2") if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError(f"Use the existing robot control Python via PTS_PYTHON; missing SDKs: {missing}")
    config_path = args.pts_config or ROOT / "configs" / f"{args.task}_15hz.yaml"
    settings = load_settings(config_path)
    if settings.task is not None and settings.task != args.task:
        raise ValueError("--task and PTS settings task do not match")
    settings = replace(settings, task=args.task)
    if mode == "mock" and args.pts_config is None:
        # Synthetic capture includes video startup/scheduling jitter. Keep the
        # hardware profile strict; record these mock-only values in pts_config.
        settings = replace(settings, max_input_age_s=1.0, max_observation_age_s=2.0,
                           max_state_camera_skew_s=0.2)
    if args.rate_cap is not None:
        settings = replace(settings, rate_cap=args.rate_cap)
    if args.decoder is not None:
        settings = replace(settings, decoder=args.decoder)
    run_config = args.run_config or Path(args.forcebench_root) / "configs/runs" / {
        "flip": "flipping_pi05.yaml", "polish": "polish_curve_pi05.yaml"}[args.task]
    resolved = load_run_configuration(run_config)
    if resolved.layers["task"]["name"] != {"flip": "flipping", "polish": "polish_curve"}[args.task]:
        raise ValueError("ForceBench run config does not match --task")
    # This resolved object belongs to this process. Existing YAML files stay intact.
    resolved.layers["baseline"].setdefault("visualizer", {})["enabled"] = args.visualize
    resolved.layers["baseline"]["transport"]["endpoint"] = args.endpoint
    resolved.layers["execution"]["dropout"]["enabled"] = False
    resolved.layers["execution"]["dropout"]["type"] = "none"
    resolved.layers["execution"]["gripper_command_threshold"] = settings.gripper_command_threshold
    settings.box(resolved.runner_execution_profile())
    if mode == "mock":
        transport = MockTransport(resolved.policy_protocol_config(), delay=args.mock_latency,
                                  decoder=settings.decoder, task=args.task)
    else:
        transport = OpenPIWebSocketPolicyClient(args.endpoint, timeout=args.warmup_timeout,
                                               capabilities=resolved.policy_protocol_config())
    bundle = build_standardized_runtime(
        resolved, storage_dir=args.output,
        robot=FreshMockRobot() if mode != "hardware" else None,
        transport=transport, dry_run=mode != "hardware",
    )
    managed = bundle.runner
    stop = install_interrupt()
    method = f"pts{'_k' if settings.decoder == 'stochastic_10_step' else ''}_{settings.rate_cap:g}x"
    session = {"session_id": session_name(args.session, method), "method": method, "mode": mode,
               "task": args.task, "setup_id": args.setup_id, "repeat_id": args.repeat_id}
    trial = None
    try:
        await managed.prepare(session)
        trial = Trial(managed.artifact_dir, task=args.task, method=method, mode=mode,
                      duration_s=args.duration, action_hz=settings.action_hz,
                      setup_id=args.setup_id, repeat_id=args.repeat_id,
                      criteria=args.criteria or f"manual_{args.task}_v1; see docs/REAL_ROBOT.md")
        managed.runner = PTSRunner.from_standard(managed.runner, settings=settings,
                                                log=managed.artifact_dir / "pts_events.jsonl")
        atomic_json(managed.artifact_dir / "pts_config.json", asdict(settings))
        print(f"PTS {mode}: warming policy before hardware ready", flush=True)
        warmup_record = await warmup(managed, settings, args.warmup_timeout)
        (managed.artifact_dir / "pts_warmup.json").write_text(json.dumps(warmup_record, indent=2))
        if stop.is_set():
            trial.finish("aborted", "operator_interrupt_before_ready")
            return
        print(f"PTS {mode}: ready; output {managed.artifact_dir}", flush=True)
        await managed.ready()
        if stop.is_set():
            trial.finish("aborted", "operator_interrupt_before_start")
            return
        result = await supervise(managed, trial, interrupt=stop)
        print(json.dumps({"artifact_dir": str(managed.artifact_dir), **result}, indent=2), flush=True)
    except BaseException as exc:
        if trial is not None and trial.document["outcome"] in {"prepared", "running"}:
            trial.finish("error", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            await managed.close()
        finally:
            # prepare/warmup errors can occur before ForceBench opens its preflight lifecycle.
            await transport.close()
            remove_interrupt()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mock", action="store_true", help="Synthetic policy, robot and cameras (default)")
    modes.add_argument("--dry-run", action="store_true", help="Real model server, synthetic robot/cameras")
    modes.add_argument("--hardware", action="store_true", help="Open configured robot/cameras, run ready pose and execute")
    import os
    parser.add_argument("--task", choices=("flip", "polish"), default="flip")
    parser.add_argument("--forcebench-root", type=Path,
                        default=Path(os.environ.get("PTS_FORCEBENCH_ROOT", ROOT.parent / "ForceBenchWeb")))
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--pts-config", type=Path)
    parser.add_argument("--rate-cap", type=float, help="Requires measured actuator bounds above 1x")
    parser.add_argument("--decoder", choices=("zero_noise_one_step", "stochastic_10_step"))
    parser.add_argument("--endpoint", default="ws://127.0.0.1:8025")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    add_trial_arguments(parser)
    parser.add_argument("--mock-latency", type=float, default=0.12)
    parser.add_argument("--warmup-timeout", type=float, default=300)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    if (not 0 < args.duration < float("inf") or not math.isfinite(args.mock_latency)
            or args.mock_latency < 0 or not math.isfinite(args.warmup_timeout) or args.warmup_timeout <= 0):
        parser.error("duration and timeout must be positive; latency must be nonnegative")
    asyncio.run(run_trial(args))


if __name__ == "__main__":
    main()
