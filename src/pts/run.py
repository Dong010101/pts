"""Run one PTS trial with the existing ForceBench hardware/recording lifecycle."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import importlib.util
from pathlib import Path
import signal
import time

import numpy as np

from .settings import load_settings

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
    from .forcebench import prepare_pts_trial
    from .mock import FreshMockRobot, MockTransport

    mode = "hardware" if args.hardware else "dry_run" if args.dry_run else "mock"
    if mode == "hardware":
        missing = [name for name in ("easyrobot", "flexivrdk", "pyrealsense2") if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError(f"Use the existing robot control Python via PTS_PYTHON; missing SDKs: {missing}")
    config_path = args.pts_config or ROOT / "configs" / ("mock_15hz.yaml" if mode == "mock" else "robot_15hz.yaml")
    settings = load_settings(config_path)
    resolved = load_run_configuration(args.run_config)
    # This resolved object belongs to this process. Existing YAML files stay intact.
    resolved.layers["baseline"].setdefault("visualizer", {})["enabled"] = args.visualize
    resolved.layers["baseline"]["transport"]["endpoint"] = args.endpoint
    resolved.layers["execution"]["dropout"]["enabled"] = False
    resolved.layers["execution"]["dropout"]["type"] = "none"
    resolved.layers["execution"]["gripper_command_threshold"] = settings.gripper_command_threshold
    if mode == "mock":
        transport = MockTransport(resolved.policy_protocol_config(), delay=args.mock_latency,
                                  decoder=settings.decoder)
    else:
        transport = OpenPIWebSocketPolicyClient(args.endpoint, timeout=args.warmup_timeout,
                                               capabilities=resolved.policy_protocol_config())
    bundle = build_standardized_runtime(
        resolved, storage_dir=args.output,
        robot=FreshMockRobot() if mode != "hardware" else None,
        transport=transport, dry_run=mode != "hardware",
    )
    managed = bundle.runner
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    session_id = args.session or datetime.now(timezone.utc).strftime("pts_%Y%m%dT%H%M%S_%fZ")
    session = {"session_id": session_id, "method": "PTS", "mode": mode}
    try:
        await prepare_pts_trial(managed, session, settings)
        print(f"PTS {mode}: warming policy before hardware ready", flush=True)
        warmup_record = await warmup(managed, settings, args.warmup_timeout)
        (managed.artifact_dir / "pts_warmup.json").write_text(json.dumps(warmup_record, indent=2))
        if stop.is_set():
            return
        print(f"PTS {mode}: ready; output {managed.artifact_dir}", flush=True)
        await managed.ready()
        if stop.is_set():
            return
        await managed.start_at(time.time())
        deadline = time.monotonic() + args.duration
        while not stop.is_set() and time.monotonic() < deadline:
            status = await managed.status()
            if status["state"] in {"stopped", "error"}:
                break
            try:
                await asyncio.wait_for(stop.wait(), min(0.1, max(0.001, deadline - time.monotonic())))
            except TimeoutError:
                pass
        await managed.stop("operator_interrupt" if stop.is_set() else "duration_limit")
        status = await managed.status()
        if status["error"]:
            raise RuntimeError(status["error"])
        result = await managed.finalize()
        print(json.dumps({"artifact_dir": str(managed.artifact_dir), "state": result["state"],
                          "mode": mode, "task_success": "not_scored"}, indent=2), flush=True)
    finally:
        await managed.close()
        # prepare/warmup errors can occur before ForceBench opens its preflight lifecycle.
        await transport.close()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mock", action="store_true", help="Synthetic policy, robot and cameras (default)")
    modes.add_argument("--dry-run", action="store_true", help="Real model server, synthetic robot/cameras")
    modes.add_argument("--hardware", action="store_true", help="Open configured robot/cameras, run ready pose and execute")
    parser.add_argument("--run-config", type=Path,
                        default=ROOT.parent / "ForceBenchWeb/configs/runs/peg_in_hole_v2_200sets_15hz_pi05.yaml")
    parser.add_argument("--pts-config", type=Path)
    parser.add_argument("--endpoint", default="ws://127.0.0.1:8025")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--session")
    parser.add_argument("--duration", type=float, default=10.0, help="Trial wall-time limit in seconds")
    parser.add_argument("--mock-latency", type=float, default=0.12)
    parser.add_argument("--warmup-timeout", type=float, default=300)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    if not 0 < args.duration < float("inf") or args.mock_latency < 0 or args.warmup_timeout <= 0:
        parser.error("duration and timeout must be positive; latency must be nonnegative")
    asyncio.run(run_trial(args))


if __name__ == "__main__":
    main()
