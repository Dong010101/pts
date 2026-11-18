"""Polish checkpoint timing/memory benchmark sharing the FreeSpeed frozen inputs."""

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    cfg = json.loads((run / "settings.json").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["cuda_visible_devices"]
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = cfg["xla_preallocate"]
    os.environ["OPENPI_DATA_HOME"] = cfg["shared_cache"]
    sys.path[:0] = [str(Path(cfg["openpi_snapshot"]) / "src"),
                    str(Path(cfg["openpi_snapshot"]) / "packages/openpi-client/src"),
                    str(Path(cfg["pts_snapshot"]) / "src")]
    spec = importlib.util.spec_from_file_location("benchmark_common", cfg["shared_benchmark_helpers"])
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    from pts.offline import CONDITIONS, execute_prefix
    from pts.openpi_policy import DecoderPolicy

    if args.condition not in (*CONDITIONS, "paired"):
        parser.error("unknown condition")
    conditions = CONDITIONS if args.condition == "paired" else (args.condition,)
    out = run / args.condition
    out.mkdir(exist_ok=False)
    directories = {c: out / c if args.condition == "paired" else out for c in conditions}
    if args.condition == "paired":
        for directory in directories.values():
            directory.mkdir()
    memory = common.GpuMemory(cfg["gpu_uuid"], cfg["memory_sampling_period_s"])
    memory.thread.start()
    try:
        began = time.perf_counter()
        import jax
        from openpi.policies import policy_config
        from openpi.training import config
        device = jax.devices()[0]
        if device.platform != "gpu":
            raise RuntimeError("Expected GPU")
        train = config.get_config(cfg["config"])
        if not train.model.pi05 or train.model.action_horizon != 50 or train.model.action_dim != 32:
            raise ValueError("Unexpected polish model contract")
        common.write_json(out / "launch.json", {
            "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "checkpoint": cfg["checkpoint"], "python": sys.executable, "device": str(device),
            "model": repr(train.model), "data": repr(train.data),
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "conditions": list(conditions), "one_loaded_policy": True,
            "sampling": {c: {"num_steps": 10 if c == "baseline_10step" else 1,
                             "noise": "frozen_gaussian" if c == "baseline_10step" else "zeros_50x32"}
                         for c in conditions},
        })
        print(f"[{args.condition}] loading shared checkpoint on {device}", flush=True)
        policy = policy_config.create_trained_policy(train, cfg["checkpoint"],
                    sample_kwargs={"num_steps": 10}, default_prompt=cfg["prompt"])
        decoder = DecoderPolicy(policy, horizon=50, action_dim=32, decoder="zero_noise_one_step")
        load_s = time.perf_counter() - began
        print(f"[{args.condition}] loaded in {load_s:.2f}s", flush=True)
        fixed = np.load(cfg["fixed_inputs"], allow_pickle=False)
        observations = [{"observation/state": state, "observation/global": fixed["global_images"][i],
                         "observation/left_wrist": fixed["wrist_images"][i], "prompt": cfg["prompt"]}
                        for i, state in enumerate(fixed["states"])]
        key, noise = jax.random.key(cfg["seed"]), fixed["noise"]

        def infer(index, condition):
            # Sequential calls in one thread; the weights and jitted model are shared.
            # The single-step/noise change is the PTS method, not a matched-noise ablation.
            policy._rng = key
            policy._sample_kwargs["num_steps"] = 10 if condition == "baseline_10step" else 1
            start = time.perf_counter_ns()
            response = (policy.infer(observations[index], noise=noise) if condition == "baseline_10step"
                        else decoder.infer(observations[index]))
            inferred = time.perf_counter_ns()
            actions, progress, diagnostic = execute_prefix(
                response["actions"], observations[index]["observation/state"], condition, cfg)
            ended = time.perf_counter_ns()
            return response["actions"], actions, progress, diagnostic, {
                "inference_ms": (inferred - start) / 1e6, "postprocess_ms": (ended - inferred) / 1e6,
                "total_ms": (ended - start) / 1e6, "model_reported_ms": response["policy_timing"]["infer_ms"],
            }

        memory.phase = "warmup"
        warmups = {c: [] for c in conditions}
        for repeat in range(cfg["warmup_rounds"]):
            for i in range(len(observations)):
                for c in conditions:
                    *_, timing = infer(i, c)
                    warmups[c].append({"round": repeat, "observation": i, **timing})
                    if repeat == 0 and i == 0:
                        print(f"[{args.condition}/{c}] first inference {timing['total_ms']:.2f}ms", flush=True)
        for c in conditions:
            common.write_json(directories[c] / "warmup.json", warmups[c])
        # Confirm the wrapper's zero-noise path preserves the original output pipeline.
        wrapper_equal = None
        if any(c != "baseline_10step" for c in conditions):
            policy._sample_kwargs["num_steps"] = 1
            policy._rng = key
            a = decoder.infer(observations[0])["actions"]
            policy._rng = key
            b = policy.infer(observations[0], noise=np.zeros((50, 32), np.float32))["actions"]
            wrapper_equal = bool(np.array_equal(a, b))
            if not wrapper_equal:
                raise AssertionError("PTS wrapper changed original postprocessing")
        memory.phase = "measure"
        before = device.memory_stats()
        records = {c: {"rows": [], "raw": [], "executed": [], "progress": [], "lengths": []} for c in conditions}
        for repeat in range(cfg["repetitions_per_observation"]):
            for i in range(len(observations)):
                offset = (repeat + i) % len(conditions)
                for order, c in enumerate(conditions[offset:] + conditions[:offset]):
                    prediction, actions, progress, diagnostic, timing = infer(i, c)
                    if not np.isfinite(prediction).all() or not np.isfinite(actions).all():
                        raise FloatingPointError("nonfinite output")
                    record = records[c]
                    n = len(actions)
                    padded = np.zeros((cfg["execution_horizon"], 7))
                    p = np.zeros(cfg["execution_horizon"])
                    padded[:n], p[:n] = actions, progress
                    record["raw"].append(prediction.copy())
                    record["executed"].append(padded)
                    record["progress"].append(p)
                    record["lengths"].append(n)
                    record["rows"].append({"repeat": repeat, "observation": i, "frame": cfg["frames"][i],
                                           "condition_order_in_pair": order, **timing, "pts": diagnostic})
        memory.sample()
        after = device.memory_stats()
        memory.phase = "save"
        common_report = {
            "status": "completed", "load_seconds": load_s,
            "nvml_total_process_peak_bytes": {phase: max((r["total_process_bytes"] for r in memory.records
                if r["phase"] == phase), default=0) for phase in ("load", "warmup", "measure")},
            "jax_memory_before_measurement": before, "jax_memory_after_measurement": after,
            "nvml_active_pids": sorted({p["pid"] for r in memory.records for p in r["nvml_processes"]}),
            "max_simultaneous_gpu_processes": max(len(r["nvml_processes"]) for r in memory.records),
            "memory_sampling_errors": memory.errors, "wrapper_equals_original_zero_noise_policy": wrapper_equal,
        }
        for c in conditions:
            r, directory = records[c], directories[c]
            np.savez_compressed(directory / "outputs.npz", raw_actions=np.stack(r["raw"]),
                executed_actions=np.stack(r["executed"]), progress=np.stack(r["progress"]),
                lengths=np.asarray(r["lengths"]), observation_indices=np.asarray([v["observation"] for v in r["rows"]]))
            (directory / "timings.jsonl").write_text("".join(json.dumps(row) + "\n" for row in r["rows"]))
            report = {**common_report, "condition": c, "num_inferences": len(r["rows"]),
                      "output_lengths": sorted(set(r["lengths"])),
                      "timing_ms": {k: common.stats([v[k] for v in r["rows"]]) for k in
                                    ("inference_ms", "postprocess_ms", "total_ms", "model_reported_ms")}}
            common.write_json(directory / "summary.json", report)
            print(f"[{c}] total median {report['timing_ms']['total_ms']['median']:.3f}ms", flush=True)
    except BaseException as error:
        common.write_json(out / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        memory.finish()
        (out / "gpu_memory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in memory.records))


if __name__ == "__main__":
    main()
