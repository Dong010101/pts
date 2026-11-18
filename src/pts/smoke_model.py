"""Validate the real decoder with synthetic inputs, without importing robot SDKs."""

import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import time

import numpy as np

from .openpi_policy import create_decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_smoke.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    policy = create_decoder(args.config, args.checkpoint)
    assert policy.policy._sample_kwargs["num_steps"] == 1
    observation = {
        "observation/global": np.zeros((224, 224, 3), np.uint8),
        "observation/left_wrist": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros(7, np.float32),
        "prompt": "pick the peg to the hole and insert it fully",
    }
    first = policy.infer(observation)
    second = policy.infer(observation)
    direct = policy.policy.infer(observation, noise=np.zeros((policy.horizon, policy.action_dim), np.float32))
    np.testing.assert_array_equal(first["actions"], second["actions"])
    np.testing.assert_array_equal(second["actions"], direct["actions"])
    report = {
        "validation": "real model, synthetic black images/zero pose; no robot or task-success evaluation",
        "metadata": policy.metadata,
        "shape": list(first["actions"].shape),
        "exact_repeat": True, "matches_original_postprocessing": True,
        "cold_decode_ms": first["pts_timing"]["decode_ms"],
        "warm_decode_ms": second["pts_timing"]["decode_ms"],
        "wall_s": time.time(),
        "versions": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "torch", "numpy", "scipy")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
