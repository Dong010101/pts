"""Checked metadata overlays for the original 15 Hz Flexiv checkpoints.

Only process-local dataclass copies are changed; shared OpenPI configs and model
transforms are left intact. Unknown configurations require explicit metadata.
"""

from dataclasses import replace
import json
import math
from pathlib import Path


TASKS = {
    "flip": ("pi05_flexiv_1arm_2cam_flip_fp2_v3_selected_fixed_15hz",
             "magikonki/flip_fp2_v3_selected_fixed_15hz"),
    "polish": ("pi05_flexiv_1arm_2cam_polish_curve_v4_15hz",
               "magikonki/polish_curve_v4_15hz"),
}
ACTION = "next_frame_absolute_tcp_pose_and_gripper_command[7]"


def adapt_config(config, checkpoint):
    task = next((task for task, (name, _) in TASKS.items() if name == config.name), None)
    if task is None:
        return config
    _, repo = TASKS[task]
    data = config.data
    if (type(data).__name__ != "FlexivSingleArm2CamDataConfig"
            or data.repo_id != repo or data.action_dim != 7 or not data.extra_delta_transform):
        raise ValueError("Flexiv training data contract differs from the verified 7D TCP configuration")
    if not config.model.pi05 or config.model.action_horizon != 50 or config.model.action_dim != 32:
        raise ValueError("expected pi0.5 with horizon 50 and padded action_dim 32")
    transforms = data.create(config.assets_dirs, config.model)
    from openpi.transforms import AbsoluteTCPPoseActions, DeltaTCPPoseActions
    absolute = [x for x in transforms.data_transforms.outputs if isinstance(x, AbsoluteTCPPoseActions)]
    delta = [x for x in transforms.data_transforms.inputs if isinstance(x, DeltaTCPPoseActions)]
    if (len(absolute) != 1 or len(delta) != 1
            or any(x.pos_dims != 3 or x.rotation_repr != "axis_angle" for x in absolute + delta)
            or transforms.asset_id != repo):
        raise ValueError("training transforms/assets do not restore absolute xyz/axis-angle targets")
    stats = Path(checkpoint) / "assets" / repo / "norm_stats.json"
    if not (Path(checkpoint) / "params").is_dir() or not stats.is_file():
        raise ValueError(f"checkpoint must contain params and task-specific statistics: {stats}")
    document = json.loads(stats.read_text())
    if not isinstance(document.get("norm_stats"), dict) or not {"state", "actions"} <= document["norm_stats"].keys():
        raise ValueError("checkpoint statistics must contain state and actions")
    metadata = dict(config.policy_metadata or {})
    for key, expected in {"action": ACTION, "control_frequency_hz": 15, "task": task}.items():
        if key in metadata and metadata[key] != expected:
            raise ValueError(f"conflicting training metadata for {key}: {metadata[key]!r}")
        metadata[key] = expected
    metadata.update(contract_source="checked_flexiv_15hz_overlay_v1", norm_stats_asset=repo,
                    output_action_dim=7, camera_order=["global", "wrist"])
    return replace(config, policy_metadata=metadata)


def load_config(name, checkpoint, *, action_space="absolute_tcp"):
    from openpi.training.config import get_config
    config = get_config(name)
    config = adapt_config(config, checkpoint) if name in {item[0] for item in TASKS.values()} else config
    metadata = dict(config.policy_metadata or {})
    if not getattr(config.model, "pi05", False):
        raise ValueError("this launcher requires a pi0.5 training configuration")
    if action_space == "absolute_tcp":
        if "absolute_tcp_pose" not in metadata.get("action", ""):
            raise ValueError("absolute_tcp requires a verified TCP action declaration")
        hz = float(metadata.get("control_frequency_hz", 0))
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError("absolute_tcp requires a positive control_frequency_hz declaration")
    return config
