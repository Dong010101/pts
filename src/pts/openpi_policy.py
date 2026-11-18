"""R1 decoder wrapper; the model's own postprocessing remains authoritative."""

import time
import numpy as np


class DecoderPolicy:
    def __init__(self, policy, *, horizon, action_dim, decoder, metadata=None):
        if decoder not in {"zero_noise_one_step", "stochastic_10_step"}:
            raise ValueError("unsupported decoder")
        self.policy = policy
        self.horizon, self.action_dim = int(horizon), int(action_dim)
        self.decoder = decoder
        self.metadata = dict(metadata or {})

    def infer(self, observation):
        started = time.monotonic()
        if self.decoder == "zero_noise_one_step":
            # Zero in normalized latent space, including all padded dimensions.
            noise = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
            result = self.policy.infer(observation, noise=noise)
        else:
            result = self.policy.infer(observation)
        result = dict(result)
        actions = np.asarray(result["actions"])
        if actions.ndim != 2 or len(actions) != self.horizon or not np.all(np.isfinite(actions)):
            raise ValueError("model returned a malformed or nonfinite action chunk")
        # OpenPI returns host numpy arrays, so this includes GPU completion and
        # output transforms, unlike a timer around an asynchronous JAX dispatch.
        result["pts_timing"] = {"decode_ms": (time.monotonic() - started) * 1000}
        return result

    def reset(self):
        reset = getattr(self.policy, "reset", None)
        if reset:
            reset()


def create_decoder(config_name, checkpoint, *, decoder="zero_noise_one_step", action_space="absolute_tcp"):
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config

    config = get_config(config_name)
    if decoder not in {"zero_noise_one_step", "stochastic_10_step"}:
        raise ValueError("unsupported decoder")
    if action_space not in {"absolute_tcp", "relative_increment"}:
        raise ValueError("unsupported action space")
    if not getattr(config.model, "pi05", False):
        raise ValueError("this launcher requires a pi0.5 training configuration")
    if action_space == "absolute_tcp" and "absolute_tcp_pose" not in config.policy_metadata.get("action", ""):
        raise ValueError("absolute_tcp requires a matching TCP policy action declaration in the OpenPI training config")
    num_steps = 1 if decoder == "zero_noise_one_step" else 10
    policy = create_trained_policy(config, checkpoint, sample_kwargs={"num_steps": num_steps})
    metadata = {
        **policy.metadata,
        "pts": {
            "decoder": decoder, "num_steps": num_steps,
            "action_horizon": config.model.action_horizon,
            "action_dim": config.model.action_dim,
            "action_space": action_space,
            "config": config_name, "checkpoint": str(checkpoint),
        },
    }
    return DecoderPolicy(policy, horizon=config.model.action_horizon, action_dim=config.model.action_dim,
                         decoder=decoder, metadata=metadata)

