from types import SimpleNamespace, ModuleType
import sys

import numpy as np
import pytest

from pts.openpi_policy import DecoderPolicy, create_decoder


def test_decoder_noise_includes_padding_and_preserves_postprocessing():
    class Policy:
        def infer(self, obs, noise):
            assert noise.shape == (50, 32)
            assert noise.dtype == np.float32 and np.count_nonzero(noise) == 0
            assert obs == {"prompt": "test"}
            return {"actions": np.full((50, 7), 42), "extra": "preserved"}
    wrapped = DecoderPolicy(Policy(), horizon=50, action_dim=32, decoder="zero_noise_one_step")
    a, b = wrapped.infer({"prompt": "test"}), wrapped.infer({"prompt": "test"})
    np.testing.assert_array_equal(a["actions"], b["actions"])
    assert a["extra"] == "preserved" and a["pts_timing"]["decode_ms"] >= 0


@pytest.mark.parametrize("decoder,steps", [("zero_noise_one_step", 1), ("stochastic_10_step", 10)])
def test_factory_sets_model_integration_steps(monkeypatch, decoder, steps):
    policy_config, training_config = ModuleType("policy_config"), ModuleType("training_config")
    config = SimpleNamespace(model=SimpleNamespace(pi05=True, action_horizon=50, action_dim=32),
                             policy_metadata={"action": "next_frame_absolute_tcp_pose_and_gripper_command[7]",
                                              "control_frequency_hz": 15})
    training_config.get_config = lambda _: config
    captured = {}
    def create(config_arg, checkpoint, **kwargs):
        captured.update(kwargs)
        assert config_arg is config and checkpoint == "/test/checkpoint"
        return SimpleNamespace(metadata={"original": 1})
    policy_config.create_trained_policy = create
    monkeypatch.setitem(sys.modules, "openpi.policies.policy_config", policy_config)
    monkeypatch.setitem(sys.modules, "openpi.training.config", training_config)
    wrapped = create_decoder("test", "/test/checkpoint", decoder=decoder)
    assert captured == {"sample_kwargs": {"num_steps": steps}}
    assert wrapped.metadata["pts"]["num_steps"] == steps
    assert wrapped.metadata["original"] == 1


def test_rejects_nonfinite_output():
    policy = SimpleNamespace(infer=lambda *a, **kw: {"actions": np.full((50, 7), np.nan)})
    with pytest.raises(ValueError, match="nonfinite"):
        DecoderPolicy(policy, horizon=50, action_dim=32, decoder="zero_noise_one_step").infer({})
