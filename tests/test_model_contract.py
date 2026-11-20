from dataclasses import dataclass, field
import json
from types import ModuleType, SimpleNamespace
import sys

import pytest

from pts.model_contract import ACTION, TASKS, adapt_config, load_config


@pytest.fixture
def contract(monkeypatch, tmp_path):
    @dataclass
    class AbsoluteTCPPoseActions:
        pos_dims: int = 3
        rotation_repr: str = "axis_angle"

    @dataclass
    class DeltaTCPPoseActions(AbsoluteTCPPoseActions):
        pass

    module = ModuleType("openpi.transforms")
    module.AbsoluteTCPPoseActions = AbsoluteTCPPoseActions
    module.DeltaTCPPoseActions = DeltaTCPPoseActions
    monkeypatch.setitem(sys.modules, "openpi.transforms", module)

    @dataclass
    class FlexivSingleArm2CamDataConfig:
        repo_id: str = TASKS["flip"][1]
        action_dim: int = 7
        extra_delta_transform: bool = True
        absolute: bool = True

        def create(self, assets, model):
            return SimpleNamespace(asset_id=self.repo_id, data_transforms=SimpleNamespace(
                inputs=[DeltaTCPPoseActions()],
                outputs=[AbsoluteTCPPoseActions()] if self.absolute else []))

    @dataclass
    class Config:
        name: str = TASKS["flip"][0]
        data: object = field(default_factory=FlexivSingleArm2CamDataConfig)
        model: object = field(default_factory=lambda: SimpleNamespace(pi05=True, action_horizon=50, action_dim=32))
        policy_metadata: object = None
        assets_dirs: object = None

    config = Config()
    (tmp_path / "params").mkdir()
    stats = tmp_path / "assets" / config.data.repo_id / "norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(json.dumps({"norm_stats": {"state": {}, "actions": {}}}))
    return config, tmp_path


def test_known_checkpoint_receives_metadata_without_mutating_training_config(contract):
    config, checkpoint = contract
    adapted = adapt_config(config, checkpoint)
    assert config.policy_metadata is None
    assert adapted.data is config.data and adapted.model is config.model
    assert adapted.policy_metadata["task"] == "flip"
    assert adapted.policy_metadata["action"] == ACTION
    assert adapted.policy_metadata["control_frequency_hz"] == 15


def test_wrong_task_statistics_are_rejected(contract):
    config, checkpoint = contract
    stats = checkpoint / "assets" / config.data.repo_id / "norm_stats.json"
    stats.unlink()
    with pytest.raises(ValueError, match="task-specific statistics"):
        adapt_config(config, checkpoint)


def test_conflicting_frequency_or_missing_inverse_transform_is_rejected(contract):
    config, checkpoint = contract
    config.policy_metadata = {"control_frequency_hz": 20}
    with pytest.raises(ValueError, match="conflicting"):
        adapt_config(config, checkpoint)
    config.policy_metadata = None
    config.data.absolute = False
    with pytest.raises(ValueError, match="restore absolute"):
        adapt_config(config, checkpoint)


def test_unknown_config_does_not_receive_assumed_tcp_metadata(monkeypatch):
    training = ModuleType("openpi.training.config")
    training.get_config = lambda _: SimpleNamespace(model=SimpleNamespace(pi05=True), policy_metadata=None)
    monkeypatch.setitem(sys.modules, "openpi.training.config", training)
    with pytest.raises(ValueError, match="verified TCP"):
        load_config("unknown", "/unused")
