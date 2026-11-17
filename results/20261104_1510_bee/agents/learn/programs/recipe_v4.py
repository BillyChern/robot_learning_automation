"""Training recipe v4: vision-driven policy (no joint angles in the model input), wrap-safe deltas, H=30.

Finding (programs/vision_check.py on lora_v3/4500): with the same state, images from another episode (bee elsewhere)
give almost the same predicted chunk as the episode's own images -> the model replays trajectories from the joint
state (copycat / proprioception shortcut) and ignores where the bee is; on the robot it closed 20 cm from the bee.
Fix: the model sees only the gripper reading as state (no joint angles). Actions stay wrap-safe joint deltas
(wrap_pi(action - q)) + absolute gripper; the server returns these deltas unchanged and programs/rollout_v4.py adds
them to the measured joints. Init from lora_v3/4500 (wrap-safe deltas already learned)."""
import dataclasses
import pathlib

import numpy as np
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

from rlpj.vla import train_config as tc
from rlpj.vla.kinova_policy import KinovaInputs, KinovaOutputs

H = 30
STEPS = 12000
INIT = "/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/lora_v3/4500/params"


def wrap(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


@dataclasses.dataclass(frozen=True)
class WrapDelta(_transforms.DataTransformFn):
    def __call__(self, data):
        if "actions" not in data:
            return data
        a = np.array(data["actions"], np.float32)
        a[..., :6] = wrap(a[..., :6] - np.asarray(data["state"], np.float32)[..., None, :6])
        data["actions"] = a
        return data


@dataclasses.dataclass(frozen=True)
class GripperOnlyState(_transforms.DataTransformFn):
    def __call__(self, data):
        data["state"] = np.asarray(data["state"], np.float32)[..., 6:7]
        return data


@dataclasses.dataclass(frozen=True)
class VisionKinovaDataConfig(tc.LeRobotKinovaDataConfig):
    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        repack = _transforms.Group(inputs=[_transforms.RepackTransform(
            {"image": "image", "wrist_image": "wrist_image", "state": "state", "actions": "actions", "prompt": "prompt"})])
        data_transforms = _transforms.Group(inputs=[KinovaInputs(model_type=model_config.model_type), WrapDelta(), GripperOnlyState()],
                                            outputs=[KinovaOutputs()])
        model_transforms = _config.ModelTransformFactory()(model_config)
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), repack_transforms=repack,
                                   data_transforms=data_transforms, model_transforms=model_transforms)


def _model():
    return pi0_config.Pi0Config(pi05=True, action_horizon=H, paligemma_variant="gemma_2b_lora",
                                action_expert_variant="gemma_300m_lora")


def configs(run_dir=None, repo_id=None):
    base = tc.configs(run_dir, repo_id)[tc.LORA_NAME]
    data = VisionKinovaDataConfig(repo_id=repo_id or tc.DEFAULT_REPO_ID, base_config=_config.DataConfig(prompt_from_task=True))
    meta = dict(tc.POLICY_METADATA, action_horizon=H, state="gripper reading / 100 only (joints ignored)",
                actions="q1..q6 DELTA rad wrap_pi(target - q_now) (rollout adds q_now), gripper command 0..1")
    name = "pi05_kinova_lora_v4"
    return {name: dataclasses.replace(
        base, name=name, data=data, model=_model(), freeze_filter=_model().get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(INIT),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=300, peak_lr=5e-5, decay_steps=STEPS, decay_lr=5e-6),
        num_train_steps=STEPS, policy_metadata=meta)}
